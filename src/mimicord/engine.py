from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from mimicord import postprocess
from mimicord.config import PersonaConfig, load_config
from mimicord.llm.base import ChatMessage, Provider
from mimicord.llm.factory import get_provider
from mimicord.paths import PersonaPaths

log = logging.getLogger(__name__)


_MENTION_RE = re.compile(r"<@[!&]?\d+>|<#\d+>|@[\w.]+")
# below this much real text, a memory lookup just matches other people's
# pings and feeds the model its own noise back
MIN_QUERY_CHARS = 12
# a retrieved window shorter than this once names are removed is ping spam
MIN_MEMORY_CHARS = 40


@dataclass
class ContextMessage:
    author: str
    content: str


def memory_query(
    context: list["ContextMessage"],
    window: int = 5,
    aliases: "set[str] | None" = None,
) -> str:
    """Recent context with mentions and the persona's own names stripped, or
    empty when nothing substantive is left. Being summoned by name is not a
    topic, and looking it up just retrieves everyone else's pings."""
    alias_re = None
    if aliases:
        pattern = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True) if a)
        if pattern:
            alias_re = re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)

    lines = []
    substantive = 0
    for message in context[-window:]:
        text = _MENTION_RE.sub(" ", message.content)
        if alias_re:
            text = alias_re.sub(" ", text)
        text = " ".join(text.split())  # collapse the gaps stripping leaves
        if not text:
            continue
        substantive += len(text)
        lines.append(f"{message.author}: {text}")
    if substantive < MIN_QUERY_CHARS:
        return ""
    return "\n".join(lines)


def is_useful_memory(text: str, aliases: "set[str] | None" = None) -> bool:
    """Drop retrieved windows that are mostly people summoning each other.

    Those match strongly on a name in the query and carry no information,
    and a promptful of them convinces the model the topic is the name.
    """
    stripped = _MENTION_RE.sub(" ", text)
    if aliases:
        pattern = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True) if a)
        if pattern:
            stripped = re.sub(rf"\b(?:{pattern})\b", " ", stripped, flags=re.IGNORECASE)
    # the rendering joins messages with " / " and prefixes an author per line
    stripped = re.sub(r"\b\w+:", " ", stripped).replace("/", " ")
    return len(" ".join(stripped.split())) >= MIN_MEMORY_CHARS


class PersonaEngine:
    """Turns recent chat context into an in-character reply.

    Prompt layout keeps the expensive parts byte-stable so provider prompt
    caches can hit: the system prompt (persona.md) and the few-shot examples
    never change between calls; memories and live context ride in the final
    user message only.
    """

    def __init__(self, name: str, *, rag_enabled: bool | None = None) -> None:
        self.paths = PersonaPaths.for_persona(name)
        self.config: PersonaConfig = load_config(self.paths.config)
        if not self.paths.persona_md.is_file():
            raise FileNotFoundError(
                f"{self.paths.persona_md} missing; write one by hand or run mimicord compile"
            )
        self.system = self.paths.persona_md.read_text(encoding="utf-8").strip()
        self.examples = self._load_examples()
        self.stats = self._load_json(self.paths.stats)
        self.provider: Provider = get_provider(self.config.llm)
        want_rag = self.config.rag.enabled if rag_enabled is None else rag_enabled
        self.rag = self._load_rag() if want_rag else None
        # every way people address this persona: its name, the account names
        # it was built from, and whatever nicknames summon it. none of these
        # are content worth searching memories on
        self._aliases = {
            self.config.name,
            *self.config.target.author_names,
            *self.config.discord.trigger_keywords,
        }
        self.last_prompt: tuple[str, list[ChatMessage]] | None = None

    @staticmethod
    def _load_json(path: Path):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _load_examples(self) -> list[ChatMessage]:
        data = self._load_json(self.paths.examples)
        if not data:
            return []
        messages: list[ChatMessage] = []
        for example in data.get("examples", []):
            transcript = "\n".join(
                f"{m['author']}: {m['content']}" for m in example["context"]
            )
            reply = example["reply"]
            if isinstance(reply, list):
                reply = "\n".join(reply)
            messages.append(ChatMessage("user", f"[chat]\n{transcript}\n[/chat]"))
            messages.append(ChatMessage("assistant", reply))
        if messages:
            messages[-1].cache_boundary = True
        return messages

    def _load_rag(self):
        try:
            from mimicord.rag import Memories
        except ImportError:
            return None
        if not self.paths.chroma_dir.is_dir():
            log.debug("no chroma index at %s, running without memories", self.paths.chroma_dir)
            return None
        return Memories(self.paths, self.config.rag)

    def reply(self, context: list[ContextMessage]) -> list[str]:
        transcript = "\n".join(f"{m.author}: {m.content}" for m in context)
        sections: list[str] = []
        if self.rag is not None:
            # query with just the tail of the conversation, that is what the
            # reply will actually be about
            query_text = memory_query(context, aliases=self._aliases)
            memories = self.rag.query(query_text) if query_text else []
            memories = [m for m in memories if is_useful_memory(m, self._aliases)]
            if memories:
                lines = "\n".join(f"- {m}" for m in memories)
                sections.append(f"[memories]\n{lines}\n[/memories]")
        sections.append(f"[chat]\n{transcript}\n[/chat]")
        messages = [*self.examples, ChatMessage("user", "\n".join(sections))]
        self.last_prompt = (self.system, messages)
        raw = self.provider.complete(
            system=self.system,
            messages=messages,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
        )
        bursts = postprocess.apply(
            raw,
            persona_name=self.config.name,
            stats=self.stats,
            max_burst=self.config.style.max_burst,
            context_authors={m.author for m in context},
        )
        if not bursts:
            log.warning("model returned nothing usable, raw was %r", raw[:200])
        return bursts
