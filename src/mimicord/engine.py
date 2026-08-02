from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mimicord import postprocess, rules
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
    images: list = field(default_factory=list)

    def render(self) -> str:
        if self.content:
            return f"{self.author}: {self.content}"
        return f"{self.author}: (posted an image)"


def collect_images(context: list["ContextMessage"], cfg) -> list:
    """The newest few pictures, newest first, capped.

    Walking the whole 25 message buffer would re-send the same meme on every
    reply for the rest of the conversation, which is how a cheap feature
    becomes an expensive one.
    """
    if not cfg.enabled or cfg.max_images <= 0:
        return []
    found: list = []
    for message in reversed(context[-cfg.lookback :]):
        for image in message.images:
            found.append(image)
            if len(found) >= cfg.max_images:
                return found
    return found


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


def now_section(moment: "datetime") -> str:
    """Tell him what day and time it is.

    He has no clock otherwise, so he fills the gap with whatever sounds
    plausible: being knackered after a shift, on a Sunday. Machine local
    time, since the persona lives wherever the bot is running.
    """
    return (
        f"[now]\n{moment:%A} {moment.day} {moment:%B %Y}, {moment:%H:%M}\n[/now]"
    )


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
        persona = self.paths.persona_md.read_text(encoding="utf-8").strip()
        self.system = rules.strip_rules(persona)
        # things the chat logs could never reveal, like his own full name
        if self.paths.extra.is_file():
            extra = self.paths.extra.read_text(encoding="utf-8").strip()
            if extra:
                self.system = f"{self.system}\n\n{extra}"
        reactions = self._reaction_block()
        if reactions:
            self.system = f"{self.system}\n\n{reactions}"
        # last, always: whatever the model reads most recently carries the most
        # weight, and these are the lines that must not bend
        self.system = f"{self.system}\n\n{rules.HARD_RULES}"
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

    def _reaction_block(self) -> str:
        if not self.config.reactions:
            return ""
        lines = []
        for reaction in self.config.reactions:
            when = f" {reaction.when}" if reaction.when else ""
            lines.append(f"- [gif:{reaction.name}]{when}")
        return (
            "## Reactions\n"
            "You can send a saved image instead of typing. Put the tag on a "
            "line by itself with nothing else on that line, either alone or "
            "after a message. Only these tags exist, never invent others, and "
            "never describe the image in words.\n" + "\n".join(lines)
        )

    def find_reaction(self, name: str):
        for reaction in self.config.reactions:
            if reaction.name == name.lower():
                return reaction
        return None

    def reaction_path(self, reaction) -> Path:
        return self.paths.media_dir / reaction.file

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

    def reply(self, context: list[ContextMessage], direction: str = "") -> list[str]:
        """Answer the recent chat. direction is an out of character nudge for
        the times nobody said anything to answer, like breaking a silence."""
        transcript = "\n".join(m.render() for m in context)
        # everything below rides in the final user message, never the system
        # prompt: a clock in a cached prefix would break the cache every minute
        sections: list[str] = [now_section(datetime.now().astimezone())]
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
        if direction:
            sections.append(f"[direction]\n{direction}\n[/direction]")
        images = collect_images(context, self.config.vision)
        if images:
            log.debug(
                "sending %d image(s), about %d tokens",
                len(images),
                sum(i.tokens for i in images),
            )
        messages = [
            *self.examples,
            ChatMessage("user", "\n".join(sections), images=images),
        ]
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
