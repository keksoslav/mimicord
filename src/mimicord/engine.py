from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mimicord import postprocess
from mimicord.config import PersonaConfig, load_config
from mimicord.llm.base import ChatMessage, Provider
from mimicord.llm.factory import get_provider
from mimicord.paths import PersonaPaths

log = logging.getLogger(__name__)


@dataclass
class ContextMessage:
    author: str
    content: str


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
            query_text = "\n".join(f"{m.author}: {m.content}" for m in context[-5:])
            memories = self.rag.query(query_text)
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
        return postprocess.apply(
            raw,
            persona_name=self.config.name,
            stats=self.stats,
            max_burst=self.config.style.max_burst,
        )
