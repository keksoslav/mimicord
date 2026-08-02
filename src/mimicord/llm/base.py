from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mimicord.vision import Image


class ProviderError(Exception):
    pass


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    # marks the last message of the stable prompt prefix so providers with
    # prefix caching (anthropic) can extend the cache over the few-shots
    cache_boundary: bool = False
    # pictures ride on the message they were posted with, which is only ever
    # the live one, never a few-shot, so the cached prefix stays byte stable
    images: list["Image"] = field(default_factory=list)


class Provider(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float | None = None,
    ) -> str: ...


def json_call(provider: Provider, *, system: str, user: str, max_tokens: int):
    """One JSON-returning call with a single repair retry on parse failure."""
    raw = provider.complete(
        system=system, messages=[ChatMessage("user", user)], max_tokens=max_tokens
    )
    try:
        return parse_json_lenient(raw)
    except ProviderError as error:
        repair = provider.complete(
            system=system,
            messages=[
                ChatMessage("user", user),
                ChatMessage("assistant", raw),
                ChatMessage(
                    "user",
                    f"That was not valid JSON ({error}). "
                    "Send only the corrected JSON, nothing else.",
                ),
            ],
            max_tokens=max_tokens,
        )
        return parse_json_lenient(repair)


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$", re.MULTILINE)


def parse_json_lenient(text: str) -> Any:
    """Parse JSON that may arrive wrapped in markdown fences or prose."""
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ProviderError(f"model did not return parseable JSON: {text[:200]!r}")
