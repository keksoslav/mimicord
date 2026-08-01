from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class ProviderError(Exception):
    pass


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    # marks the last message of the stable prompt prefix so providers with
    # prefix caching (anthropic) can extend the cache over the few-shots
    cache_boundary: bool = False


class Provider(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float | None = None,
    ) -> str: ...


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
