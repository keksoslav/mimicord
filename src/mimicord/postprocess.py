from __future__ import annotations

import re

_THINKING_RE = re.compile(
    r"<(thinking|reasoning|reflection|internal)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_STRAY_TAG_RE = re.compile(r"</?(thinking|reasoning|reflection|internal)>", re.IGNORECASE)

_AI_ISMS = (
    re.compile(r"^as an ai\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^i'?m an ai\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^let me know if .*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(is there anything else|hope (this|that) helps).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^i('| a)?m happy to help.*$", re.IGNORECASE | re.MULTILINE),
)


def clean(
    text: str,
    persona_name: str | None = None,
    context_authors: "set[str] | None" = None,
) -> str:
    """Strip model artifacts that break the illusion."""
    text = _THINKING_RE.sub("", text)
    text = _STRAY_TAG_RE.sub("", text)
    for pattern in _AI_ISMS:
        text = pattern.sub("", text)
    text = text.strip()
    if persona_name:
        echo = re.match(rf"^\s*{re.escape(persona_name)}\s*:\s*", text, re.IGNORECASE)
        if echo:
            text = text[echo.end() :]
    # the prompt shows chat as "author: message", and the model leaks that
    # format back by opening with whoever it is answering. real chat uses an
    # @mention or nothing at all
    if context_authors:
        names = "|".join(
            re.escape(a) for a in sorted(context_authors, key=len, reverse=True) if a
        )
        if names:
            text = re.sub(
                rf"^\s*(?:{names})\s*[:,]?\s+(?=\S)", "", text, count=1, flags=re.IGNORECASE
            )
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def split_bursts(text: str, max_burst: int = 3) -> list[str]:
    """One reply can come out as a few short messages, like humans type."""
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(parts) > max_burst:
        head = parts[: max_burst - 1]
        tail = " ".join(parts[max_burst - 1 :])
        parts = head + [tail]
    return parts


def apply(
    text: str,
    *,
    persona_name: str | None = None,
    stats: dict | None = None,
    max_burst: int = 3,
    context_authors: "set[str] | None" = None,
) -> list[str]:
    text = clean(text, persona_name, context_authors)
    stats = stats or {}
    capitalization = stats.get("capitalization", {})
    punctuation = stats.get("punctuation", {})
    if (
        capitalization.get("starts_lowercase", 0) > 0.8
        and text
        and text[0].isupper()
        and not text[:2].isupper()  # keep ALL CAPS words intact
    ):
        text = text[0].lower() + text[1:]
    if (
        punctuation.get("ends_with_period", 1.0) < 0.1
        and text.endswith(".")
        and not text.endswith("..")
    ):
        text = text[:-1]
    # the bot client also blocks mass pings via allowed_mentions, this is belt and suspenders
    text = text.replace("@everyone", "everyone").replace("@here", "here")
    return split_bursts(text, max_burst)
