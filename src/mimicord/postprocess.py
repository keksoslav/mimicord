from __future__ import annotations

import re

_THINKING_RE = re.compile(
    r"<(thinking|reasoning|reflection|internal)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_STRAY_TAG_RE = re.compile(r"</?(thinking|reasoning|reflection|internal)>", re.IGNORECASE)

# a burst that is only this tag is sent as an image instead of as text
REACTION_RE = re.compile(r"^\[gif:([a-zA-Z0-9_-]+)\]$")


def reaction_name(burst: str) -> str | None:
    match = REACTION_RE.match(burst.strip())
    return match.group(1).lower() if match else None


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


# echoing something this short back is just normal agreement ("kk", "ja",
# "true"). above it, a verbatim match is parroting
ECHO_MIN_CHARS = 8


def drop_echoes(
    bursts: list[str],
    *,
    said_by_me: "list[str] | None" = None,
    said_by_others: "list[str] | None" = None,
) -> list[str]:
    """Remove lines that are already in the channel word for word.

    Two tells, one fix. Repeating your own last message is the first, and
    parroting back what somebody just said is the second. Both are exact
    string matches, and both have been asked for in the prompt and ignored,
    so they get handled here where they cannot be argued with.

    Also dedups a reply against itself, since a burst that says the same
    thing twice reads exactly as badly.
    """
    seen = {t.strip().casefold() for t in (said_by_me or []) if t.strip()}
    mirrored = {
        t.strip().casefold()
        for t in (said_by_others or [])
        if len(t.strip()) >= ECHO_MIN_CHARS
    }
    kept: list[str] = []
    for burst in bursts:
        key = burst.strip().casefold()
        if not key or key in seen or key in mirrored:
            continue
        kept.append(burst)
        seen.add(key)
    return kept


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
