from __future__ import annotations

import json

from mimicord.llm.base import ChatMessage, Provider
from mimicord.rules import HARD_RULES

PERSONA_SYSTEM = """\
You write persona instructions that make a language model convincingly play \
one specific real person in casual Discord chats. You get a consolidated \
style profile (built from their real messages) and measured writing stats. \
Write in second person ("you are..."), concrete and specific, quoting exact \
words and spellings from the profile. Do not invent biography beyond what \
the profile supports.

Write these markdown sections and nothing else:
# Persona: {name}
(one short paragraph: who this is as observed in chat, their vibe)
## Voice and tone
## Language
(which languages, when they mix or switch)
## Vocabulary and tics
(exact spellings, openers, fillers)
## Topics and opinions
## People
(who they talk to and the dynamic with each)

Do not write formatting rules or forbidden-behavior rules, those get \
appended separately from measured data."""

PERSONA_MAX_TOKENS = 3000


def measured_rules(stats: dict) -> list[str]:
    """Formatting facts derived in code so the numbers are ground truth."""
    rules: list[str] = []
    if not stats or stats.get("message_count", 0) == 0:
        return rules
    length = stats.get("length", {})
    if length:
        rules.append(
            f"typical message is around {length.get('median_chars', '?')} characters, "
            f"rarely over {length.get('p90_chars', '?')}"
        )
    burst = stats.get("burst", {})
    p_multi = burst.get("p_multi", 0)
    if p_multi >= 0.25:
        rules.append(
            f"often splits a thought into several short messages in a row "
            f"({p_multi * 100:.0f}% of turns, about {burst.get('avg_burst_len')} messages)"
        )
    elif burst:
        rules.append("almost always answers in a single message")
    capitalization = stats.get("capitalization", {})
    lower = capitalization.get("starts_lowercase", 0)
    if lower >= 0.7:
        rules.append(f"starts lowercase ({lower * 100:.0f}% of messages)")
    elif lower <= 0.3:
        rules.append("usually starts with a capital letter")
    punctuation = stats.get("punctuation", {})
    period = punctuation.get("ends_with_period", 0)
    if period <= 0.15:
        rules.append("almost never ends a message with a period")
    emoji = stats.get("emoji", {})
    per_message = emoji.get("per_message", 0)
    top = [e["emoji"] for e in emoji.get("top", [])[:5]]
    if per_message <= 0.05:
        rules.append("basically never uses emoji")
    elif top:
        rules.append(
            f"emoji in about {per_message * 100:.0f}% of messages, usually {' '.join(top)}"
        )
    misc = stats.get("misc", {})
    if misc.get("markdown_rate", 0) <= 0.02:
        rules.append("never uses markdown formatting")
    language = stats.get("language", {})
    diacritics = language.get("slovene_diacritics_rate")
    if diacritics is not None:
        rules.append(
            f"{diacritics * 100:.0f}% of messages contain Slovene diacritics "
            "(match this, do not over- or under-correct spelling)"
        )
    return rules


def compile_persona(profile: dict, stats: dict, provider: Provider, name: str) -> str:
    user = (
        "Style profile:\n"
        + json.dumps(profile, ensure_ascii=False, indent=1)
        + "\n\nMeasured stats:\n"
        + json.dumps(stats, ensure_ascii=False, indent=1)
    )
    body = provider.complete(
        system=PERSONA_SYSTEM.format(name=name),
        messages=[ChatMessage("user", user)],
        max_tokens=PERSONA_MAX_TOKENS,
    ).strip()

    sections = [body]
    measured = measured_rules(stats)
    if measured:
        sections.append(
            "## Formatting habits (measured)\n" + "\n".join(f"- {r}" for r in measured)
        )
    # the same block the engine appends at prompt time, written here so the
    # file reads as a complete document. edits to it are ignored, the engine
    # uses its own copy: put persona specific rules in extra.md
    sections.append(HARD_RULES)
    return "\n\n".join(sections) + "\n"
