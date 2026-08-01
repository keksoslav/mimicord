from __future__ import annotations

import json

from mimicord.llm.base import Provider, json_call

REDUCE_SYSTEM = """\
You are consolidating many per-conversation style analyses of how {name} \
writes on Discord into one profile. The analyses may disagree because people \
talk differently in different channels; resolve contradictions by noting \
where each register shows up instead of averaging them away.

Return only a JSON object with the same keys as the inputs (tone, formality, \
languages, code_switching, verbal_tics, typical_openers, emoji_usage, \
humor_style, topics, opinions, relationships, notable_quotes, reply_style) \
plus:
  "register_shifts": ["where and how their style changes, if it does"],
  "confidence_notes": ["anything seen in only a few chunks, flag it"]

Merge duplicates, keep exact spellings of tics and quotes, prefer traits that \
recur across many chunks. Keep notable_quotes to the 10 most characteristic. \
No commentary outside the JSON."""

REDUCE_MAX_TOKENS = 4000


def reduce_profiles(chunk_results: list[dict], provider: Provider, name: str) -> dict:
    payload = json.dumps(chunk_results, ensure_ascii=False)
    return json_call(
        provider,
        system=REDUCE_SYSTEM.format(name=name),
        user=payload,
        max_tokens=REDUCE_MAX_TOKENS,
    )
