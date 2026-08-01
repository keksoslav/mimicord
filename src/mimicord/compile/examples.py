from __future__ import annotations

import json
import logging
from datetime import datetime

from mimicord.llm.base import Provider, ProviderError, json_call
from mimicord.store import Store

log = logging.getLogger(__name__)

BURST_GAP_SECONDS = 60.0
CONTEXT_WINDOW_SECONDS = 300.0
MAX_CONTEXT = 4
CANDIDATE_LIMIT = 100

CURATE_SYSTEM = """\
You are picking few-shot examples that teach a model to chat exactly like \
{name}. You get numbered candidate exchanges (context from others, then \
{name}'s real reply). Pick the {count} best ones: maximally characteristic \
of their voice, diverse in topic and mood, both languages if they mix, \
including some multi-message replies. Avoid near-duplicates and boring \
one-word acknowledgements unless that IS their style.

Return only JSON: {{"selected": [indices]}}"""

CURATE_MAX_TOKENS = 1000


def _bursts(messages) -> list[tuple[int, list]]:
    """(start_index, run) for every consecutive target run in a timeline."""
    runs = []
    run: list = []
    start = 0
    last_ts = None
    for position, message in enumerate(messages):
        ts = datetime.fromisoformat(message.timestamp)
        if message.is_target:
            gap_ok = (
                last_ts is not None
                and (ts - last_ts).total_seconds() <= BURST_GAP_SECONDS
            )
            if run and gap_ok:
                run.append(message)
            else:
                if run:
                    runs.append((start, run))
                run = [message]
                start = position
            last_ts = ts
        else:
            if run:
                runs.append((start, run))
                run = []
            last_ts = None
    if run:
        runs.append((start, run))
    return runs


def candidate_pairs(store: Store, stats: dict, limit: int = CANDIDATE_LIMIT) -> list[dict]:
    length = stats.get("length", {}) if stats else {}
    max_reply_chars = max(200, int(length.get("p90_chars", 100)) * 2)

    per_channel: dict[str, list[dict]] = {}
    for channel in store.channels():
        if channel["target"] == 0:
            continue
        messages = store.channel_messages(channel["channel_id"])
        has_others = any(not m.is_target for m in messages)
        pairs = []
        for start, run in _bursts(messages):
            reply = [m.content for m in run if m.content]
            if not reply:
                continue
            joined = "\n".join(reply)
            if len(joined) > max_reply_chars or joined.startswith("http"):
                continue
            burst_ts = datetime.fromisoformat(run[0].timestamp)
            context = []
            for message in reversed(messages[max(0, start - MAX_CONTEXT) : start]):
                if not message.content:
                    continue
                age = (burst_ts - datetime.fromisoformat(message.timestamp)).total_seconds()
                if age > CONTEXT_WINDOW_SECONDS:
                    break
                context.append({"author": message.author_name, "content": message.content})
            context.reverse()
            # solo channels (data package only) have no context to give
            if has_others and not any(
                c["author"] != run[0].author_name for c in context
            ):
                continue
            pairs.append({"context": context, "reply": reply})
        per_channel[str(channel["channel_id"])] = pairs

    # round robin across channels so one busy channel does not dominate
    result: list[dict] = []
    queues = [q for q in per_channel.values() if q]
    while queues and len(result) < limit:
        for queue in list(queues):
            if not queue:
                queues.remove(queue)
                continue
            result.append(queue.pop(0))
            if len(result) >= limit:
                break
    return result


def curate(candidates: list[dict], provider: Provider, name: str, count: int) -> list[dict]:
    numbered = [
        {"index": position, **candidate} for position, candidate in enumerate(candidates)
    ]
    try:
        data = json_call(
            provider,
            system=CURATE_SYSTEM.format(name=name, count=count),
            user=json.dumps(numbered, ensure_ascii=False),
            max_tokens=CURATE_MAX_TOKENS,
        )
        selected = [
            candidates[i]
            for i in data.get("selected", [])
            if isinstance(i, int) and 0 <= i < len(candidates)
        ]
    except ProviderError as error:
        log.warning("curation call failed (%s), falling back to a spread", error)
        selected = []
    if not selected:
        step = max(1, len(candidates) // count)
        selected = candidates[::step][:count]
    return selected[:count]


def build_examples(
    store: Store, stats: dict, provider: Provider, name: str, count: int = 20
) -> dict:
    candidates = candidate_pairs(store, stats)
    if not candidates:
        return {"examples": []}
    if len(candidates) <= count:
        selected = candidates
    else:
        selected = curate(candidates, provider, name, count)
    return {"examples": selected}
