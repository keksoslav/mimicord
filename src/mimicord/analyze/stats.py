from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from mimicord.store import Message, Store

# BMP symbols + the main supplementary emoji planes, close enough for stats
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001fbff☀-➿←-⇿⬀-⯿]"
)
_CUSTOM_EMOJI_RE = re.compile(r":[a-zA-Z0-9_]{2,}:")
_SLOVENE_RE = re.compile(r"[čšžČŠŽ]")  # czsCZS with carons
_MARKDOWN_RE = re.compile(r"```|\*\*|__|^#+\s|^[-*]\s", re.MULTILINE)

BURST_GAP_SECONDS = 60.0


def _percentile(sorted_values: list, fraction: float):
    if not sorted_values:
        return 0
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


def _rate(count: int, total: int) -> float:
    return round(count / total, 3) if total else 0.0


def _burst_stats(store: Store) -> dict:
    """Group consecutive target messages per channel: same author run with
    small gaps and nobody in between counts as one typing burst."""
    burst_lengths: list[int] = []
    for channel in store.channels():
        run = 0
        last_ts: datetime | None = None
        for message in store.channel_messages(channel["channel_id"]):
            ts = datetime.fromisoformat(message.timestamp)
            if message.is_target:
                gap_ok = (
                    last_ts is not None and (ts - last_ts).total_seconds() <= BURST_GAP_SECONDS
                )
                if run and gap_ok:
                    run += 1
                else:
                    if run:
                        burst_lengths.append(run)
                    run = 1
                last_ts = ts
            else:
                if run:
                    burst_lengths.append(run)
                    run = 0
                last_ts = None
        if run:
            burst_lengths.append(run)
    if not burst_lengths:
        return {"p_multi": 0.0, "avg_burst_len": 0.0, "bursts": 0}
    multi = sum(1 for length in burst_lengths if length > 1)
    return {
        "p_multi": _rate(multi, len(burst_lengths)),
        "avg_burst_len": round(sum(burst_lengths) / len(burst_lengths), 2),
        "bursts": len(burst_lengths),
    }


def compute(store: Store) -> dict:
    messages: list[Message] = store.target_messages()
    texts = [m.content for m in messages if m.content]
    total = len(texts)
    if total == 0:
        return {"message_count": 0}

    lengths = sorted(len(t) for t in texts)
    alpha_starts = [t for t in texts if t[0].isalpha()]

    emoji_counter: Counter[str] = Counter()
    for text in texts:
        emoji_counter.update(_EMOJI_RE.findall(text))
        emoji_counter.update(_CUSTOM_EMOJI_RE.findall(text))
    emoji_messages = sum(
        1 for t in texts if _EMOJI_RE.search(t) or _CUSTOM_EMOJI_RE.search(t)
    )

    return {
        "message_count": total,
        "length": {
            "avg_chars": round(sum(lengths) / total, 1),
            "median_chars": _percentile(lengths, 0.5),
            "p90_chars": _percentile(lengths, 0.9),
        },
        "burst": _burst_stats(store),
        "capitalization": {
            "starts_lowercase": _rate(
                sum(1 for t in alpha_starts if t[0].islower()), len(alpha_starts)
            ),
            "all_caps_rate": _rate(
                sum(1 for t in texts if len(t) > 2 and t.isupper()), total
            ),
        },
        "punctuation": {
            "ends_with_period": _rate(
                sum(1 for t in texts if t.endswith(".") and not t.endswith("..")), total
            ),
            "question_rate": _rate(sum(1 for t in texts if "?" in t), total),
            "exclaim_rate": _rate(sum(1 for t in texts if "!" in t), total),
            "ellipsis_rate": _rate(sum(1 for t in texts if "..." in t), total),
        },
        "emoji": {
            "per_message": _rate(emoji_messages, total),
            "top": [
                {"emoji": e, "count": c} for e, c in emoji_counter.most_common(10)
            ],
        },
        "language": {
            "slovene_diacritics_rate": _rate(
                sum(1 for t in texts if _SLOVENE_RE.search(t)), total
            ),
        },
        "misc": {
            "link_rate": _rate(sum(1 for t in texts if "http" in t), total),
            "attachment_rate": _rate(
                sum(1 for m in messages if m.attachments), len(messages)
            ),
            "mention_rate": _rate(sum(1 for t in texts if "@" in t), total),
            "markdown_rate": _rate(
                sum(1 for t in texts if _MARKDOWN_RE.search(t)), total
            ),
        },
    }


def summary_lines(stats: dict) -> list[str]:
    """Human readable digest for the cli."""
    if stats.get("message_count", 0) == 0:
        return ["no target messages in the corpus yet"]
    length = stats["length"]
    burst = stats["burst"]
    lines = [
        f"target messages   {stats['message_count']}",
        f"length            avg {length['avg_chars']} chars, median {length['median_chars']}, p90 {length['p90_chars']}",
        f"bursts            {burst['p_multi'] * 100:.0f}% multi message, avg {burst['avg_burst_len']} msgs",
        f"starts lowercase  {stats['capitalization']['starts_lowercase'] * 100:.0f}%",
        f"ends with period  {stats['punctuation']['ends_with_period'] * 100:.0f}%",
        f"emoji             in {stats['emoji']['per_message'] * 100:.0f}% of messages",
        f"slovene marks     in {stats['language']['slovene_diacritics_rate'] * 100:.0f}% of messages",
    ]
    top = stats["emoji"]["top"][:5]
    if top:
        lines.append("top emoji         " + "  ".join(f"{e['emoji']} x{e['count']}" for e in top))
    return lines
