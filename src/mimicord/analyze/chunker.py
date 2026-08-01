from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from mimicord.store import Message, Store

log = logging.getLogger(__name__)

GAP_MINUTES = 30.0
CHUNK_MIN = 150
CHUNK_MAX = 250
MIN_TARGET = 15


@dataclass
class Chunk:
    index: int
    channel_id: str
    channel_name: str | None
    start_ts: str
    end_ts: str
    message_count: int
    target_count: int
    text: str


def split_segments(
    messages: list[Message], gap_minutes: float = GAP_MINUTES
) -> list[list[Message]]:
    """Cut a channel timeline into conversation segments at silence gaps."""
    segments: list[list[Message]] = []
    current: list[Message] = []
    last: datetime | None = None
    for message in messages:
        ts = datetime.fromisoformat(message.timestamp)
        if last is not None and (ts - last).total_seconds() > gap_minutes * 60:
            if current:
                segments.append(current)
            current = []
        current.append(message)
        last = ts
    if current:
        segments.append(current)
    return segments


def _render(messages: list[Message], store: Store) -> str:
    lines = []
    for m in messages:
        content = m.content or "(attachment)"
        prefix = m.author_name
        if m.reply_to_id:
            ref = store.get(m.reply_to_id)
            if ref is not None:
                prefix += f" (replying to {ref.author_name})"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_chunks(
    store: Store,
    *,
    chunk_min: int = CHUNK_MIN,
    chunk_max: int = CHUNK_MAX,
    gap_minutes: float = GAP_MINUTES,
    min_target: int = MIN_TARGET,
) -> list[Chunk]:
    """Pack conversation segments into analysis chunks, chronological order."""
    packs: list[tuple[dict, list[Message]]] = []
    for channel in store.channels():
        if channel["target"] == 0:
            continue
        messages = store.channel_messages(channel["channel_id"])
        batch: list[Message] = []
        for segment in split_segments(messages, gap_minutes):
            batch.extend(segment)
            while len(batch) >= chunk_max:
                packs.append((channel, batch[:chunk_max]))
                batch = batch[chunk_max:]
            if len(batch) >= chunk_min:
                packs.append((channel, batch))
                batch = []
        if batch:
            packs.append((channel, batch))

    qualified = [
        pack for pack in packs if sum(m.is_target for m in pack[1]) >= min_target
    ]
    if not qualified and packs:
        log.warning(
            "no chunk reaches %d target messages, keeping every chunk with any",
            min_target,
        )
        qualified = [pack for pack in packs if any(m.is_target for m in pack[1])]

    chunks = [
        Chunk(
            index=0,
            channel_id=str(channel["channel_id"]),
            channel_name=channel["channel_name"],
            start_ts=messages[0].timestamp,
            end_ts=messages[-1].timestamp,
            message_count=len(messages),
            target_count=sum(m.is_target for m in messages),
            text=_render(messages, store),
        )
        for channel, messages in qualified
    ]
    chunks.sort(key=lambda c: c.end_ts)
    for position, chunk in enumerate(chunks):
        chunk.index = position
    return chunks


def sample_chunks(
    chunks: list[Chunk], cap: int = 50, recent_fraction: float = 0.5
) -> list[Chunk]:
    """Deterministic stratified sample: recent quartile is weighted because
    people drift, the rest is spread evenly across the whole history."""
    if len(chunks) <= cap:
        return chunks

    def spread(items: list[Chunk], n: int) -> list[Chunk]:
        if n >= len(items):
            return list(items)
        step = len(items) / n
        return [items[int(i * step)] for i in range(n)]

    quartile_start = int(len(chunks) * 0.75)
    older, recent = chunks[:quartile_start], chunks[quartile_start:]
    recent_budget = min(len(recent), max(1, int(cap * recent_fraction)))
    picked = spread(older, cap - recent_budget) + spread(recent, recent_budget)
    picked.sort(key=lambda c: c.index)
    return picked
