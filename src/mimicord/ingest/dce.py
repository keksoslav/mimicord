from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from mimicord.config import TargetConfig
from mimicord.store import Message

log = logging.getLogger(__name__)

KEPT_TYPES = ("Default", "Reply")


def _normalize_ts(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _author_name(author: dict) -> str:
    # field names drifted across DCE versions, take the first non empty
    for key in ("nickname", "displayName", "name"):
        value = author.get(key)
        if value:
            return str(value)
    return "unknown"


def _is_target(author: dict, target: TargetConfig) -> bool:
    author_id = str(author.get("id", ""))
    if author_id and author_id in target.author_ids:
        return True
    wanted = {n.lower() for n in target.author_names}
    candidates = {
        str(author.get(key, "")).lower()
        for key in ("name", "nickname", "displayName")
    }
    return bool(wanted & candidates)


def parse_dce(path: Path, target: TargetConfig) -> Iterator[Message]:
    """Yield normalized messages from one DiscordChatExporter JSON export."""
    with open(path, encoding="utf-8-sig") as f:  # DCE sometimes writes a BOM
        data = json.load(f)

    guild = (data.get("guild") or {}).get("name")
    if guild == "Direct Messages":
        guild = None
    channel = data.get("channel") or {}
    channel_id = str(channel.get("id", ""))
    channel_name = channel.get("name")
    if not channel_id:
        log.warning("%s has no channel id, skipping file", path)
        return

    skipped = 0
    for m in data.get("messages", []):
        if m.get("type", "Default") not in KEPT_TYPES:
            continue
        content = (m.get("content") or "").strip()
        attachments = len(m.get("attachments") or [])
        if not content and not attachments:
            continue
        timestamp = _normalize_ts(m.get("timestamp"))
        if timestamp is None:
            skipped += 1
            continue
        author = m.get("author") or {}
        reference = m.get("reference") or {}
        reply_to = None
        if m.get("type") == "Reply" and reference.get("messageId"):
            reply_to = str(reference["messageId"])
        yield Message(
            id=str(m["id"]),
            channel_id=channel_id,
            channel_name=channel_name,
            guild_name=guild,
            author_id=str(author.get("id")) if author.get("id") else None,
            author_name=_author_name(author),
            is_target=_is_target(author, target),
            content=content,
            timestamp=timestamp,
            reply_to_id=reply_to,
            attachments=attachments,
            source="dce",
        )
    if skipped:
        log.warning("%s: skipped %d messages with unparseable timestamps", path, skipped)
