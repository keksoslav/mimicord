from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from mimicord.config import TargetConfig
from mimicord.store import Message

log = logging.getLogger(__name__)


def _get(row: dict, key: str) -> str:
    """Case-insensitive lookup, package exports capitalize their keys."""
    for candidate in (key, key.capitalize(), key.upper(), key.lower()):
        if candidate in row:
            return str(row[candidate] or "")
    return ""


def _normalize_ts(raw: str) -> str | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)  # package times are UTC
    return parsed.astimezone(timezone.utc).isoformat()


def _account_identity(package_root: Path) -> tuple[str | None, str | None]:
    user_json = package_root / "account" / "user.json"
    if user_json.is_file():
        try:
            data = json.loads(user_json.read_text(encoding="utf-8-sig"))
            return (
                str(data["id"]) if data.get("id") else None,
                data.get("username") or data.get("global_name"),
            )
        except (json.JSONDecodeError, KeyError):
            log.warning("could not read %s", user_json)
    return None, None


def _channel_rows(channel_dir: Path) -> list[dict]:
    json_path = channel_dir / "messages.json"
    csv_path = channel_dir / "messages.csv"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8-sig"))
    if csv_path.is_file():  # older packages ship csv
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return []


def parse_package(root: Path, target: TargetConfig) -> Iterator[Message]:
    """Yield normalized messages from an official Discord data package.

    root may be the package root or its messages/ folder. Every message in a
    package belongs to the requesting account, so everything is is_target.
    """
    if (root / "messages").is_dir():
        package_root = root
        messages_dir = root / "messages"
    else:
        package_root = root.parent
        messages_dir = root

    index: dict = {}
    index_path = messages_dir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8-sig")) or {}

    author_id, author_name = _account_identity(package_root)
    if not author_name:
        author_name = target.author_names[0] if target.author_names else "me"

    for channel_dir in sorted(messages_dir.glob("c*")):
        if not channel_dir.is_dir():
            continue
        channel_id = channel_dir.name.lstrip("c")
        channel_name = index.get(channel_id)
        guild_name = None
        channel_json = channel_dir / "channel.json"
        if channel_json.is_file():
            try:
                meta = json.loads(channel_json.read_text(encoding="utf-8-sig"))
                guild_name = (meta.get("guild") or {}).get("name")
                channel_id = str(meta.get("id", channel_id))
            except json.JSONDecodeError:
                log.warning("could not read %s", channel_json)

        skipped = 0
        for row in _channel_rows(channel_dir):
            content = _get(row, "contents").strip()
            attachment_field = _get(row, "attachments").strip()
            attachments = 1 if attachment_field else 0
            if not content and not attachments:
                continue
            timestamp = _normalize_ts(_get(row, "timestamp"))
            message_id = _get(row, "id")
            if timestamp is None or not message_id:
                skipped += 1
                continue
            yield Message(
                id=message_id,
                channel_id=channel_id,
                channel_name=channel_name,
                guild_name=guild_name,
                author_id=author_id,
                author_name=author_name,
                is_target=True,
                content=content,
                timestamp=timestamp,
                reply_to_id=None,  # packages carry no reply references
                attachments=attachments,
                source="package",
            )
        if skipped:
            log.warning("%s: skipped %d malformed rows", channel_dir, skipped)
