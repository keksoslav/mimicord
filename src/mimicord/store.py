from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id            TEXT PRIMARY KEY,
  channel_id    TEXT NOT NULL,
  channel_name  TEXT,
  guild_name    TEXT,
  author_id     TEXT,
  author_name   TEXT NOT NULL,
  is_target     INTEGER NOT NULL,
  content       TEXT NOT NULL,
  timestamp     TEXT NOT NULL,
  reply_to_id   TEXT,
  attachments   INTEGER NOT NULL DEFAULT 0,
  source        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chan_time ON messages(channel_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_target_time ON messages(is_target, timestamp);
"""

# a message can show up in both a DCE export and the data package; the DCE row
# is richer (real author, reply refs) so it wins regardless of ingest order
UPSERT = """
INSERT INTO messages (id, channel_id, channel_name, guild_name, author_id,
                      author_name, is_target, content, timestamp, reply_to_id,
                      attachments, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  channel_id   = excluded.channel_id,
  channel_name = excluded.channel_name,
  guild_name   = excluded.guild_name,
  author_id    = excluded.author_id,
  author_name  = excluded.author_name,
  is_target    = excluded.is_target,
  content      = excluded.content,
  timestamp    = excluded.timestamp,
  reply_to_id  = excluded.reply_to_id,
  attachments  = excluded.attachments,
  source       = excluded.source
WHERE excluded.source = 'dce' AND messages.source = 'package'
"""


@dataclass
class Message:
    id: str
    channel_id: str
    channel_name: str | None
    guild_name: str | None
    author_id: str | None
    author_name: str
    is_target: bool
    content: str
    timestamp: str  # ISO 8601 UTC
    reply_to_id: str | None
    attachments: int
    source: str  # 'dce' | 'package'


def _to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        channel_id=row["channel_id"],
        channel_name=row["channel_name"],
        guild_name=row["guild_name"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        is_target=bool(row["is_target"]),
        content=row["content"],
        timestamp=row["timestamp"],
        reply_to_id=row["reply_to_id"],
        attachments=row["attachments"],
        source=row["source"],
    )


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert_many(self, messages: Iterable[Message]) -> int:
        rows = [
            (
                m.id, m.channel_id, m.channel_name, m.guild_name, m.author_id,
                m.author_name, int(m.is_target), m.content, m.timestamp,
                m.reply_to_id, m.attachments, m.source,
            )
            for m in messages
        ]
        with self.conn:
            self.conn.executemany(UPSERT, rows)
        return len(rows)

    def counts(self) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS total,
                      COALESCE(SUM(is_target), 0) AS target,
                      COUNT(DISTINCT channel_id) AS channels,
                      MIN(timestamp) AS first,
                      MAX(timestamp) AS last
               FROM messages"""
        ).fetchone()
        return dict(row)

    def channels(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT channel_id, channel_name, COUNT(*) AS total,
                      COALESCE(SUM(is_target), 0) AS target
               FROM messages GROUP BY channel_id ORDER BY total DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def channel_messages(self, channel_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE channel_id = ? ORDER BY timestamp",
            (channel_id,),
        ).fetchall()
        return [_to_message(r) for r in rows]

    def target_messages(self) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE is_target = 1 ORDER BY channel_id, timestamp"
        ).fetchall()
        return [_to_message(r) for r in rows]

    def get(self, message_id: str) -> Message | None:
        row = self.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        return _to_message(row) if row else None

    def retag(self, author_ids: list[str], author_names: list[str]) -> int:
        """Re-flag is_target after the [target] config changed."""
        names = [n.lower() for n in author_names]
        with self.conn:
            self.conn.execute("UPDATE messages SET is_target = 0 WHERE source = 'dce'")
            cursor = self.conn.execute(
                f"""UPDATE messages SET is_target = 1
                    WHERE source = 'dce' AND (
                      author_id IN ({','.join('?' * len(author_ids)) or "''"})
                      OR lower(author_name) IN ({','.join('?' * len(names)) or "''"})
                    )""",
                [*author_ids, *names],
            )
        return cursor.rowcount
