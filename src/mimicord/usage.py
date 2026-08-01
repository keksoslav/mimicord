from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class UsageLedger:
    """Monthly reply counter persisted next to the persona artifacts.

    Backs the max_replies_per_month budget so a bot goes quiet on its own
    well before a subscription credit or api budget would run dry.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._month = ""
        self._replies = 0
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._month = str(data.get("month", ""))
                self._replies = int(data.get("replies", 0))
            except (json.JSONDecodeError, ValueError):
                pass  # corrupt ledger just resets the count

    @staticmethod
    def _key(now: datetime) -> str:
        return now.strftime("%Y-%m")

    def count(self, now: datetime) -> int:
        if self._key(now) != self._month:
            return 0
        return self._replies

    def increment(self, now: datetime) -> None:
        key = self._key(now)
        if key != self._month:
            self._month = key
            self._replies = 0
        self._replies += 1
        self._path.write_text(
            json.dumps({"month": self._month, "replies": self._replies}),
            encoding="utf-8",
        )
