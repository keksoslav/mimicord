from __future__ import annotations

from datetime import datetime, timezone

from mimicord.usage import UsageLedger

AUG = datetime(2026, 8, 15, tzinfo=timezone.utc)
SEP = datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_fresh_ledger_counts_zero(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.json")
    assert ledger.count(AUG) == 0


def test_increment_persists(tmp_path):
    path = tmp_path / "usage.json"
    ledger = UsageLedger(path)
    ledger.increment(AUG)
    ledger.increment(AUG)
    assert ledger.count(AUG) == 2
    # a fresh instance reads the same numbers back
    assert UsageLedger(path).count(AUG) == 2


def test_month_rollover_resets(tmp_path):
    path = tmp_path / "usage.json"
    ledger = UsageLedger(path)
    ledger.increment(AUG)
    assert ledger.count(SEP) == 0
    ledger.increment(SEP)
    assert ledger.count(SEP) == 1
    assert UsageLedger(path).count(SEP) == 1


def test_corrupt_file_resets_quietly(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("not json", encoding="utf-8")
    ledger = UsageLedger(path)
    assert ledger.count(AUG) == 0
    ledger.increment(AUG)
    assert UsageLedger(path).count(AUG) == 1
