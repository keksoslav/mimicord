from __future__ import annotations

from pathlib import Path

import pytest

from mimicord.config import TargetConfig
from mimicord.ingest import ingest_dce, ingest_package
from mimicord.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
JANEZ_BY_ID = TargetConfig(author_ids=["111"])
JANEZ_BY_NICK = TargetConfig(author_names=["janez k."])


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "corpus.db") as s:
        yield s


def test_dce_parses_and_skips_system_messages(store):
    parsed = ingest_dce(store, [FIXTURES / "dce_small.json"], JANEZ_BY_ID)
    # 7 raw messages, 1006 is a system type and gets dropped
    assert parsed == 6
    counts = store.counts()
    assert counts["total"] == 6
    assert counts["target"] == 3  # 1002, 1003, 1005


def test_dce_directory_expansion_and_partitions(store, tmp_path):
    ingest_dce(store, [FIXTURES], JANEZ_BY_ID)  # both partition files
    counts = store.counts()
    assert counts["total"] == 8
    assert counts["target"] == 4  # + 1009


def test_dce_fields(store):
    ingest_dce(store, [FIXTURES / "dce_small.json"], JANEZ_BY_ID)
    reply = store.get("1004")
    assert reply.reply_to_id == "1002"
    assert reply.author_name == "Miha"  # nickname preferred
    assert reply.guild_name == "Test Server"
    assert reply.channel_name == "general"
    # +02:00 offset normalized to utc
    assert store.get("1002").timestamp == "2024-03-01T16:00:20+00:00"
    # attachment-only message kept with empty content
    attachment_only = store.get("1007")
    assert attachment_only.content == ""
    assert attachment_only.attachments == 1


def test_dce_target_by_nickname(store):
    ingest_dce(store, [FIXTURES / "dce_small.json"], JANEZ_BY_NICK)
    assert store.counts()["target"] == 3


def test_dce_author_name_fallback(store):
    ingest_dce(store, [FIXTURES / "dce_small_part2.json"], JANEZ_BY_ID)
    assert store.get("1010").author_name == "miha"  # no nickname in fixture


def test_package_parses_json_and_csv(store):
    parsed = ingest_package(store, FIXTURES / "package_sample", JANEZ_BY_ID)
    assert parsed == 4
    counts = store.counts()
    assert counts["total"] == 4
    assert counts["target"] == 4  # a package is always entirely the requester
    assert counts["channels"] == 2


def test_package_fields(store):
    ingest_package(store, FIXTURES / "package_sample", JANEZ_BY_ID)
    own = store.get("2001")
    assert own.author_id == "111"
    assert own.author_name == "janez123"  # from account/user.json
    assert own.channel_name == "general"  # from index.json
    assert own.guild_name == "Test Server"  # from channel.json
    assert own.timestamp == "2024-03-02T10:00:00+00:00"
    assert own.reply_to_id is None
    csv_row = store.get("3002")
    assert csv_row.content == "glej to"
    assert csv_row.attachments == 1
    assert csv_row.channel_name == "Direct Message with miha#0000"


def test_package_not_flagged_when_owner_is_not_the_target(store):
    """Ingesting your own export while building someone else's persona is a
    legitimate way to add context, but it is not their voice."""
    someone_else = TargetConfig(author_ids=["646825804369756190"])
    ingest_package(store, FIXTURES / "package_sample", someone_else)
    counts = store.counts()
    assert counts["total"] == 4
    assert counts["target"] == 0


def test_package_flagged_by_account_name(store):
    ingest_package(store, FIXTURES / "package_sample", TargetConfig(author_names=["JANEZ123"]))
    assert store.counts()["target"] == 4  # case insensitive, from account/user.json


def test_package_flagged_when_no_target_configured(store):
    ingest_package(store, FIXTURES / "package_sample", TargetConfig())
    assert store.counts()["target"] == 4  # nothing to compare against yet


def test_retag_covers_package_rows(store):
    ingest_package(store, FIXTURES / "package_sample", JANEZ_BY_ID)
    assert store.counts()["target"] == 4
    # user fixes [target] to point at someone else
    changed = store.retag(["646825804369756190"], [])
    assert changed == 0
    assert store.counts()["target"] == 0
    # and back again
    store.retag(["111"], [])
    assert store.counts()["target"] == 4


def test_package_accepts_messages_dir_directly(store):
    parsed = ingest_package(store, FIXTURES / "package_sample" / "messages", JANEZ_BY_ID)
    assert parsed == 4


def test_dce_wins_over_package_either_order(tmp_path):
    # package first, dce second: dce update applies
    with Store(tmp_path / "a.db") as store:
        ingest_package(store, FIXTURES / "package_sample", JANEZ_BY_ID)
        ingest_dce(store, [FIXTURES / "dce_small.json"], JANEZ_BY_ID)
        row = store.get("1005")
        assert row.source == "dce"
        assert row.author_name == "Janez K."
        assert store.counts()["total"] == 9  # 6 dce + 3 package-only

    # dce first, package second: package insert is blocked by the conflict rule
    with Store(tmp_path / "b.db") as store:
        ingest_dce(store, [FIXTURES / "dce_small.json"], JANEZ_BY_ID)
        ingest_package(store, FIXTURES / "package_sample", JANEZ_BY_ID)
        row = store.get("1005")
        assert row.source == "dce"
        assert store.counts()["total"] == 9


def test_retag(store):
    ingest_dce(store, [FIXTURES / "dce_small.json"], TargetConfig())
    assert store.counts()["target"] == 0
    changed = store.retag(["111"], [])
    assert changed == 3
    assert store.counts()["target"] == 3
