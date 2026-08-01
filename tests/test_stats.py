from __future__ import annotations

import pytest

from mimicord.analyze import stats
from mimicord.store import Message, Store


def msg(id_, content, *, ts, target=True, channel="c1", attachments=0):
    return Message(
        id=id_,
        channel_id=channel,
        channel_name=None,
        guild_name=None,
        author_id="111" if target else "222",
        author_name="janez" if target else "miha",
        is_target=target,
        content=content,
        timestamp=f"2024-03-01T{ts}+00:00",
        reply_to_id=None,
        attachments=attachments,
        source="dce",
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "corpus.db") as s:
        yield s


def test_empty_corpus(store):
    assert stats.compute(store) == {"message_count": 0}


def test_basic_numbers(store):
    store.upsert_many(
        [
            # burst of two, then miha interrupts, then a solo
            msg("1", "ne", ts="10:00:00"),
            msg("2", "mam kolokvij", ts="10:00:20"),
            msg("3", "a res", ts="10:00:40", target=False),
            msg("4", "ma sej bo.", ts="10:01:00"),
            # different channel, one lonely caps message with emoji
            msg("5", "LOL :kek: 💀", ts="11:00:00", channel="c2"),
        ]
    )
    result = stats.compute(store)
    assert result["message_count"] == 4
    # sorted lengths [2, 10, 11, 12] -> upper-median picks index 2
    assert result["length"]["median_chars"] == 11

    burst = result["burst"]
    assert burst["bursts"] == 3  # [1,2], [4], [5]
    assert burst["p_multi"] == pytest.approx(1 / 3, abs=0.001)

    # "ne", "mam kolokvij", "ma sej bo." lowercase; "LOL..." starts upper
    assert result["capitalization"]["starts_lowercase"] == 0.75
    assert result["punctuation"]["ends_with_period"] == 0.25

    top = {e["emoji"]: e["count"] for e in result["emoji"]["top"]}
    assert top[":kek:"] == 1
    assert top["💀"] == 1


def test_burst_gap_splits_runs(store):
    store.upsert_many(
        [
            msg("1", "prva", ts="10:00:00"),
            msg("2", "druga po pavzi", ts="10:05:00"),  # 5 min later, new burst
        ]
    )
    result = stats.compute(store)
    assert result["burst"]["bursts"] == 2
    assert result["burst"]["p_multi"] == 0.0


def test_misc_rates(store):
    store.upsert_many(
        [
            msg("1", "glej https://example.com", ts="10:00:00"),
            msg("2", "cist navadn tekst", ts="10:01:30"),
            msg("3", "", ts="10:03:00", attachments=1),
        ]
    )
    result = stats.compute(store)
    # empty content rows are excluded from text stats but count for attachments
    assert result["message_count"] == 2
    assert result["misc"]["link_rate"] == 0.5
    assert result["misc"]["attachment_rate"] == pytest.approx(1 / 3, abs=0.001)


def test_summary_lines_render(store):
    store.upsert_many([msg("1", "ne vem", ts="10:00:00")])
    lines = stats.summary_lines(stats.compute(store))
    assert any("target messages" in line for line in lines)
