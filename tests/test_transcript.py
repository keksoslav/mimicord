from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from mimicord.engine import (
    ContextMessage,
    PersonaEngine,
    describe_gap,
    render_transcript,
)

START = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

EXAMPLES = {
    "examples": [
        {"context": [{"author": "miha", "content": "a gres"}], "reply": "ne"},
    ]
}


def at(minutes: float) -> datetime:
    return START + timedelta(minutes=minutes)


def make_engine(persona_dir, fake_provider):
    (persona_dir / "examples.json").write_text(json.dumps(EXAMPLES), encoding="utf-8")
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    return engine


def test_a_normal_conversation_gets_no_markers():
    """Two people typing at each other must not get a line between every message."""
    text = render_transcript(
        [
            ContextMessage("miha", "a gres", at=at(0)),
            ContextMessage("lev", "kam", at=at(2)),
            ContextMessage("miha", "ven", at=at(3)),
        ],
        now=at(4),
    )
    assert text == "miha: a gres\nlev: kam\nmiha: ven"


def test_a_long_silence_is_marked():
    text = render_transcript(
        [
            ContextMessage("miha", "a gres", at=at(0)),
            ContextMessage("lev", "sori zdj sm vidu", at=at(60 * 24 * 2)),
        ],
        now=at(60 * 24 * 2 + 1),
    )
    assert text == "miha: a gres\n[2 days later]\nlev: sori zdj sm vidu"


def test_a_stale_buffer_says_how_old_it_is():
    """The case that started this: seeded history read as a live conversation."""
    text = render_transcript(
        [ContextMessage("miha", "a gres", at=at(0))], now=at(60 * 5)
    )
    assert text == "miha: a gres\n[that was 5 hours ago]"


def test_a_fresh_message_is_not_labelled_old():
    text = render_transcript([ContextMessage("miha", "a gres", at=at(0))], now=at(3))
    assert text == "miha: a gres"


def test_messages_with_no_timestamps_are_rendered_plainly():
    """The repl, the gui and redteam all build context without a clock."""
    text = render_transcript(
        [ContextMessage("you", "zivjo"), ContextMessage("sandman", "kk")], now=at(0)
    )
    assert text == "you: zivjo\nsandman: kk"


def test_an_untimed_message_does_not_reset_the_gap():
    """His own replies carry a time, but a caller that omits one must not
    make two distant messages look adjacent."""
    text = render_transcript(
        [
            ContextMessage("miha", "a gres", at=at(0)),
            ContextMessage("sandman", "kk"),
            ContextMessage("lev", "no?", at=at(60 * 24)),
        ]
    )
    assert "[1 day later]" in text


def test_naive_timestamps_do_not_blow_up():
    text = render_transcript(
        [
            ContextMessage("miha", "a", at=datetime(2026, 8, 4, 12, 0)),
            ContextMessage("lev", "b", at=datetime(2026, 8, 5, 12, 0)),
        ]
    )
    assert "[1 day later]" in text


def test_gaps_are_described_in_round_units():
    assert describe_gap(60 * 50) == "50 minutes"
    assert describe_gap(3600 * 3) == "3 hours"
    assert describe_gap(3600) == "1 hour"
    assert describe_gap(86400 * 2) == "2 days"
    assert describe_gap(86400 * 15) == "2 weeks"


def test_an_almost_whole_day_is_not_rounded_down_to_the_day_before():
    """Live: a gap of 3 days 23 hours was being called 3 days."""
    assert describe_gap(86400 * 4 - 180) == "4 days"
    # and the unit never rounds down to nothing
    assert describe_gap(3600 * 1.2) == "1 hour"


def test_the_gap_reaches_the_prompt(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    engine.reply([ContextMessage("miha", "a gres", at=old)])

    live = fake_provider.calls[0]["messages"][-1].content
    assert "[that was 3 days ago]" in live


def test_the_cached_prefix_is_untouched_by_gaps(persona_dir, fake_provider):
    """Markers ride in the live message, so the few-shots stay byte-stable."""
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("miha", "ena", at=datetime.now(timezone.utc))])
    engine.reply(
        [ContextMessage("miha", "dve", at=datetime.now(timezone.utc) - timedelta(days=9))]
    )

    first, second = fake_provider.calls
    assert first["system"] == second["system"]
    assert first["messages"][:-1] == second["messages"][:-1]
