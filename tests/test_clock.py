from __future__ import annotations

import json
from datetime import datetime

from mimicord.engine import ContextMessage, PersonaEngine, now_section

EXAMPLES = {
    "examples": [
        {"context": [{"author": "miha", "content": "a gres"}], "reply": "ne"},
    ]
}


def make_engine(persona_dir, fake_provider):
    (persona_dir / "examples.json").write_text(json.dumps(EXAMPLES), encoding="utf-8")
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    return engine


def test_reads_like_something_a_person_would_say():
    moment = datetime(2026, 8, 2, 15, 14)
    assert now_section(moment) == "[now]\nSunday 2 August 2026, 15:14\n[/now]"


def test_no_zero_padding_on_the_day():
    """%-d is not portable to windows, so the day is built by hand."""
    assert "August 2 2026" not in now_section(datetime(2026, 8, 2, 9, 5))
    assert now_section(datetime(2026, 8, 2, 9, 5)).startswith("[now]\nSunday 2 August")


def test_the_clock_goes_in_the_live_message(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("miha", "kaj dogaja")])

    call = fake_provider.calls[0]
    assert "[now]" in call["messages"][-1].content
    # never in the system prompt or the few-shots, that is the cached part
    assert "[now]" not in call["system"]
    assert all("[now]" not in m.content for m in call["messages"][:-1])


def test_the_cached_prefix_is_still_byte_stable(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("miha", "ena")])
    engine.reply([ContextMessage("miha", "dve")])

    first, second = fake_provider.calls
    assert first["system"] == second["system"]
    assert first["messages"][:-1] == second["messages"][:-1]


def test_the_clock_comes_before_the_conversation(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("miha", "kaj dogaja")])

    live = fake_provider.calls[0]["messages"][-1].content
    assert live.index("[now]") < live.index("[chat]")
