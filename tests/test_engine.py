from __future__ import annotations

import json

from mimicord.engine import ContextMessage, PersonaEngine

EXAMPLES = {
    "examples": [
        {
            "context": [{"author": "miha", "content": "a gres jutri?"}],
            "reply": ["ne", "mam kolokvij"],
        },
        {
            "context": [{"author": "ana", "content": "kaj je to"}],
            "reply": "ne vem lol",
        },
    ]
}


def make_engine(persona_dir, fake_provider, with_examples=True):
    if with_examples:
        (persona_dir / "examples.json").write_text(
            json.dumps(EXAMPLES), encoding="utf-8"
        )
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    return engine


def test_prompt_prefix_is_byte_stable(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("you", "kaj dogaja")])
    engine.reply([ContextMessage("you", "gremo na pivo?")])

    first, second = fake_provider.calls
    # cached prefix: identical system and identical few-shots across calls
    assert first["system"] == second["system"]
    assert first["messages"][:-1] == second["messages"][:-1]
    # only the live context differs
    assert first["messages"][-1] != second["messages"][-1]
    assert "gremo na pivo?" in second["messages"][-1].content


def test_examples_shape_and_cache_boundary(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([ContextMessage("you", "hej")])
    messages = fake_provider.calls[0]["messages"]

    # 2 examples -> 4 few-shot messages + 1 live message
    assert len(messages) == 5
    assert messages[0].role == "user"
    assert "[chat]" in messages[0].content
    assert messages[1].role == "assistant"
    assert messages[1].content == "ne\nmam kolokvij"
    # the boundary sits on the last few-shot, never on the live message
    assert messages[3].cache_boundary is True
    assert all(not m.cache_boundary for m in messages[:3])
    assert messages[4].cache_boundary is False
    assert messages[4].content.endswith("[/chat]")


def test_reply_postprocesses(persona_dir, fake_provider):
    fake_provider.reply = "<thinking>hmm</thinking>testbot: ma ja\nsaj vem"
    engine = make_engine(persona_dir, fake_provider, with_examples=False)
    bursts = engine.reply([ContextMessage("you", "no?")])
    assert bursts == ["ma ja", "saj vem"]


def test_aliases_cover_every_way_of_addressing_the_persona(persona_dir, fake_provider):
    """Nicknames that summon the bot are not topics to search memories on."""
    persona_dir.joinpath("persona.toml").write_text(
        'name = "testbot"\n'
        '[llm]\nprovider = "ollama"\n'
        '[target]\nauthor_names = ["SaNdMaN"]\n'
        '[discord]\ntrigger_keywords = ["timi", "glavonja"]\n',
        encoding="utf-8",
    )
    engine = make_engine(persona_dir, fake_provider, with_examples=False)
    assert engine._aliases == {"testbot", "SaNdMaN", "timi", "glavonja"}


def test_works_without_examples(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider, with_examples=False)
    engine.reply([ContextMessage("you", "hej")])
    assert len(fake_provider.calls[0]["messages"]) == 1
