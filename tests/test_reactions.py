from __future__ import annotations

import pytest

from mimicord.config import ConfigError, load_config
from mimicord.paths import PersonaPaths
from mimicord.postprocess import apply, reaction_name

TOML = """\
name = "testbot"

[llm]
provider = "ollama"

[[reactions]]
name = "angry"
file = "angry.gif"
when = "Lev is mean to you"

[[reactions]]
name = "shrug"
file = "shrug.gif"
"""


def write(tmp_path, text=TOML):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_reactions_parsed(tmp_path):
    cfg = load_config(write(tmp_path))
    assert [r.name for r in cfg.reactions] == ["angry", "shrug"]
    assert cfg.reactions[0].file == "angry.gif"
    assert cfg.reactions[0].when == "Lev is mean to you"
    assert cfg.reactions[1].when == ""  # optional


def test_no_reactions_by_default(tmp_path):
    assert load_config(write(tmp_path, 'name = "x"\n')).reactions == []


def test_reaction_needs_a_source(tmp_path):
    text = 'name = "x"\n[[reactions]]\nname = "angry"\n'
    with pytest.raises(ConfigError, match="exactly one of file, url or dir"):
        load_config(write(tmp_path, text))


def test_reaction_cannot_have_both_sources(tmp_path):
    text = (
        'name = "x"\n[[reactions]]\nname = "a"\n'
        'file = "a.gif"\nurl = "https://tenor.com/view/x"\n'
    )
    with pytest.raises(ConfigError, match="exactly one of file, url or dir"):
        load_config(write(tmp_path, text))


def test_reaction_url_must_be_http(tmp_path):
    text = 'name = "x"\n[[reactions]]\nname = "a"\nurl = "tenor.com/view/x"\n'
    with pytest.raises(ConfigError, match="http or https"):
        load_config(write(tmp_path, text))


def test_url_reaction_parsed(tmp_path):
    text = (
        'name = "x"\n[[reactions]]\nname = "dedi"\n'
        'url = "https://tenor.com/view/x"\nwhen = "mocking someone"\n'
    )
    reaction = load_config(write(tmp_path, text)).reactions[0]
    assert reaction.url == "https://tenor.com/view/x"
    assert reaction.file == ""
    assert reaction.when == "mocking someone"


def test_duplicate_reaction_names_rejected(tmp_path):
    text = (
        'name = "x"\n'
        '[[reactions]]\nname = "a"\nfile = "a.gif"\n'
        '[[reactions]]\nname = "a"\nfile = "b.gif"\n'
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, text))


def test_reaction_name_must_be_tag_safe(tmp_path):
    text = 'name = "x"\n[[reactions]]\nname = "very angry!"\nfile = "a.gif"\n'
    with pytest.raises(ConfigError, match="letters, digits"):
        load_config(write(tmp_path, text))


def test_reaction_names_are_lowercased(tmp_path):
    text = 'name = "x"\n[[reactions]]\nname = "ANGRY"\nfile = "a.gif"\n'
    assert load_config(write(tmp_path, text)).reactions[0].name == "angry"


def test_tag_detection():
    assert reaction_name("[gif:angry]") == "angry"
    assert reaction_name("  [gif:ANGRY]  ") == "angry"
    assert reaction_name("[gif:very-angry_2]") == "very-angry_2"


def test_tag_must_be_the_whole_message():
    """A tag buried in a sentence is text, not an image."""
    assert reaction_name("ja [gif:angry] pa to") is None
    assert reaction_name("glej [gif:angry]") is None
    assert reaction_name("nekaj drugega") is None


def test_tag_survives_postprocessing():
    stats = {"capitalization": {"starts_lowercase": 0.95}, "punctuation": {"ends_with_period": 0.0}}
    bursts = apply("ma daj\n[gif:angry]", stats=stats, persona_name="testbot")
    assert bursts == ["ma daj", "[gif:angry]"]
    assert reaction_name(bursts[1]) == "angry"


def test_engine_advertises_reactions_and_resolves_paths(personas_home, fake_provider):
    from mimicord.engine import PersonaEngine

    root = personas_home / "testbot"
    root.mkdir()
    (root / "persona.toml").write_text(TOML, encoding="utf-8")
    (root / "persona.md").write_text("# Persona: testbot", encoding="utf-8")

    engine = PersonaEngine("testbot", rag_enabled=False)
    assert "[gif:angry] Lev is mean to you" in engine.system
    assert "[gif:shrug]" in engine.system
    angry = engine.find_reaction("angry")
    # nothing on disk, so there is no usable path for it
    assert engine.reaction_path(angry) is None
    (PersonaPaths(root).media_dir).mkdir(parents=True, exist_ok=True)
    (PersonaPaths(root).media_dir / "angry.gif").write_bytes(b"gif")
    assert engine.reaction_path(angry) == PersonaPaths(root).media_dir / "angry.gif"
    assert engine.find_reaction("ANGRY") is not None  # case insensitive
    assert engine.find_reaction("nope") is None


def test_engine_without_reactions_adds_no_block(persona_dir, fake_provider):
    from mimicord.engine import PersonaEngine

    engine = PersonaEngine("testbot", rag_enabled=False)
    assert "## Reactions" not in engine.system
