from __future__ import annotations

import asyncio

import pytest

from mimicord.bot import MimicClient
from mimicord.config import ConfigError, load_config
from mimicord.engine import ContextMessage, PersonaEngine

TOML = """\
name = "testbot"

[llm]
provider = "ollama"

[style]
reaction_cooldown_seconds = 60

[[reactions]]
name = "svit"
dir = "people/svit"
when = "when Svit comes up"
cooldown = 3600

[[reactions]]
name = "angry"
file = "angry.gif"
"""


def write(tmp_path, text):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_dir_is_parsed(tmp_path):
    reaction = load_config(write(tmp_path, TOML)).reactions[0]
    assert reaction.dir == "people/svit"
    assert reaction.cooldown == 3600
    assert reaction.source == "people/svit"


def test_exactly_one_source(tmp_path):
    both = 'name = "x"\n[[reactions]]\nname = "a"\nfile = "a.gif"\ndir = "d"\n'
    with pytest.raises(ConfigError, match="file, url or dir"):
        load_config(write(tmp_path, both))
    none = 'name = "x"\n[[reactions]]\nname = "a"\n'
    with pytest.raises(ConfigError, match="file, url or dir"):
        load_config(write(tmp_path, none))


def build(persona_dir, pictures: int = 3):
    (persona_dir / "persona.toml").write_text(TOML, encoding="utf-8")
    folder = persona_dir / "media" / "people" / "svit"
    folder.mkdir(parents=True)
    for i in range(pictures):
        (folder / f"{i:02d}.jpg").write_bytes(b"jpeg")
    (folder / "notes.txt").write_text("not a picture", encoding="utf-8")
    return folder


def test_picks_a_picture_from_the_folder(persona_dir, fake_provider):
    folder = build(persona_dir)
    engine = PersonaEngine("testbot", rag_enabled=False)
    reaction = engine.find_reaction("svit")

    picked = {engine.reaction_path(reaction) for _ in range(40)}

    assert picked == set(folder.glob("*.jpg"))  # every one, and only pictures


def test_an_empty_folder_is_not_usable(persona_dir, fake_provider):
    (persona_dir / "persona.toml").write_text(TOML, encoding="utf-8")
    (persona_dir / "media" / "people" / "svit").mkdir(parents=True)
    engine = PersonaEngine("testbot", rag_enabled=False)
    assert engine.reaction_path(engine.find_reaction("svit")) is None


def test_a_missing_file_reaction_is_not_usable(persona_dir, fake_provider):
    build(persona_dir)
    engine = PersonaEngine("testbot", rag_enabled=False)
    assert engine.reaction_path(engine.find_reaction("angry")) is None


def test_the_prompt_costs_one_line_per_group(persona_dir, fake_provider):
    """The whole point: thirty photos must not become thirty instructions."""
    build(persona_dir, pictures=30)
    engine = PersonaEngine("testbot", rag_enabled=False)
    assert engine.system.count("[gif:svit]") == 1


class FakeChannel:
    id = 100

    def __str__(self) -> str:
        return "general"


def make_client(persona_dir, fake_provider) -> MimicClient:
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    client = MimicClient(engine, dry_run=True)
    client.seeded.add(FakeChannel.id)
    return client


def sent(client) -> list[str]:
    return [m.content for m in client.buffers[FakeChannel.id]]


def test_a_second_picture_is_held_back(persona_dir, fake_provider):
    build(persona_dir)
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()

    asyncio.run(client._send_bursts(channel, ["[gif:svit]"]))
    asyncio.run(client._send_bursts(channel, ["ma daj", "[gif:svit]"]))

    # the first went, the second was dropped and the text still went out
    assert sent(client) == ["[gif:svit]", "ma daj"]


def test_text_still_gets_through_while_pictures_are_on_cooldown(
    persona_dir, fake_provider
):
    build(persona_dir)
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()

    asyncio.run(client._send_bursts(channel, ["[gif:svit]"]))
    asyncio.run(client._send_bursts(channel, ["neja", "nemorm"]))

    assert sent(client) == ["[gif:svit]", "neja", "nemorm"]


def test_no_cooldown_configured_means_no_wall(persona_dir, fake_provider):
    build(persona_dir)
    (persona_dir / "persona.toml").write_text(
        TOML.replace("reaction_cooldown_seconds = 60", "").replace(
            "cooldown = 3600", ""
        ),
        encoding="utf-8",
    )
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()

    asyncio.run(client._send_bursts(channel, ["[gif:svit]"]))
    asyncio.run(client._send_bursts(channel, ["[gif:svit]"]))

    assert sent(client) == ["[gif:svit]", "[gif:svit]"]


def test_replies_are_unaffected(persona_dir, fake_provider):
    build(persona_dir)
    fake_provider.reply = "kk"
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.buffers[channel.id].append(ContextMessage("Mike_", "a gres"))

    asyncio.run(client._respond(channel, "mention"))

    assert sent(client)[-1] == "kk"


def test_a_tag_written_into_a_line_is_rescued():
    """Live: 'nikol ne pride ko rces [gif:nik]' would have posted the raw tag."""
    from mimicord.postprocess import apply, reaction_name

    bursts = apply("nikol ne pride ko rces [gif:nik]", persona_name="sandman")
    assert bursts == ["nikol ne pride ko rces", "[gif:nik]"]
    assert reaction_name(bursts[1]) == "nik"


def test_a_tag_already_on_its_own_line_is_left_alone():
    from mimicord.postprocess import apply

    assert apply("ma daj\n[gif:svit]", persona_name="sandman") == [
        "ma daj",
        "[gif:svit]",
    ]


def test_a_tag_only_message_survives():
    from mimicord.postprocess import apply

    assert apply("[gif:timi]", persona_name="sandman") == ["[gif:timi]"]


def test_two_tags_in_one_line_both_come_out():
    from mimicord.postprocess import apply

    assert apply("glej [gif:svit] pa [gif:lev]", persona_name="sandman") == [
        "glej pa",
        "[gif:svit]",
        "[gif:lev]",
    ]


def test_ordinary_text_with_brackets_is_untouched():
    from mimicord.postprocess import apply

    assert apply("[ni panike]", persona_name="sandman") == ["[ni panike]"]
