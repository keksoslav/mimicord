from __future__ import annotations

import asyncio

import pytest

from mimicord.bot import MimicClient
from mimicord.config import ConfigError, DiscordConfig, load_config
from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.triggers import MessageFacts, TriggerState, should_reply


def facts(**overrides) -> MessageFacts:
    base = dict(
        channel_id="100",
        author_is_self=False,
        author_is_bot=False,
        mentions_bot=True,
        replies_to_bot=False,
        content="nic ne odpisi na to sporocilo",
    )
    base.update(overrides)
    return MessageFacts(**base)


def cfg(**overrides) -> DiscordConfig:
    config = DiscordConfig(token_env="T")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def decide(f, c=None):
    return should_reply(f, c or cfg(), TriggerState(), 1000.0, 0.99)


def test_repeated_message_is_ignored():
    assert decide(facts(repeats_recent=True)) == (False, "repeated message")


def test_first_copy_still_answers():
    assert decide(facts())[0] is True


def test_repeat_beats_every_trigger():
    """A mention, a reply and a keyword all lose to a flood."""
    c = cfg(trigger_keywords=["sandman"], always_on_channels=["100"])
    spam = facts(repeats_recent=True, replies_to_bot=True, content="sandman")
    assert decide(spam, c)[0] is False


def test_debounce_defaults_on(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text('name = "x"\n', encoding="utf-8")
    assert load_config(path).discord.debounce_seconds == 2.5


def test_negative_debounce_rejected(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text('name = "x"\n[discord]\ndebounce_seconds = -1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="debounce_seconds"):
        load_config(path)


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


def test_will_not_send_the_same_reply_twice(persona_dir, fake_provider):
    fake_provider.reply = "ns"
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.buffers[channel.id].append(ContextMessage("Mike_", "kaj delas"))

    asyncio.run(client._respond(channel, "mention"))
    asyncio.run(client._respond(channel, "mention"))

    assert sent(client).count("ns") == 1


def test_a_different_reply_still_goes_out(persona_dir, fake_provider):
    fake_provider.replies = ["ns", "kk"]
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.buffers[channel.id].append(ContextMessage("Mike_", "kaj delas"))

    asyncio.run(client._respond(channel, "mention"))
    asyncio.run(client._respond(channel, "mention"))

    assert [m.content for m in client.buffers[channel.id] if m.author == "testbot"] == [
        "ns",
        "kk",
    ]
