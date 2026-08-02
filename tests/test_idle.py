from __future__ import annotations

import asyncio

import pytest

from mimicord.bot import MimicClient
from mimicord.config import ConfigError, DiscordConfig, load_config
from mimicord.engine import PersonaEngine
from mimicord.triggers import TriggerState, should_poke

DAY = 24 * 3600


def cfg(**overrides) -> DiscordConfig:
    config = DiscordConfig(token_env="T", idle_hours=24, idle_channels=["100"])
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def poke(c, idle_seconds=DAY, state=None, now=1000.0, monthly_count=0, channel="100"):
    return should_poke(
        channel, c, state or TriggerState(), now, idle_seconds, monthly_count
    )


def test_off_unless_configured():
    assert poke(DiscordConfig(token_env="T")) == (False, "idle pokes off")


def test_only_configured_channels():
    assert poke(cfg(), channel="200") == (False, "channel is not poked")
    assert poke(cfg())[0] is True


def test_falls_back_to_always_on_channels():
    c = cfg(idle_channels=[], always_on_channels=["100"])
    assert poke(c)[0] is True


def test_waits_for_the_whole_silence():
    assert poke(cfg(), idle_seconds=DAY - 60) == (False, "not quiet long enough")
    assert poke(cfg(), idle_seconds=DAY) == (True, "idle")


def test_fractional_hours():
    c = cfg(idle_hours=0.5)
    assert poke(c, idle_seconds=1700)[0] is False
    assert poke(c, idle_seconds=1900)[0] is True


def test_monthly_cap_blocks_the_poke():
    c = cfg(max_replies_per_month=10)
    assert poke(c, monthly_count=10) == (False, "monthly cap")


def test_hourly_cap_blocks_the_poke():
    state = TriggerState()
    for _ in range(30):
        state.note_reply("100", 990.0)
    assert poke(cfg(max_replies_per_hour=30), state=state) == (False, "hourly cap")


def test_cooldown_does_not_block_the_poke():
    """A day of silence outranks a 45 second per channel cooldown."""
    state = TriggerState()
    state.note_reply("100", 999.0)
    assert poke(cfg(), state=state)[0] is True


IDLE_TOML = """\
name = "x"

[discord]
idle_hours = 12
idle_channels = ["100", "200"]
"""


def write(tmp_path, text):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_reads_idle_settings(tmp_path):
    discord = load_config(write(tmp_path, IDLE_TOML)).discord
    assert discord.idle_hours == 12
    assert discord.poke_channels() == ["100", "200"]


def test_idle_hours_without_a_channel_is_an_error(tmp_path):
    text = 'name = "x"\n[discord]\nidle_hours = 24\n'
    with pytest.raises(ConfigError, match="idle_channels"):
        load_config(write(tmp_path, text))


def test_negative_idle_hours_rejected(tmp_path):
    text = 'name = "x"\n[discord]\nidle_hours = -1\nidle_channels = ["1"]\n'
    with pytest.raises(ConfigError, match="negative"):
        load_config(write(tmp_path, text))


class FakeChannel:
    """Enough of a channel for a dry run, which never actually sends."""

    id = 100

    def __str__(self) -> str:
        return "general"


REACTION_TOML = """\
name = "testbot"

[llm]
provider = "ollama"

[[reactions]]
name = "angry"
url = "https://tenor.com/view/angry"
"""


def make_client(persona_dir, fake_provider, toml: str | None = None) -> MimicClient:
    if toml:
        (persona_dir / "persona.toml").write_text(toml, encoding="utf-8")
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    return MimicClient(engine, dry_run=True)


def sent(client) -> list[str]:
    return [m.content for m in client.buffers[FakeChannel.id]]


def test_mention_rides_on_the_first_message(persona_dir, fake_provider):
    client = make_client(persona_dir, fake_provider)
    asyncio.run(client._send_bursts(FakeChannel(), ["ej", "kaj delas"], "<@7>"))
    assert sent(client) == ["<@7> ej", "kaj delas"]


def test_mention_skips_a_leading_gif(persona_dir, fake_provider):
    client = make_client(persona_dir, fake_provider, REACTION_TOML)
    asyncio.run(client._send_bursts(FakeChannel(), ["[gif:angry]", "ma daj"], "<@7>"))
    assert sent(client) == ["[gif:angry]", "<@7> ma daj"]


def test_gif_only_poke_pings_first(persona_dir, fake_provider):
    client = make_client(persona_dir, fake_provider, REACTION_TOML)
    asyncio.run(client._send_bursts(FakeChannel(), ["[gif:angry]"], "<@7>"))
    assert sent(client) == ["<@7>", "[gif:angry]"]


def test_unusable_gif_takes_the_ping_with_it(persona_dir, fake_provider):
    """Better to stay quiet than to ping someone with nothing attached."""
    client = make_client(persona_dir, fake_provider, REACTION_TOML)
    asyncio.run(client._send_bursts(FakeChannel(), ["[gif:nope]"], "<@7>"))
    assert sent(client) == []


def test_normal_replies_are_untouched(persona_dir, fake_provider):
    client = make_client(persona_dir, fake_provider)
    asyncio.run(client._send_bursts(FakeChannel(), ["ne", "mam kolokvij"]))
    assert sent(client) == ["ne", "mam kolokvij"]


def test_poke_pings_someone_who_talks_here(persona_dir, fake_provider):
    fake_provider.reply = "kje si crknu"
    client = make_client(persona_dir, fake_provider)
    client.cfg.idle_hours = 24
    channel = FakeChannel()
    client.seeded.add(channel.id)  # pretend history is already backfilled
    client.people[channel.id] = {7: "kekSoslav"}

    asyncio.run(client._poke(channel))

    assert sent(client) == ["<@7> kje si crknu"]
    prompt = fake_provider.calls[0]["messages"][-1].content
    assert "[direction]" in prompt
    assert "kekSoslav" in prompt
    assert "24 hours" in prompt


def test_poke_needs_somebody_to_ping(persona_dir, fake_provider):
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.seeded.add(channel.id)

    asyncio.run(client._poke(channel))

    assert sent(client) == []
    assert fake_provider.calls == []  # no llm call for an empty room
