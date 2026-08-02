from __future__ import annotations

from mimicord.config import DiscordConfig, load_config
from mimicord.triggers import MessageFacts, TriggerState, should_reply


def facts(**overrides) -> MessageFacts:
    base = dict(
        channel_id="100",
        author_is_self=False,
        author_is_bot=False,
        mentions_bot=False,
        replies_to_bot=False,
        content="timi",
        is_dm=True,
    )
    base.update(overrides)
    return MessageFacts(**base)


def cfg(**overrides) -> DiscordConfig:
    config = DiscordConfig(token_env="T", trigger_keywords=["timi"])
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def decide(f, c=None):
    return should_reply(f, c or cfg(), TriggerState(), 1000.0, 0.99)


def test_dms_are_ignored_by_default():
    assert decide(facts()) == (False, "dm")


def test_a_dm_mention_is_still_ignored():
    """A dm is not a special case worth answering, it is the risky one."""
    assert decide(facts(mentions_bot=True)) == (False, "dm")
    assert decide(facts(replies_to_bot=True)) == (False, "dm")


def test_dms_beat_the_always_on_list():
    c = cfg(always_on_channels=["100"])
    assert decide(facts(), c) == (False, "dm")


def test_the_same_message_in_a_server_is_answered():
    assert decide(facts(is_dm=False)) == (True, "keyword")


def test_dms_can_be_turned_on():
    assert decide(facts(), cfg(allow_dms=True)) == (True, "keyword")


def test_off_by_default_in_config(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text('name = "x"\n', encoding="utf-8")
    assert load_config(path).discord.allow_dms is False


def test_can_be_enabled_in_config(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text('name = "x"\n[discord]\nallow_dms = true\n', encoding="utf-8")
    assert load_config(path).discord.allow_dms is True
