from __future__ import annotations

from mimicord.config import DiscordConfig
from mimicord.triggers import MessageFacts, TriggerState, should_reply


def facts(**overrides) -> MessageFacts:
    base = dict(
        channel_id="100",
        author_is_self=False,
        author_is_bot=False,
        mentions_bot=False,
        replies_to_bot=False,
        content="kaj dogaja",
    )
    base.update(overrides)
    return MessageFacts(**base)


def cfg(**overrides) -> DiscordConfig:
    config = DiscordConfig(token_env="T")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def decide(f, c, state=None, now=1000.0, roll=0.99, monthly_count=0):
    return should_reply(f, c, state or TriggerState(), now, roll, monthly_count)


def test_ignores_own_messages():
    assert decide(facts(author_is_self=True, mentions_bot=True), cfg()) == (
        False,
        "own message",
    )


def test_ignores_bots_by_default():
    assert decide(facts(author_is_bot=True, mentions_bot=True), cfg())[0] is False


def test_bots_allowed_when_configured():
    c = cfg(ignore_bots=False)
    assert decide(facts(author_is_bot=True, mentions_bot=True), c)[0] is True


def test_allowlist_blocks_other_channels():
    c = cfg(channel_allowlist=["200"])
    assert decide(facts(mentions_bot=True), c) == (False, "channel not in allowlist")
    assert decide(facts(mentions_bot=True, channel_id="200"), c)[0] is True


def test_mention_triggers():
    assert decide(facts(mentions_bot=True), cfg()) == (True, "mention")


def test_mention_can_be_disabled():
    c = cfg(trigger_mention=False)
    assert decide(facts(mentions_bot=True), c) == (False, "no trigger")


def test_reply_to_bot_triggers():
    assert decide(facts(replies_to_bot=True), cfg()) == (True, "reply")


def test_keyword_matches_word_boundary():
    c = cfg(trigger_keywords=["janez"])
    assert decide(facts(content="JANEZ si tu?"), c) == (True, "keyword")
    assert decide(facts(content="janezovo kolo"), c) == (False, "no trigger")


def test_always_on_channel():
    c = cfg(always_on_channels=["100"])
    assert decide(facts(), c) == (True, "always on")


def test_interject_roll():
    c = cfg(interject_probability=0.1)
    assert decide(facts(), c, roll=0.05) == (True, "interject")
    assert decide(facts(), c, roll=0.5) == (False, "no trigger")


def test_no_trigger_no_reply():
    assert decide(facts(), cfg()) == (False, "no trigger")


def test_cooldown_blocks_same_channel_only():
    c = cfg(cooldown_seconds=45)
    state = TriggerState()
    state.note_reply("100", 1000.0)
    assert decide(facts(mentions_bot=True), c, state, now=1030.0) == (
        False,
        "cooldown",
    )
    assert decide(
        facts(mentions_bot=True, channel_id="300"), c, state, now=1030.0
    ) == (True, "mention")
    assert decide(facts(mentions_bot=True), c, state, now=1046.0)[0] is True


def test_hourly_cap():
    c = cfg(max_replies_per_hour=2, cooldown_seconds=0)
    state = TriggerState()
    state.note_reply("100", 1000.0)
    state.note_reply("100", 1001.0)
    assert decide(facts(mentions_bot=True), c, state, now=1002.0) == (
        False,
        "hourly cap",
    )


def test_monthly_cap_blocks():
    c = cfg(max_replies_per_month=100)
    assert decide(facts(mentions_bot=True), c, monthly_count=100) == (
        False,
        "monthly cap",
    )
    assert decide(facts(mentions_bot=True), c, monthly_count=99)[0] is True


def test_monthly_cap_zero_means_unlimited():
    assert decide(facts(mentions_bot=True), cfg(), monthly_count=10_000)[0] is True


def test_hourly_cap_expires():
    c = cfg(max_replies_per_hour=2, cooldown_seconds=0)
    state = TriggerState()
    state.note_reply("100", 1000.0)
    state.note_reply("100", 1001.0)
    # both entries fall out of the rolling hour
    assert decide(facts(mentions_bot=True), c, state, now=5000.0)[0] is True
    assert len(state.sent_at) == 0
