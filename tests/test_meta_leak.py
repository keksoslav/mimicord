from __future__ import annotations

import asyncio

import pytest

from mimicord.llm.base import ProviderError
from mimicord.postprocess import apply, strip_meta_preamble


def test_the_leak_from_the_channel():
    """Posted live. The plan and the reply welded together with no space."""
    leaked = (
        "thanos gif from Bukvic, random. Respond in character, "
        "short.ka je zdj to za gif"
    )
    assert strip_meta_preamble(leaked) == "ka je zdj to za gif"


def test_other_shapes_of_the_same_thing():
    assert strip_meta_preamble("The user is asking about cs. neja nemorm") == (
        "neja nemorm"
    )
    assert strip_meta_preamble("Respond with something short. ma daj no") == "ma daj no"


def test_a_normal_reply_is_untouched():
    for reply in ("ka je zdj to", "neja sam se mi neda", "sm bil v sihtu / pol pa nic"):
        assert strip_meta_preamble(reply) == reply


def test_the_phrase_later_in_a_message_is_left_alone():
    """Only the opening counts, so ordinary use of the words survives."""
    long = "a" * 250 + ". in character pa to"
    assert strip_meta_preamble(long) == long


def test_nothing_is_cut_without_a_sentence_end():
    """Better to post one odd message than to eat a whole reply."""
    assert strip_meta_preamble("respond in character") == "respond in character"


def test_it_runs_as_part_of_the_normal_cleanup():
    bursts = apply(
        "The user asked about juvan. Respond in character.nvem stari",
        persona_name="sandman",
    )
    assert bursts == ["nvem stari"]


class FakeSDKError(Exception):
    pass


def provider():
    from mimicord.llm.claude_code import ClaudeCodeProvider

    instance = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
    instance._model = "opus"
    return instance


def stream_of(*turns: str, then_raise: bool = False):
    import claude_agent_sdk

    async def stream(prompt, options):
        for text in turns:
            yield claude_agent_sdk.AssistantMessage(
                content=[claude_agent_sdk.TextBlock(text=text)], model="opus"
            )
        if then_raise:
            raise FakeSDKError("max turns")

    return stream


def test_only_the_last_turn_is_kept(monkeypatch):
    """A planning turn must not be glued to the front of the answer."""
    import claude_agent_sdk

    monkeypatch.setattr(
        claude_agent_sdk, "query", stream_of("let me think about this.", "ka je zdj to")
    )
    assert provider().complete(system="s", messages=[], max_tokens=10) == "ka je zdj to"


def test_a_single_turn_is_unchanged(monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", stream_of("neja nemorm"))
    assert provider().complete(system="s", messages=[], max_tokens=10) == "neja nemorm"


def test_salvage_still_keeps_the_last_turn_on_an_error(monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(
        claude_agent_sdk, "query", stream_of("jao ne", then_raise=True)
    )
    assert provider().complete(system="s", messages=[], max_tokens=10) == "jao ne"


def test_an_error_with_nothing_said_still_raises(monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", stream_of(then_raise=True))
    with pytest.raises(ProviderError, match="agent sdk call failed"):
        provider().complete(system="s", messages=[], max_tokens=10)
