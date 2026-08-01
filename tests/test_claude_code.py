from __future__ import annotations

import sys
import types

import pytest

from mimicord.llm.base import ChatMessage, ProviderError


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _AssistantMessage:
    def __init__(self, blocks):
        self.content = blocks


class _ResultMessage:
    total_cost_usd = 0.0123


@pytest.fixture
def sdk_stub(monkeypatch):
    """Fake claude_agent_sdk so tests never spawn the real harness."""
    captured: dict = {}
    module = types.ModuleType("claude_agent_sdk")

    class _Options:
        def __init__(self, **kwargs):
            captured["options"] = kwargs

    async def _query(*, prompt, options):
        captured["prompt"] = prompt
        yield _AssistantMessage([_TextBlock("zivjo"), _TextBlock(" no")])
        yield _ResultMessage()

    module.ClaudeAgentOptions = _Options
    module.query = _query
    module.AssistantMessage = _AssistantMessage
    module.TextBlock = _TextBlock
    module.ResultMessage = _ResultMessage
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return captured


def test_flattens_messages_and_locks_down_options(sdk_stub):
    from mimicord.llm.claude_code import ClaudeCodeProvider

    provider = ClaudeCodeProvider("sonnet")
    out = provider.complete(
        system="be janez",
        messages=[
            ChatMessage("user", "[chat]\nmiha: a gres?\n[/chat]"),
            ChatMessage("assistant", "ne"),
            ChatMessage("user", "[chat]\nmiha: pridi\n[/chat]"),
        ],
        max_tokens=400,
        temperature=0.9,
    )
    assert out == "zivjo no"
    options = sdk_stub["options"]
    assert options["system_prompt"] == "be janez"
    assert options["model"] == "sonnet"
    assert options["max_turns"] == 1
    assert options["allowed_tools"] == []
    assert options["setting_sources"] == []
    # few-shot pair flattened into the prompt, live message last
    prompt = sdk_stub["prompt"]
    assert "you replied:\nne" in prompt
    assert prompt.endswith("[chat]\nmiha: pridi\n[/chat]")


def test_empty_response_raises(monkeypatch):
    module = types.ModuleType("claude_agent_sdk")

    class _Options:
        def __init__(self, **kwargs):
            pass

    async def _query(*, prompt, options):
        yield _ResultMessage()

    module.ClaudeAgentOptions = _Options
    module.query = _query
    module.AssistantMessage = _AssistantMessage
    module.TextBlock = _TextBlock
    module.ResultMessage = _ResultMessage
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    from mimicord.llm.claude_code import ClaudeCodeProvider

    provider = ClaudeCodeProvider("sonnet")
    with pytest.raises(ProviderError, match="logged in"):
        provider.complete(
            system="s", messages=[ChatMessage("user", "hej")], max_tokens=100
        )


def test_factory_builds_claude_code_provider(sdk_stub):
    from mimicord.config import LLMConfig
    from mimicord.llm.claude_code import ClaudeCodeProvider
    from mimicord.llm.factory import get_provider

    provider = get_provider(LLMConfig(provider="claude-code", model="opus"))
    assert isinstance(provider, ClaudeCodeProvider)
