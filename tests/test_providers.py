from __future__ import annotations

import anthropic
import openai
import pytest

from mimicord.llm.anthropic_provider import AnthropicProvider
from mimicord.llm.base import ChatMessage, parse_json_lenient
from mimicord.llm.openai_compat import OpenAICompatProvider


class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _AnthropicResponse:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"
        self.usage = _Usage()


@pytest.fixture
def anthropic_calls(monkeypatch):
    calls = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _AnthropicResponse("zivjo")

    class _Client:
        def __init__(self, *args, **kwargs):
            self.messages = _Messages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    return calls


def test_anthropic_chat_request_shape(anthropic_calls):
    provider = AnthropicProvider("claude-opus-5")
    out = provider.complete(
        system="be janez",
        messages=[
            ChatMessage("user", "few shot", cache_boundary=True),
            ChatMessage("user", "live context"),
        ],
        max_tokens=200,
        temperature=0.7,
    )
    assert out == "zivjo"
    call = anthropic_calls[0]
    # sampling params never sent to claude
    assert "temperature" not in call
    # system is a block list carrying the cache marker
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["system"][0]["text"] == "be janez"
    # chat mode: thinking off at low effort
    assert call["thinking"] == {"type": "disabled"}
    assert call["output_config"] == {"effort": "low"}
    # cache boundary message converted to block form with the marker
    boundary, live = call["messages"]
    assert boundary["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert live["content"] == "live context"


def test_anthropic_1h_ttl(anthropic_calls):
    provider = AnthropicProvider("claude-opus-5", cache_ttl="1h")
    provider.complete(system="s", messages=[ChatMessage("user", "u")], max_tokens=10)
    marker = anthropic_calls[0]["system"][0]["cache_control"]
    assert marker == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_older_models_get_no_extras(anthropic_calls):
    provider = AnthropicProvider("claude-haiku-4-5")
    provider.complete(system="s", messages=[ChatMessage("user", "u")], max_tokens=10)
    call = anthropic_calls[0]
    assert "thinking" not in call
    assert "output_config" not in call


def test_anthropic_thinking_mode_for_analysis(anthropic_calls):
    provider = AnthropicProvider("claude-opus-5", thinking=True)
    provider.complete(system="s", messages=[ChatMessage("user", "u")], max_tokens=4000)
    call = anthropic_calls[0]
    # adaptive thinking is the model default, only effort is set
    assert "thinking" not in call
    assert call["output_config"] == {"effort": "high"}


def test_anthropic_fable_never_disables_thinking(anthropic_calls):
    provider = AnthropicProvider("claude-fable-5")
    provider.complete(system="s", messages=[ChatMessage("user", "u")], max_tokens=10)
    call = anthropic_calls[0]
    assert "thinking" not in call
    assert call["output_config"] == {"effort": "low"}


class _Choice:
    def __init__(self, text):
        class _Message:
            content = text

        self.message = _Message()
        self.finish_reason = "stop"


class _OpenAIResponse:
    def __init__(self, text):
        self.choices = [_Choice(text)]


@pytest.fixture
def openai_env(monkeypatch):
    constructed = []
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _OpenAIResponse("hej")

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, *, base_url=None, api_key=None):
            constructed.append({"base_url": base_url, "api_key": api_key})
            self.chat = _Chat()

    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(openai, "OpenAI", _Client)
    return constructed, calls


def test_openai_uses_max_completion_tokens(openai_env):
    constructed, calls = openai_env
    provider = OpenAICompatProvider("openai", "gpt-4o")
    provider.complete(
        system="sys", messages=[ChatMessage("user", "hi")], max_tokens=99, temperature=0.5
    )
    assert constructed[0]["base_url"] is None
    call = calls[0]
    assert call["max_completion_tokens"] == 99
    assert "max_tokens" not in call
    assert call["temperature"] == 0.5
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1] == {"role": "user", "content": "hi"}


def test_deepseek_base_url_and_max_tokens(openai_env):
    constructed, calls = openai_env
    provider = OpenAICompatProvider("deepseek", "deepseek-chat")
    provider.complete(system="s", messages=[ChatMessage("user", "u")], max_tokens=50)
    assert constructed[0]["base_url"] == "https://api.deepseek.com"
    assert calls[0]["max_tokens"] == 50
    assert "temperature" not in calls[0]


def test_ollama_needs_no_key(openai_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    constructed, _ = openai_env
    OpenAICompatProvider("ollama", "llama3.1")
    assert constructed[0]["base_url"] == "http://localhost:11434/v1"
    assert constructed[0]["api_key"] == "ollama"


def test_parse_json_lenient():
    assert parse_json_lenient('{"a": 1}') == {"a": 1}
    assert parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_lenient('Here you go:\n{"a": 1}\nEnjoy!') == {"a": 1}
    assert parse_json_lenient("[1, 2]") == [1, 2]
