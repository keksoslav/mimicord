from __future__ import annotations

import asyncio

import pytest

from mimicord.bot import MimicClient
from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.llm.base import ProviderError


class ExplodingTyping:
    """Discord dropping the connection while we show the typing indicator."""

    async def __aenter__(self):
        raise OSError("[WinError 121] The semaphore timeout period has expired")

    async def __aexit__(self, *exc):
        return False


class FakeChannel:
    id = 100

    def __init__(self, *, typing_works: bool = True) -> None:
        self.sent: list[str] = []
        self.typing_works = typing_works

    def typing(self):
        if not self.typing_works:
            return ExplodingTyping()
        channel = self

        class Ok:
            async def __aenter__(self):
                channel.sent.append("<typing>")

            async def __aexit__(self, *exc):
                return False

        return Ok()

    async def send(self, content=None, *, file=None):
        self.sent.append(content if content is not None else "<file>")

    def __str__(self) -> str:
        return "general"


@pytest.fixture
def nosleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr("mimicord.bot.asyncio.sleep", instant)


def make_client(persona_dir, fake_provider) -> MimicClient:
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    client = MimicClient(engine)  # live, not dry run: we want the send path
    client.seeded.add(FakeChannel.id)
    return client


def test_reply_survives_a_dead_typing_indicator(persona_dir, fake_provider, nosleep):
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel(typing_works=False)

    asyncio.run(client._send_bursts(channel, ["ne", "mam kolokvij"]))

    assert channel.sent == ["ne", "mam kolokvij"]


def test_typing_indicator_is_used_when_it_works(persona_dir, fake_provider, nosleep):
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()

    asyncio.run(client._send_bursts(channel, ["ne"]))

    assert channel.sent == ["<typing>", "ne"]


def test_respond_still_sends_when_typing_is_broken(persona_dir, fake_provider, nosleep):
    fake_provider.reply = "kk"
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel(typing_works=False)
    client.buffers[channel.id].append(ContextMessage("Mike_", "a gres cs"))

    asyncio.run(client._respond(channel, "mention"))

    assert channel.sent == ["kk"]


class FakeSDKError(Exception):
    """Stands in for the sdk's 'Reached maximum number of turns' result."""


def fake_stream(*, text: str | None):
    """An sdk query() that streams some text and then errors, like a turn
    limit hit after the model already answered."""
    import claude_agent_sdk

    async def stream(prompt, options):
        if text is not None:
            yield claude_agent_sdk.AssistantMessage(
                content=[claude_agent_sdk.TextBlock(text=text)], model="opus"
            )
        raise FakeSDKError("Claude Code returned an error result: max turns (1)")

    return stream


def provider() -> "object":
    from mimicord.llm.claude_code import ClaudeCodeProvider

    instance = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
    instance._model = "opus"
    return instance


def test_agent_sdk_error_keeps_text_it_already_streamed(monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", fake_stream(text="jao ne\nl bro"))
    assert provider().complete(system="s", messages=[], max_tokens=10) == "jao ne\nl bro"


def test_agent_sdk_error_with_no_text_still_raises(monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", fake_stream(text=None))
    with pytest.raises(ProviderError, match="agent sdk call failed"):
        provider().complete(system="s", messages=[], max_tokens=10)
