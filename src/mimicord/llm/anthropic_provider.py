from __future__ import annotations

import logging
import os

import anthropic

from mimicord.llm.base import ChatMessage, ProviderError

log = logging.getLogger(__name__)

_FIVE_FAMILY = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5")


class AnthropicProvider:
    """Claude through the official anthropic SDK.

    The system prompt always carries a cache_control marker, and a message
    flagged cache_boundary extends the cached prefix over the few-shots too.
    Short chat replies run with thinking off at low effort; analysis calls
    (thinking=True) keep the model's default reasoning at high effort.
    """

    def __init__(self, model: str, cache_ttl: str = "5m", thinking: bool = False) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self._model = model
        self._thinking = thinking
        self._cache_control: dict = {"type": "ephemeral"}
        if cache_ttl == "1h":
            self._cache_control = {"type": "ephemeral", "ttl": "1h"}

    def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        # temperature is intentionally ignored: current Claude models reject
        # sampling parameters
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[
                {"type": "text", "text": system, "cache_control": self._cache_control}
            ],
            messages=[self._to_wire(m) for m in messages],
            **self._request_extras(),
        )
        if response.stop_reason == "refusal":
            raise ProviderError("the model declined this request")
        usage = response.usage
        log.debug(
            "anthropic usage: in=%s out=%s cache_read=%s cache_write=%s",
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise ProviderError(f"empty response (stop_reason={response.stop_reason})")
        return text

    def _request_extras(self) -> dict:
        if not self._model.startswith(_FIVE_FAMILY):
            # 4.x models: omitted thinking means off, and effort is not
            # supported everywhere (haiku rejects it), so send nothing
            return {}
        if self._thinking:
            return {"output_config": {"effort": "high"}}
        if self._model.startswith("claude-fable-5"):
            # fable rejects an explicit disabled, thinking is always on there
            return {"output_config": {"effort": "low"}}
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}

    def _to_wire(self, message: ChatMessage) -> dict:
        if message.cache_boundary:
            return {
                "role": message.role,
                "content": [
                    {
                        "type": "text",
                        "text": message.content,
                        "cache_control": self._cache_control,
                    }
                ],
            }
        return {"role": message.role, "content": message.content}
