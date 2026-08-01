from __future__ import annotations

import logging
import os

import openai

from mimicord.llm.base import ChatMessage, ProviderError

log = logging.getLogger(__name__)

# provider -> (base_url, api key env var)
_BASES = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAICompatProvider:
    """openai, deepseek and ollama all speak the same chat completions API."""

    def __init__(self, provider: str, model: str) -> None:
        if provider not in _BASES:
            raise ProviderError(f"unknown openai compatible provider {provider!r}")
        base_url, key_env = _BASES[provider]
        if key_env:
            api_key = os.environ.get(key_env, "")
            if not api_key:
                raise ProviderError(f"{key_env} is not set")
        else:
            api_key = "ollama"  # ollama ignores the key but the SDK wants one
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._provider = provider
        self._model = model

    def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        wire = [{"role": "system", "content": system}]
        wire += [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        # newer openai models only accept max_completion_tokens; deepseek and
        # ollama still expect max_tokens
        if self._provider == "openai":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        response = self._client.chat.completions.create(
            model=self._model, messages=wire, **kwargs
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise ProviderError(f"empty response (finish_reason={choice.finish_reason})")
        return text
