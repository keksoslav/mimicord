from __future__ import annotations

from mimicord.config import LLMConfig
from mimicord.llm.anthropic_provider import AnthropicProvider
from mimicord.llm.base import Provider
from mimicord.llm.openai_compat import OpenAICompatProvider


def get_provider(cfg: LLMConfig, role: str = "chat") -> Provider:
    """Build a provider for a role.

    chat: runtime replies, low latency, no thinking.
    map: the cheap per chunk analysis step.
    reduce: profile merging and persona authoring, quality matters.
    """
    model = cfg.model
    thinking = False
    if role == "map":
        model = cfg.analyze.model or cfg.model
    elif role == "reduce":
        model = cfg.analyze.reduce_model or cfg.model
        thinking = True

    if cfg.provider == "anthropic":
        return AnthropicProvider(model, cache_ttl=cfg.cache_ttl, thinking=thinking)
    return OpenAICompatProvider(cfg.provider, model)
