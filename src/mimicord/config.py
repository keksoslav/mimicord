from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROVIDERS = ("anthropic", "openai", "deepseek", "ollama", "claude-code")

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.1",
    "claude-code": "sonnet",  # agent sdk accepts aliases and full model ids
}


class ConfigError(Exception):
    pass


@dataclass
class TargetConfig:
    author_ids: list[str] = field(default_factory=list)
    author_names: list[str] = field(default_factory=list)


@dataclass
class AnalyzeLLMConfig:
    model: str | None = None
    reduce_model: str | None = None


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = DEFAULT_MODELS["anthropic"]
    max_tokens: int = 400
    temperature: float | None = None
    cache_ttl: str = "5m"
    analyze: AnalyzeLLMConfig = field(default_factory=AnalyzeLLMConfig)


@dataclass
class DiscordConfig:
    token_env: str = ""
    trigger_mention: bool = True
    trigger_reply: bool = True
    trigger_keywords: list[str] = field(default_factory=list)
    interject_probability: float = 0.0
    always_on_channels: list[str] = field(default_factory=list)
    channel_allowlist: list[str] = field(default_factory=list)
    cooldown_seconds: float = 45.0
    max_replies_per_hour: int = 30
    max_replies_per_month: int = 0  # 0 = no monthly budget
    context_messages: int = 25
    ignore_bots: bool = True

    def token(self) -> str:
        if not self.token_env:
            raise ConfigError("discord.token_env is not set in persona.toml")
        value = os.environ.get(self.token_env, "")
        if not value:
            raise ConfigError(f"environment variable {self.token_env} is missing or empty")
        return value


@dataclass
class RagConfig:
    enabled: bool = True
    top_k: int = 4
    window_size: int = 8
    window_step: int = 4


@dataclass
class StyleConfig:
    max_burst: int = 3
    typing_cps: float = 7.0


@dataclass
class Reaction:
    """An image the persona can send instead of typing, by emitting a tag."""

    name: str
    file: str
    when: str = ""


@dataclass
class PersonaConfig:
    name: str
    target: TargetConfig
    llm: LLMConfig
    discord: DiscordConfig
    rag: RagConfig
    style: StyleConfig
    reactions: list[Reaction] = field(default_factory=list)


def _table(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _str_list(table: dict, key: str) -> list[str]:
    value = table.get(key, [])
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list")
    return [str(item) for item in value]


def load_config(path: Path) -> PersonaConfig:
    if not path.is_file():
        raise ConfigError(f"no persona config at {path} (run mimicord new first?)")
    with open(path, "rb") as f:
        data = tomllib.load(f)

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("persona.toml needs a top level name")

    target_data = _table(data, "target")
    target = TargetConfig(
        author_ids=_str_list(target_data, "author_ids"),
        author_names=_str_list(target_data, "author_names"),
    )

    llm_data = _table(data, "llm")
    provider = llm_data.get("provider", "anthropic")
    if provider not in PROVIDERS:
        raise ConfigError(
            f"unknown provider {provider!r}, expected one of: {', '.join(PROVIDERS)}"
        )
    cache_ttl = llm_data.get("cache_ttl", "5m")
    if cache_ttl not in ("5m", "1h"):
        raise ConfigError('llm.cache_ttl must be "5m" or "1h"')
    analyze_data = _table(llm_data, "analyze")
    llm = LLMConfig(
        provider=provider,
        model=str(llm_data.get("model", DEFAULT_MODELS[provider])),
        max_tokens=int(llm_data.get("max_tokens", 400)),
        temperature=llm_data.get("temperature"),
        cache_ttl=cache_ttl,
        analyze=AnalyzeLLMConfig(
            model=analyze_data.get("model"),
            reduce_model=analyze_data.get("reduce_model"),
        ),
    )

    discord_data = _table(data, "discord")
    interject = float(discord_data.get("interject_probability", 0.0))
    if not 0.0 <= interject <= 1.0:
        raise ConfigError("discord.interject_probability must be between 0 and 1")
    discord = DiscordConfig(
        token_env=str(discord_data.get("token_env", "")),
        trigger_mention=bool(discord_data.get("trigger_mention", True)),
        trigger_reply=bool(discord_data.get("trigger_reply", True)),
        trigger_keywords=[k.lower() for k in _str_list(discord_data, "trigger_keywords")],
        interject_probability=interject,
        always_on_channels=_str_list(discord_data, "always_on_channels"),
        channel_allowlist=_str_list(discord_data, "channel_allowlist"),
        cooldown_seconds=float(discord_data.get("cooldown_seconds", 45.0)),
        max_replies_per_hour=int(discord_data.get("max_replies_per_hour", 30)),
        max_replies_per_month=int(discord_data.get("max_replies_per_month", 0)),
        context_messages=int(discord_data.get("context_messages", 25)),
        ignore_bots=bool(discord_data.get("ignore_bots", True)),
    )

    rag_data = _table(data, "rag")
    rag = RagConfig(
        enabled=bool(rag_data.get("enabled", True)),
        top_k=int(rag_data.get("top_k", 4)),
        window_size=int(rag_data.get("window_size", 8)),
        window_step=int(rag_data.get("window_step", 4)),
    )

    style_data = _table(data, "style")
    style = StyleConfig(
        max_burst=int(style_data.get("max_burst", 3)),
        typing_cps=float(style_data.get("typing_cps", 7.0)),
    )

    reactions = []
    raw_reactions = data.get("reactions", [])
    if not isinstance(raw_reactions, list):
        raise ConfigError("[[reactions]] must be a list of tables")
    seen = set()
    for entry in raw_reactions:
        if not isinstance(entry, dict):
            raise ConfigError("each [[reactions]] entry must be a table")
        r_name = str(entry.get("name", "")).strip().lower()
        r_file = str(entry.get("file", "")).strip()
        if not r_name or not r_file:
            raise ConfigError("every [[reactions]] entry needs a name and a file")
        if not r_name.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(
                f"reaction name {r_name!r} must be letters, digits, _ or -"
            )
        if r_name in seen:
            raise ConfigError(f"duplicate reaction name {r_name!r}")
        seen.add(r_name)
        reactions.append(Reaction(r_name, r_file, str(entry.get("when", "")).strip()))

    return PersonaConfig(
        name=name,
        target=target,
        llm=llm,
        discord=discord,
        rag=rag,
        style=style,
        reactions=reactions,
    )
