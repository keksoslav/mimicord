from __future__ import annotations

import pytest

from mimicord.config import ConfigError, load_config

FULL = """\
name = "janez"

[target]
author_ids = ["123"]
author_names = ["janez", "Janez K."]

[llm]
provider = "deepseek"
model = "deepseek-chat"
max_tokens = 250
temperature = 0.9
cache_ttl = "1h"

[llm.analyze]
model = "claude-haiku-4-5"
reduce_model = "claude-opus-5"

[discord]
token_env = "DISCORD_TOKEN_JANEZ"
trigger_keywords = ["Janez"]
interject_probability = 0.05
cooldown_seconds = 30
max_replies_per_hour = 10

[rag]
enabled = false
top_k = 6

[style]
max_burst = 2
"""


def write(tmp_path, text):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_full_config(tmp_path):
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.name == "janez"
    assert cfg.target.author_ids == ["123"]
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.temperature == 0.9
    assert cfg.llm.cache_ttl == "1h"
    assert cfg.llm.analyze.model == "claude-haiku-4-5"
    assert cfg.llm.analyze.reduce_model == "claude-opus-5"
    assert cfg.discord.token_env == "DISCORD_TOKEN_JANEZ"
    assert cfg.discord.trigger_keywords == ["janez"]  # lowercased
    assert cfg.discord.interject_probability == 0.05
    assert cfg.rag.enabled is False
    assert cfg.rag.top_k == 6
    assert cfg.style.max_burst == 2


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path, 'name = "x"\n'))
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-opus-5"
    assert cfg.llm.temperature is None
    assert cfg.discord.cooldown_seconds == 45.0
    assert cfg.discord.context_messages == 25
    assert cfg.rag.enabled is True
    assert cfg.style.max_burst == 3


def test_missing_name(tmp_path):
    with pytest.raises(ConfigError, match="name"):
        load_config(write(tmp_path, '[llm]\nprovider = "openai"\n'))


def test_claude_code_provider(tmp_path):
    cfg = load_config(
        write(tmp_path, 'name = "x"\n[llm]\nprovider = "claude-code"\n')
    )
    assert cfg.llm.model == "sonnet"


def test_monthly_cap_parsed(tmp_path):
    cfg = load_config(
        write(tmp_path, 'name = "x"\n[discord]\nmax_replies_per_month = 500\n')
    )
    assert cfg.discord.max_replies_per_month == 500


def test_bad_provider(tmp_path):
    with pytest.raises(ConfigError, match="unknown provider"):
        load_config(write(tmp_path, 'name = "x"\n[llm]\nprovider = "gemini"\n'))


def test_bad_ttl(tmp_path):
    with pytest.raises(ConfigError, match="cache_ttl"):
        load_config(write(tmp_path, 'name = "x"\n[llm]\ncache_ttl = "2h"\n'))


def test_bad_interject(tmp_path):
    with pytest.raises(ConfigError, match="interject"):
        load_config(write(tmp_path, 'name = "x"\n[discord]\ninterject_probability = 3\n'))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no persona config"):
        load_config(tmp_path / "nope.toml")


def test_token_env_resolution(tmp_path, monkeypatch):
    cfg = load_config(write(tmp_path, 'name = "x"\n[discord]\ntoken_env = "T_X"\n'))
    monkeypatch.setenv("T_X", "secret")
    assert cfg.discord.token() == "secret"
    monkeypatch.delenv("T_X")
    with pytest.raises(ConfigError, match="T_X"):
        cfg.discord.token()
