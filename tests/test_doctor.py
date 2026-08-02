from __future__ import annotations

import json

import pytest

from mimicord.config import PersonaConfig, load_config
from mimicord.doctor import check_artifacts, check_discord, check_provider
from mimicord.paths import PersonaPaths


def by_name(checks) -> dict:
    return {c.name: c for c in checks}


@pytest.fixture
def paths_and_cfg(persona_dir) -> tuple[PersonaPaths, PersonaConfig]:
    paths = PersonaPaths.for_persona("testbot")
    return paths, load_config(paths.config)


def test_fresh_persona_reports_missing_pieces(paths_and_cfg):
    paths, cfg = paths_and_cfg
    checks = by_name(check_artifacts(paths, cfg))
    assert checks["corpus"].ok is None  # not yet ingested, not a failure
    assert checks["stats"].ok is False
    assert checks["few-shots"].ok is None
    assert checks["memories"].ok is False  # rag on by default


def test_starter_persona_is_flagged_as_uncompiled(paths_and_cfg):
    paths, cfg = paths_and_cfg
    checks = by_name(check_artifacts(paths, cfg))
    # the conftest persona.md is a stub, so it should nudge toward compile
    assert "compile" in checks["persona.md"].detail


def test_corpus_with_no_target_fails(paths_and_cfg):
    from mimicord.store import Message, Store

    paths, cfg = paths_and_cfg
    with Store(paths.corpus) as store:
        store.upsert_many(
            [
                Message(
                    id="1", channel_id="c", channel_name=None, guild_name=None,
                    author_id="999", author_name="someone", is_target=False,
                    content="hi", timestamp="2024-01-01T00:00:00+00:00",
                    reply_to_id=None, attachments=0, source="dce",
                )
            ]
        )
    checks = by_name(check_artifacts(paths, cfg))
    assert checks["corpus"].ok is False
    assert "author_ids" in checks["corpus"].detail


def test_examples_counted(paths_and_cfg):
    paths, cfg = paths_and_cfg
    paths.examples.write_text(
        json.dumps({"examples": [{"context": [], "reply": ["a"]}]}), encoding="utf-8"
    )
    checks = by_name(check_artifacts(paths, cfg))
    assert checks["few-shots"].ok is True
    assert "1 examples" in checks["few-shots"].detail


def test_rag_disabled_is_skipped_not_failed(persona_dir):
    persona_dir.joinpath("persona.toml").write_text(
        'name = "testbot"\n[llm]\nprovider = "ollama"\n[rag]\nenabled = false\n',
        encoding="utf-8",
    )
    paths = PersonaPaths.for_persona("testbot")
    checks = by_name(check_artifacts(paths, load_config(paths.config)))
    assert checks["memories"].ok is None


def test_provider_check_ollama_needs_no_key(paths_and_cfg):
    _, cfg = paths_and_cfg
    assert check_provider(cfg).ok is True


def test_provider_check_flags_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = tmp_path / "persona.toml"
    config.write_text('name = "x"\n[llm]\nprovider = "anthropic"\n', encoding="utf-8")
    check = check_provider(load_config(config))
    assert check.ok is False
    assert "ANTHROPIC_API_KEY" in check.detail


def test_provider_check_claude_code_needs_no_key(tmp_path):
    config = tmp_path / "persona.toml"
    config.write_text('name = "x"\n[llm]\nprovider = "claude-code"\n', encoding="utf-8")
    assert check_provider(load_config(config)).ok is True


def test_discord_check_without_token_env(paths_and_cfg):
    _, cfg = paths_and_cfg
    checks = check_discord(cfg)
    assert checks[0].ok is False
    assert "token_env" in checks[0].detail


def test_discord_check_reports_missing_intent(tmp_path, monkeypatch):
    config = tmp_path / "persona.toml"
    config.write_text(
        'name = "x"\n[discord]\ntoken_env = "T_TEST"\n', encoding="utf-8"
    )
    monkeypatch.setenv("T_TEST", "fake-token")
    monkeypatch.setattr(
        "mimicord.doctor._probe_discord",
        lambda token: _immediate({"error": "intents"}),
    )
    checks = by_name(check_discord(load_config(config)))
    assert checks["discord token"].ok is True
    assert checks["message content intent"].ok is False


def test_discord_check_offers_invite_when_in_no_servers(tmp_path, monkeypatch):
    config = tmp_path / "persona.toml"
    config.write_text('name = "x"\n[discord]\ntoken_env = "T_TEST"\n', encoding="utf-8")
    monkeypatch.setenv("T_TEST", "fake-token")
    monkeypatch.setattr(
        "mimicord.doctor._probe_discord",
        lambda token: _immediate({"user": "SaNdMaN#5845", "id": 42, "guilds": []}),
    )
    checks = by_name(check_discord(load_config(config)))
    assert checks["servers"].ok is False
    assert "client_id=42" in checks["servers"].detail
    assert "permissions=68608" in checks["servers"].detail


async def _immediate(value):
    return value


def test_oversized_reaction_is_flagged(personas_home):
    """Discord rejects attachments over 10 MB; catch it before runtime."""
    from mimicord.doctor import DISCORD_UPLOAD_LIMIT

    root = personas_home / "big"
    (root / "media").mkdir(parents=True)
    (root / "persona.toml").write_text(
        'name = "big"\n[llm]\nprovider = "ollama"\n'
        '[[reactions]]\nname = "huge"\nfile = "huge.gif"\n',
        encoding="utf-8",
    )
    (root / "media" / "huge.gif").write_bytes(b"\0" * (DISCORD_UPLOAD_LIMIT + 1))
    paths = PersonaPaths.for_persona("big")
    checks = by_name(check_artifacts(paths, load_config(paths.config)))
    assert checks["reaction huge"].ok is False
    assert "10 MB" in checks["reaction huge"].detail


def test_url_reaction_needs_no_file(personas_home):
    root = personas_home / "urlbot"
    root.mkdir()
    (root / "persona.toml").write_text(
        'name = "urlbot"\n[llm]\nprovider = "ollama"\n'
        '[[reactions]]\nname = "dedi"\nurl = "https://tenor.com/view/x"\n',
        encoding="utf-8",
    )
    paths = PersonaPaths.for_persona("urlbot")
    checks = by_name(check_artifacts(paths, load_config(paths.config)))
    assert checks["reaction dedi"].ok is True
    assert checks["reaction dedi"].detail == "url"
