from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class FakeProvider:
    reply: str = "ok"
    replies: list = field(default_factory=list)  # scripted, consumed in order
    calls: list = field(default_factory=list)

    def complete(self, *, system, messages, max_tokens, temperature=None) -> str:
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.replies:
            return self.replies.pop(0)
        return self.reply


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def personas_home(tmp_path, monkeypatch):
    """Point the personas root at a temp dir."""
    root = tmp_path / "personas"
    root.mkdir()
    monkeypatch.setenv("MIMICORD_HOME", str(root))
    return root


MINIMAL_TOML = """\
name = "testbot"

[llm]
provider = "ollama"
model = "llama3.1"
"""


@pytest.fixture
def persona_dir(personas_home):
    """A minimal persona on disk (ollama so no api key is needed)."""
    d = personas_home / "testbot"
    d.mkdir()
    (d / "persona.toml").write_text(MINIMAL_TOML, encoding="utf-8")
    (d / "persona.md").write_text("# Persona: testbot\nshort replies.", encoding="utf-8")
    return d
