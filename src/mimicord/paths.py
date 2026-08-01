from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def personas_root() -> Path:
    override = os.environ.get("MIMICORD_HOME")
    if override:
        return Path(override)
    return Path.cwd() / "personas"


@dataclass(frozen=True)
class PersonaPaths:
    root: Path

    @classmethod
    def for_persona(cls, name: str) -> "PersonaPaths":
        return cls(personas_root() / name)

    @property
    def config(self) -> Path:
        return self.root / "persona.toml"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus.db"

    @property
    def stats(self) -> Path:
        return self.root / "stats.json"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def chunks_dir(self) -> Path:
        return self.analysis_dir / "chunks"

    @property
    def profile(self) -> Path:
        return self.analysis_dir / "profile.json"

    @property
    def persona_md(self) -> Path:
        return self.root / "persona.md"

    @property
    def examples(self) -> Path:
        return self.root / "examples.json"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma"
