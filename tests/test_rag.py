from __future__ import annotations

import re
import zlib
from pathlib import Path

import pytest

from mimicord.config import RagConfig, TargetConfig
from mimicord.ingest import ingest_dce
from mimicord.paths import PersonaPaths
from mimicord.rag import Memories, build_index
from mimicord.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
JANEZ = TargetConfig(author_ids=["111"])

DIMENSIONS = 64


def hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedding so tests never download a model."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture
def persona_paths(tmp_path):
    root = tmp_path / "personas" / "testbot"
    root.mkdir(parents=True)
    return PersonaPaths(root)


@pytest.fixture
def indexed(persona_paths, tmp_path):
    cfg = RagConfig(window_size=4, window_step=2, top_k=2)
    with Store(tmp_path / "corpus.db") as store:
        ingest_dce(store, [FIXTURES], JANEZ)
        count = build_index(persona_paths, cfg, store, embedder=hash_embed)
    return persona_paths, cfg, count


def test_build_index_counts_windows(indexed):
    _, _, count = indexed
    # 7 non-empty messages in channel 999, window 4 step 2 -> starts 0 and 2,
    # both contain janez
    assert count == 2


def test_query_finds_the_right_window(indexed):
    persona_paths, cfg, _ = indexed
    memories = Memories(persona_paths, cfg, embedder=hash_embed)
    hits = memories.query("kaj je blo s tistim kolokvijem")
    assert hits
    assert any("mam kolokvij" in h for h in hits)
    # formatted with date and channel
    assert hits[0].startswith("(2024-03-01, general)")


def test_query_empty_text_returns_nothing(indexed):
    persona_paths, cfg, _ = indexed
    memories = Memories(persona_paths, cfg, embedder=hash_embed)
    assert memories.query("   ") == []


def test_rebuild_is_idempotent(indexed, tmp_path):
    persona_paths, cfg, first_count = indexed
    with Store(tmp_path / "corpus.db") as store:
        count = build_index(
            persona_paths, cfg, store, rebuild=True, embedder=hash_embed
        )
    assert count == first_count
