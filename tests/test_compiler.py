from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimicord.analyze.chunker import Chunk, build_chunks, sample_chunks, split_segments
from mimicord.analyze.mapper import analyze_chunks
from mimicord.analyze.reducer import reduce_profiles
from mimicord.compile.examples import build_examples, candidate_pairs
from mimicord.compile.persona import compile_persona, measured_rules
from mimicord.config import TargetConfig
from mimicord.ingest import ingest_dce
from mimicord.store import Store
from tests.conftest import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"
JANEZ = TargetConfig(author_ids=["111"])

CHUNK_JSON = json.dumps(
    {
        "tone": ["dry"],
        "formality": "very informal",
        "languages": [{"lang": "sl", "share": 1.0}],
        "code_switching": None,
        "verbal_tics": ["ma"],
        "typical_openers": ["ne"],
        "emoji_usage": "rare",
        "humor_style": "deadpan",
        "topics": ["uni"],
        "opinions": [],
        "relationships": [{"person": "Miha", "dynamic": "teasing"}],
        "notable_quotes": ["ma sej bo"],
        "reply_style": "short bursts",
    }
)


@pytest.fixture
def corpus(tmp_path):
    with Store(tmp_path / "corpus.db") as store:
        ingest_dce(store, [FIXTURES], JANEZ)
        yield store


def test_split_segments_on_gap(corpus):
    messages = corpus.channel_messages("999")
    segments = split_segments(messages, gap_minutes=30)
    # fixture has an hour gap between 18:04 and 19:00
    assert len(segments) == 2
    assert segments[0][0].id == "1001"
    assert segments[1][0].id == "1009"


def test_build_chunks_small_corpus_fallback(corpus):
    chunks = build_chunks(corpus)  # default min_target is far above 4
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.target_count == 4
    assert chunk.message_count == 8
    assert "Miha (replying to Janez K.): klasika" in chunk.text
    assert chunk.text.splitlines()[0] == "Miha: a gres jutri na pivo"


def test_sample_chunks_spreads_deterministically():
    chunks = [
        Chunk(i, "c", None, f"t{i:03d}", f"t{i:03d}", 10, 5, "x") for i in range(100)
    ]
    picked = sample_chunks(chunks, cap=10)
    assert len(picked) == 10
    assert picked == sample_chunks(chunks, cap=10)  # deterministic
    # half the budget goes to the recent quartile (indices 75+)
    assert sum(1 for c in picked if c.index >= 75) == 5


def test_mapper_caches_and_resumes(corpus, tmp_path):
    chunks = build_chunks(corpus)
    chunks_dir = tmp_path / "chunks"

    first = FakeProvider(reply=CHUNK_JSON)
    results = analyze_chunks(chunks, first, "janez", chunks_dir)
    assert len(results) == 1
    assert results[0]["tone"] == ["dry"]
    assert results[0]["_chunk"]["target_messages"] == 4
    assert (chunks_dir / "000.json").is_file()

    # rerun: cached file is used, provider never called
    second = FakeProvider(reply="SHOULD NOT BE CALLED")
    cached = analyze_chunks(chunks, second, "janez", chunks_dir)
    assert cached[0]["tone"] == ["dry"]
    assert second.calls == []


def test_mapper_repairs_malformed_json(corpus, tmp_path):
    chunks = build_chunks(corpus)
    provider = FakeProvider(replies=["this is not json at all", CHUNK_JSON])
    results = analyze_chunks(chunks, provider, "janez", tmp_path / "chunks")
    assert results[0]["tone"] == ["dry"]
    assert len(provider.calls) == 2
    repair_request = provider.calls[1]["messages"]
    assert repair_request[1].role == "assistant"  # failed output echoed back
    assert "not valid JSON" in repair_request[2].content


def test_reducer_passes_everything_through(fake_provider):
    fake_provider.reply = json.dumps({"tone": ["dry"], "register_shifts": []})
    profile = reduce_profiles([{"tone": ["dry"]}] * 3, fake_provider, "janez")
    assert profile["tone"] == ["dry"]
    assert "janez" in fake_provider.calls[0]["system"]


def test_measured_rules_from_stats():
    stats = {
        "message_count": 100,
        "length": {"median_chars": 19, "p90_chars": 60},
        "burst": {"p_multi": 0.4, "avg_burst_len": 2.3},
        "capitalization": {"starts_lowercase": 0.93},
        "punctuation": {"ends_with_period": 0.02},
        "emoji": {"per_message": 0.01, "top": []},
        "misc": {"markdown_rate": 0.0},
        "language": {"slovene_diacritics_rate": 0.11},
    }
    rules = "\n".join(measured_rules(stats))
    assert "19 characters" in rules
    assert "93%" in rules
    assert "never ends a message with a period" in rules
    assert "never uses emoji" in rules
    assert "never uses markdown" in rules


def test_compile_persona_appends_code_sections(fake_provider):
    fake_provider.reply = "# Persona: janez\n\ndry guy from Maribor chats."
    stats = {
        "message_count": 10,
        "length": {"median_chars": 19, "p90_chars": 60},
        "burst": {"p_multi": 0.4, "avg_burst_len": 2.0},
        "capitalization": {"starts_lowercase": 0.9},
        "punctuation": {"ends_with_period": 0.0},
        "emoji": {"per_message": 0.0, "top": []},
        "misc": {"markdown_rate": 0.0},
        "language": {"slovene_diacritics_rate": 0.2},
    }
    text = compile_persona({"tone": ["dry"]}, stats, fake_provider, "janez")
    assert text.startswith("# Persona: janez")
    assert "## Formatting habits (measured)" in text
    assert "## Never do" in text
    assert "never sound like an assistant" in text


def test_candidate_pairs_from_fixture(corpus):
    stats = {"length": {"p90_chars": 100}}
    pairs = candidate_pairs(corpus, stats)
    # janez's first burst: context by miha, reply is the two-message burst
    burst = next(p for p in pairs if p["reply"] == ["ne", "mam kolokvij"])
    assert {"author": "Miha", "content": "a gres jutri na pivo"} in burst["context"]
    # every pair in this channel has context from someone else
    assert all(
        any(c["author"] != "Janez K." for c in p["context"]) for p in pairs
    )


def test_build_examples_curates_with_llm(corpus, tmp_path):
    stats = {"length": {"p90_chars": 100}}
    provider = FakeProvider(reply=json.dumps({"selected": [0]}))
    data = build_examples(corpus, stats, provider, "janez", count=1)
    assert len(data["examples"]) == 1
    assert data["examples"][0]["reply"]
    # engine-compatible shape
    example = data["examples"][0]
    assert isinstance(example["context"], list)
    assert isinstance(example["reply"], list)


def test_build_examples_falls_back_without_llm(corpus):
    stats = {"length": {"p90_chars": 100}}
    provider = FakeProvider(reply="garbage not json {{{")
    data = build_examples(corpus, stats, provider, "janez", count=1)
    # curation happens only above the count threshold; with few candidates
    # everything is kept and the provider is not even needed
    assert data["examples"]
