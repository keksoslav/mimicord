from __future__ import annotations

import asyncio

from mimicord import pictures
from mimicord.bot import MimicClient
from mimicord.engine import PersonaEngine
from mimicord.paths import PersonaPaths

TOML = """\
name = "testbot"

[llm]
provider = "ollama"

[pictures]
enabled = true
threshold = 1.1
"""


def flat(_texts):
    """Every caption at the same distance, so ranking falls to word overlap,
    which is the part worth testing without downloading an embedding model."""
    return [[1.0, 0.0, 0.0] for _ in _texts]


def build(persona_dir, names):
    (persona_dir / "persona.toml").write_text(TOML, encoding="utf-8")
    root = persona_dir / "media" / "pictures"
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    return PersonaPaths(persona_dir)


def test_caption_comes_from_the_path(tmp_path):
    root = tmp_path / "pictures"
    (root / "svit").mkdir(parents=True)
    picture = root / "svit" / "squat_2.jpg"
    picture.write_bytes(b"x")
    assert pictures.caption_for(picture, root) == "svit squat 2"


def test_scan_finds_pictures_and_skips_junk(persona_dir):
    paths = build(persona_dir, ["Timi s pivom.jpg", "svit/squat.jpg", "notes.txt"])
    found = pictures.scan(paths)
    assert found == {
        "Timi s pivom.jpg": "Timi s pivom",
        "svit/squat.jpg": "svit squat",
    }


def test_captions_toml_overrides(persona_dir):
    paths = build(persona_dir, ["IMG_2481.jpg"])
    (persona_dir / "captions.toml").write_text(
        '[captions]\n"IMG_2481.jpg" = "timi na morju z ocali"\n', encoding="utf-8"
    )
    assert pictures.scan(paths)["IMG_2481.jpg"] == "timi na morju z ocali"


def test_overlap_counts_real_words():
    assert pictures.overlap("timi z pivom", "Timi s pivom") == 2
    assert pictures.overlap("nogometna tekma", "Timi nosi gajbo piva") == 0
    assert pictures.overlap("moja macka", "Kosara piva") == 0


def test_overlap_absorbs_slovene_endings():
    """oknu and okno are the same word, pivom and piva are the same word."""
    assert pictures.overlap("timi pri oknu", "Timi gleda cez okno") >= 2
    assert pictures.overlap("kosara piva", "Timi nosi kosaro piva") >= 2


def test_overlap_ignores_words_that_name_nothing():
    assert pictures.overlap("slika mojega", "Svit squat") == 0


def test_diacritics_do_not_matter():
    assert pictures.overlap("timi v škornjih", "Timi v zelo lepih skornjih") >= 2


def test_finds_the_picture_that_names_the_subject(persona_dir):
    paths = build(persona_dir, ["Timi s pivom.jpg", "Svit squat.jpg", "Casino.jpg"])
    pictures.build_index(paths, embedder=flat, rebuild=True)
    library = pictures.Library(paths, embedder=flat)

    match = library.find("svit squat", threshold=0)
    assert match is not None and match.path.name == "Svit squat.jpg"


def test_sends_nothing_for_something_it_does_not_have(persona_dir):
    paths = build(persona_dir, ["Timi s pivom.jpg", "Svit squat.jpg"])
    pictures.build_index(paths, embedder=flat, rebuild=True)
    library = pictures.Library(paths, embedder=flat)

    assert library.find("moja macka", threshold=0) is None
    assert library.find("ferrari", threshold=0) is None


def test_a_deleted_picture_leaves_the_index(persona_dir):
    paths = build(persona_dir, ["Timi s pivom.jpg", "Svit squat.jpg"])
    pictures.build_index(paths, embedder=flat, rebuild=True)
    (persona_dir / "media" / "pictures" / "Svit squat.jpg").unlink()
    pictures.build_index(paths, embedder=flat)

    library = pictures.Library(paths, embedder=flat)
    assert library.find("svit squat", threshold=0) is None
    assert library.find("timi pivom", threshold=0) is not None


def test_the_prompt_never_lists_the_pictures(persona_dir, fake_provider):
    """The whole point. Two hundred photos must read like two."""
    build(persona_dir, [f"picture number {i}.jpg" for i in range(200)])
    engine = PersonaEngine("testbot", rag_enabled=False)

    assert "[pic:" in engine.system
    for i in (0, 57, 199):
        assert f"picture number {i}" not in engine.system


class FakeChannel:
    id = 100

    def __str__(self) -> str:
        return "general"


def test_a_picture_request_is_resolved_and_sent(persona_dir, fake_provider):
    paths = build(persona_dir, ["Svit squat.jpg"])
    pictures.build_index(paths, embedder=flat, rebuild=True)

    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    engine.library = pictures.Library(paths, embedder=flat)
    client = MimicClient(engine, dry_run=True)
    client.seeded.add(FakeChannel.id)

    asyncio.run(client._send_bursts(FakeChannel(), ["glej", "[pic:svit squat]"]))

    assert [m.content for m in client.buffers[FakeChannel.id]] == [
        "glej",
        "[pic:svit squat]",
    ]


def test_asking_for_something_absent_sends_nothing(persona_dir, fake_provider):
    paths = build(persona_dir, ["Svit squat.jpg"])
    pictures.build_index(paths, embedder=flat, rebuild=True)

    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    engine.library = pictures.Library(paths, embedder=flat)
    client = MimicClient(engine, dry_run=True)
    client.seeded.add(FakeChannel.id)

    asyncio.run(client._send_bursts(FakeChannel(), ["glej", "[pic:moja macka]"]))

    # the text goes, the impossible picture quietly does not
    assert [m.content for m in client.buffers[FakeChannel.id]] == ["glej"]


def test_slovene_endings_are_the_same_word():
    assert pictures.same_word("oknu", "okno")
    assert pictures.same_word("pivom", "piva")
    assert pictures.same_word("hiso", "hise")
    assert pictures.same_word("kosaro", "kosara")


def test_a_shared_prefix_is_not_enough_on_its_own():
    """macka and machini share three letters. Asking for a cat used to
    return a slot machine."""
    assert not pictures.same_word("macka", "machini")
    assert not pictures.same_word("svit", "svoje")
    assert not pictures.same_word("piva", "pizza")


def test_short_words_are_matched_loosely_on_purpose():
    """A short word that is the start of a longer one does match, because
    that is exactly what a Slovene ending looks like. It means pes and pesem
    match too, which is the price. What keeps it harmless is that the word
    still has to appear in a caption somebody actually wrote."""
    assert pictures.same_word("pes", "pesem")


def test_overlap_uses_it():
    assert pictures.overlap("moja macka", "Zmaga na slot machini") == 0
    assert pictures.overlap("timi pri oknu", "Timi gleda cez okno") == 2
