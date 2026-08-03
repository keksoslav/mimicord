from __future__ import annotations

import logging
import re
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mimicord.paths import PersonaPaths

log = logging.getLogger(__name__)

COLLECTION = "pictures"
# how far down the shortlist to look for one that names the subject
CANDIDATES = 8
SUFFIXES = (".gif", ".png", ".jpg", ".jpeg", ".webp")
# where the pictures live, relative to media/
FOLDER = "pictures"

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class Match:
    path: Path
    caption: str
    distance: float


def caption_for(path: Path, root: Path) -> str:
    """What this picture is, in words, so it can be searched for.

    Filenames are the caption by default, because naming a photo is the one
    piece of work nobody minds doing and everybody has already done. Folder
    names count too, so pictures/svit/squat.jpg reads as "svit squat".
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    words = " ".join(parts)
    words = re.sub(r"[_\-]+", " ", words)
    words = re.sub(r"\s+", " ", words).strip()
    return words


def _folder(paths: PersonaPaths) -> Path:
    return paths.media_dir / FOLDER


def scan(paths: PersonaPaths) -> dict[str, str]:
    """Every picture on disk, as {relative path: caption}.

    A captions.toml next to persona.toml overrides or extends any of them,
    keyed by the same relative path, for the ones a filename cannot carry.
    """
    root = _folder(paths)
    if not root.is_dir():
        return {}
    found = {
        str(p.relative_to(root)).replace("\\", "/"): caption_for(p, root)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUFFIXES
    }
    overrides = paths.root / "captions.toml"
    if overrides.is_file():
        with open(overrides, "rb") as f:
            data = tomllib.load(f)
        for key, caption in data.get("captions", {}).items():
            key = str(key).replace("\\", "/")
            if key in found:
                found[key] = str(caption)
            else:
                log.warning("captions.toml mentions %s, which is not there", key)
    return found


def normalise(text: str) -> str:
    """Fold diacritics so a query written without them still matches."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


# words that carry no subject: mostly slovene, plus the english he mixes in
STOPWORDS = {
    "and", "the", "with", "for", "his", "her", "our", "their", "that", "this",
    "moj", "moja", "moje", "mojo", "tvoj", "tvoja", "nas", "vas", "njegov",
    "pri", "pod", "nad", "med", "brez", "kot", "pa", "ko", "ce", "ki", "ali",
    "sem", "sam", "sama", "smo", "ste", "bil", "bila", "bilo", "bo", "je",
    "slika", "slike", "fotka", "fotke", "foto", "pic", "photo", "picture",
}
MIN_STEM = 3


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalise(text).lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def same_word(a: str, b: str) -> bool:
    """One word with a Slovene ending on it, rather than a different word.

    A fixed stem length cannot do this. Four letters says oknu and okno are
    unrelated; three says macka and machini are the same thing, which is how
    asking for a cat produced a slot machine. What actually distinguishes
    them is whether the shared prefix covers nearly the whole of both words.
    """
    if a == b:
        return True
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    return shared >= MIN_STEM and shared >= min(len(a), len(b)) - 1


def overlap(query: str, caption: str) -> int:
    """How many words the description and the caption really share.

    Embedding distance alone cannot tell "trije v avtu", which we have, from
    "nogometna tekma", which we do not: measured, the football scored better.
    Counting shared words settles it, and comparing on the first few letters
    absorbs Slovene endings, so pivom matches piva and oknu matches okno.

    Used twice: anything scoring zero is not in the pile at all, and among
    those that do match, more shared words wins over merely being nearer.
    """
    wanted, have = content_words(query), content_words(caption)
    return sum(1 for w in wanted if any(same_word(w, h) for h in have))


def build_index(
    paths: PersonaPaths, *, embedder: Embedder | None = None, rebuild: bool = False
) -> int:
    from mimicord.rag import _collection as client_for

    client = client_for(paths)
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION)

    captions = scan(paths)
    if not captions:
        return 0
    ids = list(captions)
    documents = [normalise(captions[i]) for i in ids]
    kwargs: dict = {
        "ids": ids,
        "documents": documents,
        "metadatas": [{"caption": captions[i]} for i in ids],
    }
    if embedder is not None:
        kwargs["embeddings"] = embedder(documents)
    collection.upsert(**kwargs)

    # drop anything that has been deleted off disk since the last index
    stale = [i for i in collection.get()["ids"] if i not in captions]
    if stale:
        collection.delete(ids=stale)
        log.info("dropped %d picture(s) that are no longer on disk", len(stale))
    return len(ids)


class Library:
    """Find a picture from a description of it.

    This is the whole reason the prompt does not carry a catalogue. He writes
    what he wants in his own words, the nearest caption is looked up locally,
    and two hundred pictures cost exactly the same prompt as two.
    """

    def __init__(self, paths: PersonaPaths, embedder: Embedder | None = None) -> None:
        from mimicord.rag import _collection as client_for

        self._paths = paths
        self._embedder = embedder
        self._collection = client_for(paths).get_or_create_collection(COLLECTION)
        # every caption, held in memory. a few hundred short strings is
        # nothing, and scanning all of them beats hoping the right one lands
        # in an embedding shortlist
        stored = self._collection.get(include=["metadatas"])
        self._captions = {
            picture_id: (meta or {}).get("caption", picture_id)
            for picture_id, meta in zip(stored["ids"], stored["metadatas"] or [])
        }

    def count(self) -> int:
        return len(self._captions)

    def subjects(self, limit: int = 12) -> list[str]:
        """The words that come up most across the captions.

        Goes in the prompt so he knows roughly what he has photos of without
        being handed a list of them. Stays one line at any library size.
        """
        from collections import Counter

        tally: Counter[str] = Counter()
        for caption in self._captions.values():
            tally.update(content_words(caption))
        return [word for word, _ in tally.most_common(limit)]

    def find(self, description: str, threshold: float) -> Match | None:
        """Words decide what is eligible, the embedding only ranks.

        The other way round does not work: filtering to the nearest handful
        first means a picture that plainly names the subject is thrown away
        because it happened to rank twelfth. Overlap over the whole library
        is exact and costs nothing at this size.
        """
        if not normalise(description).strip():
            return None
        eligible = {
            picture_id: (overlap(description, caption), caption)
            for picture_id, caption in self._captions.items()
        }
        eligible = {k: v for k, v in eligible.items() if v[0] > 0}
        if not eligible:
            log.info("nothing in the pile is %r", description)
            return None

        distances = self._distances(description, list(eligible))
        ranked = sorted(
            (-shared, distances.get(picture_id, 0.0), picture_id, caption)
            for picture_id, (shared, caption) in eligible.items()
        )
        for _, distance, picture_id, caption in ranked:
            if threshold > 0 and distance > threshold and len(ranked) > 1:
                continue  # something nearer already named the subject better
            path = _folder(self._paths) / picture_id
            if path.is_file():
                return Match(path=path, caption=caption, distance=distance)
            log.warning("indexed picture %s is gone from disk", picture_id)
        return None

    def _distances(self, description: str, ids: list[str]) -> dict[str, float]:
        """Embedding distance for ranking only, so a failure here is survivable."""
        text = normalise(description).strip()
        try:
            kwargs: dict = {"n_results": max(len(self._captions), 1)}
            if self._embedder is not None:
                kwargs["query_embeddings"] = self._embedder([text])
            else:
                kwargs["query_texts"] = [text]
            result = self._collection.query(**kwargs)
        except Exception as error:
            log.warning("picture ranking failed (%s), going on word overlap", error)
            return {}
        got = (result.get("ids") or [[]])[0]
        dist = (result.get("distances") or [[]])[0]
        return dict(zip(got, dist))
