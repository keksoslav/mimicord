from __future__ import annotations

import logging
from collections.abc import Callable

from mimicord.config import RagConfig
from mimicord.paths import PersonaPaths
from mimicord.store import Store

log = logging.getLogger(__name__)

COLLECTION = "memories"
BATCH = 1000

# optional override so tests can embed deterministically without the model
# download; production leaves this None and chroma uses its local ONNX MiniLM
Embedder = Callable[[list[str]], list[list[float]]]


def _collection(paths: PersonaPaths):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(paths.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def build_index(
    paths: PersonaPaths,
    cfg: RagConfig,
    store: Store,
    *,
    rebuild: bool = False,
    progress: Callable[[int, int], None] | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Index sliding conversation windows where the target speaks."""
    client = _collection(paths)
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for channel in store.channels():
        if channel["target"] == 0:
            continue
        messages = [
            m for m in store.channel_messages(channel["channel_id"]) if m.content
        ]
        if not messages:
            continue
        size, step = cfg.window_size, cfg.window_step
        for start in range(0, max(len(messages) - size + 1, 1), step):
            window = messages[start : start + size]
            if not any(m.is_target for m in window):
                continue
            documents.append(
                "\n".join(f"{m.author_name}: {m.content}" for m in window)
            )
            ids.append(f"{window[0].id}:{start}")
            metadatas.append(
                {
                    "channel": channel["channel_name"] or str(channel["channel_id"]),
                    "start_ts": window[0].timestamp,
                }
            )

    for offset in range(0, len(documents), BATCH):
        batch_docs = documents[offset : offset + BATCH]
        kwargs = {
            "ids": ids[offset : offset + BATCH],
            "documents": batch_docs,
            "metadatas": metadatas[offset : offset + BATCH],
        }
        if embedder is not None:
            kwargs["embeddings"] = embedder(batch_docs)
        collection.upsert(**kwargs)
        if progress:
            progress(min(offset + BATCH, len(documents)), len(documents))
    return len(documents)


class Memories:
    """Retrieval over the indexed history: recent chat in, snippets out."""

    def __init__(
        self, paths: PersonaPaths, cfg: RagConfig, embedder: Embedder | None = None
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._collection = _collection(paths).get_or_create_collection(COLLECTION)

    def query(self, recent_text: str, top_k: int | None = None) -> list[str]:
        k = top_k or self._cfg.top_k
        if not recent_text.strip():
            return []
        try:
            if self._embedder is not None:
                result = self._collection.query(
                    query_embeddings=self._embedder([recent_text]), n_results=k * 2
                )
            else:
                result = self._collection.query(
                    query_texts=[recent_text], n_results=k * 2
                )
        except Exception as error:
            log.warning("memory query failed: %s", error)
            return []

        memories: list[str] = []
        seen_openings: set[str] = set()
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        for document, meta in zip(documents, metadatas):
            opening = document.split("\n", 1)[0]
            if opening in seen_openings:  # overlapping windows repeat lines
                continue
            seen_openings.add(opening)
            date = (meta.get("start_ts") or "")[:10]
            channel = meta.get("channel", "?")
            memories.append(f"({date}, {channel}) " + document.replace("\n", " / "))
            if len(memories) >= k:
                break
        return memories
