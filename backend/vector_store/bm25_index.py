"""BM25 index over the Chroma collection, persisted to disk.

We rebuild by streaming (id, text, metadata) from Chroma — keeps the
source of truth in one place. For the take-home corpus (a handful of
documents) this is fast enough to do on every ingest.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from backend.config import get_settings
from backend.observability.logging_config import get_logger
from backend.vector_store.chroma_store import ChromaStore

log = get_logger("bm25")


@dataclass
class BM25Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class BM25Index:
    _instance: "BM25Index | None" = None

    def __init__(self) -> None:
        s = get_settings()
        self._path = Path(s.bm25_cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False

    # --- singleton --------------------------------------------------------
    @classmethod
    def instance(cls) -> "BM25Index":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                with self._path.open("rb") as f:
                    blob = pickle.load(f)
                self._ids = blob["ids"]
                self._texts = blob["texts"]
                self._metas = blob["metas"]
                self._bm25 = BM25Okapi(blob["tokenized"])
                self._loaded = True
                return
            except Exception as e:  # noqa: BLE001
                log.warning("bm25_load_failed_rebuilding", error=str(e))
        self.rebuild()

    @classmethod
    def rebuild(cls) -> None:
        inst = cls.instance()
        store = ChromaStore()
        # Stream everything from Chroma (sync inner call wrapped in to_thread by client).
        ids, docs, metas = inst._stream_from_chroma(store)
        inst._ids = ids
        inst._texts = docs
        inst._metas = metas
        tokenized = [_tokenize(t) for t in docs]
        inst._bm25 = BM25Okapi(tokenized) if docs else None
        inst._loaded = True
        try:
            with inst._path.open("wb") as f:
                pickle.dump(
                    {"ids": ids, "texts": docs, "metas": metas, "tokenized": tokenized},
                    f,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("bm25_persist_failed", error=str(e))

    @staticmethod
    def _stream_from_chroma(store: ChromaStore) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Chroma has no streaming API; for a take-home corpus, .get() suffices.

        `coll.get()` is a synchronous, blocking call — no event loop needed.
        We keep the call shape compatible with both sync and async callers by
        running it directly. (The previous version wrapped it in
        `asyncio.run(asyncio.to_thread(...))`, which raises
        `RuntimeError: asyncio.run() cannot be called from a running event loop`
        when called from `backend.ingestion.pipeline`'s main coroutine.)
        """
        coll = store._collection  # type: ignore[attr-defined]
        data = coll.get(include=["documents", "metadatas"])
        ids = list(data.get("ids", []))
        docs = list(data.get("documents", []) or [])
        metas = list(data.get("metadatas", []) or [])
        return ids, docs, metas

    # --- search ----------------------------------------------------------
    def search(self, query: str, top_k: int = 8) -> list[BM25Hit]:
        if not self._loaded:
            self.load()
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Hit(id=self._ids[i], text=self._texts[i], metadata=self._metas[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]


import re

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s)]