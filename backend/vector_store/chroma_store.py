"""ChromaDB persistent client wrapper.

Embedding strategy: we let ChromaDB's built-in ONNX embedding function
(`all-MiniLM-L6-v2`) compute vectors, instead of calling HF ourselves.

Why this change:
  - The HF Router's `/v1/embeddings` endpoint only serves models listed
    under "Inference Providers". Most sentence-transformer checkpoints
    (BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2, e5-*, etc.) return 404.
  - Local `sentence-transformers` requires torch, which on Windows can
    fail with WinError 1114 (DLL init) — we'd rather not add torch to
    the dependency graph for embeddings alone.
  - ChromaDB's bundled ONNX runtime is already in the dependency graph
    (chromadb 0.5.7 pins onnxruntime<1.20), runs entirely on CPU, and
    produces 384-d MiniLM vectors that match the project's retrieval
    shape.

The `HFEmbeddingClient` is still available in `backend.llm.embeddings`
for code that needs explicit vectors (reranking, custom pipelines).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Silence ChromaDB's built-in telemetry BEFORE any chromadb import.
#
# Background:
#   - chromadb ships a posthog-based telemetry sender that fires
#     ClientStartEvent / ClientCreateCollectionEvent on first use.
#   - Recent posthog releases changed the `capture()` signature, so chromadb
#     raises `capture() takes 1 positional argument but 3 were given` on every
#     event — and prints a stack trace to stdout, drowning our own logs.
#
# Defence in depth (each layer catches a different failure mode):
#   1. Set the env vars chromadb checks for opt-out. Some builds honour
#      `ANONYMIZED_TELEMETRY=False`, others look for `CHROMA_TELEMETRY_DISABLED`.
#   2. Pre-register a *shim* `posthog` module so when chromadb does
#      `import posthog` it gets our stub instead of the real client. This
#      works because Python's import machinery caches the first match in
#      `sys.modules`.
# ---------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "True")


# ---------------------------------------------------------------------------
# Silence ChromaDB's "Add of existing embedding ID" warning.
#
# Background:
#   - `chromadb.segment.impl.vector.local_persistent_hnsw` emits a
#     `WARNING` for every id we upsert that already exists. The upsert
#     succeeds — this is purely informational — but the warning floods
#     the logs every time `python -m backend.ingestion.pipeline` runs
#     against an already-populated collection.
#   - We want to drop *only* this specific record; everything else from
#     chromadb (real errors, query failures) must still surface.
#
# Installing the filter here means it takes effect on the first chromadb
# import, regardless of which entry point imports this module first
# (server startup, the ingestion CLI, or a test).
# ---------------------------------------------------------------------------
class _ChromaExistingIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Add of existing embedding ID" not in record.getMessage()


logging.getLogger(
    "chromadb.segment.impl.vector.local_persistent_hnsw"
).addFilter(_ChromaExistingIDFilter())


def _install_posthog_stub() -> None:
    """Register a no-op `posthog` module before chromadb imports the real one.

    Idempotent and side-effect free: if a real posthog has already been
    imported for legitimate use elsewhere, we leave it alone.
    """
    if "posthog" in sys.modules:
        return
    import types

    stub = types.ModuleType("posthog")

    def _noop(*_a, **_kw):
        return None

    stub.capture = _noop  # type: ignore[attr-defined]
    stub.identify = _noop  # type: ignore[attr-defined]
    stub.flush = _noop  # type: ignore[attr-defined]
    stub.disable = _noop  # type: ignore[attr-defined]
    sys.modules["posthog"] = stub


from backend.config import get_settings
from backend.errors import RetrieverError
from backend.observability.logging_config import get_logger
from backend.vector_store.recovery import auto_recover_if_corrupt

log = get_logger("chroma")


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [-1, 1]


def _default_embedding_function():
    """ChromaDB's bundled ONNX MiniLM embedder (no torch)."""
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


class ChromaStore:
    """Persistent Chroma client bound to one collection."""

    def __init__(self, collection: str | None = None) -> None:
        s = get_settings()
        self._collection_name = collection or s.chroma_collection
        self._persist_dir = Path(s.chroma_persist_dir)
        # ------------------------------------------------------------------
        # Auto-recovery: if the persistent directory was left half-written by
        # a previous crash, the in-process probe would just segfault again.
        # The recovery helper probes in an isolated subprocess (so a native
        # crash can't reach us), then moves the corrupt dir aside to
        # `.bak-<stamp>` and recreates an empty one. The next ingestion
        # rebuilds the collection from `data/` transparently.
        # ------------------------------------------------------------------
        if self._persist_dir.exists():
            auto_recover_if_corrupt(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        # Install the posthog stub BEFORE chromadb grabs a reference to it.
        _install_posthog_stub()
        import chromadb

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._embed_fn = _default_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_fn,
        )
        # Cache whether the collection is healthy. A Windows process that
        # is killed mid-upsert (or an interrupted previous run) can leave
        # Chroma's HNSW files half-written; subsequent calls then segfault
        # inside chromadb's Rust code with no Python-visible exception.
        # We probe on first use and rebuild if the probe fails.
        self._verified_ok = False

    def _recreate_collection(self) -> None:
        """Drop and recreate the collection. Loses data — used as a last
        resort when the persistent HNSW is unrecoverable."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 — collection may not exist
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_fn,
        )

    def _self_heal(self) -> bool:
        """Detect and recover from a corrupt HNSW index.

        Returns True if the collection is healthy afterwards. False means
        we couldn't make it usable — callers should surface a 503.
        """
        if self._verified_ok:
            return True
        try:
            # A trivial query exercises the same HNSW read path that
            # `upsert` will use for its write path. If this raises, the
            # collection is broken.
            self._collection.query(query_texts=["__healthcheck__"], n_results=1)
        except Exception as exc:  # noqa: BLE001
            log.warning("chroma_collection_unhealthy_rebuilding", error=str(exc))
            try:
                self._recreate_collection()
                self._verified_ok = True
                return True
            except Exception as e2:  # noqa: BLE001
                log.error("chroma_recreate_failed", error=str(e2))
                return False
        self._verified_ok = True
        return True

    async def add_texts(
        self,
        *,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if not texts:
            return

        def _add() -> None:
            # Make ingestion idempotent. The pipeline assigns deterministic
            # IDs like "{stem}::chunk::{i}", so re-running `ingest_file` on
            # the same source would otherwise raise
            # `Add of existing embedding ID` from ChromaDB. The public
            # `Collection.upsert(...)` API is the documented atomic way to
            # "insert if missing, replace if present" — cleaner and faster
            # than delete-then-add.
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

        # If we know the collection is broken, heal before touching it.
        if not self._verified_ok and not self._self_heal():
            raise RetrieverError("chroma_collection_unrecoverable")
        try:
            await asyncio.to_thread(_add)
        except Exception as e:  # noqa: BLE001
            # A *catchable* failure here usually means a transient issue
            # (lock contention, corrupt index we hadn't detected). Reset
            # the verified flag so the next call re-probes — and try
            # once more after a self-heal rebuild.
            log.warning("chroma_upsert_failed_attempting_heal", error=str(e))
            self._verified_ok = False
            if self._self_heal():
                await asyncio.to_thread(_add)
                return
            raise RetrieverError(str(e)) from e

    async def query(self, text: str, top_k: int = 8) -> list[Hit]:
        if not text.strip():
            return []

        def _q() -> dict[str, Any]:
            return self._collection.query(
                query_texts=[text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        try:
            data = await asyncio.to_thread(_q)
        except Exception as e:  # noqa: BLE001
            log.error("chroma_query_failed", error=str(e))
            raise RetrieverError(str(e)) from e

        ids = (data.get("ids") or [[]])[0]
        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        dists = (data.get("distances") or [[]])[0]
        out: list[Hit] = []
        for i, d, m, dist in zip(ids, docs, metas, dists):
            # Chroma returns cosine *distance*; convert to similarity.
            sim = max(-1.0, min(1.0, 1.0 - float(dist)))
            out.append(Hit(id=i, text=d, metadata=m or {}, score=sim))
        return out

    async def count(self) -> int:
        def _c() -> int:
            return self._collection.count()

        return await asyncio.to_thread(_c)

    async def list_sources(self) -> list[dict[str, Any]]:
        """Return one row per distinct `source` metadata value.

        Used by the KB-management UI to show the user which documents are
        currently indexed. The count is exact (from `collection.get`) rather
        than estimated, because the take-home corpus is small.
        """
        def _list() -> list[dict[str, Any]]:
            data = self._collection.get(include=["metadatas"])
            counts: dict[str, int] = {}
            for m in data.get("metadatas") or []:
                src = (m or {}).get("source")
                if not src:
                    continue
                counts[src] = counts.get(src, 0) + 1
            return [
                {"source": src, "chunks": n}
                for src, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

        return await asyncio.to_thread(_list)

    async def delete_by_source(self, source: str) -> int:
        """Delete every chunk whose metadata `source` matches `source` exactly.

        Returns the number of chunks removed (0 if the source was unknown).
        Safe to call when the collection is empty. The matching is on the
        full stored path string (e.g. `data/uploads/foo.pdf`), so callers
        must pass the value returned by `list_sources()`.
        """
        if not source:
            return 0

        def _del() -> int:
            existing = self._collection.get(
                where={"source": source}, include=[]
            )
            ids = list(existing.get("ids") or [])
            if not ids:
                return 0
            # Delete in batches to stay under Chroma's per-call limits on
            # very large collections. Take-home corpus is small so a single
            # batch is fine, but keep the loop defensively.
            BATCH = 500
            for i in range(0, len(ids), BATCH):
                self._collection.delete(ids=ids[i:i + BATCH])
            return len(ids)

        return await asyncio.to_thread(_del)

    async def clear_all(self) -> int:
        """Remove every chunk in the collection. Returns the deleted count.

        Faster than `reset()` because it keeps the collection (and its
        embedding-function binding) intact — `reset()` would also have to
        re-bind the ONNX embedder, which is a noticeable cost on Windows.
        """
        def _clear() -> int:
            data = self._collection.get(include=[])
            ids = list(data.get("ids") or [])
            if not ids:
                return 0
            BATCH = 500
            for i in range(0, len(ids), BATCH):
                self._collection.delete(ids=ids[i:i + BATCH])
            return len(ids)

        return await asyncio.to_thread(_clear)

    async def reset(self) -> None:
        def _r() -> None:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embed_fn,
            )

        await asyncio.to_thread(_r)