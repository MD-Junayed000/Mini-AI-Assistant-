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


def _install_posthog_stub() -> None:
    """Register a no-op `posthog` module before chromadb imports the real one.

    Idempotent and side-effect free: if a real posthog has already been
    imported for legitimate use elsewhere, we leave it alone.
    """
    if "posthog" in sys.modules:
        return
    import types

    stub = types.ModuleType("posthog")

    def _noop(*_a, **_kw):  # noqa: ANN001
        return None

    stub.capture = _noop  # type: ignore[attr-defined]
    stub.identify = _noop  # type: ignore[attr-defined]
    stub.flush = _noop  # type: ignore[attr-defined]
    stub.disable = _noop  # type: ignore[attr-defined]
    sys.modules["posthog"] = stub


from backend.config import get_settings
from backend.errors import RetrieverError
from backend.observability.logging_config import get_logger

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
        Path(s.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        # Install the posthog stub BEFORE chromadb grabs a reference to it.
        _install_posthog_stub()
        import chromadb

        self._client = chromadb.PersistentClient(path=s.chroma_persist_dir)
        self._embed_fn = _default_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_fn,
        )

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
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

        await asyncio.to_thread(_add)

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

    async def reset(self) -> None:
        def _r() -> None:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embed_fn,
            )

        await asyncio.to_thread(_r)