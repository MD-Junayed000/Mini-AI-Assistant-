"""Reranker — local cosine similarity using the project's ChromaDB embedder.

Why local instead of an HF cross-encoder?
* The HF Inference Router does **not** expose most cross-encoders
  (e.g. ``BAAI/bge-reranker-base`` or ``cross-encoder/ms-marco-MiniLM-L-6-v2``)
  through its OpenAI-style ``/v1/rerank`` endpoint — they return 404.
* Running a real cross-encoder locally would need PyTorch, which has known
  DLL-load issues on Windows.

What we do instead: re-use ChromaDB's bundled ONNX ``all-MiniLM-L6-v2``
embedder — the same one the vector store already uses — to embed the query
and the candidates, then rank by cosine similarity. Because the vectors
live in the same 384-d space as the dense retriever, the rerank signal
is directly comparable to what the candidate was originally scored on.

Two implementations, picked at construction time:

* ``LocalCosineReranker`` (default) — embeds via the ChromaDB ONNX embedder
  and scores by dot product. No torch, no HF call, no extra dependency.
* ``NoOpReranker`` — returns ``[]`` so the caller falls back to RRF
  ordering. Used if the embedder fails to load so the chat pipeline
  never breaks.
"""
from __future__ import annotations

import asyncio
from typing import Protocol

from backend.observability.logging_config import get_logger
from backend.observability.metrics import RERANK_TOP_SCORE, STAGE_LATENCY

log = get_logger("rerank")


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot-product of two unit-length vectors.

    ChromaDB's ``DefaultEmbeddingFunction`` normalizes embeddings to unit
    length, so cosine similarity == dot product — faster than the full
    a·b / (‖a‖·‖b‖) form.
    """
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


_EMBED_FN = None


def _embed_with_chroma(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using ChromaDB's bundled ONNX MiniLM model.

    We use the bare ``DefaultEmbeddingFunction`` directly — bypassing any
    Chroma collection / HNSW index — because ``EphemeralClient`` triggers
    access violations in the bundled telemetry on Windows. The embed fn
    itself is a pure ONNX inference call and is fully thread-safe; we cache
    one instance for the process lifetime.
    """
    global _EMBED_FN
    if _EMBED_FN is None:
        from chromadb.utils import embedding_functions

        _EMBED_FN = embedding_functions.DefaultEmbeddingFunction()
    # ``__call__`` accepts a list of strings and returns a list of unit-norm
    # 384-d vectors.
    return list(_EMBED_FN(texts))


class LocalCosineReranker:
    """Embed query + candidates with ChromaDB's ONNX MiniLM, rank by cosine."""

    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []

        with STAGE_LATENCY.labels(stage="rerank").time():
            try:
                # Batch embed (query + candidates) for fewer round trips.
                all_vecs = await asyncio.to_thread(
                    _embed_with_chroma, [query, *candidates]
                )
                if not all_vecs or len(all_vecs) < len(candidates) + 1:
                    log.warning("rerank_embed_empty_result")
                    return []
                query_vec = all_vecs[0]
                cand_vecs = all_vecs[1:]
            except Exception as e:  # noqa: BLE001
                log.warning("rerank_embed_failed_using_noop", error=str(e))
                return []

        scored: list[tuple[int, float]] = []
        for i, v in enumerate(cand_vecs):
            scored.append((i, _cosine(query_vec, v)))
        # Sort descending by score, then ascending by original index for
        # stable ties.
        scored.sort(key=lambda p: (-p[1], p[0]))

        scored = scored[: max(1, min(top_k, len(scored)))]

        if scored:
            RERANK_TOP_SCORE.observe(max(s for _, s in scored))
        return scored


class NoOpReranker:
    """Returns no scores so the caller falls back to RRF ordering."""

    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        return []


def make_reranker() -> Reranker:
    """Pick the best reranker available in this environment."""
    from backend.config import get_settings  # local import to avoid cycles

    if getattr(get_settings(), "rerank_disabled", False):
        log.info("rerank_backend", chosen="noop", reason="rerank_disabled=true")
        return NoOpReranker()

    try:
        _embed_with_chroma(["ping"])
    except Exception as e:  # noqa: BLE001
        log.warning("rerank_backend_embedder_unavailable", error=str(e))
        return NoOpReranker()

    log.info("rerank_backend", chosen="local_cosine")
    return LocalCosineReranker()


__all__ = [
    "LocalCosineReranker",
    "NoOpReranker",
    "Reranker",
    "make_reranker",
]