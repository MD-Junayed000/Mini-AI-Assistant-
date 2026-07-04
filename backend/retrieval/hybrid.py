"""Hybrid retrieval: BM25 ∪ dense → RRF fusion → cross-encoder rerank → gate.

Reciprocal Rank Fusion (RRF) — the standard for hybrid retrieval when you
want each retriever to contribute even when their raw scores live on
different scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import get_settings
from backend.llm.rerank import make_reranker
from backend.observability.logging_config import get_logger
from backend.observability.metrics import RERANK_TOP_SCORE, RETRIEVAL_RESULTS, STAGE_LATENCY
from backend.vector_store.bm25_index import BM25Index
from backend.vector_store.chroma_store import ChromaStore, Hit

log = get_logger("retrieval")


@dataclass
class Retrieved:
    id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    rerank_score: float | None
    dense_score: float | None
    bm25_score: float | None


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def _fuse(dense: list[Hit], bm25: list[Any]) -> list[Retrieved]:
    by_id: dict[str, Retrieved] = {}

    for rank, h in enumerate(dense):
        by_id[h.id] = Retrieved(
            id=h.id,
            text=h.text,
            metadata=h.metadata,
            rrf_score=_rrf(rank),
            rerank_score=None,
            dense_score=h.score,
            bm25_score=None,
        )

    for rank, b in enumerate(bm25):
        existing = by_id.get(b.id)
        if existing:
            existing.rrf_score += _rrf(rank)
            existing.bm25_score = b.score
        else:
            by_id[b.id] = Retrieved(
                id=b.id,
                text=b.text,
                metadata=b.metadata,
                rrf_score=_rrf(rank),
                rerank_score=None,
                dense_score=None,
                bm25_score=b.score,
            )

    return sorted(by_id.values(), key=lambda r: r.rrf_score, reverse=True)


async def retrieve(query: str, top_k: int = 8) -> list[Retrieved]:
    """Hybrid retrieve + rerank."""
    s = get_settings()
    chroma = ChromaStore()
    bm25 = BM25Index.instance()

    with STAGE_LATENCY.labels(stage="retrieval").time():
        dense, sparse = await _parallel_retrieval(query, chroma, bm25, top_k)
        fused = _fuse(dense, sparse)

    # Track dense score distribution for dashboards.
    for h in dense:
        RETRIEVAL_RESULTS.labels(source="dense").observe(h.score)
    for b in sparse:
        RETRIEVAL_RESULTS.labels(source="bm25").observe(min(1.0, b.score / 5.0))

    # Rerank the top-k candidates.
    cand = fused[: max(top_k, 5)]
    if not cand:
        return []

    reranker = make_reranker()
    with STAGE_LATENCY.labels(stage="rerank").time():
        try:
            scored = await reranker.rerank(query, [c.text for c in cand], top_k=top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("rerank_failed_skipping", error=str(e))
            scored = []

    score_by_idx = {idx: sc for idx, sc in scored}
    for i, c in enumerate(cand):
        c.rerank_score = score_by_idx.get(i)
    cand.sort(key=lambda c: (c.rerank_score is None, -(c.rerank_score or c.rrf_score)))

    if cand and cand[0].rerank_score is not None:
        RERANK_TOP_SCORE.observe(cand[0].rerank_score)

    return cand


async def _parallel_retrieval(
    query: str,
    chroma: ChromaStore,
    bm25: BM25Index,
    top_k: int,
) -> tuple[list[Hit], list[Any]]:
    import asyncio

    async def _dense() -> list[Hit]:
        try:
            return await chroma.query(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("dense_retrieval_failed", error=str(e))
            return []

    def _sparse() -> list[Any]:
        try:
            return bm25.search(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("bm25_retrieval_failed", error=str(e))
            return []

    dense_task = asyncio.create_task(_dense())
    bm25_task = asyncio.create_task(asyncio.to_thread(_sparse))
    dense_out, sparse_out = await asyncio.gather(dense_task, bm25_task)
    return dense_out, sparse_out