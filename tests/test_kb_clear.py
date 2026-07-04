"""Tests for the KB management endpoints.

Covers:
  * `ChromaStore.delete_by_source` removes only the targeted source
  * `ChromaStore.clear_all` removes everything but keeps the collection
  * `ChromaStore.list_sources` returns one row per distinct source
  * `GET /admin/kb/sources` and the two clear endpoints round-trip
  * BM25 is rebuilt after deletion (tokenizer sees the new corpus)

These run against a temp `chroma_persist_dir` from `conftest.py::_env_defaults`
so they do not touch the real `.chroma/` directory.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from backend.vector_store.bm25_index import BM25Index
from backend.vector_store.chroma_store import ChromaStore


def _make_store() -> ChromaStore:
    """Build a fresh ChromaStore pointed at the per-test temp dir.

    The conftest has already redirected `CHROMA_PERSIST_DIR` into
    `tmp_path`, so we don't need to pass anything here.
    """
    return ChromaStore()


async def _ingest(
    store: ChromaStore,
    *,
    source: str,
    stem: str,
    texts: list[str],
) -> None:
    await store.add_texts(
        texts=texts,
        metadatas=[
            {
                "source": source,
                "chunk_index": i,
                "content_type": "chunk",
                "ocr_pages": 0,
            }
            for i, _ in enumerate(texts)
        ],
        ids=[f"{stem}::chunk::{i}" for i in range(len(texts))],
    )


# ----------------------------------------------------------------- unit tests


def test_list_sources_empty() -> None:
    store = _make_store()
    out = asyncio.run(store.list_sources())
    assert out == []


def test_list_sources_after_ingest() -> None:
    store = _make_store()
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/a.pdf",
            stem="a",
            texts=["alpha one", "alpha two"],
        )
    )
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/b.pdf",
            stem="b",
            texts=["bravo one"],
        )
    )
    out = asyncio.run(store.list_sources())
    assert {row["source"] for row in out} == {
        "data/uploads/a.pdf",
        "data/uploads/b.pdf",
    }
    counts = {row["source"]: row["chunks"] for row in out}
    assert counts["data/uploads/a.pdf"] == 2
    assert counts["data/uploads/b.pdf"] == 1


def test_delete_by_source_removes_only_target() -> None:
    store = _make_store()
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/a.pdf",
            stem="a",
            texts=["alpha one", "alpha two"],
        )
    )
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/b.pdf",
            stem="b",
            texts=["bravo one", "bravo two"],
        )
    )
    removed = asyncio.run(store.delete_by_source("data/uploads/a.pdf"))
    assert removed == 2
    # a.pdf is gone, b.pdf remains.
    out = asyncio.run(store.list_sources())
    sources = {row["source"] for row in out}
    assert sources == {"data/uploads/b.pdf"}


def test_delete_by_source_unknown_returns_zero() -> None:
    store = _make_store()
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/a.pdf",
            stem="a",
            texts=["x"],
        )
    )
    removed = asyncio.run(store.delete_by_source("data/uploads/never-existed.pdf"))
    assert removed == 0
    # Original source must still be there.
    out = asyncio.run(store.list_sources())
    assert len(out) == 1
    assert out[0]["source"] == "data/uploads/a.pdf"


def test_clear_all_wipes_collection() -> None:
    store = _make_store()
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/a.pdf",
            stem="a",
            texts=["alpha"],
        )
    )
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/b.pdf",
            stem="b",
            texts=["bravo"],
        )
    )
    removed = asyncio.run(store.clear_all())
    assert removed == 2
    out = asyncio.run(store.list_sources())
    assert out == []


# ------------------------------------------------------------- BM25 consistency


def test_bm25_rebuild_reflects_deletion() -> None:
    store = _make_store()
    # Seed enough chunks that BM25's IDF is positive (rank_bm25 can produce
    # negative IDF when N==1 and the query term appears in the only doc,
    # which would make the `scores > 0` filter drop the result). A 3-doc
    # corpus gives a stable, positive score.
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/cv.pdf",
            stem="cv",
            texts=[
                "python developer with five years of experience",
                "machine learning engineer with production track record",
                "led the api redesign and reduced latency by half",
            ],
        )
    )
    # Seed the second source with ≥2 chunks so the BM25 corpus stays
    # multi-document after the CV is deleted — rank_bm25 produces
    # non-positive IDF for N=1 docs, which would make our `> 0` score
    # filter drop the lone surviving chunk below.
    asyncio.run(
        _ingest(
            store,
            source="data/uploads/notes.txt",
            stem="notes",
            texts=[
                "the quarterly budget was approved yesterday",
                "the engineering team hired two new members this month",
                "the launch was delayed by two days due to review feedback",
            ],
        )
    )
    # Initial BM25 should find the python chunk for "python developer".
    BM25Index.rebuild()
    bm = BM25Index.instance()
    bm.load()
    hits = bm.search("python developer")
    assert hits, "BM25 should have found the python chunk before deletion"
    cv_hit_ids = {h.id for h in hits}
    # Delete the CV, rebuild, and confirm the cv hits vanish.
    asyncio.run(store.delete_by_source("data/uploads/cv.pdf"))
    BM25Index.rebuild()
    bm = BM25Index.instance()
    bm.load()
    hits_after = bm.search("python developer")
    assert not hits_after, "python chunk must be gone after delete_by_source"
    # And the remaining corpus still has the notes chunk.
    notes_hits = bm.search("quarterly budget")
    assert notes_hits, "non-deleted source should still be in BM25"


# ----------------------------------------------------------------- HTTP routes


def test_admin_kb_sources_endpoint(tmp_path: Path) -> None:
    # Use FastAPI's TestClient for the route round-trip.
    from fastapi.testclient import TestClient

    # Make sure the API is mounted against the same temp chroma dir.
    from backend.config import get_settings

    get_settings.cache_clear()
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    os.environ["BM25_CACHE_PATH"] = str(tmp_path / "chroma" / "bm25.pkl")
    get_settings.cache_clear()

    from main import app  # noqa: WPS433 — local import after env override

    with TestClient(app) as client:
        # Seed two sources.
        store = _make_store()
        asyncio.run(
            _ingest(
                store,
                source="data/uploads/x.pdf",
                stem="x",
                texts=["foo", "bar"],
            )
        )
        asyncio.run(
            _ingest(
                store,
                source="data/uploads/y.pdf",
                stem="y",
                texts=["baz"],
            )
        )

        r = client.get("/admin/kb/sources")
        assert r.status_code == 200, r.text
        body = r.json()
        sources = {row["source"] for row in body["sources"]}
        assert sources == {"data/uploads/x.pdf", "data/uploads/y.pdf"}
        assert body["total_chunks"] == 3
        assert body["total_sources"] == 2

        # Clear just x.pdf.
        r = client.post(
            "/admin/kb/clear-source", json={"source": "data/uploads/x.pdf"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["removed"] == 2

        # x.pdf is gone, y.pdf remains.
        r = client.get("/admin/kb/sources")
        assert r.status_code == 200
        sources = {row["source"] for row in r.json()["sources"]}
        assert sources == {"data/uploads/y.pdf"}

        # Clear-all wipes the rest.
        r = client.post("/admin/kb/clear")
        assert r.status_code == 200, r.text
        assert r.json()["removed"] == 1
        r = client.get("/admin/kb/sources")
        assert r.json()["sources"] == []


def test_admin_kb_clear_source_unknown_returns_zero() -> None:
    from fastapi.testclient import TestClient

    from main import app  # noqa: WPS433

    with TestClient(app) as client:
        r = client.post(
            "/admin/kb/clear-source", json={"source": "data/uploads/ghost.pdf"}
        )
        assert r.status_code == 200
        assert r.json()["removed"] == 0


def test_admin_kb_clear_source_validates_body() -> None:
    from fastapi.testclient import TestClient

    from main import app  # noqa: WPS433

    with TestClient(app) as client:
        # Empty source field is rejected by the Field(min_length=1) constraint.
        r = client.post("/admin/kb/clear-source", json={"source": ""})
        assert r.status_code in (400, 422)
