"""Top-level ingestion: PDF → chunks → Chroma + BM25 + metric increments."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from backend.config import get_settings
from backend.ingestion.chunker import chunk_text
from backend.ingestion.docling_pipeline import extract, extract_with_backend
from backend.observability.logging_config import get_logger
from backend.observability.metrics import INGEST_DOCUMENTS, STAGE_LATENCY
from backend.vector_store.chroma_store import ChromaStore
from backend.vector_store.bm25_index import BM25Index

log = get_logger("ingest")


async def ingest_file(
    path: Path,
    *,
    collection: str | None = None,
    doc_id_prefix: str | None = None,
) -> dict[str, Any]:
    """Ingest one file end-to-end.

    Returns a small dict so the API layer can tell the user *which* backend
    actually parsed the file. This matters on Windows where Docling's
    optional torch-based components raise WinError 1114 at import time;
    we silently fall back to pdfplumber and report it instead of bubbling
    a long stack trace into the user's toast.
    """
    settings = get_settings()
    collection = collection or settings.chroma_collection
    store = ChromaStore(collection=collection)

    backend_used = "docling"
    fallback_reason: str | None = None
    with STAGE_LATENCY.labels(stage="extract").time():
        try:
            doc, backend_used, fallback_reason = await extract_with_backend(path)
        except Exception as e:  # noqa: BLE001 — never let ingest crash on extractor failure
            # Surface the failure cleanly to the caller rather than a 500.
            log.error("ingest_extract_failed", error=str(e)[:200])
            INGEST_DOCUMENTS.labels(
                source_type=path.suffix.lower(), outcome="extract_failed"
            ).inc()
            return {
                "chunks": 0,
                "backend": "none",
                "fallback_reason": "extract_failed",
                "error": str(e)[:200],
            }

    figure_text = "\n".join(f"[figure] {d}" for d in doc.figure_descriptions)
    full_text = (doc.text + "\n\n" + figure_text).strip() if figure_text else doc.text
    if not full_text.strip():
        INGEST_DOCUMENTS.labels(source_type=path.suffix.lower(), outcome="empty").inc()
        return {
            "chunks": 0,
            "backend": backend_used,
            "fallback_reason": fallback_reason,
        }

    with STAGE_LATENCY.labels(stage="chunk").time():
        chunks = chunk_text(full_text, chunk_size=800, overlap=120)

    if not chunks:
        INGEST_DOCUMENTS.labels(source_type=path.suffix.lower(), outcome="empty").inc()
        return {
            "chunks": 0,
            "backend": backend_used,
            "fallback_reason": fallback_reason,
        }

    prefix = doc_id_prefix or path.stem
    metadatas = [
        {
            "source": str(path),
            "chunk_index": i,
            "content_type": "chunk",
            "ocr_pages": doc.ocr_pages,
        }
        for i, _ in enumerate(chunks)
    ]
    ids = [f"{prefix}::chunk::{i}" for i in range(len(chunks))]
    texts = [c.text for c in chunks]

    with STAGE_LATENCY.labels(stage="embed_store").time():
        try:
            await store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except Exception as e:  # noqa: BLE001
            # add_texts already attempts self-heal once before raising. If
            # the upsert still failed it usually means the persistent HNSW
            # was left half-written by a previous crash AND the worker is
            # itself still holding mmap'd file handles — quarantining
            # while we hold those locks would just create a corrupt backup.
            # Instead, surface a clear "please restart" message so the
            # operator runs `make recover-chroma` (which stops uvicorn
            # first) or restarts the app (so the auto-recovery in
            # ChromaStore.__init__ fires on a fresh process).
            log.warning("chroma_upsert_unrecoverable_restart_required", error=str(e)[:200])
            INGEST_DOCUMENTS.labels(
                source_type=path.suffix.lower(), outcome="chroma_unrecoverable"
            ).inc()
            return {
                "chunks": 0,
                "backend": backend_used,
                "fallback_reason": "chroma_restart_required",
                "error": (
                    "Chroma index is unrecoverable in this process. "
                    "Please restart the API server (uvicorn) so it can "
                    "rebuild the index on startup, or run "
                    "`make recover-chroma` to quarantine the corrupt directory."
                ),
            }

    INGEST_DOCUMENTS.labels(source_type=path.suffix.lower(), outcome="ok").inc()
    log.info(
        "ingest_complete",
        source=str(path),
        chunks=len(chunks),
        backend=backend_used,
    )
    return {
        "chunks": len(chunks),
        "backend": backend_used,
        "fallback_reason": fallback_reason,
    }


async def ingest_directory(dir_path: Path) -> int:
    """Ingest every supported file in a directory."""
    paths: list[Path] = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        paths.extend(dir_path.glob(ext))
    total = 0
    for p in paths:
        total += await ingest_file(p)
    # Rebuild BM25 after a directory pass.
    BM25Index.rebuild()
    return total


async def _cli_run(dir_path: Path) -> int:
    """Pretty CLI wrapper: prints a per-file report and exits cleanly.

    Returning 0 on success even when every file ended up on the pdfplumber
    fallback — the previous behaviour exited with code 1 on docling probe
    failure, which made `python -m backend.ingestion.pipeline` look broken
    when in fact the fallback succeeded. We only exit non-zero when no
    supported files were found *or* when every file failed to extract.
    """
    if not dir_path.exists():
        print(f"[ingest] directory not found: {dir_path}")
        return 1
    paths: list[Path] = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        paths.extend(dir_path.glob(ext))
    if not paths:
        print(f"[ingest] no .pdf/.txt/.md files in {dir_path}")
        return 0

    print(f"[ingest] {len(paths)} file(s) found in {dir_path}")
    indexed = 0
    failures = 0
    for p in sorted(paths):
        try:
            result = await ingest_file(p)
        except Exception as exc:  # noqa: BLE001 — never let the CLI crash on one bad file
            failures += 1
            print(f"[ingest] FAIL {p.name}: {exc}")
            continue
        chunks = result.get("chunks", 0)
        backend = result.get("backend", "unknown")
        reason = result.get("fallback_reason")
        indexed += chunks
        suffix = f" — backend={backend}" + (f" (reason={reason})" if reason else "")
        print(f"[ingest] {p.name}: {chunks} chunk(s){suffix}")
    BM25Index.rebuild()
    print(f"[ingest] done: {indexed} chunk(s) indexed across {len(paths)} file(s)")
    return 0 if failures == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    raise SystemExit(asyncio.run(_cli_run(target)))