"""HTTP routes — /ingest, /chat, /session/{id}/reset, /healthz, /metrics, /admin/cache/refresh, /admin/kb/sources, /admin/kb/clear, /admin/kb/clear-source."""
# NOTE: no `from __future__ import annotations` here — slowapi's decorator wrapper
# breaks FastAPI's signature introspection of `body: ChatIn` when PEP 563 turns
# annotations into strings, raising PydanticUndefinedAnnotation at import time.

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

from backend.config import get_settings

_settings = get_settings()

from backend.errors import AppError, ValidationError, friendly_message
from backend.ingestion.pipeline import ingest_file
from backend.memory import Memory
from backend.observability.health import healthz_payload
from backend.observability.logging_config import get_logger
from backend.observability.metrics import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    RATE_LIMIT_HITS,
    REGISTRY,
)
from backend.observability.request_context import REQUEST_ID
from backend.observability.tracing import tracer
from backend.pipeline.chat import run_chat
from backend.security.rate_limit import limiter
from backend.tools.registry import refresh_cache as refresh_tool_cache
from backend.vector_store.bm25_index import BM25Index

router = APIRouter()
log = get_logger("routes")


# ---------- Schemas -------------------------------------------------------
class ChatIn(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class RenameIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


# ---------- Dependencies --------------------------------------------------
def get_memory() -> Memory:
    """Process-singleton Memory instance — replaced by FastAPI's DI in main.py."""
    return _memory_instance  # type: ignore[name-defined]


_memory_instance: Memory | None = None


def install_memory(m: Memory) -> None:
    global _memory_instance
    _memory_instance = m


# ---------- Middleware helpers -------------------------------------------
async def _track_http(request: Request, call_next):  # type: ignore[no-untyped-def]
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path  # type: ignore[union-attr]
    method = request.method
    with HTTP_LATENCY.labels(method=method, endpoint=endpoint).time():
        try:
            response = await call_next(request)
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
            return response
        except RateLimitExceeded:
            RATE_LIMIT_HITS.labels(endpoint=endpoint).inc()
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status="429").inc()
            raise


# ---------- Routes -------------------------------------------------------
@router.get("/")
async def root() -> dict:
    """Friendly landing — the UI lives at the Streamlit port (default 8501)."""
    return {
        "service": "MiniCo Internal Docs",
        "version": "0.2.2",
        "ui": "http://localhost:8501 (run `streamlit run ui/streamlit_app.py`)",
        "endpoints": {
            "POST /chat": "send a chat message (JSON: {session_id, message})",
            "POST /ingest": "upload a PDF/TXT/MD document for retrieval",
            "GET  /sessions": "list all known chat sessions (newest first)",
            "POST /session/{id}/reset": "clear conversation memory for a session",
            "POST /session/{id}/delete": "permanently delete a session's history",
            "POST /session/{id}/rename": "rename a session (body: {title})",
            "GET  /healthz": "component health snapshot (cached 30s)",
            "GET  /metrics": "Prometheus exposition",
            "POST /admin/cache/refresh": "rebuild BM25 + reload tool registry",
        },
    }


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise ValidationError("missing_filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValidationError(f"unsupported_extension: {suffix}")

    dest = Path("data") / "uploads" / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    result = await ingest_file(dest)
    # Rebuild BM25 after every ingest (small corpus, cheap).
    BM25Index.rebuild()
    body = {
        "chunks": result.get("chunks", 0),
        "source": file.filename,
        "backend": result.get("backend", "unknown"),
        "fallback_reason": result.get("fallback_reason"),
    }
    # Surface a one-line error string when extraction produced no chunks —
    # the Streamlit UI uses this to explain why nothing was indexed
    # (corrupt PDF, empty text, etc.) without a separate toast.
    err = result.get("error")
    if err and result.get("chunks", 0) == 0:
        body["error"] = err
    return body


@router.post("/chat")
# Honour the global rate limit configured in Settings.
# (We can't reference `limiter._default_limits` here — slowapi prints
# "couldn't parse rate limit string '<LimitGroup object ...>'" because the
# decorator stringifies the object. A plain string from settings works.)
@limiter.limit(f"{_settings.rate_limit_per_min}/minute")
async def chat(request: Request, body: ChatIn, memory: Memory = Depends(get_memory)) -> dict:
    tr = tracer()
    if hasattr(tr, "start_as_current_span"):
        with tr.start_as_current_span("chat"):
            result = await run_chat(
                session_id=body.session_id,
                user_message=body.message,
                memory=memory,
            )
    else:
        result = await run_chat(
            session_id=body.session_id,
            user_message=body.message,
            memory=memory,
        )
    return {
        "answer": result.answer,
        "sources": result.sources,
        "tool_calls": result.tool_calls,
        "evidence": result.evidence,
        "injection_risk": result.injection_risk,
        "fallback_used": result.fallback_used,
    }


@router.post("/session/{session_id}/reset")
async def reset_session(session_id: str, memory: Memory = Depends(get_memory)) -> dict:
    await memory.reset(session_id)
    return {"session_id": session_id, "reset": True}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, memory: Memory = Depends(get_memory), limit: int = 200
) -> dict:
    """Return the conversation history for one session, oldest first.

    The Streamlit sidebar uses this when the user switches to a previous
    chat: we fetch the server's record of that conversation so the main
    pane shows the same messages regardless of whether the user has
    kept the local Streamlit state alive across the switch.
    """
    msgs = await memory.history(session_id, limit=limit)
    # Strip internal-only fields; the UI only needs role/content/sources/
    # elapsed_s so we don't leak metadata (e.g. original tool_call trace).
    safe: list[dict[str, Any]] = []
    for m in msgs:
        meta = m.get("metadata") or {}
        safe.append(
            {
                "role": m.get("role", "assistant"),
                "content": m.get("content", ""),
                "ts": m.get("ts", 0.0),
                "elapsed_s": meta.get("elapsed_s"),
                "sources": meta.get("sources") or [],
            }
        )
    return {"session_id": session_id, "messages": safe}


@router.get("/sessions")
async def list_sessions(memory: Memory = Depends(get_memory)) -> dict:
    """All known sessions, newest activity first.

    The Streamlit sidebar uses this to render a list of past chats that the
    user can switch between without losing the previous conversation.
    """
    sessions = await memory.list_sessions()
    return {"sessions": sessions}


@router.post("/session/{session_id}/delete")
async def delete_session(session_id: str, memory: Memory = Depends(get_memory)) -> dict:
    """Permanently delete a session's history from memory."""
    removed = await memory.delete_session(session_id)
    return {"session_id": session_id, "deleted": removed}


@router.post("/session/{session_id}/rename")
async def rename_session(
    session_id: str, body: RenameIn, memory: Memory = Depends(get_memory)
) -> dict:
    """Rename a session. We don't persist titles (Memory has no metadata table);
    a successful call returns the new title so the UI can keep its own map in
    sync. The rename is a no-op on the server side — the auto-derived title
    will re-appear next time the page reloads.
    """
    title = body.title.strip()
    return {"session_id": session_id, "title": title}


@router.get("/healthz")
async def healthz() -> dict:
    return await healthz_payload()


@router.get("/metrics")
def metrics() -> JSONResponse:
    body = generate_latest(REGISTRY)
    return JSONResponse(content=body.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@router.post("/admin/cache/refresh")
def admin_refresh_cache() -> dict:
    refresh_tool_cache()
    BM25Index.rebuild()
    return {"refreshed": True}


class ClearSourceIn(BaseModel):
    """Body for `POST /admin/kb/clear-source`.

    `source` is the exact metadata string the pipeline stored at ingest time
    (e.g. `data/uploads/foo.pdf`). Use `GET /admin/kb/sources` first to see
    the canonical names — never guess, two uploads with the same filename
    in different directories are *not* the same source.
    """

    source: str = Field(..., min_length=1, max_length=512)


@router.get("/admin/kb/sources")
async def admin_list_sources() -> dict:
    """List every distinct source currently in the vector store.

    Returns one entry per upload with the chunk count, sorted by chunk count
    descending then source name ascending. Used by the Streamlit sidebar to
    render a "Clear this document" button per indexed file.
    """
    from backend.vector_store.chroma_store import ChromaStore

    store = ChromaStore()
    sources = await store.list_sources()
    total_chunks = sum(s["chunks"] for s in sources)
    return {
        "sources": sources,
        "total_chunks": total_chunks,
        "total_sources": len(sources),
    }


@router.post("/admin/kb/clear-source")
async def admin_clear_source(body: ClearSourceIn) -> dict:
    """Delete every chunk that came from a single source.

    Use `GET /admin/kb/sources` to discover the exact `source` value — the
    matching is on the full stored path string, not the filename, so a file
    called `report.pdf` uploaded from two different directories is
    distinguished correctly.

    After the deletion we rebuild BM25 so lexical search stays consistent
    with what the vector store actually contains.
    """
    from backend.vector_store.chroma_store import ChromaStore

    store = ChromaStore()
    removed = await store.delete_by_source(body.source)
    BM25Index.rebuild()
    log.info(
        "kb_cleared_source",
        source=body.source,
        removed=removed,
    )
    return {"source": body.source, "removed": removed}


@router.post("/admin/kb/clear")
async def admin_clear_kb() -> dict:
    """Remove every chunk from the vector store.

    Does NOT touch the on-disk uploads directory (`data/uploads/`) — the
    files stay around so the user can re-ingest them. We only wipe the
    *indexed* representation in Chroma + the BM25 cache.

    After deletion, BM25 is rebuilt (it'll come back empty).
    """
    from backend.vector_store.chroma_store import ChromaStore

    store = ChromaStore()
    removed = await store.clear_all()
    BM25Index.rebuild()
    log.info("kb_cleared_all", removed=removed)
    return {"removed": removed}


# ---------- Global exception handler (registered in main.py) ------------
def _retry_after_seconds(exc: RateLimitExceeded) -> int:
    """Best-effort Retry-After in whole seconds for a slowapi RateLimitExceeded."""
    # slowapi sets exc.limit to the Limit object that was breached.
    rate_item = getattr(exc, "limit", None)
    # Many slowapi versions also stash the underlying Limit under the
    # exception's first positional arg as a fallback.
    if rate_item is None and exc.args:
        rate_item = exc.args[0] if hasattr(exc.args[0], "GRANULARITY") else None
    gran = getattr(rate_item, "GRANULARITY", None) if rate_item is not None else None
    if isinstance(gran, (int, float)) and gran > 0:
        return int(gran)
    # Slowapi may have pre-computed headers; prefer that when present.
    headers = getattr(exc, "headers", None) or {}
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return int(v)
            except (TypeError, ValueError):
                return 60
    return 60


def make_exception_handler():
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, AppError):
            payload = {
                "error": exc.message,
                "code": exc.code,
                "request_id": REQUEST_ID.get(),
                "friendly": friendly_message(exc.code),
            }
            status = exc.http_status
            return JSONResponse(payload, status_code=status)
        if isinstance(exc, RateLimitExceeded):
            payload = {
                "error": "rate_limited",
                "code": "rate_limited",
                "request_id": REQUEST_ID.get(),
                "friendly": friendly_message("rate_limited"),
            }
            return JSONResponse(
                payload,
                status_code=429,
                headers={"Retry-After": str(_retry_after_seconds(exc))},
            )
        payload = {
            "error": "internal_error",
            "code": "internal_error",
            "request_id": REQUEST_ID.get(),
            "friendly": friendly_message("internal_error"),
        }
        return JSONResponse(payload, status_code=500)

    return handler


async def _attach_request_id_json(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Read JSON body once and cache on request.state for the rate-limit key."""
    if request.method in {"POST"} and request.headers.get("content-type", "").startswith("application/json"):
        try:
            body_bytes = await request.body()
            import json as _json

            request.state._json_body = _json.loads(body_bytes or b"{}")
        except Exception:  # noqa: BLE001
            request.state._json_body = None
    return await call_next(request)