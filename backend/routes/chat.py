"""HTTP routes — /ingest, /chat, /session/{id}/reset, /healthz, /metrics, /admin/cache/refresh."""
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


# ---------- Schemas -------------------------------------------------------
class ChatIn(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


# ---------- Dependencies --------------------------------------------------
def get_memory() -> Memory:  # noqa: D401
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
@router.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise ValidationError("missing_filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValidationError(f"unsupported_extension: {suffix}")

    dest = Path("data") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    chunks = await ingest_file(dest)
    # Rebuild BM25 after every ingest (small corpus, cheap).
    BM25Index.rebuild()
    return {"chunks": chunks, "source": file.filename}


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


# ---------- Global exception handler (registered in main.py) ------------
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
        elif isinstance(exc, RateLimitExceeded):
            payload = {
                "error": "rate_limited",
                "code": "rate_limited",
                "request_id": REQUEST_ID.get(),
                "friendly": friendly_message("rate_limited"),
            }
            status = 429
        else:
            payload = {
                "error": "internal_error",
                "code": "internal_error",
                "request_id": REQUEST_ID.get(),
                "friendly": friendly_message("internal_error"),
            }
            status = 500
        return JSONResponse(payload, status_code=status)

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