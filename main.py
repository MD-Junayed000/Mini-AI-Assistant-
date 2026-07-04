"""FastAPI application entrypoint."""
from __future__ import annotations

import os

# Silence ChromaDB telemetry BEFORE any chromadb import anywhere in the
# import graph (chroma's __init__ spawns a posthog client at first use and
# recent posthog releases break chroma's call signature, which floods the
# log with `capture() takes 1 positional argument but 3 were given`).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "True")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import get_settings
from backend.errors import ValidationError
from backend.memory import Memory
from backend.observability.logging_config import configure_logging, get_logger
from backend.observability.request_context import RequestIDMiddleware
from backend.routes.chat import install_memory, make_exception_handler, router, _attach_request_id_json
from starlette.middleware.base import BaseHTTPMiddleware
from backend.security.rate_limit import limiter


configure_logging()
log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mem = Memory()
    install_memory(mem)
    log.info("startup", settings_keys=list(get_settings().model_dump().keys()))
    try:
        yield
    finally:
        await mem.close()


def create_app() -> FastAPI:
    app = FastAPI(title="MiniCo Internal Docs", version="0.2.2", lifespan=lifespan)

    # Rate limiter state — slowapi requires this on the app.
    app.state.limiter = limiter

    # Middleware (order matters: outermost first).
    app.add_middleware(RequestIDMiddleware)
    # Cache parsed JSON body on request.state so the slowapi key_func can read session_id.
    app.add_middleware(BaseHTTPMiddleware, dispatch=_attach_request_id_json)

    # Routes.
    app.include_router(router)

    # Rate-limit handler.
    handler = make_exception_handler()

    @app.exception_handler(RateLimitExceeded)
    async def _ratelimit_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    @app.exception_handler(ValidationError)
    async def _val_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    @app.exception_handler(Exception)
    async def _default_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _starlette_http_handler(request, exc: StarletteHTTPException):  # type: ignore[no-untyped-def]
        """Starlette HTTP errors (404 etc.) — return JSON, never HTML, never 500."""
        return JSONResponse(
            {
                "error": "not_found" if exc.status_code == 404 else "http_error",
                "code": "not_found" if exc.status_code == 404 else "http_error",
                "status": exc.status_code,
                "friendly": "That endpoint doesn't exist. Try GET / for the route catalog.",
            },
            status_code=exc.status_code,
        )

    return app


app = create_app()
