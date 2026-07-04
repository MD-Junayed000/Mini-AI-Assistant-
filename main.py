"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
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
    app = FastAPI(title="Mini AI Assistant", version="0.2.2", lifespan=lifespan)

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

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    @app.exception_handler(ValidationError)
    async def _val_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    @app.exception_handler(Exception)
    async def _default_handler(request, exc):  # type: ignore[no-untyped-def]
        return await handler(request, exc)

    return app


app = create_app()
