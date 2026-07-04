"""Per-session token-bucket rate limit (Item 2).

Uses slowapi + an in-memory backend (limits.aio.storage.MemoryStorage).
For multi-worker production you'd swap to Redis; for the take-home demo
the in-memory bucket is fine.
"""
from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import get_settings


def _key_func(request: Request) -> str:
    """Key on session_id when present, otherwise fall back to client IP."""
    # Avoid importing pydantic models here for cheap call paths.
    try:
        body = getattr(request.state, "_json_body", None)
        sid = None
        if body and isinstance(body, dict):
            sid = body.get("session_id")
    except Exception:  # noqa: BLE001
        sid = None
    return f"session:{sid}" if sid else f"ip:{get_remote_address(request)}"


_settings = get_settings()

limiter = Limiter(
    key_func=_key_func,
    default_limits=[f"{_settings.rate_limit_per_min}/minute"],
    # headers_enabled=False: slowapi's `_inject_headers` assumes the wrapped
    # handler returns a `starlette.responses.Response`. Our `/chat` route
    # returns a plain `dict`; slowapi then raises `Exception: parameter
    # 'response' must be an instance of starlette.responses.Response` and
    # the response comes back as 500. The decorator still enforces the
    # limit and raises RateLimitExceeded; we just skip the X-RateLimit
    # header injection on non-Response returns.
    headers_enabled=False,
)


__all__ = ["limiter", "RateLimitExceeded"]