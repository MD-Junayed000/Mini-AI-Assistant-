"""/healthz + per-dependency liveness, with a 10-second TTL cache.

Item 5: the Free-tier Ollama account charges per-token; the previous
implementation pinged the provider every healthz call. We now cache
the result so monitoring scrapes don't burn the budget.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import get_settings
from backend.observability.metrics import HEALTH_STATUS


@dataclass
class HealthSnapshot:
    components: dict[str, str] = field(default_factory=dict)
    overall: str = "unknown"
    taken_at: float = 0.0
    cached: bool = False


_CACHE: HealthSnapshot | None = None
_CACHE_LOCK = asyncio.Lock()


async def _probe_chroma() -> tuple[str, str]:
    try:
        # Install the posthog stub before chromadb grabs a reference to it.
        # (Keeps the chroma telemetry spam out of our healthz log.)
        try:
            import sys
            import types

            if "posthog" not in sys.modules:
                stub = types.ModuleType("posthog")
                stub.capture = lambda *a, **kw: None  # type: ignore[attr-defined]
                stub.identify = lambda *a, **kw: None  # type: ignore[attr-defined]
                sys.modules["posthog"] = stub
        except Exception:  # noqa: BLE001
            pass

        import chromadb  # noqa: WPS433  (deferred to avoid import-time cost)

        client = chromadb.PersistentClient(path=get_settings().chroma_persist_dir)
        # ListCollections is cheap and proves the on-disk store is reachable.
        client.list_collections()
        return "chroma", "up"
    except Exception:  # noqa: BLE001
        return "chroma", "down"


async def _probe_ollama() -> tuple[str, str]:
    """Probe Ollama / Ollama Cloud.

    The OpenAI-compatible endpoint shape (`/v1/models`) is the only path that
    is reliable across both local Ollama installs and `ollama.com` Cloud:
    local Ollama mounts it; Ollama Cloud rejects native `/api/tags` and returns
    404 with an HTML body, which made the previous probe always read `down`
    (status < 500 → "up" → but with 401 auth failures the metric still drifted).
    """
    import httpx

    base = get_settings().ollama_cloud_base_url.rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as cx:
            r = await cx.get(
                url,
                headers={"Authorization": f"Bearer {get_settings().ollama_cloud_api_key}"},
            )
            # 2xx = up. 401/403 = "reachable but auth bad" → still report down
            # so the operator notices the misconfigured key.
            return "ollama", "up" if 200 <= r.status_code < 300 else "down"
    except Exception:  # noqa: BLE001
        return "ollama", "down"


async def _probe_mongo() -> tuple[str, str]:
    try:
        from motor.motor_asyncio import AsyncIOMotorClient  # noqa: WPS433

        cx = AsyncIOMotorClient(get_settings().mongodb_uri, serverSelectionTimeoutMS=2000)
        await cx.admin.command("ping")
        cx.close()
        return "mongo", "up"
    except Exception:  # noqa: BLE001
        return "mongo", "down"


async def _gather_fresh() -> dict[str, str]:
    results = await asyncio.gather(
        _probe_chroma(),
        _probe_ollama(),
        _probe_mongo(),
        return_exceptions=False,
    )
    return dict(results)


async def snapshot(force_refresh: bool = False) -> HealthSnapshot:
    """Return a (possibly cached) snapshot of all dependencies."""
    global _CACHE
    ttl = get_settings().health_cache_ttl_seconds
    now = time.monotonic()

    async with _CACHE_LOCK:
        if (
            not force_refresh
            and _CACHE is not None
            and (now - _CACHE.taken_at) < ttl
        ):
            _CACHE.cached = True
            return _CACHE

        comps = await _gather_fresh()
        overall = "up" if all(v == "up" for v in comps.values()) else "degraded"
        _CACHE = HealthSnapshot(
            components=comps,
            overall=overall,
            taken_at=now,
            cached=False,
        )
        # Mirror into Prometheus
        for c, v in comps.items():
            HEALTH_STATUS.labels(component=c).set(1 if v == "up" else 0)
        return _CACHE


async def healthz_payload() -> dict[str, Any]:
    snap = await snapshot()
    return {
        "overall": snap.overall,
        "components": snap.components,
        "cached": snap.cached,
        "ttl_seconds": get_settings().health_cache_ttl_seconds,
    }
