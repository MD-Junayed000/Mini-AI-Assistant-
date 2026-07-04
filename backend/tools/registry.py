"""Tool registry + per-tool in-memory cache.

Cache invalidation is mtime-based — when the JSON file changes on disk,
the cache is rebuilt on the next call. This is fine for a take-home demo
where admins drop new files in `data/`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import get_settings


@dataclass
class _Cached:
    data: Any
    loaded_at: float
    mtime: float


_CACHE: dict[str, _Cached] = {}


def _load_json(name: str) -> Any:
    path = Path(get_settings().__class__.model_config["env_file"] or ".env").parent / "data" / name
    # Fallback: relative to cwd
    if not path.exists():
        path = Path("data") / name
    # utf-8-sig transparently strips a leading BOM if present (some Windows
    # editors — notably Notepad and certain VS Code encodings — save with one).
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _maybe_reload(key: str, filename: str) -> Any:
    path = Path("data") / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    mtime = path.stat().st_mtime
    cached = _CACHE.get(key)
    if cached and cached.mtime >= mtime:
        return cached.data
    data = _load_json(filename)
    _CACHE[key] = _Cached(data=data, loaded_at=time.time(), mtime=mtime)
    return data


def order_status(order_id: str) -> dict[str, Any]:
    """Look up an order by id; raises KeyError if missing."""
    orders = _maybe_reload("orders", "orders.json")
    for o in orders:
        if o["order_id"].upper() == order_id.upper():
            return o
    raise KeyError(order_id)


def product_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Naive substring + tag search across the product catalog."""
    products = _maybe_reload("products", "products.json")
    q = query.lower().strip()
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in products:
        hay = " ".join(
            [
                p["name"].lower(),
                p["category"].lower(),
                " ".join(p.get("tags", [])).lower(),
                p["sku"].lower(),
            ]
        )
        score = sum(1 for token in q.split() if token in hay)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]["sku"]))
    return [p for _, p in scored[:top_k]]


def refresh_cache() -> None:
    """Force the cache to reload on next access."""
    _CACHE.clear()