"""Embedding client — local sentence-transformers by default, optional HF API.

Two backends behind one async API:

* LocalBackend — uses `sentence-transformers` (already in requirements.txt).
  The first call downloads the model from Hugging Face (cached under
  ~/.cache/huggingface/) and produces 384-d vectors. No API key, no network
  rate limits, deterministic across processes.

* HFRouterBackend — calls the HF router's `/v1/embeddings` endpoint for
  users who want cloud-side embeddings. Note that the HF router only
  serves models listed under "Inference Providers"; many sentence-
  transformer models (e.g. BAAI/bge-small-en-v1.5) are NOT routable and
  return 404. Switch the model in .env, or fall back to local.

Selection is automatic:
  - If `HF_EMBEDDINGS_REMOTE=true` in .env → use HFRouterBackend
  - Otherwise → use LocalBackend (default; recommended)
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.observability.logging_config import get_logger
from backend.observability.metrics import STAGE_LATENCY

log = get_logger("embeddings")


class _Retryable(Exception):
    pass


# ---------------------------------------------------------------------------
# Local sentence-transformers backend (default)
# ---------------------------------------------------------------------------
class LocalBackend:
    """Run sentence-transformers on the local machine."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            log.info("embeddings_local_load", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        # Keep a typed reference for mypy / IDEs
        _model: Any = self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with STAGE_LATENCY.labels(stage="embed").time():
            return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        # normalize_embeddings=True keeps vectors on the unit sphere so cosine == dot
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


# ---------------------------------------------------------------------------
# HF Router backend (optional)
# ---------------------------------------------------------------------------
class HFRouterBackend:
    """Call the HF router's OpenAI-compatible `/v1/embeddings` endpoint.

    WARNING: only models exposed via HF Inference Providers are routable;
    most sentence-transformers checkpoints return 404 here. If you see 404
    in logs, either pick a different model or set `HF_EMBEDDINGS_REMOTE=false`
    in .env to fall back to the local backend.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = 30

    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        try:
            with STAGE_LATENCY.labels(stage="embed").time():
                async with httpx.AsyncClient(timeout=self._timeout) as cx:
                    r = await cx.post(self._url, headers=self._headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
        except httpx.HTTPError as e:
            raise _Retryable(str(e)) from e

        # OpenAI-compatible: {data: [{embedding: [...]}, ...]}
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as e:
            log.error("embeddings_unexpected_shape", payload=list(data.keys()))
            raise _Retryable("embeddings_unexpected_shape") from e

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0]


# ---------------------------------------------------------------------------
# Public façade — picks backend at construction time
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder() -> LocalBackend | HFRouterBackend:
    s = get_settings()
    if os.getenv("HF_EMBEDDINGS_REMOTE", "").lower() in {"1", "true", "yes"}:
        log.info("embeddings_backend", chosen="hf_router", model=s.hf_embedding_model)
        return HFRouterBackend(
            base_url=s.hf_inference_base_url,
            model=s.hf_embedding_model,
            api_key=s.hf_inference_api_key,
        )
    log.info("embeddings_backend", chosen="local", model=s.hf_embedding_model)
    return LocalBackend(s.hf_embedding_model)


# Backwards-compatible alias — code elsewhere imports `HFEmbeddingClient`.
class HFEmbeddingClient:  # noqa: D401  (kept name for compatibility)
    """Thin async wrapper around `get_embedder()`."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await get_embedder().embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        return await get_embedder().embed_one(text)


__all__ = ["HFEmbeddingClient", "LocalBackend", "HFRouterBackend", "get_embedder"]