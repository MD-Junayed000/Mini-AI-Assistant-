"""Ollama Cloud chat client — OpenAI-compatible.

Primary: gpt-oss:20b. Fallback: gpt-oss:120b.
Both support native tool-calling; we still parse JSON-intent for portability.
Model names live in `backend.config.Settings` and can be overridden via env
vars (`OLLAMA_PRIMARY_MODEL` / `OLLAMA_FALLBACK_MODEL`).
Browse live tags with: curl https://ollama.com/api/tags | jq '.models[].name'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.errors import LLMError, RateLimitError
from backend.llm.prompts import BASE_SYSTEM_PROMPT
from backend.observability.logging_config import get_logger
from backend.observability.metrics import STAGE_LATENCY
from backend.observability.tracing import tracer

log = get_logger("llm")


@dataclass
class ChatRequest:
    user: str
    history: list[dict[str, str]]  # [{role, content}, ...]
    context_blocks: list[str]  # pre-built [doc-i] excerpts


@dataclass
class ChatResponse:
    text: str
    model: str
    fallback_used: bool


class _Retryable(Exception):
    pass


class OllamaCloudChatClient:
    """Async client for Ollama Cloud with primary/fallback model + retries."""

    def __init__(self) -> None:
        s = get_settings()
        self._primary = s.ollama_primary_model
        self._fallback = s.ollama_fallback_model
        self._timeout = s.ollama_timeout_seconds
        self._client = AsyncOpenAI(
            base_url=s.ollama_cloud_base_url,
            api_key=s.ollama_cloud_api_key,
            timeout=s.ollama_timeout_seconds,
            max_retries=0,  # we control retries
        )

    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def _call_once(self, model: str, messages: list[dict[str, str]]) -> str:
        """One chat call. Retries on transient httpx errors."""
        tr = tracer()
        span_cm = (
            tr.start_as_current_span("llm.chat") if hasattr(tr, "start_as_current_span") else _nullctx()
        )
        with span_cm as span:
            if span is not None and hasattr(span, "set_attribute"):
                try:
                    span.set_attribute("llm.model", model)
                    span.set_attribute("llm.prompt_messages", len(messages))
                except Exception:  # noqa: BLE001
                    pass
            try:
                with STAGE_LATENCY.labels(stage="llm").time():
                    resp = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=800,
                    )
            except httpx.TimeoutException as e:
                raise _Retryable(str(e)) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (408, 425, 429, 500, 502, 503, 504):
                    raise _Retryable(str(e)) from e
                # 4xx is hard failure
                if e.response.status_code == 429:
                    raise RateLimitError("ollama_429") from e
                raise LLMError(f"http {e.response.status_code}") from e
            except Exception as e:  # noqa: BLE001
                raise _Retryable(str(e)) from e

        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError, KeyError) as e:
            raise LLMError("malformed_response") from e

    async def chat(self, req: ChatRequest) -> ChatResponse:
        messages: list[dict[str, str]] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
        if req.context_blocks:
            joined = "\n\n".join(req.context_blocks)
            messages.append({"role": "system", "content": f"Retrieved context:\n{joined}"})
        for m in req.history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": req.user})

        try:
            text = await self._call_once(self._primary, messages)
            return ChatResponse(text=text, model=self._primary, fallback_used=False)
        except RateLimitError:
            raise
        except _Retryable as e:
            log.warning("primary_failed_falling_back", error=str(e))
        except LLMError as e:
            log.warning("primary_failed_falling_back", error=str(e))

        try:
            text = await self._call_once(self._fallback, messages)
            return ChatResponse(text=text, model=self._fallback, fallback_used=True)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"both_models_failed: {e}") from e


# Context manager shim used when OTel is disabled (no-op).
from contextlib import contextmanager


@contextmanager
def _nullctx():
    yield