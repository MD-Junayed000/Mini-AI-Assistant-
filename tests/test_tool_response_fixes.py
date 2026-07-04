"""Regression tests for the three tool-calling fixes.

Covers:
  1. Prompt no longer instructs the model to emit [doc-N] citations.
  2. Prompt uses ORD001 (not A1001) as the sample order id.
  3. Chat pipeline substitutes a non-blank answer when the LLM returns
     an empty response (with a tool summary when a tool already ran,
     with the standard fallback line otherwise).
  4. Chat pipeline does not leak [tool-result ...] blocks into the
     user-facing reply.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from backend.llm.prompts import BASE_SYSTEM_PROMPT
from backend.pipeline import chat as chat_mod
from backend.pipeline.chat import run_chat


# ----------------------------------------------------------------- static


def test_prompt_uses_actual_order_id() -> None:
    assert '"order_id": "ORD001"' in BASE_SYSTEM_PROMPT
    assert '"order_id": "A1001"' not in BASE_SYSTEM_PROMPT


def test_prompt_does_not_instruct_citation_echo() -> None:
    assert "cite inline like" not in BASE_SYSTEM_PROMPT
    assert "Do NOT echo those markers" in BASE_SYSTEM_PROMPT


# ----------------------------------------------------------------- live


class _FakeMemory:
    def __init__(self) -> None:
        self.turns: list[Any] = []

    async def append(self, m: Any) -> None:
        self.turns.append(m)

    async def history(self, session_id: str, limit: int = 20):
        return []


class _FakeResp:
    def __init__(self, text: str, model: str = "gpt-oss:20b") -> None:
        self.text = text
        self.model = model
        self.fallback_used = False


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text

    async def chat(self, req):  # noqa: D401
        return _FakeResp(self._text)


def _run(user_message: str, llm_text: str) -> Any:
    """Run the chat pipeline with a stubbed LLM that returns llm_text."""
    mem = _FakeMemory()
    with patch.object(chat_mod, "OllamaCloudChatClient", lambda: _FakeClient(llm_text)):
        return asyncio.run(
            run_chat(
                session_id="regression",
                user_message=user_message,
                memory=mem,
            )
        )


def test_empty_llm_response_substituted_with_fallback() -> None:
    res = _run("What is your return policy?", "")
    assert res.answer.strip(), "empty LLM response should not produce a blank answer"
    assert "I don't know" in res.answer or "available information" in res.answer


def test_non_empty_llm_response_passes_through() -> None:
    res = _run("What is your return policy?", "You can return within 30 days.")
    assert "30 days" in res.answer
    # No raw internal markers should leak.
    assert "[tool-result " not in res.answer
    assert "[doc-1]" not in res.answer


def test_tool_short_circuit_does_not_leak_internal_block() -> None:
    """If the LLM response happens to contain a [tool-result ...] block,
    the pipeline must not echo it verbatim into the user-facing reply."""
    bad = "Here is the answer. [tool-result order_status] {'order_id': 'ORD001'}"
    res = _run("Where is my order ORD001?", bad)
    assert "[tool-result " not in res.answer


def test_empty_response_with_no_tool_uses_fallback() -> None:
    res = _run("Tell me a joke about ops.", "")
    assert res.answer.strip()
