"""Chat pipeline orchestrator.

Pipeline (each block is also a span in OTel when tracing is enabled):

    User message
        │
        ▼
    1. inject-check  ─→ score() with regex-weighted heuristic
        │
        ▼
    2. memory-append ─→ persist the user turn
        │
        ▼
    3. memory-load   ─→ last N turns for context
        │
        ▼
    4. tool-parse    ─→ if the user message looks like a tool JSON, run it
        │
        ▼
    5. retrieve       ─→ dense + BM25 → RRF → rerank
        │
        ▼
    6. gate           ─→ multi-signal answerability check
        │                insufficient → fallback answer, no LLM call
        ▼
    7. llm            ─→ primary model, fallback on retryable failure
        │
        ▼
    8. tool-parse-2  ─→ if the LLM emitted a tool intent, run it
        │
        ▼
    9. memory-append (assistant) + return

Every block emits a Prometheus histogram (`stage_latency_seconds`) and a
matching OTel span when tracing is on, so an operator can correlate a
slow request to the exact stage that made it slow.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from backend.errors import RetrieverEmptyError
from backend.llm.client import ChatRequest, OllamaCloudChatClient
from backend.locks import session_lock
from backend.memory import Memory, Message
from backend.observability.logging_config import get_logger
from backend.observability.metrics import ANSWERABILITY, PROMPT_INJECTION, STAGE_LATENCY
from backend.observability.tracing import tracer
from backend.retrieval.gate import evaluate as gate_evaluate
from backend.retrieval.hybrid import retrieve as hybrid_retrieve, Retrieved
from backend.security.injection_guard import score as score_injection
from backend.tools.router import dispatch as dispatch_tool, parse_tool_intent

log = get_logger("pipeline")

_FALLBACK_ANSWER = "I don't know based on the available information."

# Regex matches greetings / pleasantries / very short non-question messages
# where retrieval is meaningless. We still call the LLM so the assistant
# can respond naturally to "hello", "thanks", etc.
_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|hiya|howdy|good\s+(?:morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|cheers|bye|goodbye|see\s+ya|cya|ok(?:ay)?|"
    r"sure|got\s+it|cool|nice|great)\b[!.?,\s]*$",
    re.IGNORECASE,
)


def _is_small_talk(message: str) -> bool:
    """True when the message is a greeting/pleasantry/short ack — skip retrieval."""
    if len(message) > 60:
        return False
    if "?" in message:
        return False
    return bool(_GREETING_RE.match(message))


@contextmanager
def _nullctx():
    """No-op context manager used when OTel is disabled."""
    yield None


def _maybe_span(name: str):
    """Return a context manager: an OTel span if tracing is on, else a no-op."""
    tr = tracer()
    if hasattr(tr, "start_as_current_span"):
        return tr.start_as_current_span(name)
    return _nullctx()


@dataclass
class ChatResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    injection_risk: float = 0.0
    fallback_used: bool = False


async def run_chat(
    *,
    session_id: str,
    user_message: str,
    memory: Memory,
) -> ChatResult:
    """Top-level entrypoint — wraps everything in a root OTel span."""
    with _maybe_span("chat.request") as root_span:
        if root_span is not None and hasattr(root_span, "set_attribute"):
            try:
                root_span.set_attribute("session.id", session_id)
                root_span.set_attribute("user.message_length", len(user_message))
            except Exception:  # noqa: BLE001
                pass
        async with session_lock(session_id):
            return await _run_chat_inner(
                session_id=session_id,
                user_message=user_message,
                memory=memory,
                root_span=root_span,
            )


async def _run_chat_inner(
    *,
    session_id: str,
    user_message: str,
    memory: Memory,
    root_span: Any,
) -> ChatResult:
    # 1. Injection scoring
    with _maybe_span("chat.injection_check"):
        verdict = score_injection(user_message)
    if verdict.is_injection:
        PROMPT_INJECTION.labels(surface="user").inc()
        log.warning("prompt_injection_detected", session=session_id, signals=verdict.signals)
        if root_span is not None and hasattr(root_span, "set_attribute"):
            try:
                root_span.set_attribute("injection.score", verdict.score)
                root_span.set_attribute("injection.flagged", True)
            except Exception:  # noqa: BLE001
                pass

    # 2. Persist the user turn.
    with _maybe_span("chat.memory_append_user"):
        await memory.append(Message(session_id=session_id, role="user", content=user_message))

    # 3. Load history.
    with _maybe_span("chat.memory_load"):
        history_docs = await memory.history(session_id=session_id, limit=20)
        history = [
            {"role": d["role"], "content": d["content"]}
            for d in history_docs
            if d.get("role") in ("user", "assistant")
        ]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == user_message:
            history = history[:-1]

    # 4. Early tool parse.
    try:
        early_tool = parse_tool_intent(user_message)
    except Exception:  # noqa: BLE001
        early_tool = None

    tool_calls_made: list[dict[str, Any]] = []
    extra_context_blocks: list[str] = []

    if early_tool is not None:
        with STAGE_LATENCY.labels(stage="tool").time(), _maybe_span("chat.tool_early"):
            result = dispatch_tool(early_tool)
        tool_calls_made.append({"tool": early_tool.name, "args": early_tool.args, "result": result})
        extra_context_blocks.append(f"[tool-result {early_tool.name}] {result!r}")

    # 5. Retrieve + rerank.
    with STAGE_LATENCY.labels(stage="retrieve_rerank").time(), _maybe_span("chat.retrieve_rerank"):
        retrieved: list[Retrieved] = await hybrid_retrieve(user_message, top_k=8)

    with _maybe_span("chat.gate"):
        gate = gate_evaluate(retrieved)
    if root_span is not None and hasattr(root_span, "set_attribute"):
        try:
            root_span.set_attribute("gate.score", gate.signals.get("rerank_top", 0.0))
            root_span.set_attribute("gate.decision", gate.decision)
        except Exception:  # noqa: BLE001
            pass

    # 5b. Decide whether to send the retrieved docs to the LLM as context.
    #     We do NOT hard-short-circuit on low confidence any more — the system
    #     prompt now distinguishes "domain question" (cite or refuse) from
    #     "general chat" (answer freely). The gate still records its verdict
    #     for observability and the LLM decides how to use the context.
    is_small_talk = _is_small_talk(user_message) and not tool_calls_made

    context_blocks: list[str] = list(extra_context_blocks)
    sources: list[dict[str, Any]] = []
    # Always provide retrieved context when we have any — the LLM is the
    # arbiter of relevance now. Empty retrieval just means an empty context.
    for i, r in enumerate(retrieved[:6]):
        context_blocks.append(f"[doc-{i + 1}] {r.text}")
        sources.append(
            {
                "id": r.id,
                "preview": r.text[:160],
                "metadata": r.metadata,
                "rerank_score": r.rerank_score,
            }
        )

    # Count this as a "grounded" outcome when the gate was satisfied, else
    # "fallback" — but we never bail out without calling the LLM. The LLM
    # owns the refusal decision now.
    ANSWERABILITY.labels(
        decision="grounded" if gate.is_sufficient else "fallback"
    ).inc()

    # 6. LLM call.
    try:
        client = OllamaCloudChatClient()
        with STAGE_LATENCY.labels(stage="llm_chat").time(), _maybe_span("chat.llm"):
            resp = await client.chat(
                ChatRequest(
                    user=user_message,
                    history=history,
                    context_blocks=context_blocks,
                )
            )
        answer = resp.text
    except Exception as e:  # noqa: BLE001
        log.error("llm_chat_failed", error=str(e))
        if tool_calls_made:
            return ChatResult(
                answer=_format_tool_summary(tool_calls_made),
                sources=sources,
                tool_calls=tool_calls_made,
                evidence={"gate": gate.signals, "gate_decision": gate.decision},
                injection_risk=verdict.score,
                fallback_used=False,
            )
        raise RetrieverEmptyError("llm_chat_failed") from e

    # 7. Late tool parse.
    try:
        late_tool = parse_tool_intent(answer)
    except Exception as e:  # noqa: BLE001
        log.warning("tool_late_parse_failed", error=str(e))
        late_tool = None
    if late_tool is not None and not tool_calls_made:
        with STAGE_LATENCY.labels(stage="tool").time(), _maybe_span("chat.tool_late"):
            try:
                result = dispatch_tool(late_tool)
            except Exception as e:  # noqa: BLE001
                log.warning("tool_late_dispatch_failed", tool=late_tool.name, error=str(e))
                result = {"error": str(e)}
        tool_calls_made.append({"tool": late_tool.name, "args": late_tool.args, "result": result})
        answer = _format_tool_summary(tool_calls_made)

    # 8. Persist assistant turn.
    with _maybe_span("chat.memory_append_assistant"):
        await memory.append(
            Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                metadata={"gate": gate.signals, "tool_calls": tool_calls_made},
            )
        )

    return ChatResult(
        answer=answer,
        sources=sources,
        tool_calls=tool_calls_made,
        evidence={
            "gate": gate.signals,
            "gate_decision": gate.decision,
            "model": resp.model,
            "fallback_used": resp.fallback_used,
        },
        injection_risk=verdict.score,
        fallback_used=False,
    )


def _format_tool_summary(calls: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for c in calls:
        lines.append(f"**{c['tool']}**:")
        v = c["result"]
        if isinstance(v, list):
            for row in v[:5]:
                lines.append(f"- {row}")
        elif isinstance(v, dict):
            for k, val in v.items():
                lines.append(f"- {k}: {val}")
        else:
            lines.append(f"- {v}")
    return "\n".join(lines)
