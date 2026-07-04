"""Tool router — explicit JSON-intent dispatch.

We do NOT use LangChain or native OpenAI tool-calling; the LLM emits a
JSON object with {tool, args} and we dispatch here. The intent and args
are also validated before invocation.

Why this shape:
  - Works with any chat model (no dependency on tool-calling guarantees).
  - Easy to test in isolation.
  - Easy to observe — every dispatch produces a Prometheus increment and
    a structlog event.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from backend.errors import ToolError, ValidationError
from backend.observability.metrics import TOOL_CALLS, TOOL_LATENCY
from backend.tools.registry import order_status, product_search


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


_TOOL_SCHEMA: dict[str, dict[str, Any]] = {
    "order_status": {
        "parameters": {
            "type": "object",
            "required": ["order_id"],
            "properties": {"order_id": {"type": "string"}},
        }
    },
    "product_search": {
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
        }
    },
}


def tool_schema_json() -> str:
    """Render the tool schema as a JSON fragment for inclusion in a prompt.

    Output is the OpenAI-style ``{"tools": [{"name": ..., "parameters": ...}, ...]}``
    shape so the model reads it as a familiar tool catalog. Schema-driven,
    never hard-codes sample ids or product names — those live in
    ``data/orders.json`` and ``data/products.json``.
    """
    schema = {
        "tools": [
            {"name": name, "parameters": spec["parameters"]}
            for name, spec in _TOOL_SCHEMA.items()
        ]
    }
    return json.dumps(schema, indent=2)


# JSON fence extraction — looks for ```json ... ``` first, then { ... }.
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)

# Natural-language intent detection — runs BEFORE the LLM so the
# structured reply is deterministic and never depends on the model
# emitting correct JSON.
_ORDER_ID_RE = re.compile(r"\b([A-Z]{1,5}-?\d{2,6})\b", re.IGNORECASE)
# Query phrases that signal "look up a product in the catalog". Three groups:
#   1) lead-in verb + remaining noun phrase as the query
#      ("do you have X", "price of X", "where can I find X")
#   2) noun phrase + "in stock"/"available"  ("is the usb-c hub in stock")
#   3) generic catalog words anywhere in the sentence as a fallback
_PRODUCT_TRIGGERS = re.compile(
    r"(?:"
    r"(?:do\s+you\s+(?:have|carry|sell|stock)|"
    r"(?:price|cost|stock|availability)\s+(?:of|for)?|"
    r"(?:search|find|look\s*up)\s+(?:for\s+)?|"
    r"where\s+can\s+i\s+(?:find|get|buy)|"
    r"can\s+i\s+(?:get|buy|order))"
    r"\s+(?P<q1>.+)"
    r")|"
    r"(?:"
    r"is\s+(?:the\s+|a\s+)?(?P<q2>[\w\s'-]+?)\s+"
    r"(?:in\s+stock|in\s+store|available|in\s+inventory|on\s+hand)"
    r")|"
    r"(?:"
    r"\b(?P<kw>product|products|laptop|headphones?|keyboard|mouse|"
    r"monitor|wireless|bluetooth)\b"
    r")",
    re.IGNORECASE,
)
# Bare mention of "order" + id-shaped token (e.g. "what about my order ORD001?")
_ORDER_CONTEXT_RE = re.compile(
    r"\border\b[^.\n]*?\b([A-Z]{1,5}-?\d{2,6})\b", re.IGNORECASE
)


def detect_intent(text: str) -> ToolCall | None:
    """Heuristic natural-language intent detector.

    Returns a ToolCall when the user message clearly targets one of the
    registered tools, otherwise None. Runs after `parse_tool_intent`
    has had its chance (which handles JSON-intent directly).
    """
    if not text:
        return None

    # 1. Explicit order-id mention inside an order context.
    m = _ORDER_CONTEXT_RE.search(text)
    if m:
        return ToolCall(name="order_status", args={"order_id": m.group(1).upper()})

    # 2. Bare order-id token anywhere in the message (e.g. "ORD001?").
    m = _ORDER_ID_RE.search(text)
    if m:
        # Only treat as order if the surrounding text isn't clearly a
        # product query (the PRODUCT_RE check below would have caught
        # it anyway, so order takes precedence here).
        return ToolCall(name="order_status", args={"order_id": m.group(1).upper()})

    # 3. Product-search phrasing.
    m = _PRODUCT_TRIGGERS.search(text)
    if m:
        query = (
            m.group("q1")
            or m.group("q2")
            or m.group("kw")
            or ""
        ).strip().rstrip(".?!")
        if not query:
            return None
        # For keyword-only matches, the query is the matched word itself
        # ("laptop", "wireless"). That's fine — registry fuzzy-matches it.
        return ToolCall(name="product_search", args={"query": query, "top_k": 5})

    return None


def parse_tool_intent(text: str) -> ToolCall | None:
    """Extract a {tool, args} JSON intent from the LLM's response text.

    Returns None when the model did not request a tool. Raises
    ValidationError if a JSON object was emitted but it doesn't parse.
    """
    if not text:
        return None
    blob: str | None = None
    fence = _JSON_FENCE.search(text)
    if fence:
        blob = fence.group(1)
    else:
        obj = _JSON_OBJ.search(text)
        if obj:
            blob = obj.group(1)
    if blob is None:
        return None
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValidationError(f"tool_intent_unparseable: {e.msg}") from e

    name = payload.get("tool") or payload.get("name")
    args = payload.get("args") or payload.get("arguments") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValidationError("tool_intent_shape")
    if name not in _TOOL_SCHEMA:
        raise ValidationError(f"tool_unknown: {name}")
    schema = _TOOL_SCHEMA[name]
    for req in schema["required"]:
        if req not in args:
            raise ValidationError(f"tool_arg_missing: {req}")
    return ToolCall(name=name, args=args)


def dispatch(call: ToolCall) -> dict[str, Any]:
    """Run a tool call with metric + log side-effects."""
    with TOOL_LATENCY.labels(tool=call.name).time():
        try:
            if call.name == "order_status":
                result = order_status(call.args["order_id"])
            elif call.name == "product_search":
                result = product_search(
                    query=call.args["query"],
                    top_k=int(call.args.get("top_k", 5)),
                )
            else:
                raise ToolError(f"unhandled_tool: {call.name}")
        except ToolError:
            TOOL_CALLS.labels(tool=call.name, outcome="error").inc()
            raise
        except KeyError:
            TOOL_CALLS.labels(tool=call.name, outcome="not_found").inc()
            raise ToolError(f"not_found: {call.args.get('order_id', '?')}", tool=call.name)
        except Exception as e:  # noqa: BLE001
            TOOL_CALLS.labels(tool=call.name, outcome="error").inc()
            raise ToolError(str(e), tool=call.name) from e

    TOOL_CALLS.labels(tool=call.name, outcome="ok").inc()
    return result