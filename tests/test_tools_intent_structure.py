"""Schema-driven prompt + natural-language intent detection + structured
tool-response formatter.

These tests pin the three pieces that make the assistant "industrial" for
the orders/products surface area:

  1. ``tool_schema_json()`` exposes the live schema, not a hand-curated
     sample (no hard-coded "ORD001" / "wireless mouse" leak).
  2. ``detect_intent(text)`` catches natural-language queries that mention
     an order id or product and converts them to a ``ToolCall`` BEFORE the
     LLM ever sees them.
  3. ``_format_structured_tool_response(call, result)`` returns the exact
     field order the product spec requires:
        order_status   -> "Order Status: <status>\\nEstimated Delivery Date: <eta>"
        product_search -> "Product Name | Price | Stock Availability" table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.tools import registry, router
from backend.tools.router import (
    ToolCall,
    detect_intent,
    parse_tool_intent,
    tool_schema_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_data_dir(tmp_path: Path, monkeypatch):
    """Stand up a tiny orders/products fixture and chdir into it so the
    registry's mtime-cached loaders see a predictable shape."""

    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "orders.json").write_text(
        json.dumps(
            [
                {
                    "order_id": "ORD001",
                    "customer": "Ada Lovelace",
                    "status": "shipped",
                    "estimated_delivery": "2025-08-12",
                    "items": [{"sku": "MOUSE", "qty": 1}],
                },
                {
                    "order_id": "ORD002",
                    "customer": "Grace Hopper",
                    "status": "processing",
                    "estimated_delivery": "2025-08-15",
                    "items": [{"sku": "KB", "qty": 1}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "products.json").write_text(
        json.dumps(
            [
                {
                    "sku": "MOUSE",
                    "name": "Wireless Mouse",
                    "category": "peripherals",
                    "price": 25.0,
                    "stock": 12,
                    "tags": ["wireless", "mouse"],
                },
                {
                    "sku": "HUB",
                    "name": "USB-C Hub",
                    "category": "accessories",
                    "price": 45.0,
                    "stock": 0,
                    "tags": ["usb", "hub"],
                },
                {
                    "sku": "KB",
                    "name": "Mechanical Keyboard",
                    "category": "peripherals",
                    "price": 120.0,
                    "stock": 7,
                    "tags": ["keyboard", "mechanical"],
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    registry.refresh_cache()
    yield tmp_path
    registry.refresh_cache()  # restore after the test


# ---------------------------------------------------------------------------
# 1. Schema-driven prompt
# ---------------------------------------------------------------------------

def test_tool_schema_json_lists_both_tools():
    """Live schema must describe both tools — never a hand-typed example."""
    schema_text = tool_schema_json()
    data = json.loads(schema_text)
    assert "tools" in data, "schema must have a top-level 'tools' key"
    names = {t["name"] for t in data["tools"]}
    assert {"order_status", "product_search"}.issubset(names), (
        f"expected both tools in schema, got {names}"
    )


def test_tool_schema_json_has_required_args():
    schema = json.loads(tool_schema_json())
    by_name = {t["name"]: t for t in schema["tools"]}
    assert "order_id" in by_name["order_status"]["parameters"]["properties"]
    assert "query" in by_name["product_search"]["parameters"]["properties"]


def test_prompts_module_does_not_hard_code_samples():
    """The system prompt must NOT contain the placeholder order ids
    ('ORD001') or product names ('wireless mouse') we removed. Catches a
    regression if someone copy-pastes a sample back into the prompt."""
    from backend.llm import prompts

    rendered = prompts.BASE_SYSTEM_PROMPT.lower()
    # "ORD001" is still acceptable as an *example* of an order id shape inside
    # an explanation, but never as a quoted JSON sample with hard-coded args.
    assert '{"tool": "order_status"' not in prompts.BASE_SYSTEM_PROMPT, (
        "prompt must not contain a literal {tool: order_status, args: {...}} sample"
    )
    assert '{"tool": "product_search"' not in prompts.BASE_SYSTEM_PROMPT, (
        "prompt must not contain a literal {tool: product_search, args: {...}} sample"
    )
    # The schema JSON gets interpolated, so it WILL mention these field
    # names — that's the whole point. But there must be no placeholder
    # "wireless mouse" example value either.
    assert "wireless mouse" not in rendered, (
        "prompt must not hard-code 'wireless mouse' as an example value"
    )


# ---------------------------------------------------------------------------
# 2. Natural-language intent detection
# ---------------------------------------------------------------------------

def test_detect_intent_order_id_bare_token():
    """A bare order id shaped like ORD001 / ORD001? must short-circuit."""
    call = detect_intent("ORD001?")
    assert call is not None
    assert call.name == "order_status"
    assert call.args["order_id"].upper() == "ORD001"


def test_detect_intent_order_in_context():
    """'Where is my order ORD002' is the canonical natural-language form."""
    call = detect_intent("where is my order ORD002?")
    assert call is not None
    assert call.name == "order_status"
    assert call.args["order_id"].upper() == "ORD002"


def test_detect_intent_product_do_you_have():
    call = detect_intent("Do you have a wireless mouse?")
    assert call is not None
    assert call.name == "product_search"
    assert "wireless" in call.args["query"].lower()
    assert "mouse" in call.args["query"].lower()


def test_detect_intent_product_price_of():
    call = detect_intent("price of mechanical keyboard")
    assert call is not None
    assert call.name == "product_search"
    assert call.args["query"].lower() == "mechanical keyboard"


def test_detect_intent_product_in_stock():
    call = detect_intent("is the usb-c hub in stock?")
    assert call is not None
    assert call.name == "product_search"
    assert "usb-c hub" in call.args["query"].lower()


def test_detect_intent_product_do_you_stock():
    call = detect_intent("do you stock leather laptop sleeves?")
    assert call is not None
    assert call.name == "product_search"
    assert "leather laptop sleeves" in call.args["query"].lower()


def test_detect_intent_unrelated_returns_none():
    """General chat must not falsely trigger a tool call."""
    for msg in [
        "hi there",
        "tell me a joke",
        "what's the weather like?",
        "thanks!",
        "good morning",
    ]:
        assert detect_intent(msg) is None, f"false-positive on: {msg!r}"


def test_detect_intent_priority_over_json_for_user_message():
    """Both detectors exist; detect_intent should win for natural language
    so the structured reply is deterministic and the model never has to
    invent a JSON payload just to ask 'ORD001?'."""
    msg = "ORD001?"
    call = detect_intent(msg)
    assert call is not None and call.name == "order_status"
    # Confirm parse_tool_intent doesn't ALSO match (it shouldn't — bare
    # ORD001 with no JSON object is exactly what detect_intent handles).
    assert parse_tool_intent(msg) is None


# ---------------------------------------------------------------------------
# 3. Structured tool-response formatter
# ---------------------------------------------------------------------------

def _format(call: ToolCall, result: Any) -> str:
    """Lazy import so the test doesn't fail collection if the module is
    restructured."""
    from backend.pipeline.chat import _format_structured_tool_response

    return _format_structured_tool_response(call, result)


def test_structured_order_response_matches_spec(fake_data_dir):
    """Order queries return EXACTLY: Order Status + Estimated Delivery Date.

    No other fields leak into the user-facing reply — the product id,
    customer name, item list, etc. are not part of the contract."""
    call = ToolCall(name="order_status", args={"order_id": "ORD001"})
    result = registry.order_status("ORD001")

    out = _format(call, result)
    lines = out.splitlines()
    assert len(lines) == 2, f"expected exactly 2 lines, got {lines}"
    # Case-tolerant: live orders.json uses "Shipped" with capital S.
    assert lines[0].lower() == "order status: shipped"
    assert lines[1].startswith("Estimated Delivery Date: ")
    # Critical: no leaked noise from the underlying record.
    assert "Ada" not in out, "customer name must not leak into the reply"
    assert "MOUSE" not in out, "internal sku must not leak into the reply"
    assert "items" not in out, "internal items list must not leak"


def test_structured_order_response_handles_missing_eta(fake_data_dir):
    """If the dataset omits estimated_delivery we still emit the spec'd
    two-line shape, not a stack trace."""
    call = ToolCall(name="order_status", args={"order_id": "ORD001"})
    out = _format(call, {"status": "processing", "estimated_delivery": None})
    assert out.startswith("Order Status: processing")
    assert "Estimated Delivery Date: unknown" in out


def test_structured_product_response_with_stock(fake_data_dir):
    """A product with stock > 0 shows 'In stock (N)' — never 'stock: 12'."""
    call = ToolCall(name="product_search", args={"query": "wireless mouse"})
    matches = registry.product_search("wireless mouse", top_k=5)
    out = _format(call, matches)
    lines = out.splitlines()
    assert lines[0] == "Product Name | Price | Stock Availability"
    assert any("Wireless Mouse" in ln for ln in lines), out
    assert any("$25.00" in ln for ln in lines), out
    assert any("In stock (12)" in ln for ln in lines), out


def test_structured_product_response_out_of_stock(fake_data_dir):
    """Stock == 0 must read 'Out of stock', not 'In stock (0)' (deceptive)."""
    call = ToolCall(name="product_search", args={"query": "usb-c hub"})
    matches = registry.product_search("usb-c hub", top_k=5)
    out = _format(call, matches)
    assert "USB-C Hub" in out
    assert "Out of stock" in out
    assert "In stock (0)" not in out


def test_structured_product_response_empty_list(fake_data_dir):
    call = ToolCall(name="product_search", args={"query": "xyzzy nothing"})
    out = _format(call, [])
    assert "No matching products found" in out


def test_structured_formatter_falls_back_for_unknown_tool():
    """Adding a new tool tomorrow should not crash — fall back to the
    generic summary."""
    call = ToolCall(name="future_tool", args={"x": 1})
    out = _format(call, {"y": 2})
    assert "future_tool" in out or "y" in out
