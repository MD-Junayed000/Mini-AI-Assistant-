"""Mini AI Assistant — Streamlit UI.

Professional chat-first layout:
  - Sidebar: configuration, KB ingest, session reset, status indicators
  - Main:    message stream with role-based bubbles + collapsible sources

Run with:
    streamlit run ui/streamlit_app.py --server.port 8501
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import httpx
import streamlit as st

# ---- Page config -----------------------------------------------------------
st.set_page_config(
    page_title="Mini AI Assistant",
    page_icon=None,            # no decorative favicon emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hard-coded API endpoint — the user shouldn't need to edit this.
# Override by setting MINI_AI_API in the environment before `streamlit run`.
import os
API = os.environ.get("MINI_AI_API", "http://localhost:8000").rstrip("/")

# Role avatars. `st.chat_message` only accepts an emoji, an image URL, or
# `None` (the Streamlit default). Plain letters like "U" / "A" are treated
# as image paths and raise `StreamlitAPIException: Failed to load the
# provided avatar value as an image.`, so we let Streamlit pick its own
# default avatar per role.
_USER_AVATAR = None
_ASSISTANT_AVATAR = None


# ---- Session state helpers -------------------------------------------------
def _session_id() -> str:
    return st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])


def _messages() -> list[dict[str, Any]]:
    return st.session_state.setdefault("messages", [])


def start_new_chat() -> None:
    """Forget the previous conversation and roll a fresh session id."""
    old_sid = st.session_state.get("session_id")
    if old_sid:
        try:
            httpx.post(f"{API}/session/{old_sid}/reset", timeout=10).raise_for_status()
        except httpx.HTTPError:
            # Network blip is not fatal — the new sid will just not have
            # history on the server side either way.
            pass
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex[:12]


# ---- Health probe (cached) -------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def _health() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=4) as cx:
            r = cx.get(f"{API}/healthz")
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"overall": "down", "components": {}, "error": str(exc)}


# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    # Prominent "new chat" entry-point at the top.
    st.button(
        "+  New chat",
        use_container_width=True,
        type="primary",
        on_click=start_new_chat,
        help="Forget the current conversation and start over.",
    )

    st.divider()
    st.markdown("### Knowledge Base")
    uploaded = st.file_uploader(
        "Add a document (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        if st.button("Upload to knowledge base", use_container_width=True):
            with st.spinner("Indexing document..."):
                try:
                    with httpx.Client(timeout=180) as cx:
                        r = cx.post(
                            f"{API}/ingest",
                            files={"file": (uploaded.name, uploaded.getvalue())},
                        )
                        r.raise_for_status()
                        st.success(f"Indexed {r.json().get('chunks', 0)} chunks from {uploaded.name}")
                except httpx.HTTPError as exc:
                    st.error(str(exc))

    st.divider()
    st.markdown("### Session")
    sid = _session_id()
    st.caption(f"id: `{sid}`")

    st.divider()
    st.markdown("### Status")
    health = _health()
    overall = health.get("overall", "unknown")
    if overall == "up":
        st.markdown("**API:** :green[connected]")
    elif overall == "degraded":
        st.markdown("**API:** :orange[degraded]")
    else:
        st.markdown("**API:** :red[unreachable]")
    for name, state in (health.get("components") or {}).items():
        glyph = "ok" if state == "up" else "down"
        st.markdown(f"- `{name}`: {glyph}")
    st.caption(f"checked {datetime.now().strftime('%H:%M:%S')}")


# ---- Main pane -------------------------------------------------------------
st.markdown("## Mini AI Assistant")
st.caption(
    "Ask questions about orders, products, or the knowledge base. "
    "The assistant uses retrieval-augmented generation with structured tool calls."
)

for msg in _messages():
    role = msg.get("role", "assistant")
    avatar = _USER_AVATAR if role == "user" else _ASSISTANT_AVATAR
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.get("content", ""))
        sources = msg.get("sources") or []
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    sid_str = s.get("id", "?")
                    preview = (s.get("preview") or "")[:200]
                    st.markdown(f"- **{sid_str}** — {preview}")

prompt = st.chat_input("Ask anything about orders, products, or the knowledge base")
if prompt:
    sid = _session_id()
    messages = _messages()

    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        placeholder = st.empty()
        try:
            with httpx.Client(timeout=120) as cx:
                r = cx.post(
                    f"{API}/chat",
                    json={"session_id": sid, "message": prompt},
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as exc:
            try:
                data = exc.response.json()
            except Exception:  # noqa: BLE001
                data = {"error": str(exc), "code": "internal_error"}
            placeholder.error(data.get("friendly", "Something went wrong."))
            with st.expander("Details", expanded=False):
                st.json(data)
            messages.append(
                {"role": "assistant", "content": data.get("friendly", "Error.")}
            )
        except httpx.HTTPError as exc:
            placeholder.error(f"Network error: {exc}")
            messages.append({"role": "assistant", "content": "Network error."})
        else:
            answer = data.get("answer", "(no answer)")
            placeholder.markdown(answer)
            sources = data.get("sources") or []
            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for s in sources:
                        sid_str = s.get("id", "?")
                        preview = (s.get("preview") or "")[:200]
                        st.markdown(f"- **{sid_str}** — {preview}")
            messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

    st.session_state.messages = messages
    # NOTE: st.rerun() (Streamlit >= 1.27) replaced st.experimental_rerun() which
    # was removed in Streamlit >= 1.33. Do not call rerun here — appending to
    # session state already triggers a fresh run.