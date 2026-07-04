"""Mini AI Assistant — Streamlit UI.

Professional chat-first layout:
  - Sidebar: configuration, KB ingest, persistent session list, status indicators
  - Main:    message stream with role-based bubbles + collapsible sources

Run with:
    streamlit run ui/streamlit_app.py --server.port 8501
"""
from __future__ import annotations

import os
import time
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
API = os.environ.get("MINI_AI_API", "http://localhost:8000").rstrip("/")


# Single shared httpx client. We force `max_keepalive_connections=0` so
# every request opens a fresh TCP socket — this is the practical fix for
# WinError 10054 on Windows, where idle keep-alive sockets from the pool
# get reset by the OS between Streamlit reruns. The Connection: close
# request header alone doesn't help: it only tells the server to close
# after the response, it doesn't prevent httpx from picking up a dead
# socket from its pool for the *next* request.
_http_limits = httpx.Limits(
    max_keepalive_connections=0,
    max_connections=4,
    keepalive_expiry=1.0,
)
_http_transport = httpx.HTTPTransport(
    limits=_http_limits,
    retries=2,
)
HTTP = httpx.Client(
    transport=_http_transport,
    timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
)


def _post_with_retry(url: str, *, files: dict, headers: dict | None = None) -> httpx.Response:
    """POST a multipart upload with one retry on Windows connection drops.

    The Streamlit reruns happen dozens of times per minute; on Windows
    the underlying socket occasionally dies between reruns. Two attempts
    is plenty for an interactive upload button.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return HTTP.post(url, files=files, headers=headers)
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as exc:
            last_exc = exc
            continue
    # Re-raise the last connection-class error so the caller can render it.
    assert last_exc is not None
    raise last_exc

# Role avatars. `st.chat_message` only accepts an emoji, an image URL, or
# `None` (the Streamlit default). Plain letters like "U" / "A" are treated
# as image paths and raise `StreamlitAPIException: Failed to load the
# provided avatar value as an image.`, so we let Streamlit pick its own
# default avatar per role.
_USER_AVATAR = None
_ASSISTANT_AVATAR = None


# ---- Session state helpers -------------------------------------------------
def _new_sid() -> str:
    return uuid.uuid4().hex[:12]


def _messages_for(sid: str) -> list[dict[str, Any]]:
    """Per-session message store: every chat gets its own list.

    The previous version dropped messages into `st.session_state.messages` and
    lost them the moment the user clicked "New chat". Now each session has
    its own bucket, which we keep around for the lifetime of the page so
    switching back and forth keeps history intact.
    """
    store: dict[str, list[dict[str, Any]]] = st.session_state.setdefault(
        "chat_store", {}
    )
    return store.setdefault(sid, [])


def _ensure_titles() -> dict[str, str]:
    return st.session_state.setdefault("titles", {})


def _ensure_active_sid() -> str:
    sid = st.session_state.get("session_id") or _new_sid()
    st.session_state.session_id = sid
    return sid


def start_new_chat() -> None:
    """Create a fresh chat and make it the active one.

    Previous chats are NOT deleted: they remain in `chat_store` and in the
    sidebar's session list, so the user can go back to them.
    """
    sid = _new_sid()
    st.session_state.session_id = sid
    _messages_for(sid)  # create empty bucket so the list shows up immediately


def switch_chat(sid: str) -> None:
    """Make `sid` the active chat.

    If we don't already have messages for it in the local `chat_store`
    (typical after a page refresh, or in a fresh browser tab), fetch them
    from `GET /session/{sid}/messages` so the user sees their previous
    prompts and answers instead of a blank pane.
    """
    if not sid:
        return
    st.session_state.session_id = sid
    bucket = _messages_for(sid)
    if bucket:
        return
    try:
        r = HTTP.get(
            f"{API}/session/{sid}/messages",
            headers={"Connection": "close"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError:
        return  # server unreachable / session gone; nothing to hydrate
    server_msgs = data.get("messages") or []
    if not server_msgs:
        return
    hydrated: list[dict[str, Any]] = []
    for m in server_msgs:
        hydrated.append(
            {
                "role": m.get("role", "assistant"),
                "content": m.get("content", ""),
                "ts": m.get("ts", 0.0),
                "elapsed_s": m.get("elapsed_s"),
                "sources": m.get("sources") or [],
            }
        )
    st.session_state.chat_store[sid] = hydrated
    # Pick up the saved title from the first user prompt so the sidebar
    # doesn't show a generic "Chat ..." if the user renamed it earlier.
    titles = _ensure_titles()
    if sid not in titles:
        for m in hydrated:
            if m["role"] == "user" and m["content"]:
                titles[sid] = m["content"][:40]
                break


def delete_chat(sid: str) -> None:
    """Permanently delete a chat from server memory AND from the local cache.

    If the deleted chat was the active one, we fall back to the most recent
    remaining chat (or roll a brand new one if there are no others left).
    """
    try:
        HTTP.post(f"{API}/session/{sid}/delete", timeout=10).raise_for_status()
    except httpx.HTTPError:
        # Even if the server-side purge fails, still drop the local cache so
        # the UI stays consistent with what the user asked for.
        pass
    chat_store: dict[str, list[dict[str, Any]]] = st.session_state.get(
        "chat_store", {}
    )
    chat_store.pop(sid, None)
    titles = _ensure_titles()
    titles.pop(sid, None)

    if st.session_state.get("session_id") == sid:
        remaining = sorted(
            chat_store.keys(),
            key=lambda k: max(
                (m.get("ts", 0.0) for m in chat_store.get(k, [])),
                default=0.0,
            ),
            reverse=True,
        )
        st.session_state.session_id = remaining[0] if remaining else _new_sid()


def rename_chat(sid: str, new_title: str) -> None:
    """Set a friendly title for `sid`. Purely client-side — the server
    derives titles from the first user message and we treat the user's
    override as authoritative until they rename again."""
    new_title = new_title.strip()
    if not new_title:
        return
    try:
        HTTP.post(
            f"{API}/session/{sid}/rename",
            json={"title": new_title},
            timeout=10,
        ).raise_for_status()
    except httpx.HTTPError:
        pass
    _ensure_titles()[sid] = new_title


def _fetch_sessions() -> list[dict[str, Any]]:
    """Pull the server's session list (newest first). Falls back to local
    session ids if the server is unreachable so the sidebar still works."""
    try:
        r = HTTP.get(
            f"{API}/sessions",
            headers={"Connection": "close"},
            timeout=5,
        )
        r.raise_for_status()
        return r.json().get("sessions", []) or []
    except httpx.HTTPError:
        chat_store: dict[str, list[dict[str, Any]]] = st.session_state.get(
            "chat_store", {}
        )
        out: list[dict[str, Any]] = []
        for sid, msgs in chat_store.items():
            first_user = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                "",
            )
            out.append(
                {
                    "session_id": sid,
                    "title": first_user.strip()[:60] or f"session {sid[:8]}",
                    "turns": len(msgs),
                    "last_ts": max((m.get("ts", 0.0) for m in msgs), default=0.0),
                }
            )
        out.sort(key=lambda r: r["last_ts"], reverse=True)
        return out


def _active_label(sessions: list[dict[str, Any]], sid: str) -> str:
    titles = _ensure_titles()
    if sid in titles:
        return titles[sid]
    for s in sessions:
        if s["session_id"] == sid:
            return s["title"]
    return f"new chat ({sid[:8]})"


# ---- Health probe (cached) -------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def _health() -> dict[str, Any]:
    try:
        r = HTTP.get(
            f"{API}/healthz",
            headers={"Connection": "close"},
            timeout=4,
        )
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
        help="Start a new conversation. Previous chats stay available in the list below.",
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
                # Read the upload into a bytes buffer once, then post it on a
                # fresh TCP socket (see `_http_limits` above) with up to two
                # connection-level retries for the WinError 10054 cases that
                # hit Windows when Streamlit re-renders the sidebar many
                # times per minute.
                file_bytes = uploaded.getvalue()
                files = {
                    "file": (
                        uploaded.name,
                        file_bytes,
                        uploaded.type or "application/octet-stream",
                    )
                }
                try:
                    r = _post_with_retry(
                        f"{API}/ingest",
                        files=files,
                        headers={"Connection": "close"},
                    )
                    r.raise_for_status()
                    body = r.json()
                    chunks = body.get("chunks", 0)
                    backend = body.get("backend", "docling")
                    reason = body.get("fallback_reason")
                    detail = body.get("error")
                    if chunks == 0:
                        # Surface the failure cleanly: bad file, empty doc,
                        # or a backend that couldn't parse anything.
                        why = detail or reason or "no text could be extracted"
                        # chroma_restart_required: the in-process self-heal
                        # could not finish (this worker is still holding
                        # file handles). The operator needs to restart
                        # uvicorn or run `make recover-chroma`.
                        if reason == "chroma_restart_required":
                            st.warning(
                                f"⚠ Vector index is unrecoverable in "
                                f"this process. Restart the API server "
                                f"or run `make recover-chroma`, then "
                                f"click **Upload** again to index "
                                f"{uploaded.name}."
                            )
                        else:
                            st.error(
                                f"Couldn't index {uploaded.name}: {why}"
                            )
                    else:
                        msg = f"Indexed {chunks} chunks from {uploaded.name}"
                        if backend != "docling" and reason:
                            msg += (
                                f" — using **{backend}** (docling unavailable "
                                f"on this host; OCR-quality figures skipped)"
                            )
                        elif backend != "docling":
                            msg += f" — using **{backend}**"
                        st.success(msg)
                except httpx.HTTPError as exc:
                    st.error(str(exc))

    st.divider()
    st.markdown("### Indexed documents")
    st.caption(
        "Each row below is one uploaded file’s chunks. "
        "Use the trash icon to clear a single document, or \"Clear all\" "
        "to wipe the whole knowledge base. The original files in "
        "`data/uploads/` are NOT deleted."
    )

    @st.cache_data(ttl=5, show_spinner=False)
    def _fetch_kb_sources() -> dict[str, Any]:
        try:
            r = HTTP.get(
                f"{API}/admin/kb/sources",
                headers={"Connection": "close"},
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "sources": []}

    def _clear_kb_source(source: str) -> dict[str, Any]:
        try:
            r = HTTP.post(
                f"{API}/admin/kb/clear-source",
                json={"source": source},
                headers={"Connection": "close"},
                timeout=10,
            )
            r.raise_for_status()
            return {"ok": True, "body": r.json()}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc)}

    def _clear_kb_all() -> dict[str, Any]:
        try:
            r = HTTP.post(
                f"{API}/admin/kb/clear",
                headers={"Connection": "close"},
                timeout=10,
            )
            r.raise_for_status()
            return {"ok": True, "body": r.json()}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc)}

    kb = _fetch_kb_sources()
    if kb.get("error"):
        st.caption(f"KB status unavailable: {kb['error']}")
    sources = kb.get("sources") or []
    total_chunks = kb.get("total_chunks", 0)
    if not sources:
        st.caption("No documents indexed yet — upload a file above.")
    else:
        st.caption(
            f"{len(sources)} document(s) · {total_chunks} chunk(s) total"
        )
        # Render each indexed source as one row: name + chunk count + clear.
        # The full `source` string is opaque (it's the stored path), so we
        # show a short basename prefix and keep the full string in `key=`.
        for s in sources:
            full = s["source"]
            chunks = s["chunks"]
            # Display only the basename to keep the row narrow. The full
            # path is preserved internally so the API call can match exactly.
            short = full.replace("\\", "/").rsplit("/", 1)[-1]
            cols = st.columns([0.78, 0.22], gap="small")
            with cols[0]:
                st.markdown(
                    f"**{short}** &nbsp;<small>{chunks} chunk(s)</small>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button(
                    "✕",
                    key=f"clear_src::{full}",
                    help=f"Clear {chunks} chunk(s) from {full}",
                    use_container_width=True,
                ):
                    res = _clear_kb_source(full)
                    if res["ok"]:
                        body = res["body"]
                        st.success(
                            f"Cleared {body.get('removed', 0)} chunk(s) from {short}."
                        )
                        _fetch_kb_sources.clear()
                        st.rerun()
                    else:
                        st.error(f"Clear failed: {res['error']}")

        # Clear-all sits below the per-doc rows, visually separated so a
        # misclick on the trash icon can't take down the whole KB.
        with st.expander("Danger zone", expanded=False):
            st.caption(
                "Removes every chunk from the vector store and rebuilds "
                "BM25 from the now-empty index. Your original files in "
                "`data/uploads/` are kept — re-upload to re-index."
            )
            confirm = st.checkbox(
                "I understand this clears the entire knowledge base.",
                key="kb_clear_all_confirm",
            )
            if st.button(
                "Clear all indexed chunks",
                type="primary",
                disabled=not confirm,
                use_container_width=True,
                key="kb_clear_all_btn",
            ):
                res = _clear_kb_all()
                if res["ok"]:
                    body = res["body"]
                    st.success(
                        f"Cleared {body.get('removed', 0)} chunk(s) from the KB."
                    )
                    _fetch_kb_sources.clear()
                    st.rerun()
                else:
                    st.error(f"Clear failed: {res['error']}")

    st.divider()
    st.markdown("### Chats")
    # Make sure there's always an active session.
    active_sid = _ensure_active_sid()
    _messages_for(active_sid)

    # Refresh button so the user can pull a fresh list without losing context.
    if st.button("↻ Refresh list", use_container_width=True, key="refresh_sessions"):
        st.rerun()

    sessions = _fetch_sessions()
    titles = _ensure_titles()

    # Merge server-known sessions with any local-only ones (helps when the
    # server is unreachable).
    known = {s["session_id"]: s for s in sessions}
    for sid in list(st.session_state.get("chat_store", {}).keys()):
        if sid not in known:
            known[sid] = {
                "session_id": sid,
                "title": titles.get(sid, f"session {sid[:8]}"),
                "turns": len(st.session_state["chat_store"][sid]),
                "last_ts": 0.0,
            }
    sessions = sorted(
        known.values(),
        key=lambda s: s.get("last_ts") or 0.0,
        reverse=True,
    )

    if not sessions:
        st.caption("No chats yet — click **+ New chat** to start one.")

    # Each chat is rendered as a single clickable row: the title is the
    # switch button, and the trash icon on the right deletes it. This avoids
    # the awkward "title + Open button + delete button" three-line layout
    # and keeps switch/delete to one click each.
    for s in sessions:
        sid = s["session_id"]
        is_active = sid == active_sid
        label_default = _active_label(sessions, sid)
        cols = st.columns([0.86, 0.14], gap="small")
        with cols[0]:
            # The button label is the chat title; a leading bullet marks
            # the active chat. A second click on the active row is a no-op
            # (button is disabled) so users can't accidentally clear state.
            btn_label = (
                f"●  {label_default}" if is_active else f"○  {label_default}"
            )
            if st.button(
                btn_label,
                key=f"open_{sid}",
                use_container_width=True,
                disabled=is_active,
                type="primary" if is_active else "secondary",
                help=(
                    "Currently open"
                    if is_active
                    else "Open this chat (your other chats are saved)"
                ),
            ):
                switch_chat(sid)
                st.rerun()
        with cols[1]:
            if st.button(
                "✕",
                key=f"del_{sid}",
                help="Delete this chat permanently",
                use_container_width=True,
            ):
                delete_chat(sid)
                st.rerun()

    # Rename the active chat.
    if active_sid:
        with st.expander("Rename current chat", expanded=False):
            current_title = titles.get(
                active_sid,
                next(
                    (
                        s["title"]
                        for s in sessions
                        if s["session_id"] == active_sid
                    ),
                    "",
                ),
            )
            new_title = st.text_input(
                "New title",
                value=current_title,
                key=f"rename_in_{active_sid}",
                label_visibility="collapsed",
            )
            if st.button("Save title", key=f"save_rename_{active_sid}"):
                rename_chat(active_sid, new_title)
                st.rerun()

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

# IMPORTANT: read the active session id AFTER the sidebar (which may have
# re-routed via switch_chat).
active_sid = st.session_state.session_id
# If we just got here (page reload / new tab), `switch_chat` may not have
# fired; pull history from the server so the user sees their previous
# prompts instead of an empty pane.
if not st.session_state.get("chat_store", {}).get(active_sid):
    switch_chat(active_sid)
messages = _messages_for(active_sid)

for msg in messages:
    role = msg.get("role", "assistant")
    avatar = _USER_AVATAR if role == "user" else _ASSISTANT_AVATAR
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.get("content", ""))
        elapsed = msg.get("elapsed_s")
        if elapsed is not None:
            st.caption(f"answered in {elapsed:.1f} s")
        sources = msg.get("sources") or []
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    sid_str = s.get("id", "?")
                    preview = (s.get("preview") or "")[:200]
                    st.markdown(f"- **{sid_str}** — {preview}")

prompt = st.chat_input(
    "Ask anything about orders, products, or the knowledge base"
)
if prompt:
    ts_now = time.time()
    messages.append({"role": "user", "content": prompt, "ts": ts_now})

    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        placeholder = st.empty()
        # Surface progress with a small-font caption so the user knows the
        # request is in flight (and how long it's taking).
        t0 = time.perf_counter()
        status = placeholder.caption("🟡 requesting… 0.0 s")
        try:
            r = HTTP.post(
                f"{API}/chat",
                json={"session_id": active_sid, "message": prompt},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as exc:
            try:
                data = exc.response.json()
            except Exception:  # noqa: BLE001
                data = {"error": str(exc), "code": "internal_error"}
            elapsed = time.perf_counter() - t0
            status.markdown(
                f"<small>request failed after {elapsed:.1f} s</small>",
                unsafe_allow_html=True,
            )
            placeholder.error(data.get("friendly", "Something went wrong."))
            with st.expander("Details", expanded=False):
                st.json(data)
            messages.append(
                {
                    "role": "assistant",
                    "content": data.get("friendly", "Error."),
                    "ts": time.time(),
                    "elapsed_s": elapsed,
                }
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - t0
            status.markdown(
                f"<small>request failed after {elapsed:.1f} s</small>",
                unsafe_allow_html=True,
            )
            placeholder.error(f"Network error: {exc}")
            messages.append(
                {
                    "role": "assistant",
                    "content": "Network error.",
                    "ts": time.time(),
                    "elapsed_s": elapsed,
                }
            )
        else:
            elapsed = time.perf_counter() - t0
            # Live update one last time, then replace with the answer.
            status.markdown(
                f"<small>answered in {elapsed:.1f} s</small>",
                unsafe_allow_html=True,
            )
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
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "ts": time.time(),
                    "elapsed_s": elapsed,
                }
            )

    # Persist this conversation back into the named bucket for `active_sid`.
    st.session_state["chat_store"][active_sid] = messages
    # Adding a chat should make its timestamp fresh enough that the sidebar
    # re-sorts it to the top.
    st.rerun()
