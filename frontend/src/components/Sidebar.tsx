import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearKbAll,
  clearKbSource,
  deleteSession,
  ingestFile,
  listKbSources,
  renameSession,
  type KbSourcesResponse,
} from "../api/client";

interface LocalChat {
  sid: string;
  lastActive: number;
  title: string;
  serverKnown: boolean;
  everUsed: boolean;
}

interface Props {
  activeSid: string | null;
  titles: Record<string, string>;
  sessions: LocalChat[];
  onTitlesChange: (next: Record<string, string>) => void;
  onNewChat: () => void;
  onSwitchChat: (sid: string | null) => void;
  onDeleteChat: (sid: string) => void;
  onSessionsTouched: () => void;
  onKbChanged: () => void;
  refreshTrigger: number;
  isOpen?: boolean;           // ← NEW
  onClose?: () => void;       // ← NEW
}

type Notice = { kind: "ok" | "warn" | "err" | "info"; text: string };

function shortName(source: string): string {
  const m = source.replace(/\\/g, "/").split("/").pop();
  return m ?? source;
}

export function Sidebar(props: Props) {
  const {
    activeSid,
    titles,
    sessions,
    onTitlesChange,
    onNewChat,
    onSwitchChat,
    onDeleteChat,
    onSessionsTouched,
    onKbChanged,
    refreshTrigger,
  } = props;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [kb, setKb] = useState<KbSourcesResponse | null>(null);
  const [kbBusy, setKbBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [clearAllArmed, setClearAllArmed] = useState(false);
  const [clearSourceArmed, setClearSourceArmed] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const refreshKb = useCallback(async () => {
    try {
      const k = await listKbSources();
      setKb(k);
    } catch {}
  }, []);

  useEffect(() => {
    void refreshKb();
    const t = window.setInterval(refreshKb, 15_000);
    return () => window.clearInterval(t);
  }, [refreshKb]);

  useEffect(() => {
    void refreshKb();
  }, [refreshTrigger, refreshKb]);

  useEffect(() => {
    const handler = () => void refreshKb();
    window.addEventListener("mini_ai:kb-changed", handler);
    return () => window.removeEventListener("mini_ai:kb-changed", handler);
  }, [refreshKb]);

  const handleUpload = async (file: File) => {
    setNotice({ kind: "info", text: `Uploading ${file.name}…` });
    setKbBusy(true);
    try {
      const r = await ingestFile(file);
      if (r.chunks === 0) {
        const why = r.error ?? r.fallback_reason ?? "no text could be extracted";
        if (r.fallback_reason === "chroma_restart_required") {
          setNotice({
            kind: "warn",
            text:
              "Vector index is unrecoverable in this process. Restart the API and upload again.",
          });
        } else {
          setNotice({ kind: "err", text: `Couldn't index ${file.name}: ${why}` });
        }
      } else {
        let msg = `Indexed ${r.chunks} chunks from ${r.source}`;
        if (r.backend && r.backend !== "docling" && r.fallback_reason) {
          msg += ` — using ${r.backend} (docling unavailable; OCR figures skipped)`;
        } else if (r.backend && r.backend !== "docling") {
          msg += ` — using ${r.backend}`;
        }
        setNotice({ kind: "ok", text: msg });
      }
      onKbChanged();
      await refreshKb();
      window.setTimeout(() => void refreshKb(), 1_500);
    } catch (e) {
      setNotice({ kind: "err", text: `Upload failed: ${String(e)}` });
    } finally {
      setKbBusy(false);
    }
  };

  const handleClearSource = async (source: string) => {
    setNotice(null);
    try {
      const r = await clearKbSource(source);
      setNotice({
        kind: "ok",
        text: `Cleared ${r.removed} chunk(s) from ${shortName(source)}.`,
      });
      onKbChanged();
      await refreshKb();
    } catch (e) {
      setNotice({ kind: "err", text: `Clear failed: ${String(e)}` });
    } finally {
      setClearSourceArmed(null);
    }
  };

  const handleClearAll = async () => {
    setNotice(null);
    try {
      const r = await clearKbAll();
      setNotice({ kind: "ok", text: `Cleared ${r.removed} chunk(s) from the KB.` });
      onKbChanged();
      await refreshKb();
    } catch (e) {
      setNotice({ kind: "err", text: `Clear failed: ${String(e)}` });
    } finally {
      setClearAllArmed(false);
    }
  };

  const handleDeleteSession = async (sid: string) => {
    try {
      await deleteSession(sid);
    } catch {}
    const remaining = sessions.filter((s) => s.sid !== sid);
    const next = { ...titles };
    delete next[sid];
    onTitlesChange(next);
    onDeleteChat(sid);
    onSessionsTouched();
    if (activeSid === sid) {
      onSwitchChat(remaining[0]?.sid ?? null);
    }
  };

  const handleRename = (sid: string) => {
    const v = renameValue.trim();
    if (!v) {
      setRenameTarget(null);
      setRenameValue("");
      return;
    }
    void renameSession(sid, v).catch(() => {});
    onTitlesChange({ ...titles, [sid]: v });
    setRenameTarget(null);
    setRenameValue("");
    onSessionsTouched();
  };

  const totalSources = kb?.total_sources ?? 0;
  const totalChunks = kb?.total_chunks ?? 0;

  return (
    <aside className="sidebar">
      <button className="new-chat-btn primary" onClick={onNewChat}>
        + New chat
      </button>

      <section className="sb-section">
        <h3>Knowledge Base</h3>
        <p className="caption">Upload a PDF, TXT, or MD file to index it for retrieval.</p>
        <button
          type="button"
          className="upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={kbBusy}
        >
          {kbBusy ? "Uploading…" : "Choose file to upload"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="upload-input-hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleUpload(f);
            e.target.value = "";
          }}
        />
        {notice && <div className={`notice ${notice.kind}`}>{notice.text}</div>}
      </section>

      <hr />

      <section className="sb-section">
        <div className="kb-header">
          <h3>Indexed documents</h3>
          <span className="kb-pill" title={`${totalSources} source(s), ${totalChunks} chunk(s) total`}>
            {totalSources} doc · {totalChunks} chunk{totalChunks === 1 ? "" : "s"}
          </span>
        </div>

        {kb === null ? (
          <p className="caption">Loading indexed documents…</p>
        ) : kb.sources && kb.sources.length > 0 ? (
          <>
            <div className="kb-list">
              {kb.sources.map((s) => {
                const short = shortName(s.source);
                const armed = clearSourceArmed === s.source;
                return (
                  <div className="kb-row" key={s.source}>
                    <div className="kb-row-main">
                      <div className="src" title={s.source}>{short}</div>
                      <div className="count">{s.chunks} chunk{s.chunks === 1 ? "" : "s"}</div>
                    </div>
                    {armed ? (
                      <div className="kb-row-confirm">
                        <button
                          className="confirm"
                          title="Confirm removal"
                          onClick={() => handleClearSource(s.source)}
                        >
                          Confirm
                        </button>
                        <button
                          className="cancel"
                          title="Cancel"
                          onClick={() => setClearSourceArmed(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        className="del"
                        title={`Remove ${s.chunks} chunk(s) from ${short}`}
                        aria-label={`Remove ${short}`}
                        onClick={() => setClearSourceArmed(s.source)}
                      >
                        ✕
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
            <details className="danger-zone">
              <summary>Danger zone</summary>
              <div className="danger-zone-body">
                <label className="danger-confirm">
                  <input
                    type="checkbox"
                    checked={clearAllArmed}
                    onChange={(e) => setClearAllArmed(e.target.checked)}
                  />
                  I understand this clears the entire knowledge base.
                </label>
                <button
                  className="danger"
                  disabled={!clearAllArmed}
                  onClick={handleClearAll}
                >
                  Clear all indexed chunks
                </button>
              </div>
            </details>
          </>
        ) : (
          <p className="caption">No documents indexed yet — upload one above.</p>
        )}
      </section>

      <hr />

      <section className="sb-section">
        <h3>Chats</h3>
        <p className="caption">{sessions.length} chat{sessions.length === 1 ? "" : "s"} remembered locally.</p>
        {sessions.length === 0 ? (
          <p className="caption">No chats yet — click + New chat to start one.</p>
        ) : (
          <div className="session-list">
            {sessions.map((s) => {
              const isActive = s.sid === activeSid;
              const isRenaming = renameTarget === s.sid;
              const label = s.title;
              return (
                <div className="session-row" key={s.sid}>
                  <button
                    className={`open${isActive ? " active" : ""}${s.serverKnown ? "" : " local-only"}`}
                    disabled={isActive || isRenaming}
                    onClick={() => onSwitchChat(s.sid)}
                    title={label}
                  >
                    <span className={`dot${isActive ? " on" : ""}`} aria-hidden="true" />
                    {isRenaming ? (
                      <input
                        className="rename-input"
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRename(s.sid);
                          if (e.key === "Escape") {
                            setRenameTarget(null);
                            setRenameValue("");
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <span className="label">{label}</span>
                    )}
                  </button>
                  {!isRenaming && (
                    <>
                      <button
                        className="rename"
                        title="Rename this chat"
                        onClick={() => {
                          setRenameTarget(s.sid);
                          setRenameValue(label);
                        }}
                      >
                        ✎
                      </button>
                      <button
                        className="del"
                        title="Delete this chat"
                        onClick={() => void handleDeleteSession(s.sid)}
                      >
                        ✕
                      </button>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </aside>
  );
}
