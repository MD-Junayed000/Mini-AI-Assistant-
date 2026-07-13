import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatPane } from "./components/ChatPane";
import {
  bootstrapSession,
  listSessions,
  type SessionSummary,
} from "./api/client";

function newSid(): string {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

const ACTIVE_SID_KEY = "mini_ai.activeSid";
const TITLES_KEY = "mini_ai.titles";
const KNOWN_SIDS_KEY = "mini_ai.knownSids";

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

interface LocalChat {
  sid: string;
  lastActive: number;
  title: string;
  serverKnown: boolean;
  everUsed: boolean;
}

export default function App() {
  const [activeSid, setActiveSid] = useState<string | null>(() => {
    return localStorage.getItem(ACTIVE_SID_KEY) ?? newSid();
  });

  const [titles, setTitles] = useState<Record<string, string>>(() =>
    readJSON<Record<string, string>>(TITLES_KEY, {}),
  );

  const [knownSids, setKnownSids] = useState<string[]>(() =>
    readJSON<string[]>(KNOWN_SIDS_KEY, []),
  );

  const [serverSessions, setServerSessions] = useState<SessionSummary[]>([]);

  const [touchTick, setTouchTick] = useState(0);

  // Mobile sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshTimer = useRef<number | null>(null);

  useEffect(() => {
    if (activeSid) {
      localStorage.setItem(ACTIVE_SID_KEY, activeSid);
    }
  }, [activeSid]);

  useEffect(() => {
    localStorage.setItem(TITLES_KEY, JSON.stringify(titles));
  }, [titles]);

  useEffect(() => {
    localStorage.setItem(KNOWN_SIDS_KEY, JSON.stringify(knownSids));
  }, [knownSids]);

  useEffect(() => {
    if (!activeSid) return;

    setKnownSids((prev) =>
      prev.includes(activeSid) ? prev : [activeSid, ...prev],
    );
  }, [activeSid]);

  const mergedSessions = useMemo<LocalChat[]>(() => {
    const byId = new Map<string, LocalChat>();

    for (const s of serverSessions) {
      byId.set(s.session_id, {
        sid: s.session_id,
        lastActive: (s.last_ts ?? 0) * 1000,
        title:
          titles[s.session_id] ??
          s.title ??
          `Session ${s.session_id.slice(0, 8)}`,
        serverKnown: true,
        everUsed: true,
      });
    }

    for (const sid of knownSids) {
      const existing = byId.get(sid);

      if (existing) {
        if (titles[sid]) {
          existing.title = titles[sid];
        }
      } else {
        byId.set(sid, {
          sid,
          lastActive: 0,
          title: titles[sid] ?? `Session ${sid.slice(0, 8)}`,
          serverKnown: false,
          everUsed: false,
        });
      }
    }

    const order = new Map<string, number>();
    knownSids.forEach((sid, index) => order.set(sid, index));

    return Array.from(byId.values()).sort((a, b) => {
      if (b.lastActive !== a.lastActive) {
        return b.lastActive - a.lastActive;
      }

      return (order.get(a.sid) ?? 999) - (order.get(b.sid) ?? 999);
    });
  }, [serverSessions, knownSids, titles, touchTick]);

  const refreshSessions = useCallback(async () => {
    try {
      const result = await listSessions();
      setServerSessions(result.sessions ?? []);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    void refreshSessions();

    refreshTimer.current = window.setInterval(refreshSessions, 15000);

    return () => {
      if (refreshTimer.current !== null) {
        window.clearInterval(refreshTimer.current);
      }
    };
  }, [refreshSessions]);

  const handleNewChat = useCallback(() => {
    const previous = activeSid;
    const sid = newSid();

    if (previous) {
      setKnownSids((sids) =>
        sids.includes(previous) ? sids : [previous, ...sids],
      );
    }

    setActiveSid(sid);
    setSidebarOpen(false);
    setTouchTick((n) => n + 1);

    void bootstrapSession(sid)
      .then(() => {
        void refreshSessions();
      })
      .catch(() => {});
  }, [activeSid, refreshSessions]);

  const handleSwitchChat = useCallback((sid: string | null) => {
    if (!sid) {
      setActiveSid(null);
      setSidebarOpen(false);
      return;
    }

    setActiveSid((current) => {
      if (current && current !== sid) {
        setKnownSids((sids) =>
          sids.includes(current) ? sids : [current, ...sids],
        );
      }

      return sid;
    });

    setSidebarOpen(false);
  }, []);

  const handleTitlesChange = useCallback(
    (next: Record<string, string>) => {
      setTitles(next);
    },
    [],
  );

  const handleDeleteChat = useCallback((sid: string) => {
    setKnownSids((sids) => sids.filter((s) => s !== sid));

    setActiveSid((current) => (current === sid ? null : current));

    setTouchTick((n) => n + 1);
  }, []);

  const handleSessionsTouched = useCallback(() => {
    setTouchTick((n) => n + 1);
    void refreshSessions();
  }, [refreshSessions]);

  const handleKbChanged = useCallback(() => {
    window.dispatchEvent(new CustomEvent("mini_ai:kb-changed"));
  }, []);

  const sidebarTitles = useMemo<Record<string, string>>(() => {
    const out: Record<string, string> = {};

    for (const chat of mergedSessions) {
      out[chat.sid] = chat.title;
    }

    return out;
  }, [mergedSessions]);

  return (
    <div className="app">
      {sidebarOpen && (
          <div
              className="sidebar-overlay"
              onClick={() => setSidebarOpen(false)}
          />
      )}
      <Sidebar
        activeSid={activeSid}
        titles={sidebarTitles}
        sessions={mergedSessions}
        onTitlesChange={handleTitlesChange}
        onNewChat={handleNewChat}
        onSwitchChat={handleSwitchChat}
        onDeleteChat={handleDeleteChat}
        onSessionsTouched={handleSessionsTouched}
        onKbChanged={handleKbChanged}
        refreshTrigger={touchTick}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <ChatPane
        sessionId={activeSid}
        onSessionsTouched={handleSessionsTouched}
        onMenuClick={() => setSidebarOpen(true)}
      />
    </div>
  );
}
