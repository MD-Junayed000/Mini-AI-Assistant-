"""Auto-recovery for a corrupted ChromaDB persistent directory.

When a Windows process is killed mid-upsert (e.g. by an uncatchable native
fault in torch/docling), Chroma's HNSW files can be left half-written.
The next read or write then segfaults inside chromadb's Rust code, which
Python cannot catch — so /ingest returns 200 with chunks=0 instead of
failing cleanly, and the operator has to manually move the directory aside
before the next request can succeed.

This module offers `auto_recover_if_corrupt(persist_dir)`:

  - probes the directory with a cheap `chromadb.PersistentClient.heartbeat()`
    call in an isolated subprocess (so a native crash can never reach the
    FastAPI worker);
  - if the probe fails, renames `persist_dir` to `persist_dir.bak-<stamp>`
    and recreates a fresh empty directory in its place.

The next ingestion request transparently builds the collection from
`data/` because `ChromaStore.__init__` always calls `get_or_create_collection`.

The quarantine is move-aside, never wipe: the corrupt files are kept under
the `.bak-<stamp>` suffix so the operator can inspect or restore them later.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from backend.observability.logging_config import get_logger

log = get_logger("chroma.recovery")

# Subprocess script: probe chromadb without poisoning our process.
_PROBE_SCRIPT = (
    "import json, sys\n"
    "try:\n"
    "    import chromadb\n"
    "    c = chromadb.PersistentClient(path=sys.argv[1])\n"
    "    c.heartbeat()\n"
    "    print(json.dumps({'ok': True}))\n"
    "except BaseException as e:\n"
    "    print(json.dumps({'ok': False, 'err': '%s: %s' % (type(e).__name__, e)}))\n"
)


def _probe_chroma_isolated(persist_dir: Path, timeout: float = 15.0) -> tuple[bool, str | None]:
    """Spawn a child interpreter to probe chromadb. Returns (healthy, reason)."""
    if not persist_dir.exists():
        # Nothing to probe — a missing dir is always "healthy" because
        # the next ingest will recreate the collection.
        return True, None
    # An empty directory is the expected post-recovery state; chromadb's
    # `heartbeat()` on it would raise "no such table" or similar. We
    # treat an empty dir as already-healthy so subsequent calls become
    # no-ops instead of creating a fresh backup every time.
    if not any(persist_dir.iterdir()):
        return True, None
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE_SCRIPT, str(persist_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "ANONYMIZED_TELEMETRY": "False", "CHROMA_TELEMETRY_DISABLED": "True"},
        )
    except subprocess.TimeoutExpired:
        return False, f"chroma_probe_timeout_{int(timeout)}s"
    except FileNotFoundError as exc:
        return False, f"chroma_probe_no_python: {exc}"

    stdout = (proc.stdout or "").strip()
    if stdout.startswith("{"):
        try:
            payload = json.loads(stdout.splitlines()[-1])
            if payload.get("ok"):
                return True, None
            return False, payload.get("err", "chroma_probe_failed")
        except json.JSONDecodeError:
            pass

    # Native crash — exit non-zero, no usable JSON. Treat as unhealthy.
    stderr_first = next(
        (ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()),
        f"chroma_probe_exitcode_{proc.returncode}",
    )
    return False, f"chroma_probe_native: {stderr_first[:160]}"


def auto_recover_if_corrupt(
    persist_dir: Path | str,
    *,
    timeout: float = 15.0,
    force: bool = False,
) -> bool:
    """Quarantine a corrupt Chroma directory.

    Returns True if a recovery action was taken (directory was moved aside).
    Returns False if the directory is healthy or already missing.

    The function is **idempotent and safe**:
      - Move-aside only, never delete.
      - The renamed directory keeps a `.bak-<UTC-stamp>` suffix so it can
        be inspected or restored by hand.
      - A fresh empty directory is created in its place so the next ingest
        call builds the collection from `data/` transparently.
      - Failures are logged but never raised; callers should fall back to
        the in-process error path.

    Set `force=True` to quarantine unconditionally (e.g. when an operator
    runs this from the recovery script after a known crash).
    """
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        # Missing dir is the expected post-recovery state — do nothing.
        return False

    healthy, reason = _probe_chroma_isolated(persist_dir, timeout=timeout)
    if healthy and not force:
        return False

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_backup = persist_dir.with_name(f"{persist_dir.name}.bak-{stamp}")
    backup = base_backup
    # Disambiguate if two recoveries happen within the same second — the
    # second one would otherwise fail with WinError 183 on the copytree
    # fallback path.
    suffix = 1
    while backup.exists():
        backup = base_backup.with_name(f"{base_backup.name}-{suffix}")
        suffix += 1
    log.warning(
        "chroma_auto_recovery_quarantine",
        reason=reason or "forced",
        source=str(persist_dir),
        backup=str(backup),
    )
    try:
        shutil.move(str(persist_dir), str(backup))
    except OSError as exc:
        # shutil.move on Windows is MoveFileExW; if the destination already
        # exists OR a file in the source is locked (mismatched WinError 32
        # from the test, 17/39 in production), we fall back to a copy +
        # rmtree so the quarantine still succeeds.
        log.warning("chroma_auto_recovery_move_failed_falling_back", error=str(exc))
        try:
            shutil.copytree(str(persist_dir), str(backup), dirs_exist_ok=False)
            shutil.rmtree(str(persist_dir), ignore_errors=True)
        except OSError as e2:
            log.error("chroma_auto_recovery_copytree_failed", error=str(e2))
            return False
    # Recreate the empty directory so ChromaStore can init cleanly.
    persist_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "chroma_auto_recovery_complete",
        backup=str(backup),
        next_step="next /ingest call rebuilds the collection from data/",
    )
    return True