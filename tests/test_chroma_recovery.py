"""Tests for the Chroma auto-recovery helper.

These tests exercise the **move-aside** quarantine behaviour. The probe
itself is tested only for the "directory missing" path — the corrupt-HNSW
case would need a fabricated broken index, which is fragile to construct
in CI. We trust that subprocess.run + PersistentClient.heartbeat() does
the right thing; the *recovery* logic (rename, recreate empty dir, idempotent)
is what matters here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.vector_store.recovery import auto_recover_if_corrupt


def test_recovery_is_noop_when_dir_missing(tmp_path: Path):
    assert auto_recover_if_corrupt(tmp_path / "nonexistent") is False
    assert not (tmp_path / "nonexistent").exists()


def test_recovery_quarantines_corrupt_dir(tmp_path: Path):
    persist = tmp_path / ".chroma"
    persist.mkdir()
    # Plant a recognisable file so we can confirm the move carried it.
    (persist / "chromadb.sqlite3").write_bytes(b"corrupt-bytes-for-test")

    # `force=True` skips the (potentially healthy) probe so this test is
    # deterministic regardless of the local chromadb build.
    assert auto_recover_if_corrupt(persist, force=True) is True

    # Original dir is recreated empty.
    assert persist.exists()
    assert list(persist.iterdir()) == []

    # Quarantined copy exists with the .bak- prefix and the original file.
    backups = list(tmp_path.glob(".chroma.bak-*"))
    assert len(backups) == 1
    assert (backups[0] / "chromadb.sqlite3").read_bytes() == b"corrupt-bytes-for-test"


def test_recovery_is_idempotent(tmp_path: Path):
    persist = tmp_path / ".chroma"
    persist.mkdir()
    (persist / "blob").write_text("x")

    auto_recover_if_corrupt(persist, force=True)
    # Second call: the new empty dir is healthy, so nothing should happen.
    assert auto_recover_if_corrupt(persist) is False
    assert persist.exists()
    assert list(persist.iterdir()) == []
    # Exactly one backup — the recovery should not loop and stack
    # .bak-<stamp>/.bak-<stamp>/ backups on every call.
    assert len(list(tmp_path.glob(".chroma.bak-*"))) == 1


def test_recovery_preserves_evidence(tmp_path: Path):
    """The corrupt directory must be MOVED, never deleted, so an operator
    can later inspect or restore it."""
    persist = tmp_path / ".chroma"
    persist.mkdir()
    payload = b"forensic-bytes-do-not-lose-me"
    (persist / "data.parquet").write_bytes(payload)

    auto_recover_if_corrupt(persist, force=True)

    [backup] = list(tmp_path.glob(".chroma.bak-*"))
    assert (backup / "data.parquet").read_bytes() == payload


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="windows-only smoke")
def test_recovery_falls_back_to_copytree_when_move_fails(tmp_path: Path):
    """If shutil.move fails (e.g. a file is still mmap'd by another
    process), the recovery helper must fall back to copytree + rmtree so
    the quarantine still succeeds — leaving the next ingest able to
    rebuild the collection.

    We simulate the move failure by replacing shutil.move with a stub
    that raises OSError. The fallback path should copy the corrupt
    directory and then remove the original.
    """
    persist = tmp_path / ".chroma"
    persist.mkdir()
    (persist / "data_level0.bin").write_bytes(b"corrupt")

    import backend.vector_store.recovery as rec

    original_move = rec.shutil.move
    rec.shutil.move = lambda *a, **kw: (_ for _ in ()).throw(OSError("locked"))
    try:
        assert auto_recover_if_corrupt(persist, force=True) is True
    finally:
        rec.shutil.move = original_move

    # Original dir is recreated empty.
    assert persist.exists()
    assert list(persist.iterdir()) == []
    # And the corrupt files survived in the .bak-* copy.
    [backup] = list(tmp_path.glob(".chroma.bak-*"))
    assert (backup / "data_level0.bin").read_bytes() == b"corrupt"