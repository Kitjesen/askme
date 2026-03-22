"""Tests for bug fixes in EpisodicMemory._restore_active_buffer()
and _write_category_knowledge().

Bugs covered:
  1. _restore_active_buffer(): corrupt JSON lines are skipped; valid lines kept
  2. _restore_active_buffer(): episodes older than retention window filtered out
  3. _write_category_knowledge(): OSError is caught, not propagated to caller
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("askme.memory.episodic_memory.project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {
            "app": {"data_dir": str(tmp_path / "data")},
            "memory": {"episodic": {"reflect_min_events": 5}},
        },
    )
    from askme.memory.episodic_memory import EpisodicMemory
    return EpisodicMemory()


def _episode_line(desc: str, ts: float, importance: float = 0.3) -> str:
    """Minimal valid JSONL line matching Episode.to_dict() format."""
    return json.dumps({
        "ts": ts,
        "time": "12:00:00",
        "type": "action",
        "desc": desc,
        "ctx": {},
        "importance": importance,
        "stability": 3600.0,
        "access_count": 0,
        "last_accessed": ts,
    }, ensure_ascii=False)


def _active_file(tmp_path: Path) -> Path:
    return tmp_path / "data" / "memory" / "episodes" / "_active.jsonl"


# ── Bug 1: corrupt lines skipped, valid lines kept ────────────────────────────


def test_restore_skips_corrupt_lines_but_keeps_valid(tmp_path, monkeypatch):
    """Single corrupt line must not discard the entire buffer."""
    now = time.time()
    active = _active_file(tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join([
            _episode_line("first valid", now),
            "{CORRUPT JSON <<<",
            _episode_line("second valid", now),
        ]) + "\n",
        encoding="utf-8",
    )

    mem = _make_memory(tmp_path, monkeypatch)

    assert mem.buffer_size == 2, (
        f"Expected 2 valid episodes, got {mem.buffer_size}. "
        "Corrupt line must not clear the buffer."
    )
    descs = {ep.description for ep in mem.get_recent(10)}
    assert "first valid" in descs
    assert "second valid" in descs


def test_restore_all_corrupt_gives_empty_buffer(tmp_path, monkeypatch):
    """All-corrupt journal results in empty buffer (no crash)."""
    active = _active_file(tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text("{{invalid1\n{{invalid2\n", encoding="utf-8")

    mem = _make_memory(tmp_path, monkeypatch)

    assert mem.buffer_size == 0


# ── Bug 2: expired episodes filtered on restore ───────────────────────────────


def test_restore_filters_expired_episodes(tmp_path, monkeypatch):
    """Episodes older than retention window must not be loaded on restart."""
    now = time.time()
    active = _active_file(tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join([
            _episode_line("stale 25h ago", now - 25 * 3600),
            _episode_line("fresh 30min ago", now - 30 * 60),
        ]) + "\n",
        encoding="utf-8",
    )

    mem = _make_memory(tmp_path, monkeypatch)

    assert mem.buffer_size == 1, (
        f"Expected 1 fresh episode, got {mem.buffer_size}."
    )
    assert mem.get_recent(5)[0].description == "fresh 30min ago"


def test_restore_keeps_all_fresh_episodes(tmp_path, monkeypatch):
    """All episodes within retention window are fully restored."""
    now = time.time()
    active = _active_file(tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    n = 4
    active.write_text(
        "\n".join(_episode_line(f"ep{i}", now - i * 60) for i in range(n)) + "\n",
        encoding="utf-8",
    )

    mem = _make_memory(tmp_path, monkeypatch)

    assert mem.buffer_size == n


def test_restore_empty_journal_is_noop(tmp_path, monkeypatch):
    """Empty _active.jsonl must not crash and leaves buffer at zero."""
    active = _active_file(tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text("", encoding="utf-8")

    mem = _make_memory(tmp_path, monkeypatch)

    assert mem.buffer_size == 0


# ── Bug 3: _write_category_knowledge OSError does not propagate ───────────────


def test_write_category_knowledge_missing_dir_does_not_raise(tmp_path, monkeypatch):
    """FileNotFoundError (OSError subclass) must be caught silently."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem._knowledge_dir = tmp_path / "nonexistent" / "deeply" / "nested"

    # Must not raise
    mem._write_category_knowledge("environment", ["走廊有门"], [])


def test_write_category_knowledge_permission_error_does_not_raise(tmp_path, monkeypatch):
    """PermissionError (OSError subclass) must be caught silently."""
    mem = _make_memory(tmp_path, monkeypatch)

    with patch.object(Path, "write_text", side_effect=PermissionError("read-only")):
        mem._write_category_knowledge("entities", ["主人叫森哥"], [])


def test_update_knowledge_continues_after_write_error(tmp_path, monkeypatch):
    """_update_knowledge() must not raise when _write_category_knowledge fails."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem._knowledge_dir = tmp_path / "nope"  # does not exist

    # Must not raise even when all category writes fail with FileNotFoundError
    mem._update_knowledge({
        "new_facts": [
            {"fact": "走廊有摄像头", "category": "environment"},
            {"fact": "主人叫森哥", "category": "entities"},
        ],
        "patterns": [],
        "updates": [],
    })
