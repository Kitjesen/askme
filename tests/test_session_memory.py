"""Tests for the session memory (Layer 2) system."""

from __future__ import annotations


def test_save_and_load_summary(tmp_path, monkeypatch):
    """Session summaries are saved as .md files and loaded back."""
    monkeypatch.setattr(
        "askme.memory.session.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.session.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.session import SessionMemory

    sm = SessionMemory()
    sm.save_direct("用户询问了天气和时间。用户偏好简洁回答。")

    summaries = sm.get_recent_summaries()
    assert "用户询问了天气和时间" in summaries
    assert "历史会话摘要" in summaries


def test_empty_summaries(tmp_path, monkeypatch):
    """No session files -> empty string."""
    monkeypatch.setattr(
        "askme.memory.session.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.session.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.session import SessionMemory

    sm = SessionMemory()
    assert sm.get_recent_summaries() == ""


def test_max_sessions_limit(tmp_path, monkeypatch):
    """Only the most recent N sessions are included."""
    monkeypatch.setattr(
        "askme.memory.session.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.session.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.session import MAX_RECENT_SESSIONS, SessionMemory

    sm = SessionMemory()

    # Create more than MAX_RECENT_SESSIONS files
    sessions_dir = tmp_path / "data" / "sessions"
    for i in range(MAX_RECENT_SESSIONS + 3):
        filepath = sessions_dir / f"2026-03-{i + 1:02d}_120000.md"
        filepath.write_text(f"Session {i + 1}", encoding="utf-8")

    summaries = sm.get_recent_summaries()
    # Should only include the most recent ones
    assert f"Session {MAX_RECENT_SESSIONS + 3}" in summaries
    assert "Session 1" not in summaries


def test_format_messages():
    """Message formatting for summarization."""
    from askme.memory.session import SessionMemory

    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "现在几点"},
        {"role": "tool", "content": "ignored"},
    ]
    result = SessionMemory._format_messages(messages)
    assert "用户: 你好" in result
    assert "助手: 你好！" in result
    assert "ignored" not in result
