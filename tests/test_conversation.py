"""Tests for ConversationManager — sliding window compression."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def _make_conv(tmp_path, monkeypatch, max_history=40):
    """Create a ConversationManager with paths in tmp_path."""
    monkeypatch.setattr(
        "askme.llm.conversation.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.llm.conversation.get_config",
        lambda: {"conversation": {
            "history_file": str(tmp_path / "data" / "conv.json"),
            "max_history": max_history,
        }},
    )

    from askme.llm.conversation import ConversationManager

    return ConversationManager()


def test_basic_add_and_get(tmp_path, monkeypatch):
    """Basic add/get messages works."""
    conv = _make_conv(tmp_path, monkeypatch)

    conv.add_user_message("hello")
    conv.add_assistant_message("hi")

    msgs = conv.get_messages("system prompt")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_clear(tmp_path, monkeypatch):
    """clear() removes all history."""
    conv = _make_conv(tmp_path, monkeypatch)
    conv.add_user_message("hello")
    conv.clear()
    msgs = conv.get_messages("sys")
    assert len(msgs) == 1  # only system prompt


async def test_maybe_compress_below_threshold(tmp_path, monkeypatch):
    """No compression happens when history is short."""
    conv = _make_conv(tmp_path, monkeypatch)

    for i in range(10):
        conv.add_user_message(f"msg {i}")
        conv.add_assistant_message(f"reply {i}")

    mock_llm = AsyncMock()
    await conv.maybe_compress(mock_llm)

    # LLM should not have been called
    mock_llm.chat.assert_not_called()
    assert len(conv.history) == 20


async def test_maybe_compress_triggers(tmp_path, monkeypatch):
    """Compression triggers when history exceeds threshold."""
    monkeypatch.setattr(
        "askme.llm.conversation.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.llm.conversation.get_config",
        lambda: {"conversation": {
            "history_file": str(tmp_path / "data" / "conv.json"),
            "max_history": 100,
        }},
    )
    # Lower threshold for testing
    monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", 20)
    monkeypatch.setattr("askme.llm.conversation.KEEP_RECENT", 6)

    from askme.llm.conversation import ConversationManager, SUMMARY_TAG

    conv = ConversationManager()

    # Add 30 messages (15 user + 15 assistant)
    for i in range(15):
        conv.add_user_message(f"用户消息 {i}")
        conv.add_assistant_message(f"助手回复 {i}")

    assert len(conv.history) == 30

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "用户讨论了多个话题，包括天气和时间。"

    await conv.maybe_compress(mock_llm)

    # Should have compressed: 1 summary + KEEP_RECENT recent messages
    assert len(conv.history) == 7  # 1 summary + 6 recent
    assert conv.history[0]["content"].startswith(SUMMARY_TAG)
    # Recent messages should be preserved
    assert "用户消息 14" in conv.history[-2]["content"]
    assert "助手回复 14" in conv.history[-1]["content"]


async def test_maybe_compress_preserves_existing_summary(tmp_path, monkeypatch):
    """Second compression includes the previous summary in the prompt."""
    monkeypatch.setattr(
        "askme.llm.conversation.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.llm.conversation.get_config",
        lambda: {"conversation": {
            "history_file": str(tmp_path / "data" / "conv.json"),
            "max_history": 100,
        }},
    )
    monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", 10)
    monkeypatch.setattr("askme.llm.conversation.KEEP_RECENT", 4)

    from askme.llm.conversation import ConversationManager, SUMMARY_TAG

    conv = ConversationManager()

    # Pre-set history with an existing summary
    conv.history = [
        {"role": "assistant", "content": f"{SUMMARY_TAG} 之前讨论了天气"},
    ]
    for i in range(8):
        conv.history.append({"role": "user", "content": f"msg {i}"})
        conv.history.append({"role": "assistant", "content": f"reply {i}"})

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "之前讨论了天气，后来又讨论了编程。"

    await conv.maybe_compress(mock_llm)

    # The existing summary should have been included in the prompt
    call_args = mock_llm.chat.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "之前讨论了天气" in user_content


async def test_maybe_compress_failure_safe(tmp_path, monkeypatch):
    """Compression failure doesn't corrupt history."""
    monkeypatch.setattr(
        "askme.llm.conversation.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.llm.conversation.get_config",
        lambda: {"conversation": {
            "history_file": str(tmp_path / "data" / "conv.json"),
            "max_history": 100,
        }},
    )
    monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", 10)

    from askme.llm.conversation import ConversationManager

    conv = ConversationManager()
    for i in range(10):
        conv.add_user_message(f"msg {i}")
        conv.add_assistant_message(f"reply {i}")

    original_len = len(conv.history)

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = Exception("API error")

    await conv.maybe_compress(mock_llm)

    # History should be unchanged on failure
    assert len(conv.history) == original_len
