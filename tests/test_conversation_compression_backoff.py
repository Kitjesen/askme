"""Tests for ConversationManager compression back-off mechanism."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock


def _make_conv(tmp_path, monkeypatch, *, max_history=100, threshold=20, keep_recent=6):
    """Create a ConversationManager with lower thresholds suitable for testing."""
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
    monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", threshold)
    monkeypatch.setattr("askme.llm.conversation.KEEP_RECENT", keep_recent)

    from askme.llm.conversation import ConversationManager

    return ConversationManager()


def _fill_history(conv, n_pairs: int) -> None:
    """Add n_pairs user+assistant messages to exceed compress threshold."""
    for i in range(n_pairs):
        conv.history.append({"role": "user", "content": f"消息 {i}"})
        conv.history.append({"role": "assistant", "content": f"回复 {i}"})


async def test_compress_backoff_after_failure(tmp_path, monkeypatch):
    """压缩失败後、_compress_backoff_until が未来の時刻に設定される。"""
    conv = _make_conv(tmp_path, monkeypatch)
    _fill_history(conv, 15)  # 30 msgs > threshold=20

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = Exception("LLM error")

    t_before = time.monotonic()
    await conv.maybe_compress(mock_llm)
    t_after = time.monotonic()

    # Backoff must be set to a future time after the call
    assert conv._compress_backoff_until > t_before
    # 60-second backoff window from source: monotonic() + 60.0
    assert conv._compress_backoff_until <= t_after + 61.0


async def test_compress_skips_during_backoff(tmp_path, monkeypatch):
    """圧縮失敗後すぐ再呼び出しすると LLM が呼ばれない。"""
    conv = _make_conv(tmp_path, monkeypatch)
    _fill_history(conv, 15)

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = Exception("LLM error")

    # First call — fails, sets backoff
    await conv.maybe_compress(mock_llm)
    first_call_count = mock_llm.chat.call_count
    assert first_call_count == 1  # was called once and failed

    # Reset side_effect so it would succeed if called
    mock_llm.chat.side_effect = None
    mock_llm.chat.return_value = "圧縮サマリー"

    # Second call — should be skipped due to backoff
    await conv.maybe_compress(mock_llm)
    assert mock_llm.chat.call_count == 1  # still 1, not called again


async def test_compress_retries_after_backoff_expires(tmp_path, monkeypatch):
    """バックオフ期間が過ぎると次回呼び出し時に圧縮が再試行される。"""
    conv = _make_conv(tmp_path, monkeypatch)
    _fill_history(conv, 15)

    # Manually set backoff to a past time (already expired)
    conv._compress_backoff_until = time.monotonic() - 1.0

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "圧縮されたサマリー"

    await conv.maybe_compress(mock_llm)

    # LLM should have been called — compression was attempted
    mock_llm.chat.assert_called_once()
