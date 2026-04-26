"""Tests for CloudASR — config parsing, available flag, session state logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from askme.voice.cloud_asr import CloudASR

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_asr(**kwargs) -> CloudASR:
    """Create a CloudASR with injectable config."""
    cfg = {
        "enabled": True,
        "api_key": "test-key-123",
        "model": "paraformer-realtime-v2",
        "sample_rate": 16000,
        "language_hints": ["zh", "en"],
    }
    cfg.update(kwargs)
    return CloudASR(config=cfg)


# ── Init and config ───────────────────────────────────────────────────────────

class TestCloudASRInit:
    def test_enabled_with_key(self):
        asr = _make_asr()
        assert asr._enabled is True
        assert asr._api_key == "test-key-123"

    def test_enabled_without_key_disables(self):
        asr = _make_asr(api_key="")
        # No API key → disabled
        assert asr._enabled is False

    def test_disabled_config(self):
        asr = _make_asr(enabled=False)
        assert asr._enabled is False

    def test_custom_model(self):
        asr = _make_asr(model="paraformer-v1")
        assert asr._model == "paraformer-v1"

    def test_default_sample_rate(self):
        asr = _make_asr()
        assert asr._sample_rate == 16000

    def test_custom_sample_rate(self):
        asr = _make_asr(sample_rate=8000)
        assert asr._sample_rate == 8000

    def test_language_hints(self):
        asr = _make_asr(language_hints=["zh"])
        assert asr._language_hints == ["zh"]

    def test_empty_config_uses_defaults(self):
        asr = CloudASR(config={})
        assert asr._enabled is False  # default enabled=False
        assert asr._model == "paraformer-realtime-v2"
        assert asr._sample_rate == 16000

    def test_none_config_uses_defaults(self):
        asr = CloudASR(config=None)
        assert asr._enabled is False


# ── available property ────────────────────────────────────────────────────────

class TestAvailable:
    def test_available_when_enabled_and_key_set(self):
        asr = _make_asr()
        assert asr.available is True

    def test_not_available_when_disabled(self):
        asr = _make_asr(enabled=False)
        assert asr.available is False

    def test_not_available_without_key(self):
        asr = _make_asr(api_key="")
        assert asr.available is False


# ── start_session without websocket ──────────────────────────────────────────

class TestStartSession:
    def test_start_session_returns_false_when_not_available(self):
        asr = _make_asr(enabled=False)
        result = asr.start_session()
        assert result is False

    def test_start_session_returns_false_on_import_error(self):
        asr = _make_asr()
        with patch.dict("sys.modules", {"websocket": None}):
            result = asr.start_session()
        assert result is False

    def test_start_session_disables_on_import_error(self):
        asr = _make_asr()
        with patch.dict("sys.modules", {"websocket": None}):
            asr.start_session()
        assert asr._enabled is False

    def test_start_session_returns_false_on_connection_error(self):
        asr = _make_asr()
        mock_ws_module = MagicMock()
        mock_ws = MagicMock()
        mock_ws.connect.side_effect = OSError("connection refused")
        mock_ws_module.WebSocket.return_value = mock_ws

        with patch.dict("sys.modules", {"websocket": mock_ws_module}):
            result = asr.start_session()
        assert result is False


# ── feed ──────────────────────────────────────────────────────────────────────

class TestFeed:
    def test_feed_noop_when_no_session(self):
        asr = _make_asr()
        # Should not raise when no active session
        asr.feed(b"\x00\x01\x02")

    def test_feed_noop_when_ws_none(self):
        asr = _make_asr()
        asr._session_active = False
        asr._ws = None
        asr.feed(b"\x00\x01\x02")  # Should not raise

    def test_feed_sends_bytes_when_active(self):
        asr = _make_asr()
        mock_ws = MagicMock()
        asr._ws = mock_ws
        asr._session_active = True
        asr.feed(b"\x00\x01")
        mock_ws.send_binary.assert_called_once_with(b"\x00\x01")

    def test_feed_sets_error_on_exception(self):
        asr = _make_asr()
        mock_ws = MagicMock()
        mock_ws.send_binary.side_effect = OSError("ws closed")
        asr._ws = mock_ws
        asr._session_active = True
        asr.feed(b"\x00")
        assert asr._error is not None


# ── finish_session ────────────────────────────────────────────────────────────

class TestFinishSession:
    def test_returns_empty_when_no_active_session(self):
        asr = _make_asr()
        result = asr.finish_session()
        assert result == ""

    def test_returns_accumulated_text(self):
        asr = _make_asr()
        asr._result_text = "已积累的文字"
        asr._session_active = False
        result = asr.finish_session()
        assert result == "已积累的文字"

    def test_finish_session_cleans_up(self):
        asr = _make_asr()
        mock_ws = MagicMock()
        asr._ws = mock_ws
        asr._session_active = True
        asr._result_text = "结果"
        # result_ready is not set, so wait will timeout immediately with timeout=0
        asr._result_ready.set()  # pre-set so finish_session doesn't hang
        asr.finish_session(timeout=0.01)
        assert asr._session_active is False


# ── cancel_session ────────────────────────────────────────────────────────────

class TestCancelSession:
    def test_cancel_clears_session(self):
        asr = _make_asr()
        mock_ws = MagicMock()
        asr._ws = mock_ws
        asr._session_active = True
        asr.cancel_session()
        assert asr._session_active is False
        assert asr._ws is None

    def test_cancel_closes_ws(self):
        asr = _make_asr()
        mock_ws = MagicMock()
        asr._ws = mock_ws
        asr._session_active = True
        asr.cancel_session()
        mock_ws.close.assert_called_once()


# ── _receive_loop (unit test message parsing) ─────────────────────────────────

class TestReceiveLoop:
    def _run_receive_loop_with_messages(self, messages: list) -> CloudASR:
        """Run the receive loop synchronously with a sequence of mock messages."""
        import json as _json

        asr = _make_asr()
        asr._session_start = __import__("time").monotonic()
        asr._session_active = True

        responses = iter([_json.dumps(m) for m in messages] + [Exception("done")])

        mock_ws = MagicMock()
        def recv_side_effect():
            val = next(responses)
            if isinstance(val, Exception):
                raise val
            return val
        mock_ws.recv = recv_side_effect
        asr._ws = mock_ws

        asr._receive_loop()
        return asr

    def test_final_sentence_appended(self):
        messages = [
            {
                "header": {"event": "result-generated"},
                "payload": {"output": {"sentence": {"text": "你好", "sentence_end": True}}},
            },
            {"header": {"event": "task-finished"}},
        ]
        asr = self._run_receive_loop_with_messages(messages)
        assert "你好" in asr._result_text

    def test_task_finished_stops_loop(self):
        messages = [{"header": {"event": "task-finished"}}]
        asr = self._run_receive_loop_with_messages(messages)
        # result_ready should be set
        assert asr._result_ready.is_set()

    def test_task_failed_sets_error(self):
        messages = [
            {
                "header": {"event": "task-failed", "error_message": "auth failed"},
            }
        ]
        asr = self._run_receive_loop_with_messages(messages)
        assert asr._error is not None

    def test_interim_not_appended_to_result(self):
        messages = [
            {
                "header": {"event": "result-generated"},
                "payload": {"output": {"sentence": {"text": "interim", "sentence_end": False}}},
            },
            {"header": {"event": "task-finished"}},
        ]
        asr = self._run_receive_loop_with_messages(messages)
        # Interim should NOT be in _result_text
        assert "interim" not in asr._result_text

    def test_bytes_messages_skipped(self):
        """Bytes messages should be skipped without crashing."""
        asr = _make_asr()
        asr._session_start = __import__("time").monotonic()
        asr._session_active = True

        call_count = [0]
        def recv_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return b"\x00\x01"  # bytes — should skip
            raise Exception("done")

        mock_ws = MagicMock()
        mock_ws.recv = recv_side_effect
        asr._ws = mock_ws

        asr._receive_loop()  # Should not raise
        assert asr._result_ready.is_set()
