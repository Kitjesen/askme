"""Tests for ASRManager: noise filtering, cloud/local fallback, punctuation, reset."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
from askme.voice.asr_manager import (
    _SINGLE_CHAR_COMMANDS,
    ASRManager,
    ASRPartial,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    cloud_available: bool = False,
    config: dict[str, object] | None = None,
) -> ASRManager:
    """Create an ASRManager with mocked backends."""
    with (
        patch("askme.voice.asr_manager.ASREngine") as mock_asr_cls,
        patch("askme.voice.asr_manager.CloudASR") as mock_cloud_cls,
        patch("askme.voice.asr_manager.PunctuationRestorer") as mock_punct_cls,
    ):
        # Local ASR mock
        mock_asr = MagicMock()
        mock_stream = MagicMock()
        mock_asr.create_stream.return_value = mock_stream
        mock_asr.is_ready.return_value = False
        mock_asr.is_endpoint.return_value = False
        mock_asr.get_result.return_value = ""
        mock_asr_cls.return_value = mock_asr

        # Cloud ASR mock
        mock_cloud = MagicMock()
        type(mock_cloud).available = PropertyMock(return_value=cloud_available)
        mock_cloud.start_session.return_value = cloud_available
        mock_cloud.finish_session.return_value = ""
        mock_cloud_cls.return_value = mock_cloud

        # Punctuation mock
        mock_punct = MagicMock()
        type(mock_punct).available = PropertyMock(return_value=True)
        mock_punct.restore.side_effect = lambda t: t + "."
        mock_punct_cls.return_value = mock_punct

        mgr = ASRManager(config or {})

    # Expose mocks for test assertions
    mgr._test_mocks = {  # type: ignore[attr-defined]
        "asr": mock_asr,
        "stream": mock_stream,
        "cloud": mock_cloud,
        "punct": mock_punct,
    }
    return mgr


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


class TestNoiseFilter:
    """Verify noise filtering logic extracted from audio_agent."""

    def test_noise_utterance_is_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u55ef", awaiting_confirmation=False) is True

    def test_confirmation_word_awaiting_is_not_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u597d\u7684", awaiting_confirmation=True) is False

    def test_confirmation_word_not_awaiting_is_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u597d\u7684", awaiting_confirmation=False) is True

    def test_single_char_command_not_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u505c", awaiting_confirmation=False) is False

    def test_short_noise_is_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u55ef\u55ef", awaiting_confirmation=False) is True

    def test_normal_text_not_noise(self):
        mgr = _make_manager()
        assert mgr._filter_noise("\u4f60\u597d\u4e16\u754c", awaiting_confirmation=False) is False

    def test_unknown_single_char_is_noise(self):
        mgr = _make_manager()
        # Single char not in _SINGLE_CHAR_COMMANDS should be noise
        assert mgr._filter_noise("\u7f8e", awaiting_confirmation=False) is True

    def test_all_single_char_commands_pass(self):
        mgr = _make_manager()
        for cmd in _SINGLE_CHAR_COMMANDS:
            assert mgr._filter_noise(cmd, awaiting_confirmation=False) is False, (
                f"Single-char command '{cmd}' should not be noise"
            )


# ---------------------------------------------------------------------------
# Punctuation restoration
# ---------------------------------------------------------------------------


class TestPunctuation:
    """Verify punctuation restoration is called on valid text."""

    def test_punctuation_called_on_valid_text(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        # Simulate local ASR returning valid text via finish_and_get_result
        mocks["asr"].get_result.return_value = "\u4f60\u597d\u4e16\u754c"
        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.text == "\u4f60\u597d\u4e16\u754c."  # punct mock appends "."
        mocks["punct"].restore.assert_called_once_with("\u4f60\u597d\u4e16\u754c")

    def test_punctuation_not_called_on_noise(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].get_result.return_value = "\u55ef"
        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.is_noise is True
        mocks["punct"].restore.assert_not_called()


# ---------------------------------------------------------------------------
# Cloud / local fallback
# ---------------------------------------------------------------------------


class TestCloudLocalFallback:
    """Verify cloud-preferred, local-fallback behaviour."""

    def test_cloud_preferred_over_local(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["cloud"].finish_session.return_value = "\u4e91\u7aef\u7ed3\u679c"
        mocks["asr"].get_result.return_value = "\u672c\u5730\u7ed3\u679c"

        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.source == "cloud"
        assert "\u4e91\u7aef\u7ed3\u679c" in result.text

    def test_local_fallback_when_cloud_empty(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["cloud"].finish_session.return_value = ""
        mocks["asr"].get_result.return_value = "\u672c\u5730\u7ed3\u679c"

        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.source == "local"
        assert "\u672c\u5730\u7ed3\u679c" in result.text

    def test_local_fallback_when_cloud_whitespace(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["cloud"].finish_session.return_value = "   "
        mocks["asr"].get_result.return_value = "\u672c\u5730\u7ed3\u679c"

        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.source == "local"

    def test_finish_uses_configured_cloud_timeout(self):
        mgr = _make_manager(
            cloud_available=True,
            config={"cloud_asr": {"finish_timeout": 8.5}},
        )
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["cloud"].finish_session.return_value = "\u4e91\u7aef\u7ed3\u679c"

        mgr.start_session()
        mgr.finish_and_get_result()

        mocks["cloud"].finish_session.assert_called_once_with(timeout=8.5)

    def test_local_fallback_when_cloud_unavailable(self):
        mgr = _make_manager(cloud_available=False)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].get_result.return_value = "\u672c\u5730\u7ed3\u679c"

        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is not None
        assert result.source == "local"

    def test_none_when_both_empty(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["cloud"].finish_session.return_value = ""
        mocks["asr"].get_result.return_value = ""

        mgr.start_session()
        result = mgr.finish_and_get_result()

        assert result is None


# ---------------------------------------------------------------------------
# Reset / session lifecycle
# ---------------------------------------------------------------------------


class TestReset:
    """Verify reset clears streams and state."""

    def test_reset_clears_state(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mgr.start_session()
        assert mgr._recognition_active is True

        mgr.reset()

        assert mgr._recognition_active is False
        assert mgr._cloud_active is False
        assert mgr._start_time == 0.0
        mocks["asr"].reset.assert_called()
        mocks["asr"].create_stream.assert_called()

    def test_reset_creates_new_stream(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        old_stream = mgr._stream
        new_stream = MagicMock()
        mocks["asr"].create_stream.return_value = new_stream

        mgr.reset()

        assert mgr._stream is new_stream


# ---------------------------------------------------------------------------
# Force endpoint
# ---------------------------------------------------------------------------


class TestForceEndpoint:
    """Verify forced endpoint behaviour (max speech duration guard)."""

    def test_force_endpoint_returns_text(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].get_result.return_value = "\u8fd9\u662f\u4e00\u6bb5\u8bdd"
        mgr.start_session()
        result = mgr.force_endpoint()

        assert result is not None
        assert result.source == "local"
        assert "\u8fd9\u662f\u4e00\u6bb5\u8bdd" in result.text

    def test_force_endpoint_cancels_cloud(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].get_result.return_value = "\u6d4b\u8bd5"
        mgr.start_session()
        mgr.force_endpoint()

        mocks["cloud"].cancel_session.assert_called_once()

    def test_force_endpoint_none_for_short_text(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].get_result.return_value = "\u554a"
        mgr.start_session()
        result = mgr.force_endpoint()

        assert result is None


def test_abort_session_cancels_provider_without_resetting_local_stream() -> None:
    mgr = _make_manager(cloud_available=True)
    mocks = mgr._test_mocks  # type: ignore[attr-defined]
    mgr._recognition_active = True
    # finish_and_get_result hands this flag off before it waits for cloud I/O.
    mgr._cloud_active = False
    mocks["asr"].reset.reset_mock()
    mocks["asr"].create_stream.reset_mock()

    mgr.abort_session()

    assert mgr._recognition_active is False
    assert mgr._cloud_active is False
    mocks["cloud"].cancel_session.assert_called_once_with()
    mocks["asr"].reset.assert_not_called()
    mocks["asr"].create_stream.assert_not_called()


def test_abort_during_cloud_finish_suppresses_local_fallback() -> None:
    mgr = _make_manager(cloud_available=True)
    mocks = mgr._test_mocks  # type: ignore[attr-defined]
    finish_entered = threading.Event()
    release_finish = threading.Event()
    result: list[object] = []

    def blocking_finish(*, timeout: float) -> str:
        del timeout
        finish_entered.set()
        release_finish.wait(timeout=1.0)
        return "stale cloud result"

    mocks["cloud"].finish_session.side_effect = blocking_finish
    mgr.start_session()
    worker = threading.Thread(target=lambda: result.append(mgr.finish_and_get_result()))
    worker.start()
    assert finish_entered.wait(timeout=1.0)

    mgr.abort_session()
    release_finish.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result == [None]
    mocks["asr"].get_result.assert_not_called()


def test_abort_linearizes_against_late_preconnect_success() -> None:
    mgr = _make_manager(cloud_available=True)
    mocks = mgr._test_mocks  # type: ignore[attr-defined]
    start_entered = threading.Event()
    release_start = threading.Event()

    def blocked_start() -> bool:
        start_entered.set()
        release_start.wait(timeout=1.0)
        return True

    mocks["cloud"].start_session.side_effect = blocked_start
    worker = threading.Thread(target=mgr.preconnect_cloud)
    worker.start()
    assert start_entered.wait(timeout=1.0)

    mgr.abort_session()
    release_start.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert mgr._recognition_active is False
    assert mgr._cloud_active is False
    assert mocks["cloud"].cancel_session.call_count == 2

    mocks["cloud"].start_session.side_effect = None
    mocks["cloud"].start_session.return_value = True
    mgr.preconnect_cloud()
    assert mgr._cloud_active is True
    assert mocks["cloud"].start_session.call_count == 2


def test_abort_linearizes_against_late_start_session_success() -> None:
    mgr = _make_manager(cloud_available=True)
    mocks = mgr._test_mocks  # type: ignore[attr-defined]
    start_entered = threading.Event()
    release_start = threading.Event()

    def blocked_start() -> bool:
        start_entered.set()
        release_start.wait(timeout=1.0)
        return True

    mocks["cloud"].start_session.side_effect = blocked_start
    worker = threading.Thread(target=mgr.start_session)
    worker.start()
    assert start_entered.wait(timeout=1.0)

    mgr.abort_session()
    release_start.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert mgr._recognition_active is False
    assert mgr._cloud_active is False
    assert mocks["cloud"].cancel_session.call_count == 2

    mocks["cloud"].start_session.side_effect = None
    mocks["cloud"].start_session.return_value = True
    mgr.start_session()
    assert mgr._recognition_active is True
    assert mgr._cloud_active is True
    assert mocks["cloud"].start_session.call_count == 2


# ---------------------------------------------------------------------------
# Check endpoint (local streaming)
# ---------------------------------------------------------------------------


class TestCheckEndpoint:
    """Verify local ASR endpoint detection."""

    def test_check_endpoint_returns_result(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].is_endpoint.return_value = True
        mocks["asr"].get_result.return_value = "\u4f60\u597d"

        mgr.start_session()
        result = mgr.check_endpoint()

        assert result is not None
        assert result.text == "\u4f60\u597d"
        assert result.source == "local"

    def test_check_endpoint_none_when_no_endpoint(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mocks["asr"].is_endpoint.return_value = False

        mgr.start_session()
        result = mgr.check_endpoint()

        assert result is None

    def test_check_endpoint_does_not_preempt_active_cloud_session(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        mocks["asr"].is_endpoint.return_value = True
        mocks["asr"].get_result.return_value = "你记住"

        mgr.start_session()
        result = mgr.check_endpoint()

        assert mgr._cloud_active is True
        assert result is None
        mocks["asr"].get_result.assert_not_called()


class TestPartialResult:
    def test_cloud_partial_is_exposed_without_finishing_session(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        mocks["cloud"].status_snapshot.return_value = {
            "partial_text": "\u4f60\u597d",
            "partial_age_ms": 180.0,
        }

        mgr.start_session()
        partial = mgr.partial_result()

        assert partial == ASRPartial(
            text="\u4f60\u597d",
            source="cloud_partial",
            age_ms=180.0,
        )
        mocks["cloud"].finish_session.assert_not_called()

    def test_commit_partial_cancels_cloud_without_waiting_for_final(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        mgr.start_session()

        result = mgr.commit_partial(
            ASRPartial(text="\u4f60\u597d", source="cloud_partial", age_ms=200.0)
        )

        assert result is not None
        assert result.source == "cloud_partial"
        assert result.text.startswith("\u4f60\u597d")
        assert mgr._recognition_active is False
        mocks["cloud"].cancel_session.assert_called_once()
        mocks["cloud"].finish_session.assert_not_called()


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------


class TestStatusSnapshot:
    """Verify ASR health details are safe to expose to UI/health checks."""

    def test_status_snapshot_reports_local_provider_by_default(self):
        mgr = _make_manager(cloud_available=False)

        snapshot = mgr.status_snapshot()

        assert snapshot["provider"] == "local"
        assert snapshot["recognition_active"] is False
        assert snapshot["cloud_active"] is False
        assert snapshot["local"]["provider"] == "sherpa_onnx"
        assert snapshot["local"]["available"] is True

    def test_status_snapshot_embeds_cloud_snapshot_when_available(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        mocks["cloud"].status_snapshot.return_value = {
            "provider": "dashscope_paraformer",
            "available": True,
            "partial_text": "partial",
            "first_partial_ms": 87.5,
        }

        mgr.start_session()
        snapshot = mgr.status_snapshot()

        assert snapshot["provider"] == "cloud+local"
        assert snapshot["recognition_active"] is True
        assert snapshot["cloud_active"] is True
        assert snapshot["cloud"]["partial_text"] == "partial"
        assert snapshot["cloud"]["first_partial_ms"] == 87.5
        assert snapshot["elapsed_ms"] is not None


# ---------------------------------------------------------------------------
# Session guard tests
# ---------------------------------------------------------------------------


class TestSessionGuards:
    """Verify methods are safe to call before start_session."""

    def test_feed_audio_before_start_session_is_noop(self):
        mgr = _make_manager()
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        assert mgr._recognition_active is False
        # Should not crash, should silently return
        mgr.feed_audio(
            np.zeros(160, dtype=np.float32),
            np.zeros(160, dtype=np.int16),
            16000,
        )
        mocks["stream"].accept_waveform.assert_not_called()

    def test_finish_before_start_returns_none(self):
        mgr = _make_manager()
        assert mgr._recognition_active is False
        result = mgr.finish_and_get_result()
        assert result is None

    def test_force_endpoint_before_start_returns_none(self):
        mgr = _make_manager()
        assert mgr._recognition_active is False
        result = mgr.force_endpoint()
        assert result is None

    def test_check_endpoint_before_start_returns_none(self):
        mgr = _make_manager()
        assert mgr._recognition_active is False
        result = mgr.check_endpoint()
        assert result is None

    def test_feed_audio_feeds_local_and_cloud(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        mgr.start_session()

        f32 = np.zeros(160, dtype=np.float32)
        i16 = np.zeros(160, dtype=np.int16)
        mgr.feed_audio(f32, i16, 16000)

        mocks["stream"].accept_waveform.assert_called_once_with(16000, f32)
        mocks["cloud"].feed.assert_called_once()

    def test_preconnected_cloud_does_not_feed_silence_by_default(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mgr.preconnect_cloud()
        mgr.feed_cloud_only(np.zeros(160, dtype=np.int16))

        mocks["cloud"].feed.assert_not_called()

    def test_volcengine_does_not_idle_preconnect_by_default(self):
        mgr = _make_manager(
            cloud_available=True,
            config={"cloud_asr": {"provider": "volcengine"}},
        )
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mgr.preconnect_cloud()

        mocks["cloud"].start_session.assert_not_called()
        assert mgr._cloud_active is False

    def test_volcengine_idle_preconnect_can_be_explicitly_enabled(self):
        mgr = _make_manager(
            cloud_available=True,
            config={
                "cloud_asr": {
                    "provider": "volcengine",
                    "preconnect": True,
                }
            },
        )
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mgr.preconnect_cloud()

        mocks["cloud"].start_session.assert_called_once()
        assert mgr._cloud_active is True

    def test_preconnected_cloud_can_feed_silence_when_enabled(self):
        mgr = _make_manager(
            cloud_available=True,
            config={"cloud_asr": {"feed_silence": True}},
        )
        mocks = mgr._test_mocks  # type: ignore[attr-defined]

        mgr.preconnect_cloud()
        mgr.feed_cloud_only(np.zeros(160, dtype=np.int16))

        mocks["cloud"].feed.assert_called_once()

    def test_cloud_start_failure_falls_back(self):
        mgr = _make_manager(cloud_available=True)
        mocks = mgr._test_mocks  # type: ignore[attr-defined]
        # Cloud reports available but start_session fails
        mocks["cloud"].start_session.return_value = False
        mgr.start_session()
        assert mgr._cloud_active is False

        # Feed audio — should only go to local, not cloud
        f32 = np.zeros(160, dtype=np.float32)
        i16 = np.zeros(160, dtype=np.int16)
        mgr.feed_audio(f32, i16, 16000)
        mocks["cloud"].feed.assert_not_called()
