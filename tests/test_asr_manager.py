"""Tests for ASRManager: noise filtering, cloud/local fallback, punctuation, reset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from askme.voice.asr_manager import (
    ASRManager,
    ASRResult,
    _CONFIRMATION_WORDS,
    _MIN_VALID_TEXT_LEN,
    _NOISE_UTTERANCES,
    _SINGLE_CHAR_COMMANDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(cloud_available: bool = False) -> ASRManager:
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

        mgr = ASRManager({})

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
