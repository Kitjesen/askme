"""Tests for VADController state machine."""

from unittest.mock import MagicMock, patch

import numpy as np

from askme.voice.vad_controller import VADController, VADEvent


def _chunk(val: int = 0, size: int = 512) -> np.ndarray:
    """Create a fake int16 audio chunk."""
    return np.full(size, val, dtype=np.int16)


def _make_controller(
    *,
    noise_gate_peak: int = 0,
    barge_in_hold_s: float = 0.15,
    max_speech_duration: float = 30.0,
    vad_speech: bool = False,
) -> tuple[VADController, MagicMock]:
    """Build a VADController with a mocked VADEngine.

    Returns (controller, mock_vad_engine).
    """
    with patch("askme.voice.vad_controller.VADEngine") as MockVAD:
        mock_vad = MockVAD.return_value
        mock_vad.is_speech_detected.return_value = vad_speech
        ctrl = VADController({
            "noise_gate_peak": noise_gate_peak,
            "barge_in_hold_s": barge_in_hold_s,
            "max_speech_duration": max_speech_duration,
        })
    return ctrl, mock_vad


# ---------------------------------------------------------------------------
# Basic silence / speech transitions
# ---------------------------------------------------------------------------


class TestSilence:
    def test_silence_on_quiet_input(self):
        ctrl, vad = _make_controller(vad_speech=False)
        event = ctrl.feed(_chunk(), peak=50)
        assert event is VADEvent.SILENCE
        assert not ctrl.speech_active

    def test_silence_repeated(self):
        ctrl, vad = _make_controller(vad_speech=False)
        for _ in range(5):
            assert ctrl.feed(_chunk(), peak=0) is VADEvent.SILENCE
        assert not ctrl.speech_active


class TestSpeechStart:
    def test_speech_start_when_vad_triggers(self):
        ctrl, vad = _make_controller(vad_speech=True)
        event = ctrl.feed(_chunk(1000), peak=2000)
        assert event is VADEvent.SPEECH_START
        assert ctrl.speech_active

    def test_speech_start_no_noise_gate(self):
        """Without noise gate, any VAD-triggered frame starts speech."""
        ctrl, vad = _make_controller(vad_speech=True, noise_gate_peak=0)
        event = ctrl.feed(_chunk(10), peak=10)
        assert event is VADEvent.SPEECH_START


class TestSpeechContinue:
    def test_speech_continue_on_sustained(self):
        ctrl, vad = _make_controller(vad_speech=True)
        ctrl.feed(_chunk(1000), peak=2000)  # SPEECH_START
        event = ctrl.feed(_chunk(1000), peak=2000)
        assert event is VADEvent.SPEECH_CONTINUE
        assert ctrl.speech_active


class TestSpeechEnd:
    def test_speech_end_when_vad_drops(self):
        ctrl, vad = _make_controller(vad_speech=True)
        ctrl.feed(_chunk(1000), peak=2000)  # SPEECH_START

        vad.is_speech_detected.return_value = False
        event = ctrl.feed(_chunk(), peak=50)
        assert event is VADEvent.SPEECH_END
        assert not ctrl.speech_active


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------


class TestNoiseGate:
    def test_noise_gate_blocks_speech_start(self):
        ctrl, vad = _make_controller(vad_speech=True, noise_gate_peak=500)
        event = ctrl.feed(_chunk(100), peak=100)
        assert event is VADEvent.SILENCE
        assert not ctrl.speech_active

    def test_noise_gate_passes_above_threshold(self):
        ctrl, vad = _make_controller(vad_speech=True, noise_gate_peak=500)
        event = ctrl.feed(_chunk(1000), peak=1000)
        assert event is VADEvent.SPEECH_START
        assert ctrl.speech_active

    def test_noise_gate_does_not_affect_speech_continue(self):
        """Once speech_active, noise gate does not interrupt."""
        ctrl, vad = _make_controller(vad_speech=True, noise_gate_peak=500)
        ctrl.feed(_chunk(1000), peak=1000)  # SPEECH_START
        # Now peak drops below gate -- should still CONTINUE
        event = ctrl.feed(_chunk(100), peak=100)
        assert event is VADEvent.SPEECH_CONTINUE


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


class TestBargeIn:
    def test_barge_in_start_during_tts(self):
        ctrl, vad = _make_controller(vad_speech=True, barge_in_hold_s=0.15)
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=True)
        assert event is VADEvent.BARGE_IN_START
        assert not ctrl.speech_active
        assert len(ctrl.barge_in_buffer) == 1

    def test_barge_in_confirmed_after_hold(self):
        ctrl, vad = _make_controller(vad_speech=True, barge_in_hold_s=0.10)
        t0 = 100.0
        # First frame: starts barge-in
        ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t0)
        # Second frame: still within hold, accumulates
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t0 + 0.05)
        assert event is VADEvent.SILENCE  # not yet confirmed
        assert len(ctrl.barge_in_buffer) == 2
        # Third frame: hold exceeded
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t0 + 0.12)
        assert event is VADEvent.BARGE_IN_CONFIRMED
        assert ctrl.speech_active

    def test_barge_in_dismissed_on_vad_drop(self):
        ctrl, vad = _make_controller(vad_speech=True, barge_in_hold_s=0.15)
        ctrl.feed(_chunk(1000), peak=2000, tts_active=True)  # BARGE_IN_START

        vad.is_speech_detected.return_value = False
        event = ctrl.feed(_chunk(), peak=50, tts_active=True)
        assert event is VADEvent.BARGE_IN_DISMISSED
        assert not ctrl.speech_active
        assert len(ctrl.barge_in_buffer) == 0

    def test_barge_in_buffer_accumulates(self):
        ctrl, vad = _make_controller(vad_speech=True, barge_in_hold_s=1.0)
        t = 100.0
        ctrl.feed(_chunk(1), peak=2000, tts_active=True, _now=t)
        ctrl.feed(_chunk(2), peak=2000, tts_active=True, _now=t + 0.05)
        ctrl.feed(_chunk(3), peak=2000, tts_active=True, _now=t + 0.10)
        assert len(ctrl.barge_in_buffer) == 3
        # Verify content preserved
        assert ctrl.barge_in_buffer[0][0] == 1
        assert ctrl.barge_in_buffer[1][0] == 2
        assert ctrl.barge_in_buffer[2][0] == 3

    def test_no_barge_in_without_tts(self):
        """When TTS is not active, speech starts immediately."""
        ctrl, vad = _make_controller(vad_speech=True)
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=False)
        assert event is VADEvent.SPEECH_START
        assert ctrl.speech_active


# ---------------------------------------------------------------------------
# Max speech duration
# ---------------------------------------------------------------------------


class TestMaxDuration:
    def test_max_duration_exceeded(self):
        ctrl, vad = _make_controller(vad_speech=True, max_speech_duration=30.0)
        t0 = 100.0
        ctrl.feed(_chunk(1000), peak=2000, _now=t0)  # SPEECH_START
        # Jump past 30s
        event = ctrl.feed(_chunk(1000), peak=2000, _now=t0 + 31.0)
        assert event is VADEvent.MAX_DURATION_EXCEEDED
        assert not ctrl.speech_active

    def test_max_duration_not_exceeded_within_limit(self):
        ctrl, vad = _make_controller(vad_speech=True, max_speech_duration=30.0)
        t0 = 100.0
        ctrl.feed(_chunk(1000), peak=2000, _now=t0)
        event = ctrl.feed(_chunk(1000), peak=2000, _now=t0 + 29.0)
        assert event is VADEvent.SPEECH_CONTINUE
        assert ctrl.speech_active


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_state(self):
        ctrl, vad = _make_controller(vad_speech=True, barge_in_hold_s=1.0)
        # Drive into barge-in pending state
        ctrl.feed(_chunk(1000), peak=2000, tts_active=True)
        assert len(ctrl.barge_in_buffer) == 1

        ctrl.reset()
        assert not ctrl.speech_active
        assert len(ctrl.barge_in_buffer) == 0

    def test_reset_allows_fresh_speech_start(self):
        ctrl, vad = _make_controller(vad_speech=True)
        ctrl.feed(_chunk(1000), peak=2000)  # SPEECH_START
        assert ctrl.speech_active

        vad.is_speech_detected.return_value = False
        ctrl.feed(_chunk(), peak=0)  # SPEECH_END

        ctrl.reset()
        vad.is_speech_detected.return_value = True
        event = ctrl.feed(_chunk(1000), peak=2000)
        assert event is VADEvent.SPEECH_START
        assert ctrl.speech_active


# ---------------------------------------------------------------------------
# VADEngine delegation
# ---------------------------------------------------------------------------


class TestVADDelegation:
    def test_accept_waveform_called(self):
        ctrl, vad = _make_controller(vad_speech=False)
        chunk = _chunk(500)
        ctrl.feed(chunk, peak=500)
        vad.accept_waveform.assert_called_once_with(chunk)

    def test_is_speech_detected_called(self):
        ctrl, vad = _make_controller(vad_speech=False)
        ctrl.feed(_chunk(), peak=0)
        vad.is_speech_detected.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_noise_gate_during_barge_in_pending(self):
        """When barge-in is pending and a low-peak frame arrives with VAD speech,
        the noise gate fires first (line 109) and returns SILENCE without
        accumulating into the barge-in buffer. This is correct: the noise gate
        is a hard gate that blocks all non-speech-active transitions."""
        ctrl, vad = _make_controller(
            vad_speech=True, noise_gate_peak=500, barge_in_hold_s=1.0,
        )
        t0 = 100.0
        # First frame: high peak, starts barge-in
        event = ctrl.feed(_chunk(1000), peak=1000, tts_active=True, _now=t0)
        assert event is VADEvent.BARGE_IN_START
        assert len(ctrl.barge_in_buffer) == 1

        # Second frame: low peak (below noise gate) -- noise gate blocks before
        # barge-in accumulation path, so buffer stays at 1.
        event = ctrl.feed(_chunk(100), peak=100, tts_active=True, _now=t0 + 0.05)
        assert event is VADEvent.SILENCE
        assert len(ctrl.barge_in_buffer) == 1  # NOT accumulated

    def test_immediate_max_duration(self):
        """Extremely short max_speech_duration triggers on the next frame."""
        ctrl, vad = _make_controller(vad_speech=True, max_speech_duration=0.001)
        t0 = 100.0
        event = ctrl.feed(_chunk(1000), peak=2000, _now=t0)
        assert event is VADEvent.SPEECH_START

        # Next frame just 10ms later -- exceeds 0.001s max
        event = ctrl.feed(_chunk(1000), peak=2000, _now=t0 + 0.01)
        assert event is VADEvent.MAX_DURATION_EXCEEDED
        assert not ctrl.speech_active

    def test_reset_clears_barge_in_timing(self):
        """After a barge-in cycle and reset, a fresh barge-in uses new timing."""
        ctrl, vad = _make_controller(
            vad_speech=True, barge_in_hold_s=0.10,
        )
        t0 = 100.0
        # Start a barge-in
        ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t0)
        assert len(ctrl.barge_in_buffer) == 1

        # Reset clears everything
        ctrl.reset()
        assert len(ctrl.barge_in_buffer) == 0
        assert not ctrl.speech_active

        # Fresh barge-in at a much later time
        t1 = 200.0
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t1)
        assert event is VADEvent.BARGE_IN_START
        assert len(ctrl.barge_in_buffer) == 1

        # Confirm with correct hold relative to t1, not t0
        event = ctrl.feed(_chunk(1000), peak=2000, tts_active=True, _now=t1 + 0.12)
        assert event is VADEvent.BARGE_IN_CONFIRMED
        assert ctrl.speech_active
