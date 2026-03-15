"""Tests for AudioProcessor — unified audio preprocessing pipeline."""

import numpy as np
import pytest

from askme.voice.audio_processor import AudioProcessor


class TestAudioProcessorDefaults:
    """Default (all-disabled) configuration should pass audio through."""

    def test_passthrough_no_config(self):
        """Empty config = no filter, no gate, no echo gate."""
        ap = AudioProcessor()
        samples = np.random.randn(1600).astype(np.float32) * 0.1
        out_f32, out_i16, peak, gated = ap.process(samples)
        assert out_f32 is samples  # same object, no filter applied
        assert out_i16.dtype == np.int16
        assert peak >= 0
        assert gated is False

    def test_noise_gate_disabled_by_default(self):
        ap = AudioProcessor()
        assert ap.noise_gate_peak == 0
        assert ap.is_noise_gated(100) is False


class TestAudioFilterIntegration:
    """HPF enabled via config processes audio."""

    def test_hpf_enabled(self):
        ap = AudioProcessor({"audio_filter": {"enabled": True, "highpass_freq": 80}})
        t = np.arange(1600, dtype=np.float32) / 16000
        # 50Hz hum + 1kHz speech
        samples = (0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        out_f32, out_i16, peak, gated = ap.process(samples)
        assert out_f32.dtype == np.float32
        assert len(out_f32) == 1600
        assert gated is False

    def test_hpf_disabled_passthrough(self):
        ap = AudioProcessor({"audio_filter": {"enabled": False}})
        samples = np.random.randn(1600).astype(np.float32) * 0.1
        out_f32, _, _, _ = ap.process(samples)
        assert out_f32 is samples


class TestNoiseGate:
    """Noise gate blocks low-peak frames."""

    def test_noise_gate_blocks_low_peak(self):
        ap = AudioProcessor({"noise_gate_peak": 500})
        assert ap.noise_gate_peak == 500
        assert ap.is_noise_gated(200) is True

    def test_noise_gate_passes_high_peak(self):
        ap = AudioProcessor({"noise_gate_peak": 500})
        assert ap.is_noise_gated(800) is False

    def test_noise_gate_setter(self):
        ap = AudioProcessor()
        assert ap.noise_gate_peak == 0
        ap.noise_gate_peak = 600
        assert ap.noise_gate_peak == 600
        assert ap.is_noise_gated(400) is True
        assert ap.is_noise_gated(700) is False


class TestEchoGate:
    """Echo gate suppresses low-energy mic during TTS playback."""

    def test_echo_gate_blocks_during_tts(self):
        ap = AudioProcessor({"echo_gate_peak": 800})
        # Low-energy frame during TTS → gated
        samples = np.zeros(1600, dtype=np.float32)  # silent
        _, _, peak, gated = ap.process(samples, tts_active=True, speech_active=False)
        assert gated is True

    def test_echo_gate_passes_speech_during_tts(self):
        ap = AudioProcessor({"echo_gate_peak": 800})
        # High-energy frame during TTS → barge-in, not gated
        samples = (np.ones(1600, dtype=np.float32) * 0.5)
        _, _, peak, gated = ap.process(samples, tts_active=True, speech_active=False)
        assert peak > 800
        assert gated is False

    def test_echo_gate_no_tts(self):
        ap = AudioProcessor({"echo_gate_peak": 800})
        samples = np.zeros(1600, dtype=np.float32)
        _, _, _, gated = ap.process(samples, tts_active=False, speech_active=False)
        assert gated is False

    def test_echo_gate_speech_active_bypass(self):
        """Once speech_active, echo gate must not suppress (barge-in in progress)."""
        ap = AudioProcessor({"echo_gate_peak": 800})
        samples = np.zeros(1600, dtype=np.float32)
        _, _, _, gated = ap.process(samples, tts_active=True, speech_active=True)
        assert gated is False

    def test_echo_gate_disabled(self):
        ap = AudioProcessor({"echo_gate_peak": 0})
        samples = np.zeros(1600, dtype=np.float32)
        _, _, _, gated = ap.process(samples, tts_active=True, speech_active=False)
        assert gated is False


class TestAutoCalibration:
    """Noise gate auto-calibration from ambient noise."""

    def test_auto_calibrate_flag(self):
        ap = AudioProcessor({"noise_gate_peak": "auto"})
        assert ap.noise_gate_peak == 0  # not yet calibrated
        assert ap._auto_calibrate_gate is True
        assert ap._noise_calibrator is not None

    def test_auto_calibrate_completes(self):
        ap = AudioProcessor({"noise_gate_peak": "auto"})
        # Feed 20 frames of low-energy noise (NoiseGateCalibrator default)
        rng = np.random.RandomState(42)
        for i in range(19):
            chunk = (rng.randn(1600) * 50).astype(np.int16)
            result = ap.auto_calibrate_gate(chunk)
            assert result is None
        # 20th frame should complete calibration
        chunk = (rng.randn(1600) * 50).astype(np.int16)
        result = ap.auto_calibrate_gate(chunk)
        assert result is not None
        assert ap.noise_gate_peak == result
        assert ap._noise_calibrator is None  # consumed

    def test_auto_calibrate_noop_when_not_auto(self):
        ap = AudioProcessor({"noise_gate_peak": 500})
        chunk = (np.random.randn(1600) * 50).astype(np.int16)
        assert ap.auto_calibrate_gate(chunk) is None


class TestNoiseReduction:
    """Spectral subtraction calibration + processing."""

    def test_feed_calibration(self):
        ap = AudioProcessor({
            "noise_reduction": {"enabled": True, "calibration_frames": 3},
        })
        assert ap._subtractor is not None
        assert ap._subtractor.calibrated is False
        for _ in range(3):
            ap.feed_calibration(np.random.randn(1600).astype(np.float32) * 0.01)
        assert ap._subtractor.calibrated is True

    def test_noise_reduction_processes_after_calibration(self):
        ap = AudioProcessor({
            "noise_reduction": {"enabled": True, "calibration_frames": 3, "frame_size": 512, "hop_size": 256},
        })
        # Calibrate with quiet noise
        for _ in range(3):
            ap.feed_calibration(np.random.randn(1600).astype(np.float32) * 0.001)
        # Process a frame with noise + tone
        t = np.arange(1600, dtype=np.float32) / 16000
        noisy = (0.1 * np.sin(2 * np.pi * 1000 * t) + np.random.randn(1600).astype(np.float32) * 0.001).astype(np.float32)
        _, out_i16, peak, _ = ap.process(noisy)
        assert out_i16.dtype == np.int16
        assert len(out_i16) == 1600

    def test_noise_reduction_disabled_passthrough(self):
        ap = AudioProcessor({"noise_reduction": {"enabled": False}})
        assert ap._subtractor is None


class TestConfigParsing:
    """Config edge cases."""

    def test_none_config(self):
        ap = AudioProcessor(None)
        assert ap._filter is None
        assert ap._subtractor is None
        assert ap.noise_gate_peak == 0

    def test_empty_config(self):
        ap = AudioProcessor({})
        assert ap._filter is None
        assert ap._subtractor is None

    def test_full_config(self):
        ap = AudioProcessor({
            "audio_filter": {"enabled": True, "highpass_freq": 100, "mode": "bandpass"},
            "noise_reduction": {"enabled": True, "alpha": 1.5, "beta": 0.05},
            "noise_gate_peak": 400,
            "echo_gate_peak": 900,
        })
        assert ap._filter is not None
        assert ap._subtractor is not None
        assert ap.noise_gate_peak == 400
        assert ap._echo_gate_peak == 900
