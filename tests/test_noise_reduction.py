"""Tests for spectral subtraction noise reduction and noise gate calibration."""

import numpy as np
import pytest

from askme.voice.noise_reduction import NoiseGateCalibrator, SpectralSubtractor


# ---------------------------------------------------------------------------
# SpectralSubtractor
# ---------------------------------------------------------------------------

class TestSpectralSubtractor:
    def test_disabled_by_default(self):
        ss = SpectralSubtractor()
        assert ss.enabled is False
        assert ss.calibrated is False

    def test_enabled_from_config(self):
        ss = SpectralSubtractor({"enabled": True})
        assert ss.enabled is True

    def test_disabled_returns_input_unchanged(self):
        ss = SpectralSubtractor({"enabled": False})
        audio = np.random.randn(1600).astype(np.float32)
        result = ss.process(audio)
        assert result is audio

    def test_uncalibrated_returns_input_unchanged(self):
        ss = SpectralSubtractor({"enabled": True})
        audio = np.random.randn(1600).astype(np.float32)
        result = ss.process(audio)
        assert result is audio

    def test_calibration_needs_enough_frames(self):
        ss = SpectralSubtractor({"enabled": True, "calibration_frames": 5})
        noise = np.random.randn(1600).astype(np.float32) * 0.01
        for i in range(4):
            assert ss.feed_calibration(noise) is False
        assert ss.feed_calibration(noise) is True
        assert ss.calibrated is True

    def test_calibration_with_int16(self):
        ss = SpectralSubtractor({"enabled": True, "calibration_frames": 3})
        noise = (np.random.randn(1600) * 100).astype(np.int16)
        for _ in range(3):
            ss.feed_calibration(noise)
        assert ss.calibrated is True

    def test_process_float32(self):
        np.random.seed(42)  # deterministic
        ss = SpectralSubtractor({
            "enabled": True,
            "calibration_frames": 5,
            "alpha": 2.0,
            "beta": 0.02,
            "frame_size": 256,
            "hop_size": 128,
        })
        # Calibrate with steady-state noise
        for _ in range(5):
            noise = np.random.randn(3200).astype(np.float32) * 0.1
            ss.feed_calibration(noise)

        # Process pure noise (no signal) — should be clearly reduced
        test_noise = np.random.randn(3200).astype(np.float32) * 0.1
        result = ss.process(test_noise)

        assert result.dtype == np.float32
        assert len(result) == len(test_noise)
        # Pure noise energy should be significantly reduced
        assert np.mean(result**2) < np.mean(test_noise**2)

    def test_process_int16(self):
        ss = SpectralSubtractor({
            "enabled": True,
            "calibration_frames": 3,
            "frame_size": 256,
            "hop_size": 128,
        })
        noise = (np.random.randn(1600) * 100).astype(np.int16)
        for _ in range(3):
            ss.feed_calibration(noise)

        signal = (np.sin(np.arange(1600) * 0.1) * 10000).astype(np.int16)
        result = ss.process(signal)
        assert result.dtype == np.int16
        assert len(result) == len(signal)

    def test_process_short_audio(self):
        """Audio shorter than frame_size should be returned unchanged."""
        ss = SpectralSubtractor({
            "enabled": True,
            "calibration_frames": 3,
            "frame_size": 512,
        })
        noise = np.random.randn(1600).astype(np.float32) * 0.01
        for _ in range(3):
            ss.feed_calibration(noise)

        short = np.zeros(100, dtype=np.float32)
        result = ss.process(short)
        assert result is short

    def test_reset(self):
        ss = SpectralSubtractor({"enabled": True, "calibration_frames": 3})
        noise = np.random.randn(1600).astype(np.float32) * 0.01
        for _ in range(3):
            ss.feed_calibration(noise)
        assert ss.calibrated is True
        ss.reset()
        assert ss.calibrated is False

    def test_feed_calibration_when_disabled(self):
        ss = SpectralSubtractor({"enabled": False})
        noise = np.random.randn(1600).astype(np.float32)
        assert ss.feed_calibration(noise) is False

    def test_feed_calibration_after_calibrated(self):
        ss = SpectralSubtractor({"enabled": True, "calibration_frames": 2})
        noise = np.random.randn(1600).astype(np.float32) * 0.01
        ss.feed_calibration(noise)
        ss.feed_calibration(noise)
        assert ss.calibrated is True
        # Further calibration calls return True immediately
        assert ss.feed_calibration(noise) is True


# ---------------------------------------------------------------------------
# NoiseGateCalibrator
# ---------------------------------------------------------------------------

class TestNoiseGateCalibrator:
    def test_needs_enough_frames(self):
        cal = NoiseGateCalibrator(num_frames=5)
        noise = (np.random.randn(1600) * 100).astype(np.int16)
        for _ in range(4):
            assert cal.feed(noise) is None
        result = cal.feed(noise)
        assert result is not None
        assert isinstance(result, int)

    def test_threshold_above_noise(self):
        cal = NoiseGateCalibrator(num_frames=10, sigma_factor=2.5)
        # Simulate consistent low-level noise (peak ~100-200)
        for _ in range(10):
            noise = (np.random.randn(1600) * 50).astype(np.int16)
            result = cal.feed(noise)

        assert result is not None
        # Threshold should be above typical noise peaks
        assert result >= 100
        assert result <= 2000

    def test_clamp_minimum(self):
        cal = NoiseGateCalibrator(num_frames=3, sigma_factor=0.0)
        # Very quiet noise — threshold would be ~mean which is very low
        for i in range(3):
            noise = (np.random.randn(1600) * 5).astype(np.int16)
            result = cal.feed(noise)
        assert result is not None
        assert result >= 100  # clamped minimum

    def test_clamp_maximum(self):
        cal = NoiseGateCalibrator(num_frames=3, sigma_factor=10.0)
        for i in range(3):
            loud = (np.random.randn(1600) * 20000).astype(np.int16)
            result = cal.feed(loud)
        assert result is not None
        assert result <= 2000  # clamped maximum

    def test_reset(self):
        cal = NoiseGateCalibrator(num_frames=3)
        noise = (np.random.randn(1600) * 100).astype(np.int16)
        cal.feed(noise)
        cal.feed(noise)
        cal.reset()
        # After reset, needs full num_frames again
        assert cal.feed(noise) is None
        assert cal.feed(noise) is None
        result = cal.feed(noise)
        assert result is not None
