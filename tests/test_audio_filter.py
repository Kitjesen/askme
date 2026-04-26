"""Tests for audio input filter (DC removal + HPF + bandpass)."""

import numpy as np

from askme.voice.audio_filter import AudioFilter


class TestAudioFilter:
    def test_enabled_by_default(self):
        af = AudioFilter()
        assert af.enabled is True

    def test_disabled(self):
        af = AudioFilter({"enabled": False})
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        result = af.process(audio)
        assert result is audio  # unchanged

    def test_dc_removal(self):
        """DC offset should be removed, leaving only AC content."""
        af = AudioFilter({"enabled": True, "highpass_freq": 10, "sample_rate": 16000})
        # Signal with large DC offset + small AC
        t = np.arange(1600, dtype=np.float32) / 16000
        signal = 0.5 + 0.01 * np.sin(2 * np.pi * 1000 * t)  # DC=0.5
        # Process multiple chunks so filter converges
        for _ in range(20):
            result = af.process(signal)
        # DC should be mostly removed
        assert abs(np.mean(result)) < 0.05

    def test_highpass_removes_low_frequency(self):
        """80Hz HPF should attenuate 50Hz motor hum."""
        np.random.seed(42)
        af = AudioFilter({"enabled": True, "highpass_freq": 80, "sample_rate": 16000})
        t = np.arange(16000, dtype=np.float32) / 16000  # 1 second
        hum = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50Hz motor hum
        speech = 0.1 * np.sin(2 * np.pi * 500 * t)  # 500Hz speech
        mixed = (hum + speech).astype(np.float32)

        # Process in chunks to let filter converge
        chunk_size = 1600
        filtered_chunks = []
        for i in range(0, len(mixed), chunk_size):
            chunk = mixed[i:i + chunk_size]
            if len(chunk) == chunk_size:
                filtered_chunks.append(af.process(chunk))
        filtered = np.concatenate(filtered_chunks)

        # Measure 50Hz energy before and after (skip first 3200 samples for convergence)
        skip = 3200
        hum_before = np.mean(mixed[skip:] ** 2)
        hum_filtered = np.mean(filtered[skip:] ** 2)
        # Filtered should have less energy (50Hz removed, 500Hz kept)
        assert hum_filtered < hum_before

    def test_highpass_preserves_speech_frequencies(self):
        """80Hz HPF should pass 300Hz+ speech frequencies."""
        af = AudioFilter({"enabled": True, "highpass_freq": 80, "sample_rate": 16000})
        t = np.arange(16000, dtype=np.float32) / 16000
        speech = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        chunk_size = 1600
        filtered_chunks = []
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i:i + chunk_size]
            if len(chunk) == chunk_size:
                filtered_chunks.append(af.process(chunk))
        filtered = np.concatenate(filtered_chunks)

        # 1000Hz energy should be mostly preserved (>80% of original)
        skip = 3200
        energy_before = np.mean(speech[skip:] ** 2)
        energy_after = np.mean(filtered[skip:] ** 2)
        assert energy_after > 0.8 * energy_before

    def test_bandpass_mode(self):
        af = AudioFilter({
            "enabled": True,
            "mode": "bandpass",
            "bandpass_low": 300,
            "bandpass_high": 3400,
            "sample_rate": 16000,
        })
        t = np.arange(16000, dtype=np.float32) / 16000
        # Mix of below-band (50Hz), in-band (1000Hz), above-band (7000Hz)
        low = 0.3 * np.sin(2 * np.pi * 50 * t)
        mid = 0.3 * np.sin(2 * np.pi * 1000 * t)
        high = 0.3 * np.sin(2 * np.pi * 7000 * t)
        mixed = (low + mid + high).astype(np.float32)

        chunk_size = 1600
        filtered_chunks = []
        for i in range(0, len(mixed), chunk_size):
            chunk = mixed[i:i + chunk_size]
            if len(chunk) == chunk_size:
                filtered_chunks.append(af.process(chunk))
        filtered = np.concatenate(filtered_chunks)

        # Total energy should be reduced (out-of-band removed)
        skip = 3200
        assert np.mean(filtered[skip:] ** 2) < np.mean(mixed[skip:] ** 2)

    def test_output_dtype_float32(self):
        af = AudioFilter({"enabled": True})
        samples = np.random.randn(1600).astype(np.float32) * 0.1
        result = af.process(samples)
        assert result.dtype == np.float32

    def test_output_length_preserved(self):
        af = AudioFilter({"enabled": True})
        samples = np.random.randn(1600).astype(np.float32) * 0.1
        result = af.process(samples)
        assert len(result) == len(samples)

    def test_reset(self):
        af = AudioFilter({"enabled": True})
        samples = np.random.randn(1600).astype(np.float32) * 0.1
        af.process(samples)
        af.reset()
        assert np.all(af._hp_state == 0)
        assert af._dc_prev_in == 0.0
        assert af._dc_prev_out == 0.0

    def test_state_persistence_across_chunks(self):
        """Filter state should persist across calls for smooth filtering."""
        af = AudioFilter({"enabled": True})
        chunk1 = np.random.randn(1600).astype(np.float32) * 0.1
        chunk2 = np.random.randn(1600).astype(np.float32) * 0.1
        af.process(chunk1)
        state_after_1 = af._hp_state.copy()
        af.process(chunk2)
        state_after_2 = af._hp_state.copy()
        # State should change between chunks
        assert not np.allclose(state_after_1, state_after_2)
