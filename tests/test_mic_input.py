"""Tests for MicInput module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.voice.mic_input import MicInput


class TestMicInputStatic:
    def test_to_int16(self):
        samples = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        result = MicInput.to_int16(samples)
        assert result.dtype == np.int16
        assert result[0] == 16384  # 0.5 * 32768
        assert result[1] == -16384
        assert result[2] == 0

    def test_get_peak(self):
        samples = np.array([100, -500, 300, -200], dtype=np.int16)
        assert MicInput.get_peak(samples) == 500

    def test_get_peak_silence(self):
        samples = np.zeros(1600, dtype=np.int16)
        assert MicInput.get_peak(samples) == 0


class TestMicInputInit:
    def test_default_config(self):
        mic = MicInput()
        assert mic.sample_rate == 16000
        assert mic.chunk_samples == 1600  # 100ms at 16kHz
        assert mic.is_open is False

    def test_custom_config(self):
        mic = MicInput(device=2, sample_rate=44100, chunk_ms=50)
        assert mic.sample_rate == 44100
        assert mic.chunk_samples == 2205  # 50ms at 44100

    def test_from_config(self):
        cfg = {"voice": {"input_device": 0, "asr": {"sample_rate": 16000}}}
        mic = MicInput.from_config(cfg)
        assert mic._device == 0
        assert mic.sample_rate == 16000

    def test_from_config_string_device(self):
        cfg = {"voice": {"input_device": "hw:1,0"}}
        mic = MicInput.from_config(cfg)
        assert mic._device == "hw:1,0"

    def test_from_config_null_device(self):
        cfg = {"voice": {}}
        mic = MicInput.from_config(cfg)
        assert mic._device is None


class TestPreRoll:
    def test_buffer_pre_roll(self):
        mic = MicInput(pre_roll_chunks=3)
        for i in range(5):
            mic.buffer_pre_roll(np.full(100, i, dtype=np.float32))
        # Only last 3 kept (maxlen=3)
        chunks = mic.flush_pre_roll()
        assert len(chunks) == 3
        assert chunks[0][0] == 2
        assert chunks[2][0] == 4

    def test_flush_clears(self):
        mic = MicInput()
        mic.buffer_pre_roll(np.zeros(100, dtype=np.float32))
        mic.flush_pre_roll()
        assert len(mic.flush_pre_roll()) == 0

    def test_pre_roll_copies(self):
        mic = MicInput()
        original = np.ones(100, dtype=np.float32)
        mic.buffer_pre_roll(original)
        original[:] = 0  # modify original
        chunks = mic.flush_pre_roll()
        assert chunks[0][0] == 1.0  # buffer has the copy, not modified


class TestMicInputOpen:
    def test_not_open_raises(self):
        mic = MicInput()
        with pytest.raises(RuntimeError, match="not open"):
            mic.read_chunk()

    @patch("askme.voice.mic_input.sd.InputStream")
    def test_open_context(self, mock_stream_cls):
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.read.return_value = (np.zeros((1600, 1), dtype=np.float32), None)
        mock_stream_cls.return_value = mock_stream

        mic = MicInput(device=0, sample_rate=16000)
        assert mic.is_open is False
        with mic.open():
            assert mic.is_open is True
            chunk = mic.read_chunk()
            assert chunk.shape == (1600,)
            assert chunk.dtype == np.float32
        assert mic.is_open is False

    @patch("askme.voice.mic_input.sd.InputStream")
    def test_open_with_audio_router(self, mock_stream_cls):
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream_cls.return_value = mock_stream

        router = MagicMock()
        mic = MicInput(audio_router=router)
        with mic.open():
            pass
        router.wait_for_input_ready.assert_called_once_with(timeout=10.0)
