"""Tests for MicInput module."""

import io
import queue
from types import SimpleNamespace
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

    def test_from_config_input_transport(self):
        cfg = {"voice": {"input_transport": "usb_direct"}}
        mic = MicInput.from_config(cfg)
        assert mic._input_transport == "usb_direct"

    def test_rejects_channel_select_outside_configured_channels(self):
        with pytest.raises(ValueError, match="mic_channel_select"):
            MicInput(mic_native_rate=48000, mic_channels=2, mic_channel_select=2)

    def test_rejects_zero_mic_channels(self):
        with pytest.raises(ValueError, match="mic_channels"):
            MicInput(mic_native_rate=48000, mic_channels=0)

    def test_rejects_negative_channel_select(self):
        with pytest.raises(ValueError, match="mic_channel_select"):
            MicInput(mic_native_rate=48000, mic_channels=2, mic_channel_select=-1)


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

        mic = MicInput(device=0, sample_rate=16000, input_transport="sounddevice")
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

    @patch("askme.voice.mic_input.sd.InputStream")
    def test_router_permission_precedes_physical_stream_open(self, mock_stream_cls):
        order: list[str] = []
        router = MagicMock()
        router.wait_for_input_ready.side_effect = lambda **kwargs: order.append("wait") or True
        mock_stream = MagicMock()
        mock_stream_cls.side_effect = lambda **kwargs: order.append("open") or mock_stream

        mic = MicInput(audio_router=router, input_transport="sounddevice")
        with mic.open():
            pass

        assert order[:2] == ["wait", "open"]

    @patch("askme.voice.mic_input.sd.InputStream")
    def test_exclusive_output_waits_for_inflight_microphone_open(
        self,
        mock_stream_cls,
    ):
        import threading

        from askme.voice.audio_router import AudioRouter

        start_entered = threading.Event()
        release_start = threading.Event()
        output_entered = threading.Event()
        lifecycle: list[str] = []

        class _BlockingStream:
            def start(self):
                lifecycle.append("start")
                start_entered.set()
                assert release_start.wait(timeout=1.0)

            def stop(self):
                lifecycle.append("stop")

            def close(self):
                lifecycle.append("close")

        router = AudioRouter()
        mic = MicInput(
            audio_router=router,
            input_transport="sounddevice",
        )
        mock_stream_cls.return_value = _BlockingStream()
        router.set_input_controller(suspend=mic.stop, resume=None)

        start_thread = threading.Thread(target=mic.start)

        def _take_output() -> None:
            with router.output_session():
                assert mic.is_open is False
                output_entered.set()

        output_thread = threading.Thread(target=_take_output)
        start_thread.start()
        assert start_entered.wait(timeout=1.0)
        output_thread.start()
        assert not output_entered.wait(timeout=0.05)

        release_start.set()
        start_thread.join(timeout=1.0)
        output_thread.join(timeout=1.0)

        assert not start_thread.is_alive()
        assert not output_thread.is_alive()
        assert output_entered.is_set()
        assert lifecycle == ["start", "stop", "close"]

    def test_stop_releases_stream_even_when_driver_stop_raises(self):
        lifecycle: list[str] = []

        class _BrokenStream:
            def stop(self):
                lifecycle.append("stop")
                raise RuntimeError("device disappeared")

            def close(self):
                lifecycle.append("close")

        mic = MicInput(input_transport="sounddevice")
        mic._stream = _BrokenStream()

        mic.stop()

        assert mic._stream is None
        assert lifecycle == ["stop", "close"]

    @patch("askme.voice.mic_input.sd.InputStream")
    def test_start_closes_partial_stream_when_driver_start_fails(
        self,
        mock_stream_cls,
    ):
        lifecycle: list[str] = []

        class _FailedStartStream:
            def start(self):
                lifecycle.append("start")
                raise RuntimeError("no such device")

            def close(self):
                lifecycle.append("close")

        mock_stream_cls.return_value = _FailedStartStream()
        mic = MicInput(input_transport="sounddevice")

        with pytest.raises(RuntimeError, match="no such device"):
            mic.start()

        assert mic._stream is None
        assert lifecycle == ["start", "close"]

    def test_read_chunk_raises_after_callback_starvation(self):
        mic = MicInput(input_transport="sounddevice")
        stream = SimpleNamespace(active=True)
        audio_queue = MagicMock()
        audio_queue.get.side_effect = queue.Empty
        mic._stream = stream
        mic._audio_queue = audio_queue

        assert np.count_nonzero(mic.read_chunk()) == 0
        assert np.count_nonzero(mic.read_chunk()) == 0
        with pytest.raises(RuntimeError, match="callback stopped"):
            mic.read_chunk()

        assert audio_queue.get.call_count == 3


class TestMicInputUsbDirect:
    def test_usb_direct_read_chunk_from_helper(self, monkeypatch):
        pcm = np.arange(1600, dtype=np.int16).tobytes()
        captured: dict[str, list[str]] = {}

        class _FakeProc:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(pcm)
                self.stderr = io.BytesIO()
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return _FakeProc()

        mic = MicInput(input_transport="usb_direct")
        monkeypatch.setattr(mic, "_ensure_usb_audio_binary", lambda: "helper")
        monkeypatch.setattr("askme.voice.mic_input.subprocess.Popen", fake_popen)

        mic.start()
        try:
            chunk = mic.read_chunk()
        finally:
            mic.stop()

        assert captured["args"] == [
            "helper",
            "--play-ms",
            "0",
            "--capture-ms",
            "0",
            "--capture-stdout",
        ]
        assert chunk.shape == (1600,)
        assert chunk.dtype == np.float32
        assert chunk[1] == pytest.approx(1 / 32768.0)

    def test_auto_transport_uses_usb_when_alsa_has_no_cards(self, monkeypatch):
        mic = MicInput()
        started: dict[str, bool] = {}

        def fake_start_usb():
            started["called"] = True
            mic._usb_audio_proc = MagicMock()

        monkeypatch.setattr(mic, "_alsa_input_available", lambda: False)
        monkeypatch.setattr(mic, "_mcp01_visible", lambda: True)
        monkeypatch.setattr(mic, "_start_usb_direct", fake_start_usb)

        mic.start()
        try:
            assert started["called"] is True
            assert mic.is_open is True
        finally:
            mic._usb_audio_proc = None

    def test_alsa_input_probe_handles_portaudio_device_error(self, monkeypatch):
        mic = MicInput(device=0)

        monkeypatch.setattr(
            "askme.voice.mic_input.os",
            SimpleNamespace(name="posix"),
        )
        monkeypatch.setattr(
            "askme.voice.mic_input.Path.read_text",
            lambda self, **_kwargs: " 0 [MCP01 ]: USB-Audio - MCP01\n",
        )

        def _raise_portaudio_error(*_args, **_kwargs):
            raise RuntimeError("Error querying device 0")

        monkeypatch.setattr(
            "askme.voice.mic_input.sd.query_devices",
            _raise_portaudio_error,
        )

        assert mic._alsa_input_available() is False

    def test_alsa_input_requires_configured_channel_count(self, monkeypatch):
        mic = MicInput(mic_native_rate=48000, mic_channels=2)

        # Patch only the module-local platform probe. Mutating the process-wide
        # ``os.name`` makes pathlib try to construct PosixPath on Windows and
        # can crash pytest's own failure reporting before this assertion runs.
        monkeypatch.setattr(
            "askme.voice.mic_input.os",
            SimpleNamespace(name="posix"),
        )
        monkeypatch.setattr(
            "askme.voice.mic_input.Path.read_text",
            lambda self, **_kwargs: " 1 [MCP01 ]: USB-Audio - MCP01\n",
        )
        monkeypatch.setattr(
            "askme.voice.mic_input.sd.query_devices",
            lambda *args, **_kwargs: [{"name": "MCP01", "max_input_channels": 1}],
        )

        assert mic._alsa_input_available() is False

    def test_alsa_input_accepts_matching_channel_count(self, monkeypatch):
        mic = MicInput(mic_native_rate=48000, mic_channels=2)

        monkeypatch.setattr(
            "askme.voice.mic_input.os",
            SimpleNamespace(name="posix"),
        )
        monkeypatch.setattr(
            "askme.voice.mic_input.Path.read_text",
            lambda self, **_kwargs: " 0 [HKMIC ]: USB-Audio - HKMIC\n",
        )
        monkeypatch.setattr(
            "askme.voice.mic_input.sd.query_devices",
            lambda *args, **_kwargs: [{"name": "HKMIC", "max_input_channels": 2}],
        )

        assert mic._alsa_input_available() is True
