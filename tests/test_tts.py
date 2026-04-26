"""Regression tests for the TTS engine (local + edge backends)."""

from __future__ import annotations

import threading
import time
import types

import pytest


def test_local_backend_generates_and_queues(monkeypatch):
    """Local backend: sherpa-onnx generate → resample → queue."""

    from askme.voice.tts import TTSEngine

    class _FakeAudio:
        def __init__(self):
            self.samples = [0.1] * 4410  # 0.1s at 44100 Hz
            self.sample_rate = 44100

    class _FakeTts:
        def generate(self, text, sid=0, speed=1.0):
            return _FakeAudio()

    # Force edge backend (no model dir) then patch in local
    engine = TTSEngine({"backend": "edge"})
    try:
        engine._backend = "local"
        engine._local_tts = _FakeTts()
        engine._local_sample_rate = 44100

        generation = engine._get_generation()
        engine._generate_audio("你好世界", generation)

        assert engine._has_buffered_audio()
        with engine._buffer_lock:
            samples = engine.tts_buffer[0]
        # Should be resampled from 44100 to 24000
        expected_len = int(4410 * 24000 / 44100)
        assert abs(len(samples) - expected_len) <= 1
    finally:
        engine.shutdown()


def test_edge_backend_calls_edge_tts_and_decodes(monkeypatch):
    """Edge backend: edge-tts stream → MP3 accumulate → decode → queue."""
    pytest.importorskip("edge_tts", reason="edge_tts not installed")
    from askme.voice.tts import TTSEngine

    captured: list[bytes] = []

    async def fake_stream(self):
        yield {"type": "audio", "data": b"ABC"}
        yield {"type": "WordBoundary", "data": None}
        yield {"type": "audio", "data": b"DEF"}

    monkeypatch.setattr("edge_tts.Communicate.stream", fake_stream)

    import miniaudio

    class _Decoded:
        samples = b"\x01\x00" * 100

    monkeypatch.setattr(miniaudio, "decode", lambda data, **kw: _Decoded())

    engine = TTSEngine({"backend": "edge"})
    try:
        generation = engine._get_generation()
        engine._generate_audio("hello", generation)
        assert engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_drain_buffers_invalidates_inflight_generation(monkeypatch):
    from askme.voice.tts import TTSEngine

    started = threading.Event()
    queued = []

    def fake_generate(self, text, generation):
        started.set()
        time.sleep(0.15)
        # Try to queue after drain
        if self._is_generation_current(generation):
            queued.append(text)

    engine = TTSEngine({"backend": "edge"})
    try:
        monkeypatch.setattr(engine, "_generate_audio", types.MethodType(fake_generate, engine))
        engine.speak("old turn")
        assert started.wait(timeout=1.0)

        engine.drain_buffers()
        engine.tts_text_queue.join()
        time.sleep(0.05)

        assert queued == []
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_playback_loop_uses_configured_output_device(monkeypatch):
    """_playback_loop passes the configured output_device to sd.play."""
    import numpy as np

    import askme.voice.tts as tts_mod
    from askme.voice.tts import TTSEngine

    played_kwargs: dict[str, object] = {}

    def fake_play(data, samplerate, device=None):
        played_kwargs["device"] = device
        # Stop playback loop after the first chunk so the test terminates.
        engine._is_playing = False

    monkeypatch.setattr(tts_mod.sd, "play", fake_play)
    monkeypatch.setattr(tts_mod.sd, "wait", lambda: None)

    engine = TTSEngine({"backend": "edge", "output_device": 3})
    engine._aplay_bin = None  # disable aplay so we exercise the sd.play branch
    try:
        # Queue a real audio chunk so the loop doesn't spin on empty buffer.
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert played_kwargs.get("device") == 3


def test_auto_fallback_when_model_missing():
    """Backend auto-falls back to edge when model directory doesn't exist."""
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "local", "model_dir": "/nonexistent/path"})
    try:
        assert engine._backend == "edge"
    finally:
        engine.shutdown()
