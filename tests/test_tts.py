"""Regression tests for the TTS engine (local + edge backends)."""

from __future__ import annotations

import threading
import time
import types


def test_local_backend_generates_and_queues(monkeypatch):
    """Local backend: sherpa-onnx generate → resample → queue."""
    from askme.voice.tts import TTSEngine
    import numpy as np

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
    from askme.voice.tts import TTSEngine

    captured: list[bytes] = []

    async def fake_stream(self):
        yield {"type": "audio", "data": b"ABC"}
        yield {"type": "WordBoundary", "data": None}
        yield {"type": "audio", "data": b"DEF"}

    monkeypatch.setattr("edge_tts.Communicate.stream", fake_stream)

    import miniaudio
    import numpy as np

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
    from askme.voice.tts import TTSEngine
    import askme.voice.tts as tts_mod

    captured: dict[str, object] = {}

    class _FakeStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_sleep(ms):
        engine._is_playing = False

    monkeypatch.setattr(tts_mod.sd, "OutputStream", _FakeStream)
    monkeypatch.setattr(tts_mod.sd, "sleep", fake_sleep)

    engine = TTSEngine({"backend": "edge", "output_device": 3})
    try:
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert captured["device"] == 3


def test_auto_fallback_when_model_missing():
    """Backend auto-falls back to edge when model directory doesn't exist."""
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "local", "model_dir": "/nonexistent/path"})
    try:
        assert engine._backend == "edge"
    finally:
        engine.shutdown()
