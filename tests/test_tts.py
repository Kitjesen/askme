"""Regression tests for the TTS engine (local + edge backends)."""

from __future__ import annotations

import queue
import threading
import time
import types

import pytest


def test_cloud_tts_coalesces_adjacent_response_fragments() -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._tts_text_coalesce_seconds = 0.05
    engine._tts_text_coalesce_max_chars = 80
    engine.tts_text_queue = queue.Queue()
    engine.tts_text_queue.put((7, "后半句。"))

    text, acknowledgements, stop_after_item = engine._coalesce_tts_text(7, "前半句，")

    assert text == "前半句，后半句。"
    assert acknowledgements == 1
    assert stop_after_item is False
    engine.tts_text_queue.task_done()


def test_tts_rejects_internal_protocol_and_silent_markers() -> None:
    from askme.voice.tts import TTSEngine

    assert TTSEngine._is_speakable_text("正常中文回答") is True
    assert TTSEngine._is_speakable_text("[SILENT]") is False
    assert TTSEngine._is_speakable_text("<｜｜DSML｜｜tool_calls>") is False
    assert TTSEngine._is_speakable_text("<tool_call>internal</tool_call>") is False


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
    """_playback_loop passes the configured output_device to OutputStream."""
    import askme.voice.tts as tts_mod
    from askme.voice.tts import TTSEngine

    stream_kwargs: dict[str, object] = {}

    class FakeOutputStream:
        def __init__(self, **kwargs):
            stream_kwargs.update(kwargs)

        def __enter__(self):
            engine._is_playing = False
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(tts_mod.sd, "OutputStream", FakeOutputStream)

    engine = TTSEngine({"backend": "edge", "output_device": 3})
    engine._aplay_bin = None  # disable aplay so we exercise sounddevice streaming
    try:
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert stream_kwargs.get("device") == 3


def test_sounddevice_stream_open_failure_reports_transport_failure(monkeypatch):
    import askme.voice.tts as tts_mod
    from askme.voice.tts import TTSEngine

    failed = threading.Event()

    class FailingOutputStream:
        def __init__(self, **kwargs):
            del kwargs
            raise OSError("device or resource busy")

    monkeypatch.setattr(tts_mod.sd, "OutputStream", FailingOutputStream)
    engine = TTSEngine({"backend": "edge"})
    engine._aplay_bin = None
    engine.set_render_transport_failure_callback(lambda exc: failed.set())
    try:
        engine._is_playing = True
        engine._playback_loop()

        assert failed.wait(timeout=1.0)
        assert engine.status_snapshot()["render_reference"]["transport_failure_latched"] is True
    finally:
        engine.shutdown()


def test_playback_loop_uses_usb_direct_transport(monkeypatch):
    """output_transport=usb_direct bypasses aplay/sounddevice playback."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, int] = {}

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "usb_direct_preroll_seconds": 0.0,
        }
    )

    def fake_usb_play(chunk):
        played["samples"] = len(chunk)
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_speech", fake_usb_play)
    engine._aplay_bin = "aplay"
    try:
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert played["samples"] == 100


def test_wait_done_waits_for_usb_direct_chunk_after_buffer_pop(monkeypatch):
    """wait_done must not return while USB direct playback is in progress."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    started = threading.Event()
    finished = threading.Event()

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "usb_direct_preroll_seconds": 0.0,
            "usb_direct_coalesce_timeout": 0.1,
        }
    )

    def fake_usb_play(_chunk):
        started.set()
        time.sleep(0.15)
        finished.set()
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_speech", fake_usb_play)
    try:
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        engine.start_playback()
        assert started.wait(timeout=1.0)

        start = time.monotonic()
        engine.wait_done(timeout=2.0)
        elapsed = time.monotonic() - start
    finally:
        engine.shutdown()

    assert finished.is_set()
    assert elapsed >= 0.10


def test_wait_done_times_out_while_synthesis_queue_is_busy(monkeypatch):
    """A stuck TTS backend must not make one-shot CLI playback hang forever."""
    from askme.voice.tts import TTSEngine

    started = threading.Event()
    release = threading.Event()

    def fake_generate(self, _text, _generation):
        started.set()
        release.wait(timeout=1.0)

    engine = TTSEngine({"backend": "edge"})
    try:
        monkeypatch.setattr(engine, "_generate_audio", types.MethodType(fake_generate, engine))
        engine.speak("slow synthesis")
        assert started.wait(timeout=1.0)

        start = time.monotonic()
        done = engine.wait_done(timeout=0.05)
        elapsed = time.monotonic() - start
    finally:
        release.set()
        engine.shutdown()

    assert done is False
    assert elapsed < 0.3


def test_usb_direct_playback_coalesces_chunks_and_adds_preroll(monkeypatch):
    """MiniMax streamed chunks should become one continuous MCP01 USB play."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 16000,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_coalesce_timeout": 0.2,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        engine.tts_buffer.append(np.ones(160, dtype=np.float32) * 0.2)
        engine.tts_buffer.append(np.ones(160, dtype=np.float32) * 0.3)
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    chunk = played["chunk"]
    assert len(chunk) == 1600 + 320
    assert np.max(np.abs(chunk[:1600])) < 0.04
    assert np.allclose(chunk[1600:1760], 0.2)
    assert np.allclose(chunk[1760:], 0.3)


def test_usb_direct_background_prewarm_is_opt_in(monkeypatch):
    """A separate prewarm stream must not make the first speech stream clip."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}
    warm_calls: list[int] = []

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_coalesce_timeout": 0.05,
        }
    )

    def fake_warm(chunk):
        warm_calls.append(len(chunk))
        return True

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_warming", fake_warm)
    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        engine._is_playing = True
        playback = threading.Thread(target=engine._playback_loop)
        playback.start()
        time.sleep(0.06)
        with engine._buffer_lock:
            engine.tts_buffer.append(np.ones(10, dtype=np.float32) * 0.2)
        playback.join(timeout=2.0)
    finally:
        engine._is_playing = False
        engine.shutdown()

    assert not playback.is_alive()
    assert warm_calls == []
    chunk = played["chunk"]
    assert len(chunk) == 100 + 10
    assert np.max(np.abs(chunk[:100])) < 0.04
    assert np.allclose(chunk[100:], 0.2)


def test_usb_direct_background_prewarm_can_be_enabled(monkeypatch):
    """Legacy separate prewarm remains available for devices that need it."""
    from askme.voice.tts import TTSEngine

    warm_calls: list[int] = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_background_prewarm": True,
        }
    )

    def fake_warm(chunk):
        warm_calls.append(len(chunk))
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_warming", fake_warm)
    try:
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert warm_calls == [100]


def test_usb_direct_warm_playback_adds_stream_guard(monkeypatch):
    """Warm USB sessions still need a short guard for new stream start-up."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 16000,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_stream_guard_seconds": 0.02,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        engine._last_aplay_close = time.monotonic()
        ok = engine._play_chunk_usb_direct_with_preroll(np.ones(160, dtype=np.float32) * 0.2)
    finally:
        engine.shutdown()

    assert ok is True
    chunk = played["chunk"]
    assert len(chunk) == 320 + 160
    assert np.max(np.abs(chunk[:320])) < 0.04
    assert np.allclose(chunk[320:], 0.2)


def test_usb_direct_speech_leadin_ignores_feedback_warm_state(monkeypatch):
    """A prior feedback helper must not downgrade the next speech lead-in."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: list[np.ndarray] = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_speech_leadin_seconds": 0.1,
            "usb_direct_stream_guard_seconds": 0.02,
            "usb_direct_coalesce_timeout": 0.05,
        }
    )

    def fake_usb_play(chunk):
        played.append(chunk.copy())
        engine._last_aplay_close = time.monotonic()
        if len(played) >= 2:
            engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        assert engine.play_feedback_audio(np.ones(10, dtype=np.float32) * 0.1, 1000)
        engine.tts_buffer.append(np.ones(10, dtype=np.float32) * 0.2)
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    feedback, speech = played
    assert len(feedback) == 100 + 10
    assert len(speech) == 100 + 10
    assert np.max(np.abs(speech[:100])) < 0.04
    assert np.allclose(speech[100:], 0.2)


def test_usb_direct_speech_does_not_trust_live_stream_by_default(monkeypatch):
    """A live stdin stream alone still sends idle silence, so speech stays protected."""
    import time

    import numpy as np
    from askme.voice.tts import TTSEngine

    class FakeProc:
        def poll(self):
            return None

    played: dict[str, np.ndarray] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_persistent_stream": True,
            "usb_direct_speech_leadin_seconds": 1.0,
            "usb_direct_speech_warm_leadin_seconds": 0.12,
            "usb_direct_speech_wake_signal_seconds": 0.8,
            "usb_direct_speech_wake_signal_gain": 0.2,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        with engine._usb_audio_lock:
            engine._usb_audio_stream_proc = FakeProc()
            engine._usb_audio_stream_ready_at = time.monotonic()
        engine._last_aplay_close = time.monotonic()
        assert engine._play_chunk_usb_direct_speech(np.ones(10, dtype=np.float32) * 0.2)
    finally:
        engine.shutdown()

    chunk = played["chunk"]
    assert len(chunk) == 1000 + 10
    assert np.max(np.abs(chunk[:800])) > 0.15
    assert np.allclose(chunk[1000:], 0.2)


def test_usb_direct_speech_onset_cushion_protects_first_audible_samples():
    """USB direct speech can prepend a low-volume sacrificial first onset."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_onset_cushion_seconds": 0.2,
            "usb_direct_speech_onset_cushion_gain": 0.2,
            "usb_direct_speech_onset_gap_seconds": 0.05,
        }
    )
    try:
        chunk = np.concatenate(
            [
                np.zeros(100, dtype=np.float32),
                np.ones(300, dtype=np.float32) * 0.5,
            ]
        )
        cushion = engine._usb_direct_speech_onset_cushion_chunk(chunk)
    finally:
        engine.shutdown()

    assert len(cushion) == 200 + 50
    audible = np.flatnonzero(np.abs(cushion[:200]) > 0.001)
    assert audible[0] == 20
    assert np.max(np.abs(cushion[:200])) <= 0.11
    assert 0.0 < np.max(np.abs(cushion[200:])) < 0.03


def test_usb_direct_speech_path_inserts_cushion_between_leadin_and_speech(monkeypatch):
    """The real USB speech path sends lead-in, cushion, gap, then full speech."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_leadin_seconds": 0.1,
            "usb_direct_speech_onset_cushion_seconds": 0.2,
            "usb_direct_speech_onset_cushion_gain": 0.2,
            "usb_direct_speech_onset_gap_seconds": 0.05,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        speech = np.ones(300, dtype=np.float32) * 0.5
        assert engine._play_chunk_usb_direct_speech(speech)
    finally:
        engine.shutdown()

    chunk = played["chunk"]
    assert len(chunk) == 100 + 200 + 50 + 300
    assert np.max(np.abs(chunk[:100])) < 0.04
    assert np.max(np.abs(chunk[100:300])) <= 0.11
    assert 0.0 < np.max(np.abs(chunk[300:350])) < 0.03
    assert np.allclose(chunk[350:], 0.5)


def test_usb_direct_speech_gain_boosts_only_speech_body(monkeypatch):
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_leadin_seconds": 0.1,
            "usb_direct_speech_gain": 8.0,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        speech = np.ones(100, dtype=np.float32) * 0.1
        assert engine._play_chunk_usb_direct_speech(speech)
    finally:
        engine.shutdown()

    chunk = played["chunk"]
    assert np.max(np.abs(chunk[:100])) < 0.04
    assert np.all(chunk[100:] > 0.6)
    assert np.max(np.abs(chunk[100:])) <= 1.0


def test_usb_direct_speech_leadin_can_include_wake_signal():
    """A shaped wake signal can open MCP01's speaker gate before speech."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_leadin_seconds": 0.5,
            "usb_direct_speech_wake_signal_seconds": 0.2,
            "usb_direct_speech_wake_signal_gain": 0.1,
            "usb_direct_speech_wake_signal_hz": 50,
            "usb_direct_speech_wake_gap_seconds": 0.05,
        }
    )
    try:
        leadin = engine._usb_direct_speech_leadin_chunk()
    finally:
        engine.shutdown()

    assert len(leadin) == 500
    assert 0.07 <= np.max(np.abs(leadin[:200])) <= 0.11
    assert 0.0 < np.max(np.abs(leadin[200:250])) < 0.03
    assert 0.0 < np.max(np.abs(leadin[250:])) < 0.03


def test_usb_direct_speech_leadin_can_be_silent_when_wake_gains_are_zero():
    """Sunrise can keep the timing guard without an audible pre-speech artifact."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_leadin_seconds": 0.5,
            "usb_direct_speech_wake_signal_seconds": 0.2,
            "usb_direct_speech_wake_signal_gain": 0.0,
            "usb_direct_speech_wake_noise_gain": 0.0,
            "usb_direct_speech_wake_gap_seconds": 0.05,
        }
    )
    try:
        leadin = engine._usb_direct_speech_leadin_chunk()
    finally:
        engine.shutdown()

    assert len(leadin) == 500
    assert np.max(np.abs(leadin)) == 0.0


def test_usb_direct_speech_leadin_shortens_when_stream_is_warm():
    """A trusted warm stream can use a shorter active wake guard."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_speech_leadin_seconds": 1.0,
            "usb_direct_speech_warm_leadin_seconds": 0.12,
            "usb_direct_speech_wake_signal_seconds": 0.8,
            "usb_direct_speech_wake_signal_gain": 0.2,
        }
    )
    try:
        cold = engine._usb_direct_speech_leadin_chunk()
        warm = engine._usb_direct_speech_leadin_chunk(warm=True)
    finally:
        engine.shutdown()

    assert len(cold) == 1000
    assert len(warm) == 120
    assert np.max(np.abs(cold[:800])) > 0.15
    assert np.max(np.abs(warm)) > 0.15


def test_playback_loop_falls_back_to_usb_when_aplay_pipe_breaks(monkeypatch):
    """On Sunrise, ALSA can expose aplay but fail when no card exists."""
    import askme.voice.tts as tts_mod
    import numpy as np
    from askme.voice.tts import TTSEngine

    class _BrokenStdin:
        def write(self, _data):
            raise BrokenPipeError()

        def flush(self):
            pass

    class _BrokenProc:
        stdin = _BrokenStdin()

    played: dict[str, int] = {}

    def fake_popen(*_args, **_kwargs):
        return _BrokenProc()

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_device": "plughw:1,0",
            "output_transport": "auto",
            "usb_direct_preroll_seconds": 0.0,
        }
    )

    def fake_usb_play(chunk):
        played["samples"] = len(chunk)
        engine._is_playing = False
        return True

    monkeypatch.setattr(tts_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine, "_play_chunk_usb_direct_speech", fake_usb_play)
    engine._aplay_bin = "aplay"
    try:
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert played["samples"] == 100


def test_auto_transport_uses_usb_when_plughw_has_no_alsa_card(monkeypatch):
    """If ALSA reports no card, auto mode should not trust aplay."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, int] = {}

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_device": "plughw:1,0",
            "output_transport": "auto",
            "usb_direct_preroll_seconds": 0.0,
        }
    )

    def fake_usb_play(chunk):
        played["samples"] = len(chunk)
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_alsa_output_available", lambda: False)
    monkeypatch.setattr(engine, "_play_chunk_usb_direct_speech", fake_usb_play)
    engine._aplay_bin = "aplay"
    try:
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert played["samples"] == 100


def test_usb_direct_pcm_is_48k_stereo():
    """MCP01 direct USB helper expects 48 kHz stereo S16_LE PCM."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge", "sample_rate": 24000})
    try:
        pcm_bytes = engine._chunk_to_usb_stereo_pcm(np.ones(240, dtype=np.float32) * 0.5)
    finally:
        engine.shutdown()

    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert pcm.size == 480 * 2
    assert np.array_equal(pcm[0::2], pcm[1::2])


def test_usb_direct_persistent_stream_writes_pcm_without_one_shot(monkeypatch):
    """Persistent USB mode writes to the live helper instead of reopening USB."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    class _Stdin:
        def __init__(self):
            self.data = bytearray()
            self.flushes = 0

        def write(self, data):
            self.data.extend(bytes(data))

        def flush(self):
            self.flushes += 1

    class _Proc:
        def __init__(self):
            self.stdin = _Stdin()

        def poll(self):
            return None

    proc = _Proc()
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 48000,
            "usb_direct_persistent_stream": True,
            "usb_direct_stream_drain_grace_seconds": 0.0,
        }
    )
    monkeypatch.setattr(engine, "_start_usb_audio_stream_locked", lambda: proc)
    monkeypatch.setattr(
        engine,
        "_play_chunk_usb_direct_one_shot_locked",
        lambda _chunk: pytest.fail("one-shot playback should not run"),
    )
    try:
        assert engine._play_chunk_usb_direct_locked(np.ones(480, dtype=np.float32) * 0.25)
    finally:
        engine.shutdown()

    pcm = np.frombuffer(bytes(proc.stdin.data), dtype=np.int16)
    assert pcm.size == 480 * 2
    assert np.array_equal(pcm[0::2], pcm[1::2])
    assert proc.stdin.flushes > 0


def test_usb_direct_persistent_stream_falls_back_to_one_shot(monkeypatch):
    """If the persistent helper fails to start, playback still has a fallback."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, int] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 48000,
            "usb_direct_persistent_stream": True,
        }
    )
    monkeypatch.setattr(engine, "_start_usb_audio_stream_locked", lambda: None)

    def fake_one_shot(chunk):
        played["samples"] = len(chunk)
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_one_shot_locked", fake_one_shot)
    try:
        assert engine._play_chunk_usb_direct_locked(np.ones(480, dtype=np.float32))
    finally:
        engine.shutdown()

    assert played["samples"] == 480


def test_usb_one_shot_anchors_reference_after_first_device_write(monkeypatch):
    import io

    import askme.voice.tts as tts_mod
    import numpy as np
    from askme.voice.tts import TTSEngine

    order: list[str] = []

    class Stdin:
        def write(self, payload):
            order.append("write")
            return len(payload)

        def flush(self):
            order.append("flush")

    class Proc:
        def __init__(self):
            self.stdin = Stdin()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = 0

        def communicate(self, payload, timeout):
            del payload, timeout
            order.append("communicate")
            return b"ok", b""

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "output_transport": "usb_direct",
            "phrase_cache_enabled": False,
        }
    )
    monkeypatch.setattr(engine, "_ensure_usb_audio_binary", lambda: "helper")
    monkeypatch.setattr(tts_mod.subprocess, "Popen", lambda *args, **kwargs: Proc())
    monkeypatch.setattr(
        engine,
        "_publish_render_reference",
        lambda samples, **kwargs: order.append("publish"),
    )
    try:
        assert engine._play_chunk_usb_direct_one_shot_locked(np.ones(320, dtype=np.float32))
    finally:
        engine.shutdown()

    assert order[:3] == ["write", "flush", "publish"]


def test_usb_direct_persistent_playback_loop_can_prewarm_same_stream(monkeypatch):
    """Persistent prewarm starts the stream before speech chunks arrive."""
    from askme.voice.tts import TTSEngine

    start_calls: list[bool] = []
    warm_calls: list[int] = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 1000,
            "usb_direct_persistent_stream": True,
            "usb_direct_background_prewarm": True,
            "usb_direct_preroll_seconds": 0.1,
        }
    )

    monkeypatch.setattr(
        engine,
        "_start_usb_audio_stream_locked",
        lambda: start_calls.append(True) or object(),
    )

    def fake_warm(chunk):
        warm_calls.append(len(chunk))
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_warming", fake_warm)
    try:
        engine._is_playing = True
        engine._playback_loop()
    finally:
        engine.shutdown()

    assert start_calls
    assert warm_calls == [100]


def test_feedback_audio_uses_usb_direct_with_preroll(monkeypatch):
    """ACK/thinking chimes should use the same USB direct path as TTS."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: dict[str, np.ndarray] = {}
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 44100,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_stream_guard_seconds": 0.02,
        }
    )

    def fake_usb_play(chunk):
        played["chunk"] = chunk.copy()
        return True

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        ok = engine.play_feedback_audio(np.ones(441, dtype=np.float32) * 0.2, 44100)
    finally:
        engine.shutdown()

    assert ok is True
    assert len(played["chunk"]) == 4410 + 441
    assert not engine._usb_direct_warming.is_set()


def test_feedback_audio_clears_warming_when_router_fails():
    """A failed output-session claim must not leave the DAC marked warm."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    class _FailingRouter:
        def output_session(self):
            return self

        def __enter__(self):
            raise RuntimeError("router busy")

        def __exit__(self, *_args):
            return False

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 44100,
            "usb_direct_preroll_seconds": 0.1,
        },
        audio_router=_FailingRouter(),
    )
    try:
        with pytest.raises(RuntimeError, match="router busy"):
            engine.play_feedback_audio(np.ones(441, dtype=np.float32) * 0.2, 44100)
        assert not engine._usb_direct_warming.is_set()
    finally:
        engine.shutdown()


def test_feedback_audio_serializes_cold_preroll(monkeypatch):
    """Concurrent USB feedback calls should not both pay cold-DAC preroll."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    played: list[int] = []
    played_lock = threading.Lock()
    start = threading.Barrier(2)
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "usb_direct",
            "sample_rate": 44100,
            "usb_direct_preroll_seconds": 0.1,
            "usb_direct_stream_guard_seconds": 0.02,
        }
    )

    def fake_usb_play(chunk):
        with played_lock:
            played.append(len(chunk))
        time.sleep(0.05)
        engine._last_aplay_close = time.monotonic()
        return True

    def call_feedback():
        start.wait(timeout=1.0)
        assert engine.play_feedback_audio(np.ones(441, dtype=np.float32) * 0.2, 44100)

    monkeypatch.setattr(engine, "_play_chunk_usb_direct_locked", fake_usb_play)
    try:
        t1 = threading.Thread(target=call_feedback)
        t2 = threading.Thread(target=call_feedback)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
    finally:
        engine.shutdown()

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert played == [4410 + 441, 882 + 441]
    assert not engine._usb_direct_warming.is_set()


def test_feedback_audio_returns_false_when_usb_direct_inactive():
    """Non-USB transports keep the legacy chime playback path."""
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge", "output_transport": "sounddevice"})
    try:
        assert engine.play_feedback_audio(np.zeros(100, dtype=np.float32), 44100) is False
    finally:
        engine.shutdown()


def test_auto_fallback_when_model_missing():
    """Backend auto-falls back to edge when model directory doesn't exist."""
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "local", "model_dir": "/nonexistent/path"})
    try:
        assert engine._backend == "edge"
    finally:
        engine.shutdown()


def test_output_tail_silence_is_queued_for_current_generation():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16000,
            "output_tail_silence_seconds": 0.25,
        }
    )

    generation = engine._get_generation()
    engine._queue_output_tail_silence(generation)

    assert len(engine.tts_buffer) == 1
    tail = engine.tts_buffer.popleft()
    assert len(tail) == 4000
    assert np.count_nonzero(tail) == 0


def test_aplay_prebuffer_waits_only_while_synthesis_is_active():
    import queue
    import threading
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._sample_rate = 1000
    engine._aplay_start_buffer_seconds = 2.5
    engine._aplay_wait_for_synthesis_complete = False
    engine.tts_text_queue = queue.Queue()
    engine.tts_buffer = deque()
    engine._buffer_lock = threading.Lock()
    engine.tts_text_queue.put((1, "测试"))
    with engine._buffer_lock:
        engine.tts_buffer.append(np.zeros(2000, dtype=np.float32))
    assert engine._aplay_prebuffer_pending() is True
    with engine._buffer_lock:
        engine.tts_buffer.append(np.zeros(500, dtype=np.float32))
    assert engine._aplay_prebuffer_pending() is False
    engine.tts_text_queue.get_nowait()
    engine.tts_text_queue.task_done()


def test_aplay_complete_utterance_mode_waits_until_synthesis_finishes():
    import queue
    import threading
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._sample_rate = 1000
    engine._aplay_start_buffer_seconds = 0.0
    engine._aplay_wait_for_synthesis_complete = True
    engine.tts_text_queue = queue.Queue()
    engine.tts_buffer = deque([np.zeros(8000, dtype=np.float32)])
    engine._buffer_lock = threading.Lock()
    engine.tts_text_queue.put((1, "完整回复"))
    assert engine._aplay_prebuffer_pending() is True
    engine.tts_text_queue.get_nowait()
    engine.tts_text_queue.task_done()
    assert engine._aplay_prebuffer_pending() is False


def test_aplay_drain_uses_configured_timeout_without_killing():
    from askme.voice.tts import TTSEngine

    class FakeProcess:
        def __init__(self):
            self.timeout = None
            self.killed = False

        def wait(self, timeout):
            self.timeout = timeout
            return 0

        def kill(self):
            self.killed = True

    engine = object.__new__(TTSEngine)
    engine._aplay_drain_timeout_seconds = 30.0
    process = FakeProcess()

    assert engine._wait_for_aplay_drain(process) is True
    assert process.timeout == 30.0
    assert process.killed is False


def test_aplay_drain_timeout_reports_transport_failure() -> None:
    import subprocess

    from askme.voice.tts import TTSEngine

    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("aplay", timeout)

        def kill(self) -> None:
            self.killed = True

    engine = object.__new__(TTSEngine)
    engine._aplay_drain_timeout_seconds = 0.01
    engine._stop_requested = threading.Event()
    failures: list[str] = []
    engine.report_render_transport_failure = lambda reason, exc=None: failures.append(reason)
    process = FakeProcess()

    assert engine._wait_for_aplay_drain(process) is False
    assert process.killed is True
    assert failures == ["aplay_drain_timeout"]


def test_aplay_nonzero_exit_reports_transport_failure_unless_stop_is_intentional() -> None:
    from askme.voice.tts import TTSEngine

    class FakeProcess:
        def wait(self, timeout):
            del timeout
            return 7

        def kill(self) -> None:
            raise AssertionError("an exited process must not be killed")

    engine = object.__new__(TTSEngine)
    engine._aplay_drain_timeout_seconds = 1.0
    engine._stop_requested = threading.Event()
    failures: list[str] = []
    engine.report_render_transport_failure = lambda reason, exc=None: failures.append(reason)

    assert engine._wait_for_aplay_drain(FakeProcess()) is False
    assert failures == ["aplay_exit_7"]

    failures.clear()
    assert engine._wait_for_aplay_drain(FakeProcess(), intentional_stop=True) is False
    assert failures == []


def test_queue_cached_pcm_resamples_into_normal_playback_buffer():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16000,
            "output_tail_silence_seconds": 0.1,
        }
    )
    try:
        assert engine.queue_cached_pcm(
            np.ones(800, dtype=np.float32) * 0.2,
            8000,
            cache_key="greeting",
        )
        with engine._buffer_lock:
            speech, tail = list(engine.tts_buffer)
    finally:
        engine.shutdown()

    assert len(speech) == 1600
    assert np.allclose(speech, 0.2)
    assert len(tail) == 1600
    assert np.count_nonzero(tail) == 0


def test_local_generation_check_and_enqueue_are_atomic(monkeypatch):
    import numpy as np
    from askme.voice.tts import TTSEngine

    class _Audio:
        samples = np.ones(160, dtype=np.float32)

    class _LocalTts:
        def generate(self, text, sid=0, speed=1.0):
            return _Audio()

    engine = TTSEngine({"backend": "edge", "phrase_cache_enabled": False})
    engine._local_tts = _LocalTts()
    engine._local_sample_rate = engine._sample_rate
    generation = engine._get_generation()
    real_is_current = engine._is_generation_current
    checks = 0

    def invalidate_after_final_check(candidate: int) -> bool:
        nonlocal checks
        checks += 1
        current = real_is_current(candidate)
        if checks == 2 and current:
            engine.drain_buffers()
            return True
        return current

    monkeypatch.setattr(engine, "_is_generation_current", invalidate_after_final_check)
    try:
        engine._generate_local("old turn", generation)
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_queue_cached_pcm_rejects_invalid_audio():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge"})
    try:
        assert engine.queue_cached_pcm(np.empty(0, dtype=np.float32), 16000) is False
        assert engine.queue_cached_pcm(np.zeros((2, 2), dtype=np.float32), 16000) is False
        assert engine.queue_cached_pcm(np.asarray([np.nan], dtype=np.float32), 16000) is False
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_cached_phrase_queues_pcm_without_tts_provider(monkeypatch, tmp_path):
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16000,
            "output_tail_silence_seconds": 0.0,
            "phrase_cache_dir": str(tmp_path),
        }
    )
    text = "\u4f60\u597d\uff0c\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u60a8\uff1f"
    storage_key = engine._phrase_cache_storage_key(text, "greeting")
    assert engine._phrase_cache.put(
        storage_key,
        np.ones(320, dtype=np.float32) * 0.1,
        16000,
    )
    monkeypatch.setattr(
        engine,
        "_generate_audio",
        lambda *_args, **_kwargs: pytest.fail("cached phrase must not call a TTS provider"),
    )
    try:
        assert engine.queue_cached_phrase(text, cache_key="greeting") is True
        with engine._buffer_lock:
            queued = engine.tts_buffer.popleft()
    finally:
        engine.shutdown()

    assert len(queued) == 320
    assert np.allclose(queued, 0.1)


def test_cached_phrase_pcm_returns_a_defensive_copy_without_queueing(tmp_path):
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16000,
            "phrase_cache_dir": str(tmp_path),
        }
    )
    text = "收到，我来看看。"
    storage_key = engine._phrase_cache_storage_key(text, "feedback-waiting")
    source = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
    assert engine._phrase_cache.put(storage_key, source, 16000)
    try:
        first = engine.cached_phrase_pcm(
            text,
            cache_key="feedback-waiting",
            target_sample_rate=8000,
        )
        assert first is not None
        first_samples, first_rate = first
        first_samples[0] = 1.0

        second = engine.cached_phrase_pcm(
            text,
            cache_key="feedback-waiting",
            target_sample_rate=8000,
        )
        assert second is not None
        second_samples, second_rate = second

        assert first_rate == second_rate == 8000
        assert len(first_samples) == len(second_samples) == 80
        assert second_samples[0] != 1.0
        assert not engine._has_buffered_audio()
        assert engine.tts_text_queue.empty()
        assert (
            engine.cached_phrase_pcm(
                "not cached",
                cache_key="feedback-waiting",
            )
            is None
        )
    finally:
        engine.shutdown()


def test_drain_buffers_clears_cached_pcm_and_advances_generation():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge"})
    try:
        generation = engine._get_generation()
        assert engine.queue_cached_pcm(np.ones(32, dtype=np.float32), 24000)
        engine.drain_buffers()

        assert engine._get_generation() > generation
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_minimax_websocket_reuses_one_started_task_for_sequential_fragments(
    monkeypatch,
):
    import json
    import sys
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    pcm_hex = (np.ones(64, dtype="<i2") * 1200).tobytes().hex()
    create_calls: list[object] = []

    class FakeSocket:
        def __init__(self):
            self.connected = True
            self.events: list[dict[str, object]] = []
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                ]
            )

        def send(self, raw):
            event = json.loads(raw)
            self.events.append(event)
            if event["event"] == "task_continue":
                self.responses.append(
                    {
                        "data": {"audio": pcm_hex},
                        "is_final": True,
                        "base_resp": {"status_code": 0},
                    }
                )
            elif event["event"] == "task_finish":
                self.responses.append({"event": "task_finished", "base_resp": {"status_code": 0}})

        def recv(self):
            return json.dumps(self.responses.popleft())

        def abort(self):
            self.connected = False

        def close(self, *args, **kwargs):
            self.connected = False

    sockets = []

    def create_connection(*args, **kwargs):
        create_calls.append((args, kwargs))
        socket = FakeSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_audio_format": "pcm",
            "phrase_cache_enabled": False,
        }
    )
    generation = engine._get_generation()
    try:
        assert engine._run_async(engine._generate_minimax_websocket("first", generation))
        engine.prepare_turn()
        generation = engine._get_generation()
        assert engine._run_async(engine._generate_minimax_websocket("second", generation))

        socket = sockets[0]
        sent_events = [event["event"] for event in socket.events]
        assert len(create_calls) == 1
        assert sent_events.count("task_start") == 1
        assert sent_events.count("task_continue") == 2
        assert sent_events.count("task_finish") == 0

        with engine._minimax_ws_state_lock:
            engine._minimax_ws_last_used -= engine._minimax_ws_idle_timeout_seconds + 1.0
        assert engine._run_async(engine._generate_minimax_websocket("after idle", generation))
        assert len(create_calls) == 2
        assert [event["event"] for event in socket.events].count("task_finish") == 1
        assert [event["event"] for event in sockets[1].events].count("task_start") == 1
    finally:
        engine.shutdown()


def test_prewarm_provider_session_opens_and_reuses_minimax_websocket(monkeypatch):
    import json
    import sys
    from collections import deque

    from askme.voice.tts import TTSEngine

    sockets = []

    class FakeSocket:
        def __init__(self):
            self.connected = True
            self.events: list[dict[str, object]] = []
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                ]
            )

        def send(self, raw):
            self.events.append(json.loads(raw))

        def recv(self):
            return json.dumps(self.responses.popleft())

        def close(self, *args, **kwargs):
            self.connected = False

    def create_connection(*args, **kwargs):
        socket = FakeSocket()
        sockets.append((socket, kwargs))
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    try:
        opened = engine.prewarm_provider_session()
        reused = engine.prewarm_provider_session()

        assert opened["ok"] is True
        assert opened["status"] == "opened"
        assert reused["ok"] is True
        assert reused["status"] == "reused"
        assert len(sockets) == 1
        socket, kwargs = sockets[0]
        assert kwargs["timeout"] == 10
        assert [event["event"] for event in socket.events] == ["task_start"]
        assert opened["buffered_samples_delta"] == 0
        assert reused["buffered_samples_delta"] == 0
        assert not engine._has_buffered_audio()

        refreshed = engine.prewarm_provider_session(force_refresh=True)

        assert refreshed["ok"] is True
        assert refreshed["status"] == "refreshed"
        assert refreshed["reused"] is False
        assert refreshed["buffered_samples_delta"] == 0
        assert len(sockets) == 2
        refreshed_socket, refreshed_kwargs = sockets[1]
        assert refreshed_kwargs["timeout"] == 10
        assert socket.connected is False
        assert refreshed_socket.connected is True
        assert [event["event"] for event in socket.events] == [
            "task_start",
            "task_finish",
        ]
        assert [event["event"] for event in refreshed_socket.events] == ["task_start"]
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_prewarm_provider_session_skips_when_disabled_or_not_ready(monkeypatch):
    import sys

    from askme.voice.tts import TTSEngine

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=lambda *a, **kw: calls.append((a, kw))),
    )

    disabled = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": False,
            "phrase_cache_enabled": False,
        }
    )
    missing_key = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    sse = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "sse",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    try:
        assert disabled.prewarm_provider_session()["reason"] == "disabled"
        assert missing_key.prewarm_provider_session()["ok"] is False
        assert sse.prewarm_provider_session()["reason"] == "transport_not_websocket"
        assert calls == []
    finally:
        disabled.shutdown()
        missing_key.shutdown()
        sse.shutdown()


def test_prewarm_provider_session_skips_when_synthesis_lock_is_busy(monkeypatch):
    import sys

    from askme.voice.tts import TTSEngine

    calls: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(
            create_connection=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    try:
        assert engine._minimax_ws_use_lock.acquire(timeout=1.0)
        result = engine.prewarm_provider_session()

        assert result == {
            "ok": False,
            "status": "skipped",
            "reason": "synthesis_busy",
        }
        assert calls == []
        assert not engine._has_buffered_audio()
    finally:
        if engine._minimax_ws_use_lock.locked():
            engine._minimax_ws_use_lock.release()
        engine.shutdown()


def test_slow_prewarm_never_blocks_real_minimax_synthesis(monkeypatch):
    import json
    import sys
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    prewarm_handshake_started = threading.Event()
    release_prewarm = threading.Event()
    sockets: list[object] = []

    class PrewarmSocket:
        connected = True

        def __init__(self) -> None:
            self.recv_count = 0
            self.events: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.events.append(json.loads(raw))

        def recv(self) -> str:
            self.recv_count += 1
            if self.recv_count == 1:
                prewarm_handshake_started.set()
                assert release_prewarm.wait(timeout=2.0)
                return json.dumps({"event": "connected_success", "base_resp": {"status_code": 0}})
            return json.dumps({"event": "task_started", "base_resp": {"status_code": 0}})

        def close(self, *args, **kwargs) -> None:
            self.connected = False

    class LiveSocket:
        connected = True

        def __init__(self) -> None:
            pcm_hex = (np.ones(64, dtype="<i2") * 1200).tobytes().hex()
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                    {
                        "event": "task_continued",
                        "base_resp": {"status_code": 0},
                        "data": {"audio": pcm_hex},
                        "is_final": True,
                    },
                ]
            )
            self.events: list[dict[str, object]] = []

        def send(self, raw: str) -> None:
            self.events.append(json.loads(raw))

        def recv(self) -> str:
            return json.dumps(self.responses.popleft())

        def close(self, *args, **kwargs) -> None:
            self.connected = False

    def create_connection(*args, **kwargs):
        socket = PrewarmSocket() if not sockets else LiveSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    prewarm_result: dict[str, object] = {}
    synthesis_result: dict[str, object] = {}
    try:
        prewarm_thread = threading.Thread(
            target=lambda: prewarm_result.update(engine.prewarm_provider_session())
        )
        prewarm_thread.start()
        assert prewarm_handshake_started.wait(timeout=1.0)

        generation = engine._get_generation()
        synthesis_done = threading.Event()

        def synthesize() -> None:
            synthesis_result["ok"] = engine._run_async(
                engine._generate_minimax_websocket("真实首句。", generation)
            )
            synthesis_done.set()

        synthesis_thread = threading.Thread(target=synthesize)
        synthesis_thread.start()
        assert synthesis_done.wait(timeout=1.0), "real synthesis waited behind prewarm"
        assert synthesis_result["ok"] is True

        release_prewarm.set()
        prewarm_thread.join(timeout=1.0)
        synthesis_thread.join(timeout=1.0)

        assert prewarm_result["status"] == "superseded_by_live_session"
        assert len(sockets) == 2
        assert sockets[0].connected is False
        assert sockets[1].connected is True
        assert engine._has_buffered_audio()
    finally:
        release_prewarm.set()
        engine.shutdown()


def test_minimax_websocket_reconnects_once_before_falling_back(monkeypatch):
    import json
    import sys
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    pcm_hex = (np.ones(64, dtype="<i2") * 1200).tobytes().hex()
    sockets: list[FakeSocket] = []

    class FakeSocket:
        def __init__(self, *, fail_first_fragment: bool):
            self.connected = True
            self.fail_first_fragment = fail_first_fragment
            self.failed = False
            self.recv_count = 0
            self.events: list[dict[str, object]] = []
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                ]
            )

        def send(self, raw):
            event = json.loads(raw)
            self.events.append(event)
            if event["event"] == "task_continue":
                self.responses.append(
                    {
                        "data": {"audio": pcm_hex},
                        "is_final": True,
                        "base_resp": {"status_code": 0},
                    }
                )

        def recv(self):
            if self.fail_first_fragment and not self.failed and self.recv_count >= 2:
                self.failed = True
                raise OSError("connection reset")
            self.recv_count += 1
            return json.dumps(self.responses.popleft())

        def abort(self):
            self.connected = False

        def close(self, *args, **kwargs):
            self.connected = False

    def create_connection(*args, **kwargs):
        socket = FakeSocket(fail_first_fragment=not sockets)
        sockets.append(socket)
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_audio_format": "pcm",
            "phrase_cache_enabled": False,
        }
    )
    try:
        assert engine._run_async(
            engine._generate_minimax_websocket("retry me", engine._get_generation())
        )
        assert len(sockets) == 2
        assert (
            sum(event["event"] == "task_continue" for socket in sockets for event in socket.events)
            == 2
        )
    finally:
        engine.shutdown()


def test_drain_buffers_invalidates_warm_minimax_websocket(monkeypatch):
    import json
    import sys
    from collections import deque

    import numpy as np
    from askme.voice.tts import TTSEngine

    pcm_hex = (np.ones(64, dtype="<i2") * 1200).tobytes().hex()
    sockets = []

    class FakeSocket:
        def __init__(self):
            self.connected = True
            self.aborted = False
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                ]
            )

        def send(self, raw):
            if json.loads(raw)["event"] == "task_continue":
                self.responses.append(
                    {
                        "data": {"audio": pcm_hex},
                        "is_final": True,
                        "base_resp": {"status_code": 0},
                    }
                )

        def recv(self):
            return json.dumps(self.responses.popleft())

        def abort(self):
            self.aborted = True
            self.connected = False

        def close(self, *args, **kwargs):
            self.connected = False

    def create_connection(*args, **kwargs):
        socket = FakeSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_audio_format": "pcm",
            "phrase_cache_enabled": False,
        }
    )
    try:
        generation = engine._get_generation()
        assert engine._run_async(engine._generate_minimax_websocket("old turn", generation))

        engine.drain_buffers()

        assert sockets[0].aborted is True
        assert engine._run_async(
            engine._generate_minimax_websocket("new turn", engine._get_generation())
        )
        assert len(sockets) == 2
    finally:
        engine.shutdown()


def test_stale_generation_cannot_commit_decoded_minimax_audio():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge", "phrase_cache_enabled": False})
    pending = []
    state = {"pending_len": 0, "queued_samples": 0, "first_flush": True}
    try:
        stale_generation = engine._get_generation()
        engine.drain_buffers()

        assert not engine._commit_minimax_samples_for_generation(
            stale_generation,
            pending,
            state,
            samples=np.ones(engine._MINIMAX_MIN_STREAM_SAMPLES, dtype=np.float32),
            flush=True,
        )
        assert not engine._has_buffered_audio()
        assert pending == []
    finally:
        engine.shutdown()


def test_minimax_stream_thresholds_are_playback_rate_milliseconds():
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "phrase_cache_enabled": False,
            "minimax_stream_first_chunk_ms": 36,
            "minimax_stream_later_chunk_ms": 54,
        }
    )
    try:
        assert engine._minimax_stream_chunk_samples(first=True) == 36
        assert engine._minimax_stream_chunk_samples(first=False) == 54
    finally:
        engine.shutdown()


def test_minimax_stream_default_remains_legacy_2400_samples():
    from askme.voice.tts import TTSEngine

    for sample_rate in (24_000, 44_100):
        engine = TTSEngine(
            {
                "backend": "edge",
                "sample_rate": sample_rate,
                "phrase_cache_enabled": False,
            }
        )
        try:
            assert engine._minimax_stream_chunk_samples(first=True) == 2_400
            assert engine._minimax_stream_chunk_samples(first=False) == 2_400
        finally:
            engine.shutdown()


def test_minimax_stream_first_and_later_thresholds_share_state_machine():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "phrase_cache_enabled": False,
            "minimax_stream_first_chunk_ms": 36,
            "minimax_stream_later_chunk_ms": 54,
            "minimax_leading_silence_preserve_seconds": 0.01,
        }
    )
    pending: list[np.ndarray] = []
    state = engine._new_minimax_stream_state()
    try:
        engine._queue_minimax_samples(
            np.ones(36, dtype=np.float32) * 0.2,
            pending,
            state,
        )
        with engine._buffer_lock:
            assert [len(chunk) for chunk in engine.tts_buffer] == [36]

        engine._queue_minimax_samples(
            np.ones(53, dtype=np.float32) * 0.2,
            pending,
            state,
        )
        with engine._buffer_lock:
            assert [len(chunk) for chunk in engine.tts_buffer] == [36]
        engine._queue_minimax_samples(
            np.ones(1, dtype=np.float32) * 0.2,
            pending,
            state,
        )
        with engine._buffer_lock:
            assert [len(chunk) for chunk in engine.tts_buffer] == [36, 54]
    finally:
        engine.shutdown()


def test_minimax_stream_final_flushes_below_threshold():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "phrase_cache_enabled": False,
            "minimax_stream_first_chunk_ms": 36,
        }
    )
    generation = engine._get_generation()
    pending: list[np.ndarray] = []
    state = engine._new_minimax_stream_state()
    try:
        assert engine._commit_minimax_samples_for_generation(
            generation,
            pending,
            state,
            samples=np.ones(12, dtype=np.float32) * 0.2,
            flush=True,
        )
        with engine._buffer_lock:
            assert [len(chunk) for chunk in engine.tts_buffer] == [12]
    finally:
        engine.shutdown()


def test_minimax_stream_does_not_flush_silence_before_first_onset():
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "phrase_cache_enabled": False,
            "minimax_stream_first_chunk_ms": 36,
            "minimax_leading_silence_preserve_seconds": 0.01,
            "minimax_onset_threshold": 0.01,
        }
    )
    pending: list[np.ndarray] = []
    state = engine._new_minimax_stream_state()
    try:
        engine._queue_minimax_samples(
            np.zeros(36, dtype=np.float32),
            pending,
            state,
        )
        assert pending
        with engine._buffer_lock:
            assert not engine.tts_buffer

        engine._queue_minimax_samples(
            np.ones(8, dtype=np.float32) * 0.2,
            pending,
            state,
        )
        with engine._buffer_lock:
            [chunk] = list(engine.tts_buffer)
        assert len(chunk) == 18
        assert np.allclose(chunk[:10], 0.0)
        assert np.allclose(chunk[10:], 0.2)
    finally:
        engine.shutdown()


def test_phrase_cache_signature_covers_backend_acoustic_settings(tmp_path):
    from askme.voice.tts import TTSEngine

    base = {
        "backend": "edge",
        "voice": "zh-CN-YunjianNeural",
        "rate": "+0%",
        "sample_rate": 24_000,
        "phrase_cache_dir": str(tmp_path),
    }
    first = TTSEngine(base)
    changed = TTSEngine({**base, "voice": "zh-CN-YunxiNeural"})
    try:
        key = first._phrase_cache_storage_key("好的。", "quick-stable")
        assert key.startswith("quick-stable-v2-")
        assert key != changed._phrase_cache_storage_key("好的。", "quick-stable")

        first._backend = "minimax"
        minimax_key = first._phrase_cache_storage_key("好的。", "quick-stable")
        first._minimax_vol = 2.0
        assert minimax_key != first._phrase_cache_storage_key("好的。", "quick-stable")
    finally:
        first.shutdown()
        changed.shutdown()


def test_phrase_prime_refuses_to_cache_fallback_audio(monkeypatch, tmp_path):
    import numpy as np
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "phrase_cache_dir": str(tmp_path),
        }
    )
    monkeypatch.setattr(
        engine,
        "_generate_audio",
        lambda _text, generation: (
            engine._append_audio_for_generation(
                generation,
                np.ones(20, dtype=np.float32) * 0.2,
            )
            and "local"
        ),
    )
    try:
        result = engine.prime_cached_phrase("好的。", cache_key="quick-stable")
        assert result["cached"] is False
        assert result["reason"] == "backend_fallback_not_cached"
        assert not list(tmp_path.glob("*.npz"))
    finally:
        engine.shutdown()


def test_render_reference_callback_is_async_and_receives_final_float32_chunk():
    import numpy as np
    from askme.voice.tts import TTSEngine

    received = []
    delivered = threading.Event()

    def callback(samples, sample_rate):
        time.sleep(0.15)
        received.append((samples, sample_rate))
        delivered.set()

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16000,
            "volume": 0.5,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback)
    out = np.zeros((4, 1), dtype=np.float32)
    with engine._buffer_lock:
        engine.tts_buffer.append(np.ones(4, dtype=np.float32))
    try:
        started = time.monotonic()
        engine.play_audio_callback(out, 4, None, None)
        elapsed = time.monotonic() - started

        assert elapsed < 0.1
        assert delivered.wait(timeout=1.0)
        samples, sample_rate = received[0]
        assert samples.dtype == np.float32
        assert np.allclose(samples, 0.5)
        assert sample_rate == 16000
    finally:
        engine.shutdown()


def test_sounddevice_callback_schedules_reference_from_dac_time():
    import numpy as np
    from askme.voice.tts import TTSEngine

    captured = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine._publish_render_reference = lambda chunk, **kwargs: captured.append(
        (np.asarray(chunk).copy(), kwargs)
    )
    out = np.zeros((160, 1), dtype=np.float32)
    with engine._buffer_lock:
        engine.tts_buffer.append(np.ones(160, dtype=np.float32))
    try:
        before = time.monotonic()
        engine.play_audio_callback(
            out,
            160,
            {"currentTime": 12.0, "outputBufferDacTime": 12.05},
            None,
        )
        after = time.monotonic()

        assert len(captured) == 1
        render_at = captured[0][1]["render_at"]
        assert before + 0.045 <= render_at <= after + 0.055
    finally:
        engine.shutdown()


def test_non_usb_feedback_reference_waits_for_physical_playback_signal():
    import numpy as np
    from askme.voice.tts import TTSEngine

    received = []
    delivered = threading.Event()

    def callback(samples, sample_rate):
        received.append((samples, sample_rate))
        delivered.set()

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "sounddevice",
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback)
    feedback = np.linspace(-0.2, 0.2, 441, dtype=np.float32)
    try:
        assert engine.play_feedback_audio(feedback, 44100) is False
        assert not delivered.wait(timeout=0.05)
        engine.publish_feedback_render_reference(feedback, 44100)
        assert delivered.wait(timeout=1.0)
        samples, sample_rate = received[0]
        assert sample_rate == 44100
        assert np.allclose(samples, feedback)
    finally:
        engine.shutdown()


def test_render_reference_is_paced_in_ten_millisecond_frames():
    import numpy as np
    from askme.voice.tts import TTSEngine

    received: list[tuple[float, int]] = []
    delivered = threading.Event()

    def callback(samples, sample_rate):
        assert sample_rate == 16_000
        received.append((time.monotonic(), len(samples)))
        if sum(length for _, length in received) >= 320:
            delivered.set()

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback)
    try:
        engine._publish_render_reference(np.ones(320, dtype=np.float32))

        assert delivered.wait(timeout=1.0)
        assert [length for _, length in received] == [160, 160]
        assert received[1][0] - received[0][0] >= 0.005
        assert engine.status_snapshot()["render_reference"]["delivered_frames"] == 2
    finally:
        engine.shutdown()


def test_render_reference_queue_overflow_reports_failure_and_unhealthy_status():
    import numpy as np
    from askme.voice.tts import TTSEngine

    callback_entered = threading.Event()
    release_callback = threading.Event()
    failed = threading.Event()

    def callback(samples, sample_rate):
        del samples, sample_rate
        callback_entered.set()
        release_callback.wait(timeout=1.0)

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "render_reference_queue_size": 1,
            "render_reference_max_lag_ms": 1_000,
            "phrase_cache_enabled": False,
        }
    )

    def slow_failure_callback(exc):
        del exc
        time.sleep(0.25)
        failed.set()

    engine.set_render_reference_callback(callback, on_failure=slow_failure_callback)
    frame = np.ones(160, dtype=np.float32)
    try:
        engine._publish_render_reference(frame)
        assert callback_entered.wait(timeout=1.0)
        engine._publish_render_reference(frame)
        started = time.monotonic()
        engine._publish_render_reference(frame)
        publish_elapsed = time.monotonic() - started

        assert failed.wait(timeout=1.0)
        assert publish_elapsed < 0.1
        status = engine.status_snapshot()["render_reference"]
        assert status["healthy"] is False
        assert status["dropped_items"] == 1
    finally:
        release_callback.set()
        engine.shutdown()


def test_render_transport_failure_invalidates_timeline_and_requests_downgrade():
    from askme.voice.tts import TTSEngine

    failed = threading.Event()
    resets: list[str] = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(
        lambda samples, rate: None,
        on_failure=lambda exc: failed.set(),
        on_reset=lambda: resets.append("reset"),
    )
    before_epoch = engine.status_snapshot()["render_reference"]["epoch"]
    try:
        engine.report_render_transport_failure("speaker disconnected")

        assert failed.wait(timeout=1.0)
        status = engine.status_snapshot()["render_reference"]
        assert status["epoch"] > before_epoch
        assert status["healthy"] is False
        assert status["last_reset_reason"] == "transport_failure"
        assert resets == ["reset"]
    finally:
        engine.shutdown()


def test_render_transport_failure_prefers_dedicated_handler_over_native_aec() -> None:
    from askme.voice.tts import TTSEngine

    transport_failed = threading.Event()
    transport_failures: list[str] = []
    aec_failures: list[str] = []
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(
        lambda samples, rate: None,
        on_failure=lambda exc: aec_failures.append(str(exc)),
    )

    def on_transport_failure(exc: BaseException) -> None:
        transport_failures.append(str(exc))
        transport_failed.set()

    engine.set_render_transport_failure_callback(on_transport_failure)
    try:
        engine.report_render_transport_failure("speaker disconnected")
        engine.report_render_transport_failure("duplicate")

        assert transport_failed.wait(timeout=1.0)
        assert transport_failures == ["TTS render transport failed: speaker disconnected"]
        assert aec_failures == []
        status = engine.status_snapshot()["render_reference"]
        assert status["last_reset_reason"] == "transport_failure"
        assert status["transport_failure_latched"] is True
    finally:
        engine.shutdown()


def test_drain_invalidates_queued_render_reference_and_resets_aec_timeline():
    import numpy as np
    from askme.voice.tts import TTSEngine

    callback_entered = threading.Event()
    release_callback = threading.Event()
    resets: list[str] = []
    delivered_frames = 0

    def callback(samples, sample_rate):
        nonlocal delivered_frames
        del samples, sample_rate
        delivered_frames += 1
        callback_entered.set()
        release_callback.wait(timeout=1.0)

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "render_reference_max_lag_ms": 1_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback, on_reset=lambda: resets.append("reset"))
    try:
        engine._publish_render_reference(np.ones(800, dtype=np.float32))
        assert callback_entered.wait(timeout=1.0)

        engine.drain_buffers()
        release_callback.set()
        deadline = time.monotonic() + 1.0
        while (
            engine.status_snapshot()["render_reference"]["stale_items"] == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        assert delivered_frames == 1
        assert resets == ["reset"]
        status = engine.status_snapshot()["render_reference"]
        assert status["last_reset_reason"] == "turn_aborted"
        assert status["stale_items"] >= 1
    finally:
        release_callback.set()
        engine.shutdown()


def test_render_reference_reset_is_ordered_after_inflight_delivery():
    import numpy as np
    from askme.voice.tts import TTSEngine

    callback_entered = threading.Event()
    release_callback = threading.Event()
    reset_finished = threading.Event()
    order: list[str] = []

    def callback(samples, sample_rate):
        del samples, sample_rate
        callback_entered.set()
        release_callback.wait(timeout=1.0)
        order.append("old_feed")

    def reset():
        order.append("reset")
        reset_finished.set()

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "render_reference_max_lag_ms": 1_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback, on_reset=reset)
    try:
        engine._publish_render_reference(np.ones(160, dtype=np.float32))
        assert callback_entered.wait(timeout=1.0)

        engine.drain_buffers()
        assert not reset_finished.is_set()
        release_callback.set()

        assert reset_finished.wait(timeout=1.0)
        assert order == ["old_feed", "reset"]
    finally:
        release_callback.set()
        engine.shutdown()


def test_callback_change_cannot_discard_new_epoch_publish():
    import numpy as np
    from askme.voice.tts import TTSEngine

    setter_in_discard = threading.Event()
    release_setter = threading.Event()
    delivered = threading.Event()
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(lambda samples, rate: None)
    original_discard = engine._discard_render_reference_queue_locked

    def blocking_discard():
        setter_in_discard.set()
        release_setter.wait(timeout=1.0)
        return original_discard()

    engine._discard_render_reference_queue_locked = blocking_discard
    setter_thread = threading.Thread(
        target=lambda: engine.set_render_reference_callback(lambda samples, rate: delivered.set())
    )
    publisher_thread = threading.Thread(
        target=lambda: engine._publish_render_reference(np.ones(160, dtype=np.float32))
    )
    try:
        setter_thread.start()
        assert setter_in_discard.wait(timeout=1.0)
        publisher_thread.start()
        release_setter.set()
        setter_thread.join(timeout=1.0)
        publisher_thread.join(timeout=1.0)

        assert delivered.wait(timeout=1.0)
    finally:
        release_setter.set()
        engine.shutdown()


def test_shutdown_marks_reference_unhealthy_and_consumes_exit_sentinel():
    import numpy as np
    from askme.voice.tts import TTSEngine

    callback_entered = threading.Event()
    release_callback = threading.Event()

    def callback(samples, sample_rate):
        del samples, sample_rate
        callback_entered.set()
        release_callback.wait(timeout=3.0)

    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 16_000,
            "phrase_cache_enabled": False,
        }
    )
    engine.set_render_reference_callback(callback)
    engine._publish_render_reference(np.ones(160, dtype=np.float32))
    assert callback_entered.wait(timeout=1.0)

    shutdown_thread = threading.Thread(target=engine.shutdown)
    shutdown_thread.start()
    shutdown_thread.join(timeout=2.0)
    try:
        assert not shutdown_thread.is_alive()
        assert engine.status_snapshot()["render_reference"]["healthy"] is False

        release_callback.set()
        deadline = time.monotonic() + 1.0
        while engine._render_reference_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not engine._render_reference_thread.is_alive()
        assert engine._render_reference_queue.unfinished_tasks == 0
    finally:
        release_callback.set()


def test_force_refresh_keeps_live_minimax_socket_when_real_use_wins(monkeypatch):
    import json
    import sys
    from collections import deque

    from askme.voice.tts import TTSEngine

    refresh_started = threading.Event()
    release_refresh = threading.Event()
    sockets = []

    class FakeSocket:
        def __init__(self, *, block_connect):
            self.block_connect = block_connect
            self.connected = True
            self.events = []
            self.responses = deque(
                [
                    {"event": "connected_success", "base_resp": {"status_code": 0}},
                    {"event": "task_started", "base_resp": {"status_code": 0}},
                ]
            )

        def send(self, raw):
            self.events.append(json.loads(raw))

        def recv(self):
            if self.block_connect and len(self.responses) == 2:
                refresh_started.set()
                assert release_refresh.wait(timeout=2.0)
            return json.dumps(self.responses.popleft())

        def close(self, *args, **kwargs):
            self.connected = False

    def create_connection(*args, **kwargs):
        socket = FakeSocket(block_connect=bool(sockets))
        sockets.append(socket)
        return socket

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        types.SimpleNamespace(create_connection=create_connection),
    )
    engine = TTSEngine(
        {
            "backend": "minimax",
            "minimax_api_key": "test-key",
            "minimax_tts_transport": "websocket",
            "minimax_live_session_prewarm_enabled": True,
            "phrase_cache_enabled": False,
        }
    )
    refresh_result = {}
    refresh_thread = threading.Thread(
        target=lambda: refresh_result.update(engine.prewarm_provider_session(force_refresh=True))
    )
    try:
        assert engine.prewarm_provider_session()["status"] == "opened"
        live_socket = sockets[0]

        refresh_thread.start()
        assert refresh_started.wait(timeout=1.0)
        with engine._minimax_ws_use_lock:
            engine._mark_minimax_websocket_used(live_socket)

        release_refresh.set()
        refresh_thread.join(timeout=2.0)

        assert not refresh_thread.is_alive()
        assert refresh_result["status"] == "superseded_by_live_session"
        assert engine._minimax_ws_connection is live_socket
        assert live_socket.connected is True
        assert sockets[1].connected is False
    finally:
        release_refresh.set()
        refresh_thread.join(timeout=2.0)
        engine.shutdown()


def test_default_provider_idle_windows_outlive_warm_refresh_interval() -> None:
    from askme.voice.tts import TTSEngine

    engine = TTSEngine({"backend": "edge", "phrase_cache_enabled": False})
    try:
        assert engine._minimax_ws_idle_timeout_seconds == 90.0
        assert engine._volcengine_tts_idle_timeout_seconds == 90.0
    finally:
        engine.shutdown()
