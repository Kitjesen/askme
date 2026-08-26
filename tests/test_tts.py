"""Regression tests for the TTS engine (local + edge backends)."""

from __future__ import annotations

import queue
import threading
import time
import types

import pytest


def test_stop_playback_clears_stale_stop_request() -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._playback_lock = threading.Lock()
    engine._is_playing = False
    engine._playback_thread = None
    engine._stop_requested = threading.Event()
    engine._stop_requested.set()
    engine._kill_aplay = lambda: None
    engine._kill_usb_audio = lambda: None

    engine.stop_playback()

    assert engine._stop_requested.is_set() is False


def test_interrupted_tts_drops_late_streamed_text_until_next_playback() -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._discard_text_until_restart = threading.Event()
    engine._discard_text_until_restart.set()
    engine.tts_text_queue = queue.Queue()
    engine._generation = 4
    engine._generation_lock = threading.Lock()

    engine.speak("旧回答的后续分片")

    assert engine.tts_text_queue.empty()


def test_cloud_tts_coalesces_adjacent_response_fragments() -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._tts_text_coalesce_seconds = 0.05
    engine._tts_text_coalesce_max_chars = 80
    engine.tts_text_queue = queue.Queue()
    engine.tts_text_queue.put((7, "后半句。"))

    text, acknowledgements, stop_after_item = engine._coalesce_tts_text(
        7, "前半句，"
    )

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
        ok = engine._play_chunk_usb_direct_with_preroll(
            np.ones(160, dtype=np.float32) * 0.2
        )
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
        assert engine._play_chunk_usb_direct_locked(
            np.ones(480, dtype=np.float32) * 0.25
        )
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

    engine = TTSEngine({
        "backend": "edge",
        "sample_rate": 16000,
        "output_tail_silence_seconds": 0.25,
    })

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
        lambda *_args, **_kwargs: pytest.fail(
            "cached phrase must not call a TTS provider"
        ),
    )
    try:
        assert engine.queue_cached_phrase(text, cache_key="greeting") is True
        with engine._buffer_lock:
            queued = engine.tts_buffer.popleft()
    finally:
        engine.shutdown()

    assert len(queued) == 320
    assert np.allclose(queued, 0.1)


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
