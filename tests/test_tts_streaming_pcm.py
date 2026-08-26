"""Safe streaming PCM enqueue behavior for TTSEngine."""

import numpy as np
import pytest

from askme.voice.output.tts import TTSEngine


class _WritableAudioProcess:
    class _Stdin:
        def write(self, payload: bytes) -> int:
            return len(payload)

        def flush(self) -> None:
            return None

    stdin = _Stdin()


def test_streaming_pcm_played_ms_follows_dac_clock_and_excludes_tail(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = [10.0]
    monkeypatch.setattr(
        "askme.voice.output.tts.time.monotonic",
        lambda: clock[0],
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_tail_silence_seconds": 0.1,
            "phrase_cache_enabled": False,
        }
    )
    generation = engine.begin_streaming_pcm()

    try:
        assert engine.queue_streaming_pcm(
            np.ones(100, dtype=np.float32),
            1_000,
            generation=generation,
            final=True,
        )
        rendered = np.empty((200, 1), dtype=np.float32)
        engine.play_audio_callback(
            rendered,
            200,
            {"currentTime": 5.0, "outputBufferDacTime": 5.2},
            None,
        )

        assert engine.streaming_pcm_played_ms(generation) == 0
        clock[0] = 10.25
        assert engine.streaming_pcm_played_ms(generation) == 50
        clock[0] = 11.0
        assert engine.streaming_pcm_played_ms(generation) == 100

        engine.prepare_turn()
        assert engine.streaming_pcm_played_ms(generation) == 0
    finally:
        engine.shutdown()


def test_streaming_pcm_aplay_clock_does_not_count_cold_start_preroll(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = [20.0]
    monkeypatch.setattr(
        "askme.voice.output.tts.time.monotonic",
        lambda: clock[0],
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_tail_silence_seconds": 0.0,
            "phrase_cache_enabled": False,
        }
    )
    generation = engine.begin_streaming_pcm()
    proc = _WritableAudioProcess()

    try:
        assert engine.queue_streaming_pcm(
            np.ones(100, dtype=np.float32),
            1_000,
            generation=generation,
        )
        preroll_start = engine._write_aplay_with_render_clock(
            proc,
            np.zeros(1_500, dtype=np.float32),
        )
        speech_start = engine._write_aplay_with_render_clock(
            proc,
            np.ones(100, dtype=np.float32),
        )
        engine._record_streaming_pcm_render(
            100,
            render_at=speech_start,
            window_reserved=True,
        )

        assert preroll_start == pytest.approx(20.0)
        assert speech_start == pytest.approx(21.5)
        clock[0] = 21.0
        assert engine.streaming_pcm_played_ms(generation) == 0
        clock[0] = 21.55
        assert engine.streaming_pcm_played_ms(generation) == 50
    finally:
        engine.shutdown()


def test_streaming_pcm_usb_progress_is_conservative_until_physical_drain(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = [30.0]
    monkeypatch.setattr(
        "askme.voice.output.tts.time.monotonic",
        lambda: clock[0],
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_transport": "usb_direct",
            "output_tail_silence_seconds": 0.0,
            "phrase_cache_enabled": False,
        }
    )
    generation = engine.begin_streaming_pcm()
    observed_during_write: list[int] = []

    def _play_usb(_chunk: np.ndarray) -> bool:
        clock[0] = 30.05
        observed_during_write.append(
            engine.streaming_pcm_played_ms(generation)
        )
        clock[0] = 30.2
        engine._is_playing = False
        return True

    monkeypatch.setattr(engine, "_should_use_usb_direct", lambda: True)
    monkeypatch.setattr(engine, "_collect_usb_direct_chunk", lambda chunk: chunk)
    monkeypatch.setattr(engine, "_play_chunk_usb_direct_speech", _play_usb)

    try:
        assert engine.queue_streaming_pcm(
            np.ones(100, dtype=np.float32),
            1_000,
            generation=generation,
        )
        engine._is_playing = True
        engine._playback_loop()

        assert observed_during_write == [0]
        assert engine.streaming_pcm_played_ms(generation) == 100
    finally:
        engine.shutdown()


def test_streaming_pcm_appends_tail_silence_only_when_final():
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_tail_silence_seconds": 0.1,
            "phrase_cache_enabled": False,
        }
    )
    generation = engine.begin_streaming_pcm()

    try:
        assert engine.queue_streaming_pcm(
            np.array([0.1, 0.2], dtype=np.float32),
            1_000,
            generation=generation,
        )
        assert engine.queue_streaming_pcm(
            np.array([0.3, 0.4], dtype=np.float32),
            1_000,
            generation=generation,
        )
        assert engine.status_snapshot()["buffered_samples"] == 4

        assert engine.queue_streaming_pcm(
            np.empty(0, dtype=np.float32),
            1_000,
            generation=generation,
            final=True,
        )
        assert engine.status_snapshot()["buffered_samples"] == 104

        rendered = np.empty((104, 1), dtype=np.float32)
        engine.play_audio_callback(rendered, 104, None, None)
    finally:
        engine.shutdown()

    assert np.allclose(rendered[:4, 0], [0.1, 0.2, 0.3, 0.4])
    assert np.count_nonzero(rendered[4:, 0]) == 0


def test_streaming_pcm_final_is_accepted_once_per_generation():
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_tail_silence_seconds": 0.1,
            "phrase_cache_enabled": False,
        }
    )
    generation = engine.begin_streaming_pcm()

    try:
        assert engine.queue_streaming_pcm(
            np.ones(4, dtype=np.float32),
            1_000,
            generation=generation,
            final=True,
        )
        buffered_after_final = engine.status_snapshot()["buffered_samples"]

        assert not engine.queue_streaming_pcm(
            np.empty(0, dtype=np.float32),
            1_000,
            generation=generation,
            final=True,
        )
        assert engine.status_snapshot()["buffered_samples"] == buffered_after_final
    finally:
        engine.shutdown()


def test_streaming_pcm_rejects_chunks_from_an_invalidated_generation():
    engine = TTSEngine(
        {
            "backend": "edge",
            "sample_rate": 1_000,
            "output_tail_silence_seconds": 0.1,
            "phrase_cache_enabled": False,
        }
    )
    stale_generation = engine.begin_streaming_pcm()

    try:
        engine.prepare_turn()

        assert not engine.queue_streaming_pcm(
            np.ones(4, dtype=np.float32),
            1_000,
            generation=stale_generation,
            final=True,
        )
        assert engine.status_snapshot()["buffered_samples"] == 0

        current_generation = engine.begin_streaming_pcm()
        assert current_generation != stale_generation
        assert engine.queue_streaming_pcm(
            np.ones(4, dtype=np.float32),
            1_000,
            generation=current_generation,
            final=True,
        )
    finally:
        engine.shutdown()
