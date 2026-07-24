"""Integration contracts between TTSEngine and the resident output worker."""

from __future__ import annotations

import threading

import numpy as np


class _RecordingAdapter:
    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.drop_count = 0
        self.writes: list[bytes] = []
        self.first_write = threading.Event()
        self.block_writes = False
        self.release_write = threading.Event()

    def open(self) -> None:
        self.open_count += 1

    def write(self, pcm: bytes) -> None:
        self.writes.append(bytes(pcm))
        self.first_write.set()
        if self.block_writes:
            self.release_write.wait(timeout=1.0)

    def drop(self) -> None:
        self.drop_count += 1
        self.release_write.set()

    def close(self) -> None:
        self.close_count += 1


def _build_engine(adapter: _RecordingAdapter):
    from askme.voice.tts import TTSEngine

    return TTSEngine(
        {
            "backend": "edge",
            "output_transport": "aplay",
            "resident_output_enabled": True,
            "resident_output_sample_rate": 48_000,
            "resident_output_channels": 2,
            "resident_output_period_ms": 10,
            "resident_output_buffer_ms": 80,
            "resident_output_idle_keepalive": True,
            "resident_output_warm_hold_seconds": 60.0,
            "resident_output_full_duplex_verified": True,
            "resident_output_cold_preroll_ms": 0,
            "resident_output_warm_leadin_ms": 0,
        },
        audio_output_adapter=adapter,
    )


def test_tts_reuses_one_resident_stream_across_playback_lifecycles() -> None:
    adapter = _RecordingAdapter()
    engine = _build_engine(adapter)
    try:
        for amplitude in (0.1, 0.2):
            with engine._buffer_lock:
                engine.tts_buffer.append(
                    np.full(960, amplitude, dtype=np.float32)
                )
            engine.start_playback()
            assert engine.wait_done(timeout=1.0)
            engine.stop_playback()

        status = engine.status_snapshot()
        assert status["resident_output"]["enabled"] is True
        assert status["resident_output"]["stream_open"] is True
    finally:
        engine.shutdown()

    assert adapter.open_count == 1
    assert adapter.close_count == 1
    assert len(adapter.writes) >= 4


def test_tts_drain_cancels_resident_generation_and_next_turn_recovers() -> None:
    adapter = _RecordingAdapter()
    adapter.block_writes = True
    engine = _build_engine(adapter)
    try:
        with engine._buffer_lock:
            engine.tts_buffer.append(np.full(48_000, 0.1, dtype=np.float32))
        engine.start_playback()
        assert adapter.first_write.wait(timeout=1.0)

        engine.drain_buffers()
        engine.stop_playback()

        adapter.block_writes = False
        with engine._buffer_lock:
            engine.tts_buffer.append(np.full(960, 0.2, dtype=np.float32))
        engine.start_playback()
        assert engine.wait_done(timeout=1.0)
        engine.stop_playback()
    finally:
        engine.shutdown()

    assert adapter.drop_count >= 1
    assert adapter.open_count >= 2
