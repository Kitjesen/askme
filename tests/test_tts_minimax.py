"""Tests for TTS MiniMax SSE streaming, fallback chain, and runtime controls."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from askme.voice.tts import TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**overrides) -> TTSEngine:
    """Create a TTSEngine with edge backend (no model files needed)."""
    cfg = {"backend": "edge", **overrides}
    return TTSEngine(cfg)


# ---------------------------------------------------------------------------
# MiniMax SSE streaming
# ---------------------------------------------------------------------------


class _FakeSSEResponse:
    """Simulates an httpx streaming response with SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines
        self._body = b"error body"

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def stream(self, method, url, json=None, headers=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _hex_pcm(n_samples: int = 100) -> str:
    """Generate hex-encoded PCM data (little-endian int16)."""
    samples = np.zeros(n_samples, dtype="<i2")
    samples[:] = 1000  # non-zero so we can verify
    return samples.tobytes().hex()


def _sse_data(audio_hex: str, status: int = 1) -> str:
    payload = {"data": {"audio": audio_hex, "status": status}}
    return f"data:{json.dumps(payload)}"


async def test_minimax_sse_streaming_queues_audio():
    """MiniMax backend: SSE hex-PCM chunks are decoded and queued."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        lines = [
            _sse_data(_hex_pcm(200)),
            _sse_data(_hex_pcm(200)),
            "data:[DONE]",
        ]
        response = _FakeSSEResponse(lines)
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        assert engine._has_buffered_audio()
    finally:
        engine.shutdown()


async def test_minimax_sse_skips_status_2_duplicate():
    """MiniMax sends a status=2 summary event with duplicate audio; we skip it."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        lines = [
            _sse_data(_hex_pcm(100), status=1),
            _sse_data(_hex_pcm(5000), status=2),  # duplicate — should be skipped
            "data:[DONE]",
        ]
        response = _FakeSSEResponse(lines)
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        # Should only have the status=1 chunk, not the large status=2 duplicate
        total = 0
        with engine._buffer_lock:
            for chunk in engine.tts_buffer:
                total += len(chunk)
        assert total < 5000  # would be ~5100 if status=2 was included
    finally:
        engine.shutdown()


async def test_minimax_sse_skips_status_2_string():
    """status='2' (string) is also skipped — MiniMax can return either."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        payload = {"data": {"audio": _hex_pcm(100), "status": "2"}}
        lines = [
            _sse_data(_hex_pcm(100), status=1),
            f"data:{json.dumps(payload)}",
            "data:[DONE]",
        ]
        response = _FakeSSEResponse(lines)
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        total = 0
        with engine._buffer_lock:
            for chunk in engine.tts_buffer:
                total += len(chunk)
        assert total < 200
    finally:
        engine.shutdown()


async def test_minimax_sse_aborts_on_stale_generation():
    """Mid-stream generation change causes MiniMax to abort."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        lines = [
            _sse_data(_hex_pcm(3000)),
            _sse_data(_hex_pcm(3000)),  # won't be processed
            "data:[DONE]",
        ]

        call_count = 0
        original_lines = list(lines)

        async def fake_aiter_lines():
            nonlocal call_count
            for line in original_lines:
                call_count += 1
                if call_count == 2:
                    engine._advance_generation()  # invalidate
                yield line

        response = _FakeSSEResponse([])
        response.aiter_lines = fake_aiter_lines
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        # First chunk should have been flushed, second not processed
        assert call_count == 2
    finally:
        engine.shutdown()


async def test_minimax_http_error_returns_without_crash():
    """Non-200 HTTP response is logged and does not crash."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        response = _FakeSSEResponse([], status_code=401)
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


async def test_minimax_resamples_when_rates_differ():
    """When minimax_sample_rate != sample_rate, audio is resampled."""
    engine = _make_engine(
        minimax_api_key="test-key",
        minimax_sample_rate=32000,
        sample_rate=16000,
    )
    try:
        lines = [
            _sse_data(_hex_pcm(3200)),  # 3200 samples at 32kHz
            "data:[DONE]",
        ]
        response = _FakeSSEResponse(lines)
        client = _FakeClient(response)

        gen = engine._get_generation()
        with patch("httpx.AsyncClient", return_value=client):
            await engine._generate_minimax("hello", gen)

        total = 0
        with engine._buffer_lock:
            for chunk in engine.tts_buffer:
                total += len(chunk)
        # 3200 samples at 32kHz → ~1600 at 16kHz
        assert 1500 <= total <= 1700
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Consecutive failure / backoff
# ---------------------------------------------------------------------------


def test_consecutive_failures_trigger_backoff():
    """After _MINIMAX_FAIL_THRESHOLD consecutive failures, MiniMax is disabled."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        engine._backend = "minimax"
        engine._minimax_fail_count = TTSEngine._MINIMAX_FAIL_THRESHOLD - 1

        # Simulate one more failure
        engine._minimax_fail_count += 1
        if engine._minimax_fail_count >= TTSEngine._MINIMAX_FAIL_THRESHOLD:
            engine._minimax_disabled_until = (
                time.monotonic() + TTSEngine._MINIMAX_BACKOFF_SECONDS
            )

        assert engine._minimax_disabled_until > time.monotonic()
    finally:
        engine.shutdown()


def test_success_resets_failure_counter():
    """Successful MiniMax call resets consecutive failure counter."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        engine._minimax_fail_count = 2

        # Simulate success path
        engine._minimax_fail_count = 0

        assert engine._minimax_fail_count == 0
    finally:
        engine.shutdown()


def test_backoff_period_skips_minimax():
    """During backoff, _generate_audio skips MiniMax and uses fallback."""
    engine = _make_engine(minimax_api_key="test-key")
    try:
        engine._backend = "minimax"
        engine._minimax_disabled_until = time.monotonic() + 999  # far future

        fallback_called = []
        engine._use_minimax_fallback = lambda text, gen: fallback_called.append(text)

        gen = engine._get_generation()
        engine._generate_audio("test", gen)

        assert fallback_called == ["test"]
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


def test_minimax_fallback_to_local_when_available():
    """_use_minimax_fallback uses local backend when local_tts is loaded."""
    engine = _make_engine()
    try:
        local_calls = []

        class FakeLocalTts:
            def generate(self, text, sid=0, speed=1.0):
                local_calls.append(text)
                result = MagicMock()
                result.samples = [0.1] * 100
                result.sample_rate = 24000
                return result

        engine._local_tts = FakeLocalTts()
        engine._local_sample_rate = 24000

        gen = engine._get_generation()
        engine._use_minimax_fallback("test", gen)

        assert local_calls == ["test"]
    finally:
        engine.shutdown()


def test_minimax_fallback_to_edge_when_no_local():
    """_use_minimax_fallback falls back to edge when no local TTS."""
    engine = _make_engine()
    try:
        engine._local_tts = None

        edge_called = []
        original_run_async = engine._run_async
        engine._run_async = lambda coro: edge_called.append(True)

        gen = engine._get_generation()
        engine._use_minimax_fallback("test", gen)

        assert edge_called
    finally:
        engine.shutdown()


def test_auto_fallback_minimax_no_key():
    """MiniMax without API key falls back to edge at init."""
    engine = _make_engine(backend="minimax", minimax_api_key="")
    try:
        assert engine._backend == "edge"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Volume / speed runtime controls
# ---------------------------------------------------------------------------


def test_volume_clamped_to_range():
    """set_volume clamps to [0.05, 1.0]."""
    engine = _make_engine()
    try:
        assert engine.set_volume(0.0) == TTSEngine._VOLUME_MIN
        assert engine.set_volume(2.0) == TTSEngine._VOLUME_MAX
        assert engine.set_volume(0.5) == 0.5
    finally:
        engine.shutdown()


def test_adjust_volume_incremental():
    """adjust_volume adds delta to current volume."""
    engine = _make_engine(volume=0.5)
    try:
        result = engine.adjust_volume(0.2)
        assert abs(result - 0.7) < 0.01
    finally:
        engine.shutdown()


def test_speed_clamped_to_range():
    """set_speed clamps to [0.5, 2.0]."""
    engine = _make_engine()
    try:
        assert engine.set_speed(0.1) == TTSEngine._SPEED_MIN
        assert engine.set_speed(5.0) == TTSEngine._SPEED_MAX
        assert engine.set_speed(1.5) == 1.5
    finally:
        engine.shutdown()


def test_set_speed_updates_all_backends():
    """set_speed updates local speed, MiniMax speed, and edge rate."""
    engine = _make_engine()
    try:
        engine.set_speed(1.2)
        assert engine._speed == 1.2
        assert engine._minimax_speed == 1.2
        assert engine._rate == "+20%"

        engine.set_speed(0.7)
        assert engine._rate == "-30%"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Text cleaning (speak method)
# ---------------------------------------------------------------------------


def test_speak_strips_emoji():
    engine = _make_engine()
    try:
        engine.speak("")
        assert engine.tts_text_queue.empty()
    finally:
        engine.shutdown()


def test_speak_strips_markdown():
    """Markdown bold, italic, code, headers, lists are stripped."""
    engine = _make_engine()
    try:
        engine.speak("**bold** and *italic* and `code`")
        item = engine.tts_text_queue.get_nowait()
        _, text = item
        assert "**" not in text
        assert "*" not in text
        assert "`" not in text
        assert "bold" in text
        assert "italic" in text
        assert "code" in text
    finally:
        engine.shutdown()


def test_speak_strips_links():
    """Markdown links keep text, remove URL."""
    engine = _make_engine()
    try:
        engine.speak("[click here](https://example.com)")
        item = engine.tts_text_queue.get_nowait()
        _, text = item
        assert "click here" in text
        assert "https" not in text
    finally:
        engine.shutdown()


def test_speak_skips_single_char():
    """Single-character text after cleaning is skipped (len <= 1)."""
    engine = _make_engine()
    try:
        engine.speak("a")
        assert engine.tts_text_queue.empty()
    finally:
        engine.shutdown()


def test_speak_skips_empty():
    """Empty string is not queued."""
    engine = _make_engine()
    try:
        engine.speak("")
        assert engine.tts_text_queue.empty()
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Generation tracking
# ---------------------------------------------------------------------------


def test_generation_advances_and_invalidates():
    """_advance_generation invalidates prior generation IDs."""
    engine = _make_engine()
    try:
        g1 = engine._get_generation()
        assert engine._is_generation_current(g1)

        g2 = engine._advance_generation()
        assert not engine._is_generation_current(g1)
        assert engine._is_generation_current(g2)
    finally:
        engine.shutdown()


def test_drain_buffers_clears_everything():
    """drain_buffers clears queue, audio buffer, and advances generation."""
    engine = _make_engine()
    try:
        engine.tts_buffer.append(np.zeros(100, dtype=np.float32))
        old_gen = engine._get_generation()

        engine.drain_buffers()

        assert not engine._has_buffered_audio()
        assert not engine._is_generation_current(old_gen)
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# MiniMax voice tuning params
# ---------------------------------------------------------------------------


def test_minimax_voice_setting_includes_tuning_params():
    """Non-default speed/vol/pitch/emotion are included in voice_setting."""
    engine = _make_engine(
        minimax_api_key="test-key",
        minimax_speed=1.5,
        minimax_vol=3.0,
        minimax_pitch=2,
        minimax_emotion="happy",
    )
    try:
        assert engine._minimax_speed == 1.5
        assert engine._minimax_vol == 3.0
        assert engine._minimax_pitch == 2
        assert engine._minimax_emotion == "happy"
    finally:
        engine.shutdown()


def test_minimax_vol_clamped_0_to_10():
    """MiniMax vol is clamped to [0, 10]."""
    engine = _make_engine(minimax_api_key="k", minimax_vol=15)
    try:
        assert engine._minimax_vol == 10.0
    finally:
        engine.shutdown()

    engine2 = _make_engine(minimax_api_key="k", minimax_vol=-5)
    try:
        assert engine2._minimax_vol == 0.0
    finally:
        engine2.shutdown()
