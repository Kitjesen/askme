"""Lossless, generation-scoped playback holds for ``TTSEngine``."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from askme.voice.output.tts import PlaybackHoldToken, TTSEngine


class _CallbackOutputStream:
    def __init__(
        self,
        *,
        entered: threading.Event,
        capture: list[Callable[..., None]],
        **kwargs: Any,
    ) -> None:
        self._entered = entered
        capture.append(kwargs["callback"])

    def __enter__(self) -> _CallbackOutputStream:
        self._entered.set()
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _start_callback_engine(
    monkeypatch: pytest.MonkeyPatch,
    **config: object,
) -> tuple[TTSEngine, Callable[..., None]]:
    entered = threading.Event()
    callbacks: list[Callable[..., None]] = []
    monkeypatch.setattr(
        "askme.voice.output.tts.sd.OutputStream",
        lambda **kwargs: _CallbackOutputStream(
            entered=entered,
            capture=callbacks,
            **kwargs,
        ),
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "sounddevice",
            "phrase_cache_enabled": False,
            **config,
        }
    )
    engine.start_playback()
    assert entered.wait(timeout=1.0)
    assert _wait_until(
        lambda: bool(engine.status_snapshot()["playback_hold"]["supported"])
    )
    return engine, callbacks[0]


def _acquire_hold(
    engine: TTSEngine,
    callback: Callable[..., None],
    *,
    frames: int = 1,
) -> PlaybackHoldToken:
    result: list[PlaybackHoldToken | None] = []
    pause_thread = threading.Thread(
        target=lambda: result.append(engine.pause_playback(timeout_s=1.0))
    )
    pause_thread.start()
    assert _wait_until(
        lambda: bool(engine.status_snapshot()["playback_hold"]["active"])
    )
    callback(np.empty((frames, 1), dtype=np.float32), frames, None, None)
    pause_thread.join(timeout=1.0)
    assert not pause_thread.is_alive()
    assert len(result) == 1
    token = result[0]
    assert token is not None
    return token


def test_hold_silences_callback_and_resume_plays_remainder_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)

    try:
        assert engine.queue_cached_pcm(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            1_000,
        )
        initial: np.ndarray = np.empty((2, 1), dtype=np.float32)
        callback(initial, 2, None, None)
        assert np.allclose(initial[:, 0], [0.1, 0.2])

        result: list[PlaybackHoldToken | None] = []
        pause_thread = threading.Thread(
            target=lambda: result.append(engine.pause_playback(timeout_s=1.0))
        )
        pause_thread.start()
        assert _wait_until(
            lambda: bool(engine.status_snapshot()["playback_hold"]["active"])
        )

        muted: np.ndarray = np.full((3, 1), 99.0, dtype=np.float32)
        callback(muted, 3, None, None)
        pause_thread.join(timeout=1.0)

        assert not pause_thread.is_alive()
        assert len(result) == 1
        token = result[0]
        assert token is not None
        assert np.count_nonzero(muted) == 0
        assert engine.status_snapshot()["buffered_samples"] == 3
        assert engine.resume_playback(token)

        resumed: np.ndarray = np.empty((3, 1), dtype=np.float32)
        callback(resumed, 3, None, None)
        exhausted: np.ndarray = np.full((3, 1), 99.0, dtype=np.float32)
        callback(exhausted, 3, None, None)

        assert np.allclose(resumed[:, 0], [0.3, 0.4, 0.5])
        assert np.count_nonzero(exhausted) == 0
    finally:
        engine.shutdown()


def test_hold_is_invalidated_when_callback_has_no_pcm_to_preserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch)

    try:
        result: list[PlaybackHoldToken | None] = []
        pause_thread = threading.Thread(
            target=lambda: result.append(engine.pause_playback(timeout_s=1.0))
        )
        pause_thread.start()
        assert _wait_until(
            lambda: bool(engine.status_snapshot()["playback_hold"]["active"])
        )

        callback(np.empty((4, 1), dtype=np.float32), 4, None, None)
        pause_thread.join(timeout=1.0)

        assert not pause_thread.is_alive()
        assert result == [None]
        assert engine.status_snapshot()["playback_hold"]["active"] is False
    finally:
        engine.shutdown()


def test_drain_invalidates_token_and_new_generation_gets_a_new_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 1_000)
        stale_token = _acquire_hold(engine, callback)

        engine.drain_buffers()

        assert not engine.resume_playback(stale_token)
        assert engine.status_snapshot()["buffered_samples"] == 0
        assert engine.queue_cached_pcm(np.ones(3, dtype=np.float32), 1_000)
        current_token = _acquire_hold(engine, callback)

        assert current_token.generation != stale_token.generation
        assert current_token.epoch != stale_token.epoch
        assert engine.resume_playback(current_token)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("transport", ["aplay", "usb_direct"])
def test_blocking_output_transports_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
) -> None:
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": transport,
            "phrase_cache_enabled": False,
        }
    )
    if transport == "aplay":
        monkeypatch.setattr(engine, "_aplay_bin", "aplay")

    try:
        engine.start_playback()
        assert _wait_until(
            lambda: engine.status_snapshot()["playback_hold"]["render_mode"]
            == "unsupported"
        )

        started = time.monotonic()
        assert engine.pause_playback(timeout_s=0.5) is None
        assert time.monotonic() - started < 0.2
        hold_status = engine.status_snapshot()["playback_hold"]
        assert hold_status["supported"] is False
        assert hold_status["rejected"] == 1
    finally:
        engine.shutdown()


def test_concurrent_duplicate_pause_resume_and_abort_calls_linearize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)
    pause_gate = threading.Barrier(5)
    pause_results: list[PlaybackHoldToken | None] = []

    def pause() -> None:
        pause_gate.wait()
        pause_results.append(engine.pause_playback(timeout_s=1.0))

    try:
        assert engine.queue_cached_pcm(np.ones(8, dtype=np.float32), 1_000)
        pause_threads = [threading.Thread(target=pause) for _ in range(4)]
        for thread in pause_threads:
            thread.start()
        pause_gate.wait()
        assert _wait_until(
            lambda: engine.status_snapshot()["playback_hold"]["attempts"] == 4
        )

        callback(np.empty((2, 1), dtype=np.float32), 2, None, None)
        for thread in pause_threads:
            thread.join(timeout=1.0)

        assert all(not thread.is_alive() for thread in pause_threads)
        assert len(pause_results) == 4
        token = pause_results[0]
        assert token is not None
        assert pause_results == [token] * 4
        assert engine.pause_playback(timeout_s=0.0) == token

        resume_results: list[bool] = []
        resume_threads = [
            threading.Thread(
                target=lambda: resume_results.append(engine.resume_playback(token))
            )
            for _ in range(2)
        ]
        for thread in resume_threads:
            thread.start()
        for thread in resume_threads:
            thread.join(timeout=1.0)

        assert sorted(resume_results) == [False, True]
        replacement = _acquire_hold(engine, callback)
        assert engine.abort_playback_hold(replacement)
        assert not engine.abort_playback_hold(replacement)
        hold_status = engine.status_snapshot()["playback_hold"]
        assert hold_status["active"] is False
        assert hold_status["acquired"] == 2
        assert hold_status["resumed"] == 1
        assert hold_status["aborted"] == 1
    finally:
        engine.shutdown()


def test_pause_timeout_cancels_request_without_consuming_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)

    try:
        expected = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        assert engine.queue_cached_pcm(expected, 1_000)

        assert engine.pause_playback(timeout_s=0.02) is None
        hold_status = engine.status_snapshot()["playback_hold"]
        assert hold_status["active"] is False
        assert hold_status["timeouts"] == 1

        rendered: np.ndarray = np.empty((3, 1), dtype=np.float32)
        callback(rendered, 3, None, None)
        assert np.allclose(rendered[:, 0], expected)
    finally:
        engine.shutdown()


def test_wait_done_stays_blocked_during_hold_and_shutdown_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)
    wait_started = threading.Event()
    wait_results: list[bool] = []

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 1_000)
        _acquire_hold(engine, callback)

        def wait_done() -> None:
            wait_started.set()
            wait_results.append(engine.wait_done(timeout=1.0))

        wait_thread = threading.Thread(target=wait_done)
        wait_thread.start()
        assert wait_started.wait(timeout=1.0)
        time.sleep(0.05)
        assert wait_thread.is_alive()

        engine.shutdown()
        wait_thread.join(timeout=1.0)

        assert not wait_thread.is_alive()
        assert wait_results == [True]
        assert engine.status_snapshot()["playback_hold"]["active"] is False
    finally:
        engine.shutdown()


def test_stop_playback_releases_a_pending_pause_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _callback = _start_callback_engine(monkeypatch, sample_rate=1_000)
    pause_results: list[PlaybackHoldToken | None] = []

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 1_000)
        pause_thread = threading.Thread(
            target=lambda: pause_results.append(
                engine.pause_playback(timeout_s=1.0)
            )
        )
        pause_thread.start()
        assert _wait_until(
            lambda: bool(engine.status_snapshot()["playback_hold"]["active"])
        )

        engine.stop_playback()
        pause_thread.join(timeout=1.0)

        assert not pause_thread.is_alive()
        assert pause_results == [None]
        assert engine.status_snapshot()["playback_hold"]["active"] is False
    finally:
        engine.shutdown()


def test_stop_immediately_invalidates_an_acknowledged_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 1_000)
        token = _acquire_hold(engine, callback)

        engine.stop_immediately()

        assert _wait_until(
            lambda: not engine.status_snapshot()["playback_hold"]["active"]
        )
        assert not engine.resume_playback(token)
    finally:
        engine.shutdown()


def test_output_stream_exception_releases_pending_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class _FailingOutputStream:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> _FailingOutputStream:
            entered.set()
            release.wait(timeout=1.0)
            raise RuntimeError("test output failure")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "askme.voice.output.tts.sd.OutputStream",
        _FailingOutputStream,
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "sounddevice",
            "phrase_cache_enabled": False,
        }
    )
    pause_results: list[PlaybackHoldToken | None] = []

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 24_000)
        engine.start_playback()
        assert entered.wait(timeout=1.0)
        pause_thread = threading.Thread(
            target=lambda: pause_results.append(
                engine.pause_playback(timeout_s=1.0)
            )
        )
        pause_thread.start()
        assert _wait_until(
            lambda: bool(engine.status_snapshot()["playback_hold"]["active"])
        )

        release.set()
        pause_thread.join(timeout=1.0)

        assert not pause_thread.is_alive()
        assert pause_results == [None]
        assert _wait_until(
            lambda: engine.status_snapshot()["playback_hold"]["render_mode"]
            == "stopped"
        )
    finally:
        release.set()
        engine.shutdown()


def test_short_duplicate_timeout_does_not_cancel_another_pause_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, callback = _start_callback_engine(monkeypatch, sample_rate=1_000)
    primary_results: list[PlaybackHoldToken | None] = []

    try:
        assert engine.queue_cached_pcm(np.ones(4, dtype=np.float32), 1_000)
        primary = threading.Thread(
            target=lambda: primary_results.append(
                engine.pause_playback(timeout_s=1.0)
            )
        )
        primary.start()
        assert _wait_until(
            lambda: engine.status_snapshot()["playback_hold"]["attempts"] == 1
        )

        assert engine.pause_playback(timeout_s=0.01) is None
        assert engine.status_snapshot()["playback_hold"]["active"] is True

        callback(np.empty((1, 1), dtype=np.float32), 1, None, None)
        primary.join(timeout=1.0)

        assert not primary.is_alive()
        assert len(primary_results) == 1
        token = primary_results[0]
        assert token is not None
        assert engine.resume_playback(token)
    finally:
        engine.shutdown()
