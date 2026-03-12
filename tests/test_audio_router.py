"""Tests for AudioRouter — ownership protocol and error classification."""

from __future__ import annotations

import threading
import time

import pytest

from askme.voice.audio_router import AudioErrorKind, AudioRouter


# ── Ownership ────────────────────────────────────────────────────────────────


def test_initially_input_ready() -> None:
    r = AudioRouter()
    assert r.wait_for_input_ready(timeout=0) is True
    assert r.is_output_active is False


def test_output_session_blocks_input() -> None:
    r = AudioRouter()
    with r.output_session():
        assert r.is_output_active is True
        assert r.wait_for_input_ready(timeout=0) is False


def test_output_session_releases_on_exit() -> None:
    r = AudioRouter()
    with r.output_session():
        pass
    assert r.is_output_active is False
    assert r.wait_for_input_ready(timeout=0) is True


def test_output_session_releases_on_exception() -> None:
    r = AudioRouter()
    try:
        with r.output_session():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert r.is_output_active is False
    assert r.wait_for_input_ready(timeout=0) is True


def test_nested_output_sessions_reference_counted() -> None:
    r = AudioRouter()
    with r.output_session():
        with r.output_session():
            assert r.is_output_active is True
        # Still active — outer session not done yet
        assert r.is_output_active is True
    # Both released
    assert r.is_output_active is False


def test_wait_for_input_ready_unblocks_after_output() -> None:
    r = AudioRouter()
    released_at: list[float] = []
    unblocked_at: list[float] = []

    def _hold() -> None:
        with r.output_session():
            time.sleep(0.1)
        released_at.append(time.monotonic())

    def _wait() -> None:
        r.wait_for_input_ready(timeout=2.0)
        unblocked_at.append(time.monotonic())

    t1 = threading.Thread(target=_hold)
    t2 = threading.Thread(target=_wait)
    t1.start()
    time.sleep(0.02)  # ensure _hold enters output_session first
    t2.start()
    t1.join(timeout=1)
    t2.join(timeout=1)

    assert len(released_at) == 1
    assert len(unblocked_at) == 1
    # _wait should unblock no earlier than _hold releases
    assert unblocked_at[0] >= released_at[0] - 0.01  # 10ms tolerance


def test_wait_for_input_ready_timeout() -> None:
    r = AudioRouter()
    with r.output_session():
        result = r.wait_for_input_ready(timeout=0.05)
    assert result is False


# ── Error classification ─────────────────────────────────────────────────────


@pytest.mark.parametrize("msg, expected", [
    # XRUN patterns
    ("Unanticipated host error [PaErrorCode -9999]", AudioErrorKind.XRUN),
    ("Illegal combination of I/O devices [PaErrorCode -9993]", AudioErrorKind.XRUN),
    ("XRUN detected in capture stream", AudioErrorKind.XRUN),
    ("AlsaRestart failed at line 3313", AudioErrorKind.XRUN),
    ("alsa_snd_pcm_start failed", AudioErrorKind.XRUN),
    # DEVICE_LOST
    ("ALSA lib confmisc.c: Cannot get card index for 1", AudioErrorKind.DEVICE_LOST),
    ("No such device hw:1,0", AudioErrorKind.DEVICE_LOST),
    # DEVICE_BUSY
    ("Invalid number of channels [PaErrorCode -9998]", AudioErrorKind.DEVICE_BUSY),
    ("Device or resource busy (EBUSY)", AudioErrorKind.DEVICE_BUSY),
    # TTS_FAIL
    ("minimax API returned 429", AudioErrorKind.TTS_FAIL),
    ("aplay exited with code 1", AudioErrorKind.TTS_FAIL),
    ("edge-tts connection refused", AudioErrorKind.TTS_FAIL),
    # UNKNOWN
    ("unexpected memory allocation failure", AudioErrorKind.UNKNOWN),
    ("some random unrelated error", AudioErrorKind.UNKNOWN),
])
def test_classify_error(msg: str, expected: AudioErrorKind) -> None:
    exc = RuntimeError(msg)
    assert AudioRouter.classify_error(exc) == expected


def test_classify_error_case_insensitive() -> None:
    assert AudioRouter.classify_error(RuntimeError("XRUN DETECTED")) == AudioErrorKind.XRUN
    assert AudioRouter.classify_error(RuntimeError("MiniMax TTS ERROR")) == AudioErrorKind.TTS_FAIL
