from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from askme.voice.orchestration.interrupt_recovery import (
    InterruptionRecoveryCoordinator,
    InterruptionRecoveryState,
)


class _PlaybackHoldFake:
    def __init__(self, *, supported: bool = True) -> None:
        self.supported = supported
        self.pause_calls: list[float] = []
        self.resume_calls: list[object] = []
        self.abort_calls: list[object] = []
        self.token = object()

    def pause_playback(self, *, timeout_s: float) -> object | None:
        self.pause_calls.append(timeout_s)
        return self.token if self.supported else None

    def resume_playback(self, token: object) -> bool:
        self.resume_calls.append(token)
        return token is self.token

    def abort_playback_hold(self, token: object) -> bool:
        self.abort_calls.append(token)
        return token is self.token


def test_false_detection_pauses_once_then_resumes_exact_generation() -> None:
    playback = _PlaybackHoldFake()
    guard = InterruptionRecoveryCoordinator(playback)

    assert guard.begin_detection()
    assert guard.begin_detection()
    guard.confirm()
    assert guard.state is InterruptionRecoveryState.VALIDATING
    assert playback.pause_calls == [0.05]
    assert playback.abort_calls == []

    assert guard.recover("asr_noise_filtered")

    assert playback.resume_calls == [playback.token]
    assert playback.abort_calls == []
    assert guard.state is InterruptionRecoveryState.IDLE
    assert guard.status_snapshot() == {
        "state": "idle",
        "hold_active": False,
        "hold_supported": True,
        "detections": 1,
        "confirmations": 1,
        "commits": 0,
        "recoveries": 1,
        "hold_timeouts": 0,
        "resume_failures": 0,
        "abort_failures": 0,
        "last_reason": "asr_noise_filtered",
    }


def test_validated_interruption_aborts_hold_without_resuming() -> None:
    playback = _PlaybackHoldFake()
    guard = InterruptionRecoveryCoordinator(playback)

    guard.begin_detection()
    guard.confirm()

    assert guard.commit("accepted_transcript")
    assert playback.abort_calls == [playback.token]
    assert playback.resume_calls == []
    assert guard.state is InterruptionRecoveryState.IDLE
    assert guard.status_snapshot()["commits"] == 1


def test_unsupported_transport_still_validates_before_commit_or_recovery() -> None:
    playback = _PlaybackHoldFake(supported=False)
    guard = InterruptionRecoveryCoordinator(playback)

    assert not guard.begin_detection()
    guard.confirm()
    assert guard.state is InterruptionRecoveryState.VALIDATING

    assert guard.recover("empty_asr")
    assert playback.resume_calls == []
    assert playback.abort_calls == []
    assert guard.status_snapshot()["hold_supported"] is False

    guard.confirm()
    assert guard.commit("accepted_transcript")
    assert playback.abort_calls == []
    assert guard.status_snapshot()["commits"] == 1


def test_hold_timeout_releases_audio_but_keeps_validation_open() -> None:
    now = [10.0]
    playback = _PlaybackHoldFake()
    guard = InterruptionRecoveryCoordinator(
        playback,
        hold_timeout_s=2.0,
        clock=lambda: now[0],
    )
    guard.begin_detection()
    guard.confirm()

    now[0] = 11.99
    assert not guard.expire_hold()
    now[0] = 12.01
    assert guard.expire_hold()

    assert playback.resume_calls == [playback.token]
    assert guard.state is InterruptionRecoveryState.VALIDATING
    assert guard.status_snapshot()["hold_timeouts"] == 1
    assert guard.commit("accepted_after_hold_timeout")
    assert playback.abort_calls == []


def test_concurrent_duplicate_detection_is_linearized() -> None:
    playback = _PlaybackHoldFake()
    guard = InterruptionRecoveryCoordinator(playback)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: guard.begin_detection(), range(32)))

    assert outcomes == [True] * 32
    assert playback.pause_calls == [0.05]
    assert guard.status_snapshot()["detections"] == 1
    assert guard.recover("dismissed")


def test_close_aborts_an_active_hold_and_is_idempotent() -> None:
    playback = _PlaybackHoldFake()
    guard = InterruptionRecoveryCoordinator(playback)
    guard.begin_detection()

    assert guard.close()
    assert not guard.close()
    assert playback.abort_calls == [playback.token]
    assert playback.resume_calls == []
    assert guard.state is InterruptionRecoveryState.IDLE
