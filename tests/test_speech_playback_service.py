from __future__ import annotations

import asyncio

import pytest

from askme.ports.speech_playback import (
    PlaybackTarget,
    SpeechActor,
    SpeechDelivery,
    SpeechPlaybackError,
    SpeechPlaybackRequest,
    SpeechPlaybackState,
    SpeechPriority,
)
from askme.voice.playback.service import SpeechPlaybackService


class _FakeAudio:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stopped = False

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool:
        return False

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)

    def stop_immediately(self) -> None:
        self.stopped = True


async def _terminal_job(service: SpeechPlaybackService, playback_id: str):
    for _ in range(200):
        job = await service.status(playback_id)
        if job.state.terminal:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("playback did not reach a terminal state")


@pytest.mark.asyncio
async def test_submit_verbatim_playback_returns_queued_then_completes() -> None:
    audio = _FakeAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="xiaosuan-1",
        device_id="xiaosuan-1",
        site_id="site-a",
    )
    await service.start()

    accepted = await service.submit(
        SpeechPlaybackRequest(
            text="您好，请让一下。",
            target=PlaybackTarget(
                robot_id="xiaosuan-1",
                device_id="xiaosuan-1",
                site_id="site-a",
            ),
            idempotency_key="speak-001",
        )
    )
    completed = await _terminal_job(service, accepted.playback_id)
    await service.shutdown()

    assert accepted.state is SpeechPlaybackState.QUEUED
    assert completed.state is SpeechPlaybackState.COMPLETED
    assert completed.timestamps.queued_at is not None
    assert completed.timestamps.synthesis_started_at is not None
    assert completed.timestamps.playback_started_at is not None
    assert completed.timestamps.completed_at is not None
    assert completed.target.device_id == "xiaosuan-1"
    assert completed.text_chars == len("您好，请让一下。")


@pytest.mark.asyncio
async def test_idempotency_replays_same_job_and_rejects_changed_request() -> None:
    audio = _FakeAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
    )
    await service.start()
    target = PlaybackTarget(robot_id="robot-1", device_id="speaker-1")

    first = await service.submit(
        SpeechPlaybackRequest(text="same text", target=target, idempotency_key="same-key")
    )
    replay = await service.submit(
        SpeechPlaybackRequest(text="same text", target=target, idempotency_key="same-key")
    )
    with pytest.raises(SpeechPlaybackError) as conflict:
        await service.submit(
            SpeechPlaybackRequest(text="另一句话", target=target, idempotency_key="same-key")
        )
    await _terminal_job(service, first.playback_id)
    await service.shutdown()

    assert replay.playback_id == first.playback_id
    assert conflict.value.code == "idempotency_conflict"
    assert audio.spoken == ["same text"]


@pytest.mark.asyncio
async def test_rejects_missing_or_non_local_target_before_playback() -> None:
    audio = _FakeAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
        site_id="site-a",
    )

    with pytest.raises(SpeechPlaybackError) as mismatch:
        await service.submit(
            SpeechPlaybackRequest(
                text="不能误播",
                target=PlaybackTarget(
                    robot_id="robot-2",
                    device_id="speaker-2",
                    site_id="site-a",
                ),
            )
        )

    assert mismatch.value.code == "target_not_local"
    assert audio.spoken == []


class _BlockingAudio(_FakeAudio):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)
        self.started.set()
        await self.release.wait()

    def stop_immediately(self) -> None:
        super().stop_immediately()
        self.release.set()


@pytest.mark.asyncio
async def test_cancel_active_playback_fences_late_completion() -> None:
    audio = _BlockingAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
    )
    await service.start()
    accepted = await service.submit(
        SpeechPlaybackRequest(
            text="这是一段会被停止的播报",
            target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
        )
    )
    await asyncio.wait_for(audio.started.wait(), timeout=1.0)

    cancelled = await service.cancel(accepted.playback_id, reason="operator_cancelled")
    terminal = await _terminal_job(service, accepted.playback_id)
    await service.shutdown()

    assert cancelled.state is SpeechPlaybackState.CANCELLED
    assert terminal.state is SpeechPlaybackState.CANCELLED
    assert audio.stopped is True


@pytest.mark.asyncio
async def test_safety_playback_preempts_lower_priority_job() -> None:
    audio = _BlockingAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
    )
    await service.start()
    target = PlaybackTarget(robot_id="robot-1", device_id="speaker-1")
    ordinary = await service.submit(
        SpeechPlaybackRequest(text="普通通知", target=target)
    )
    await asyncio.wait_for(audio.started.wait(), timeout=1.0)

    safety = await service.submit(
        SpeechPlaybackRequest(
            text="请立即停止并注意安全",
            target=target,
            priority=SpeechPriority.SAFETY,
            queue_policy="replace_noncritical",
        )
    )
    safety_done = await _terminal_job(service, safety.playback_id)
    ordinary_done = await service.status(ordinary.playback_id)
    await service.shutdown()

    assert ordinary_done.state is SpeechPlaybackState.CANCELLED
    assert safety_done.state is SpeechPlaybackState.COMPLETED
    assert audio.spoken == ["普通通知", "请立即停止并注意安全"]


class _SynthTTS:
    def __init__(self) -> None:
        self.primed: list[tuple[str, str]] = []

    def prime_cached_phrase(self, text: str, *, cache_key: str):
        self.primed.append((text, cache_key))
        return {
            "cached": True,
            "created": True,
            "cache_key": cache_key,
            "samples": 4,
            "sample_rate": 16000,
        }

    def cached_phrase_pcm(self, text: str, *, cache_key: str):
        return ([0.0, 0.25, -0.25, 0.0], 16000)


@pytest.mark.asyncio
async def test_synthesize_only_creates_downloadable_wav_without_playback(tmp_path) -> None:
    audio = _FakeAudio()
    audio.tts = _SynthTTS()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
        artifact_dir=tmp_path,
    )
    await service.start()

    accepted = await service.submit(
        SpeechPlaybackRequest(
            text="Preview audio",
            target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
            delivery=SpeechDelivery.SYNTHESIZE_ONLY,
            idempotency_key="wav-1",
        )
    )
    completed = await _terminal_job(service, accepted.playback_id)
    artifact = await service.artifact_file(accepted.playback_id)
    await service.shutdown()

    assert completed.state is SpeechPlaybackState.COMPLETED
    assert completed.artifact["format"] == "wav"
    assert completed.artifact["download_url"].endswith("/audio")
    assert artifact.path.read_bytes().startswith(b"RIFF")
    assert audio.spoken == []


@pytest.mark.asyncio
async def test_operator_cannot_cancel_another_operators_playback() -> None:
    audio = _BlockingAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
    )
    await service.start()
    accepted = await service.submit(
        SpeechPlaybackRequest(
            text="Owned playback",
            target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
            actor=SpeechActor(operator_id="operator-a", roles=frozenset({"operator"})),
        )
    )
    await asyncio.wait_for(audio.started.wait(), timeout=1.0)

    with pytest.raises(SpeechPlaybackError) as denied:
        await service.cancel(
            accepted.playback_id,
            reason="not_mine",
            actor=SpeechActor(operator_id="operator-b", roles=frozenset({"operator"})),
        )
    await service.cancel(
        accepted.playback_id,
        reason="supervisor_override",
        actor=SpeechActor(operator_id="supervisor-1", roles=frozenset({"supervisor"})),
    )
    await service.shutdown()

    assert denied.value.code == "cancel_not_allowed"


class _ControlTTS:
    def __init__(self) -> None:
        self.speed = 1.0
        self.pitch = 0.0
        self.volume = 0.8

    def set_speed(self, value):
        self.speed = float(value)

    def set_pitch(self, value):
        self.pitch = float(value)

    def set_volume(self, value):
        self.volume = float(value)


class _ControlledAudio(_FakeAudio):
    def __init__(self) -> None:
        super().__init__()
        self.tts = _ControlTTS()
        self.observed_controls = None

    def set_speed(self, value):
        self.tts.set_speed(value)

    def set_pitch(self, value):
        self.tts.set_pitch(value)

    def set_volume(self, value):
        self.tts.set_volume(value)

    async def speak_and_wait(self, text: str) -> None:
        self.observed_controls = (self.tts.speed, self.tts.pitch, self.tts.volume)
        await super().speak_and_wait(text)


@pytest.mark.asyncio
async def test_per_job_voice_controls_are_bounded_applied_and_restored() -> None:
    audio = _ControlledAudio()
    service = SpeechPlaybackService(
        audio=audio,
        robot_id="robot-1",
        device_id="speaker-1",
    )
    await service.start()
    accepted = await service.submit(
        SpeechPlaybackRequest(
            text="Controlled voice",
            target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
            speed=1.2,
            pitch=2.0,
            volume=0.5,
        )
    )
    await _terminal_job(service, accepted.playback_id)
    await service.shutdown()

    assert audio.observed_controls == (1.2, 2.0, 0.5)
    assert (audio.tts.speed, audio.tts.pitch, audio.tts.volume) == (1.0, 0.0, 0.8)


@pytest.mark.asyncio
async def test_idempotency_survives_service_restart_without_replaying_audio(tmp_path) -> None:
    ledger = tmp_path / "playback-ledger.json"
    target = PlaybackTarget(robot_id="robot-1", device_id="speaker-1")
    request = SpeechPlaybackRequest(
        text="Persisted request",
        target=target,
        idempotency_key="durable-key",
    )
    first_audio = _FakeAudio()
    first_service = SpeechPlaybackService(
        audio=first_audio,
        robot_id="robot-1",
        device_id="speaker-1",
        ledger_path=ledger,
    )
    await first_service.start()
    first = await first_service.submit(request)
    await _terminal_job(first_service, first.playback_id)
    await first_service.shutdown()

    second_audio = _FakeAudio()
    second_service = SpeechPlaybackService(
        audio=second_audio,
        robot_id="robot-1",
        device_id="speaker-1",
        ledger_path=ledger,
    )
    await second_service.start()
    replay = await second_service.submit(request)
    await second_service.shutdown()

    assert replay.playback_id == first.playback_id
    assert first_audio.spoken == ["Persisted request"]
    assert second_audio.spoken == []
