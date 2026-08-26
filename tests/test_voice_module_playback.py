from __future__ import annotations

from askme.ports import (
    PlaybackTarget,
    SpeechDelivery,
    SpeechPlaybackJob,
    SpeechPlaybackState,
    SpeechPlaybackTimestamps,
    SpeechPriority,
)
from askme.runtime.modules.voice_module import VoiceModule


class _PlaybackPort:
    def __init__(self) -> None:
        self.request = None
        self.cancelled = None

    async def submit(self, request):
        self.request = request
        return SpeechPlaybackJob(
            playback_id="spk_1",
            state=SpeechPlaybackState.QUEUED,
            target=request.target,
            delivery=request.delivery,
            priority=request.priority,
            text_chars=len(request.text),
            request_hash="hash",
            idempotency_key=request.idempotency_key,
            timestamps=SpeechPlaybackTimestamps(queued_at="now"),
        )

    async def status(self, playback_id):
        return SpeechPlaybackJob(
            playback_id=playback_id,
            state=SpeechPlaybackState.PLAYING,
            target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
            delivery=SpeechDelivery.PLAYBACK,
            priority=SpeechPriority.NORMAL,
            text_chars=4,
            request_hash="hash",
            idempotency_key="key",
            timestamps=SpeechPlaybackTimestamps(queued_at="now"),
        )

    async def cancel(self, playback_id, *, reason, actor=None):
        self.cancelled = (playback_id, reason, actor)
        job = await self.status(playback_id)
        return SpeechPlaybackJob(
            **{
                **job.__dict__,
                "state": SpeechPlaybackState.CANCELLED,
            }
        )


def _module():
    module = VoiceModule()
    module._speech_playback = _PlaybackPort()
    return module


async def test_voice_module_builds_literal_playback_request_from_api_payload() -> None:
    module = _module()

    result = await module.speak_payload(
        {
            "text": "Hello robot",
            "semantics": "verbatim",
            "robot_id": "robot-1",
            "device_id": "speaker-1",
            "site_id": "site-a",
            "idempotency_key": "request-1",
            "operator_id": "operator-1",
            "operator_auth": {
                "operator": {"operator_id": "operator-1", "roles": ["operator"]}
            },
        }
    )

    request = module._speech_playback.request
    assert result["playback_id"] == "spk_1"
    assert request.text == "Hello robot"
    assert request.target == PlaybackTarget(
        robot_id="robot-1",
        device_id="speaker-1",
        site_id="site-a",
    )
    assert request.actor.operator_id == "operator-1"
    assert request.actor.roles == frozenset({"operator"})
    assert request.delivery is SpeechDelivery.PLAYBACK


async def test_voice_module_keeps_synthesize_only_and_cancel_on_same_port() -> None:
    module = _module()
    payload = {
        "text": "Preview",
        "robot_id": "robot-1",
        "device_id": "speaker-1",
        "idempotency_key": "wav-1",
        "operator_id": "operator-1",
    }

    synthesized = await module.synthesize_speech_payload(payload)
    cancelled = await module.cancel_speech_playback_payload(
        "spk_1",
        {"reason": "operator_cancelled", "operator_id": "operator-1"},
    )

    assert synthesized["delivery"] == "synthesize_only"
    assert module._speech_playback.request.delivery is SpeechDelivery.SYNTHESIZE_ONLY
    assert cancelled["state"] == "cancelled"
    assert module._speech_playback.cancelled[0:2] == ("spk_1", "operator_cancelled")
