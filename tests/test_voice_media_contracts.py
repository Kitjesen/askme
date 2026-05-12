from __future__ import annotations

from askme.voice.media_contracts import (
    InterruptDecision,
    VoiceMediaFrame,
    VoiceMediaStatus,
    VoiceTurnEvent,
    VoiceTurnEventType,
)


def test_voice_media_frame_reports_duration() -> None:
    frame = VoiceMediaFrame(
        pcm=b"\x00\x00" * 1600,
        sample_rate=16000,
        channels=1,
    )

    assert frame.duration_ms == 100.0


def test_voice_media_status_serializes_transport_fields() -> None:
    status = VoiceMediaStatus(
        media_transport="local_sounddevice",
        session_id="run-1",
        participant_count=1,
        input_transport="sounddevice",
        output_transport="sounddevice",
        metadata={"asr_provider": "cloud+local"},
    )

    assert status.to_dict() == {
        "media_transport": "local_sounddevice",
        "session_id": "run-1",
        "room_id": "",
        "participant_count": 1,
        "packet_loss": None,
        "jitter_ms": None,
        "input_transport": "sounddevice",
        "output_transport": "sounddevice",
        "metadata": {"asr_provider": "cloud+local"},
    }


def test_voice_turn_event_serializes_enum_value() -> None:
    event = VoiceTurnEvent(
        event_type=VoiceTurnEventType.ASR_FINAL,
        voice_turn_id="turn-1",
        offset_ms=123.456,
        transcript="inspect area A",
        is_final=True,
        confidence=0.92,
        provider="dashscope_paraformer",
    )

    assert event.to_dict()["event_type"] == "asr_final"
    assert event.to_dict()["offset_ms"] == 123.46
    assert event.to_dict()["is_final"] is True


def test_interrupt_decision_serializes_runtime_action_boundary() -> None:
    decision = InterruptDecision(
        accepted=True,
        reason="barge_in",
        stopped_playback=True,
        cancelled_generation=True,
        requires_runtime_action=False,
    )

    assert decision.to_dict()["accepted"] is True
    assert decision.to_dict()["requires_runtime_action"] is False
