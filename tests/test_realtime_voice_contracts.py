from __future__ import annotations

from collections.abc import Iterator

from askme.ports import RealtimeApprovalPort, RealtimeVoiceFrontendPort
from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeDialogueSession,
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)


class _FakeSession:
    def start(self, context: RealtimeVoiceSessionContext) -> bool:
        return bool(context.session_id)

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        return bool(frame.pcm)

    def finish_input(self) -> None:
        return None

    def interrupt(self, reason: str) -> None:
        return None

    def next_event(self, timeout: float | None = None) -> RealtimeVoiceEvent | None:
        return None

    def events(self) -> Iterator[RealtimeVoiceEvent]:
        return iter(())

    def close(self, reason: str = "shutdown") -> None:
        return None

    def status_snapshot(self) -> dict[str, object]:
        return {"available": True}


class _FakeApproval:
    initial_text = "你好"
    completed = True

    def wait(self, timeout: float | None = None) -> str:
        return self.initial_text


class _FakeRealtimeFrontend:
    last_turn_realtime_generation = 2
    last_turn_realtime_baseline_generation = 1

    def realtime_general_chat_ready(self) -> bool:
        return True

    def realtime_capture_active(self) -> bool:
        return True

    def try_realtime_general_chat(
        self,
        local_text: str,
        *,
        expected_generation: int = 0,
    ) -> _FakeApproval | None:
        return _FakeApproval() if local_text and expected_generation > 0 else None

    def discard_realtime_turn(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None:
        return None

    def abort_realtime_playback(self, reason: str) -> None:
        return None

    def realtime_playback_started(self) -> bool:
        return True


def test_realtime_session_context_has_safe_robot_defaults() -> None:
    context = RealtimeVoiceSessionContext(session_id="turn-1")

    assert context.input_mode == "audio"
    assert context.input_sample_rate == 16_000
    assert context.output_sample_rate == 24_000
    assert context.output_format == "pcm_s16le"
    assert context.allow_tool_calls is False
    assert context.allow_hardware_dispatch is False


def test_realtime_audio_event_serializes_metadata_without_raw_pcm() -> None:
    event = RealtimeVoiceEvent(
        event_type=RealtimeVoiceEventType.OUTPUT_AUDIO,
        session_id="session-1",
        generation=3,
        provider="volcengine_s2s",
        provider_event_id=352,
        audio=VoiceMediaFrame(
            pcm=b"\x00\x00" * 480,
            sample_rate=24_000,
            channels=1,
        ),
    )

    payload = event.to_dict()

    assert payload["event_type"] == "output_audio"
    assert payload["audio"] == {
        "sample_rate": 24_000,
        "channels": 1,
        "bytes": 960,
        "duration_ms": 20.0,
    }
    assert "pcm" not in repr(payload)


def test_realtime_dialogue_session_contract_is_runtime_checkable() -> None:
    assert isinstance(_FakeSession(), RealtimeDialogueSession)


def test_realtime_frontend_contract_is_explicit_and_runtime_checkable() -> None:
    frontend = _FakeRealtimeFrontend()

    assert isinstance(_FakeApproval(), RealtimeApprovalPort)
    assert isinstance(frontend, RealtimeVoiceFrontendPort)
    assert frontend.last_turn_realtime_generation == 2
    assert frontend.last_turn_realtime_baseline_generation == 1
