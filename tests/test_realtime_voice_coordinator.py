from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.coordinator import RealtimeDialogueCoordinator


class _FakeSession:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.start_contexts: list[RealtimeVoiceSessionContext] = []
        self.close_reasons: list[str] = []
        self.offered: list[VoiceMediaFrame] = []
        self.interrupt_reasons: list[str] = []
        self.deleted_items: list[str] = []
        self.delete_success = True
        self.truncate_calls: list[tuple[str, int]] = []
        self.truncate_success = True
        self._events: queue.Queue[RealtimeVoiceEvent | None] = queue.Queue()

    def start(self, context: RealtimeVoiceSessionContext) -> bool:
        self.started = True
        self.closed = False
        self.context = context
        self.start_contexts.append(context)
        return True

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        self.offered.append(frame)
        return self.started and not self.closed

    def finish_input(self) -> None:
        return None

    def interrupt(self, reason: str) -> None:
        self.interrupt_reasons.append(reason)

    def delete_conversation_turn(
        self,
        item_id: str,
        *,
        timeout: float = 1.0,
    ) -> bool:
        self.deleted_items.append(item_id)
        return self.delete_success

    def truncate_response(self, *, reply_id: str, audio_end_ms: int) -> bool:
        self.truncate_calls.append((reply_id, audio_end_ms))
        return self.truncate_success

    def next_event(self, timeout: float | None = None) -> RealtimeVoiceEvent | None:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def events(self) -> Iterator[RealtimeVoiceEvent]:
        return iter(())

    def close(self, reason: str = "shutdown") -> None:
        self.closed = True
        self.close_reasons.append(reason)
        self._events.put(None)

    def status_snapshot(self) -> dict[str, object]:
        return {"available": True, "active": self.started and not self.closed}

    def emit(self, event: RealtimeVoiceEvent) -> None:
        self._events.put(event)


class _ExplodingSession(_FakeSession):
    def next_event(self, timeout: float | None = None) -> RealtimeVoiceEvent | None:
        raise RuntimeError("event consumer failed")


class _QwenSession(_FakeSession):
    def status_snapshot(self) -> dict[str, object]:
        return {
            **super().status_snapshot(),
            "provider": "qwen3_5_omni",
            "model": "qwen3.5-omni-flash-realtime",
            "provider_session_id": "qwen-session-1",
            "api_key": "must-not-leak",
        }


class _FailsFirstStartSession(_FakeSession):
    def start(self, context: RealtimeVoiceSessionContext) -> bool:
        self.start_contexts.append(context)
        if len(self.start_contexts) == 1:
            self.started = False
            self.closed = True
            return False
        self.started = True
        self.closed = False
        self.context = context
        return True


class _BlockingDeleteSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.first_delete_started = threading.Event()
        self.release_first_delete = threading.Event()

    def delete_conversation_turn(
        self,
        item_id: str,
        *,
        timeout: float = 1.0,
    ) -> bool:
        self.deleted_items.append(item_id)
        if len(self.deleted_items) == 1:
            self.first_delete_started.set()
            return self.release_first_delete.wait(timeout=timeout)
        return True


class _BlockingCloseSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = threading.Event()
        self.close_release = threading.Event()

    def close(self, reason: str = "shutdown") -> None:
        self.closed = True
        self.close_reasons.append(reason)
        self.close_started.set()
        assert self.close_release.wait(timeout=1.0)
        self._events.put(None)


def _event(
    event_type: RealtimeVoiceEventType,
    *,
    session_id: str = "session-1",
    generation: int = 1,
    transcript: str = "",
    text: str = "",
    pcm: bytes | None = None,
) -> RealtimeVoiceEvent:
    audio = (
        VoiceMediaFrame(pcm=pcm, sample_rate=24_000, channels=1)
        if pcm is not None
        else None
    )
    return RealtimeVoiceEvent(
        event_type=event_type,
        session_id=session_id,
        generation=generation,
        provider="volcengine_s2s",
        transcript=transcript,
        text=text,
        audio=audio,
        is_final=event_type is RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
        metadata={
            "question_id": f"q-{generation}",
            "reply_id": f"r-{generation}",
        },
    )


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_status_snapshot_exposes_safe_provider_identity_only() -> None:
    session = _QwenSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )

    assert coordinator.start() is True
    snapshot = coordinator.status_snapshot()
    assert snapshot["provider"] == "qwen3_5_omni"
    assert snapshot["model"] == "qwen3.5-omni-flash-realtime"
    assert snapshot["provider_session_id"] == "qwen-session-1"
    assert "must-not-leak" not in repr(snapshot)
    coordinator.close()


def test_coordinator_streams_clean_audio_without_blocking_capture() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="shadow",
    )

    assert coordinator.start() is True
    frame = VoiceMediaFrame(pcm=b"\x00\x00" * 160, sample_rate=16_000)

    assert coordinator.offer_audio(frame) is True
    assert session.offered == [frame]
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        _event(RealtimeVoiceEventType.OUTPUT_AUDIO, pcm=b"shadow-audio")
    )
    _wait_until(lambda: coordinator.status_snapshot()["shadow_audio_frames"] == 1)
    assert coordinator.status_snapshot()["pending_audio_frames"] == 0
    coordinator.close()


def test_provider_audio_is_held_until_general_chat_is_approved() -> None:
    session = _FakeSession()
    rendered: list[tuple[bytes, bool]] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, final: rendered.append((frame.pcm, final)),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            transcript="今天天气怎么样",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            text="今天天气不错。",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            pcm=b"\x01\x00" * 480,
        )
    )
    _wait_until(lambda: coordinator.status_snapshot()["pending_audio_frames"] == 1)

    assert rendered == []
    approval = coordinator.approve_general_chat(
        "今天天气怎么样",
        wait_timeout=0.5,
    )

    assert approval is not None
    assert approval.initial_text == "今天天气不错。"
    assert rendered == [(b"\x01\x00" * 480, False)]

    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            pcm=b"\x02\x00" * 480,
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_DONE,
            text="今天天气不错。适合出门。",
        )
    )
    assert approval.wait(timeout=1.0) == "今天天气不错。适合出门。"
    _wait_until(lambda: rendered[-1][1] is True)
    assert rendered[-2:] == [
        (b"\x02\x00" * 480, False),
        (b"", True),
    ]
    coordinator.close()


def test_prepared_provider_audio_stays_buffered_until_explicit_release() -> None:
    session = _FakeSession()
    rendered: list[tuple[bytes, bool]] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, final: rendered.append((frame.pcm, final)),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=3))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=3,
            transcript="你好",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=3,
            text="你好呀。",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            generation=3,
            pcm=b"\x01\x00" * 480,
        )
    )
    _wait_until(lambda: coordinator.status_snapshot()["pending_audio_frames"] == 1)

    prepared = coordinator.prepare_general_chat(
        "你好",
        expected_generation=3,
        wait_timeout=0.5,
    )

    assert prepared is not None
    assert rendered == []
    assert coordinator.status_snapshot()["approved"] is False
    assert coordinator.release_general_chat(prepared) is True
    assert rendered == [(b"\x01\x00" * 480, False)]
    assert coordinator.status_snapshot()["approved"] is True
    coordinator.close()


def test_finish_input_false_is_propagated_by_coordinator() -> None:
    class _RejectingFinishSession(_FakeSession):
        def finish_input(self) -> bool:
            return False

    coordinator = RealtimeDialogueCoordinator(
        _RejectingFinishSession(),
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="shadow",
    )
    assert coordinator.start() is True

    assert coordinator.finish_input() is False
    coordinator.close()


def test_interim_transcript_cannot_approve_audio_before_final_transcript() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
            transcript="hello",
        )
    )
    session.emit(_event(RealtimeVoiceEventType.RESPONSE_STARTED, text="hi"))
    session.emit(_event(RealtimeVoiceEventType.OUTPUT_AUDIO, pcm=b"pending"))

    approvals: list[object] = []
    approval_thread = threading.Thread(
        target=lambda: approvals.append(
            coordinator.approve_general_chat("hello", wait_timeout=0.5)
        )
    )
    approval_thread.start()
    _wait_until(lambda: coordinator.status_snapshot()["pending_audio_frames"] == 1)
    time.sleep(0.03)

    assert rendered == []
    assert approval_thread.is_alive()

    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            transcript="hello",
        )
    )
    approval_thread.join(timeout=1.0)

    assert approvals[0] is not None
    assert rendered == [b"pending"]
    coordinator.close()


def test_interim_transcript_timeout_quarantines_and_rolls_back_turn() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=2))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
            generation=2,
            transcript="unfinished",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=2,
            text="speculative",
        )
    )

    assert coordinator.approve_general_chat("unfinished", wait_timeout=0.05) is None

    assert session.interrupt_reasons[-1] == "approval_timeout"
    _wait_until(lambda: session.deleted_items == ["q-2"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    coordinator.close()


def test_transcript_mismatch_discards_speculative_audio_and_interrupts() -> None:
    session = _BlockingDeleteSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            transcript="打开仓库大门",
        )
    )
    session.emit(
        _event(RealtimeVoiceEventType.OUTPUT_AUDIO, pcm=b"\x01\x00" * 480)
    )

    approval = coordinator.approve_general_chat("讲个笑话", wait_timeout=0.5)

    assert approval is None
    assert rendered == []
    assert session.interrupt_reasons[-1] == "transcript_mismatch"
    assert session.first_delete_started.wait(timeout=1.0)
    assert coordinator.status_snapshot()["quarantined"] is True
    assert coordinator.offer_audio(
        VoiceMediaFrame(pcm=b"\x00\x00" * 160, sample_rate=16_000)
    ) is False
    session.release_first_delete.set()
    _wait_until(lambda: session.deleted_items == ["q-1"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    coordinator.close()


def test_interrupt_fences_late_audio_from_cancelled_generation() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=4))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=4,
            transcript="你好",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=4,
            text="你好呀。",
        )
    )
    approval = coordinator.approve_general_chat("你好", wait_timeout=0.5)
    assert approval is not None

    coordinator.discard_current("robot_task")
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            generation=4,
            pcm=b"late",
        )
    )
    time.sleep(0.05)

    assert b"late" not in rendered
    assert session.interrupt_reasons[-1] == "robot_task"
    _wait_until(lambda: session.deleted_items == ["q-4"])
    coordinator.close()


def test_truncate_current_preserves_heard_history_without_deleting_turn() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=6))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=6,
            transcript="continue",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=6,
            text="long answer",
        )
    )
    assert coordinator.approve_general_chat(
        "continue",
        expected_generation=6,
        wait_timeout=0.5,
    ) is not None

    assert coordinator.truncate_current(
        "barge_in",
        audio_end_ms=240,
        expected_generation=6,
    ) is True
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            generation=6,
            pcm=b"late",
        )
    )
    time.sleep(0.05)

    assert rendered == []
    assert session.truncate_calls == [("r-6", 240)]
    assert session.interrupt_reasons == ["barge_in"]
    assert session.deleted_items == []
    snapshot = coordinator.status_snapshot()
    assert snapshot["truncation_count"] == 1
    assert snapshot["quarantined"] is False
    coordinator.close()


def test_failed_truncate_falls_back_to_safe_conversation_delete() -> None:
    session = _FakeSession()
    session.truncate_success = False
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=7))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=7,
            transcript="answer",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=7,
            text="response",
        )
    )
    assert coordinator.approve_general_chat(
        "answer",
        expected_generation=7,
        wait_timeout=0.5,
    ) is not None

    assert coordinator.truncate_current(
        "barge_in",
        audio_end_ms=120,
        expected_generation=7,
    ) is False

    assert session.truncate_calls == [("r-7", 120)]
    _wait_until(lambda: session.deleted_items == ["q-7"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    assert session.closed is False
    coordinator.close()


def test_zero_played_audio_never_calls_truncate_and_deletes_unheard_turn() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=8))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=8,
            transcript="unheard",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=8,
            text="response",
        )
    )
    assert coordinator.approve_general_chat(
        "unheard",
        expected_generation=8,
        wait_timeout=0.5,
    ) is not None

    assert coordinator.truncate_current(
        "pre_playback_cancel",
        audio_end_ms=0,
        expected_generation=8,
    ) is False

    assert session.truncate_calls == []
    _wait_until(lambda: session.deleted_items == ["q-8"])
    coordinator.close()


def test_unapproved_interrupted_response_is_rolled_back_before_fallback() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=9))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=9,
            transcript="hello",
        )
    )
    session.emit(_event(RealtimeVoiceEventType.INTERRUPTED, generation=9))

    assert coordinator.approve_general_chat(
        "hello",
        expected_generation=9,
        wait_timeout=0.5,
    ) is None

    _wait_until(lambda: session.deleted_items == ["q-9"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    coordinator.close()


def test_approval_rejects_a_newer_full_duplex_generation() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=8))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=8,
            transcript="重复一句",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=8,
            text="好的。",
        )
    )

    approval = coordinator.approve_general_chat(
        "重复一句",
        expected_generation=7,
        wait_timeout=0.5,
    )

    assert approval is None
    assert rendered == []
    assert coordinator.status_snapshot()["generation"] == 8
    _wait_until(lambda: session.closed)
    assert session.deleted_items == []
    assert coordinator.status_snapshot()["last_error"] == "rollback_generation_mismatch"
    assert session.close_reasons == ["rollback_generation_mismatch"]


def test_pending_audio_prefix_cannot_be_overtaken_by_live_audio() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    first_pending_entered = threading.Event()
    release_pending = threading.Event()

    def _sink(frame: VoiceMediaFrame, _final: bool) -> None:
        rendered.append(frame.pcm)
        if frame.pcm == b"pending":
            first_pending_entered.set()
            assert release_pending.wait(timeout=1.0)

    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        audio_sink=_sink,
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=9))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            generation=9,
            transcript="你好",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            generation=9,
            text="你好。",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            generation=9,
            pcm=b"pending",
        )
    )
    _wait_until(lambda: coordinator.status_snapshot()["pending_audio_frames"] == 1)

    approval_result: list[object] = []
    approval_thread = threading.Thread(
        target=lambda: approval_result.append(
            coordinator.approve_general_chat(
                "你好",
                expected_generation=9,
                wait_timeout=0.5,
            )
        )
    )
    approval_thread.start()
    assert first_pending_entered.wait(timeout=1.0)
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            generation=9,
            pcm=b"live",
        )
    )
    time.sleep(0.03)
    assert rendered == [b"pending"]

    release_pending.set()
    approval_thread.join(timeout=1.0)
    _wait_until(lambda: rendered == [b"pending", b"live"])
    assert approval_result[0] is not None
    coordinator.close()


def test_pending_audio_limit_falls_back_instead_of_building_latency() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
        pending_output_ms=20,
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            transcript="你好",
        )
    )
    session.emit(
        _event(RealtimeVoiceEventType.OUTPUT_AUDIO, pcm=b"\x01\x00" * 480)
    )
    session.emit(
        _event(RealtimeVoiceEventType.OUTPUT_AUDIO, pcm=b"\x02\x00" * 480)
    )
    _wait_until(lambda: coordinator.status_snapshot()["overflow_count"] == 1)

    assert coordinator.approve_general_chat("你好", wait_timeout=0.2) is None
    assert session.interrupt_reasons[-1] == "pending_audio_overflow"
    _wait_until(lambda: session.deleted_items == ["q-1"])
    snapshot = coordinator.status_snapshot()
    assert snapshot["pending_audio_ms"] == 0.0
    assert "你好" not in repr(snapshot)
    coordinator.close()


def test_close_is_idempotent_and_stops_consumer() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="shadow",
    )
    coordinator.start()

    coordinator.close()
    coordinator.close()

    assert session.closed is True
    assert coordinator.status_snapshot()["active"] is False


def test_provider_error_closes_optional_lane_and_keeps_cascade_available() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED))
    session.emit(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.ERROR,
            session_id="session-1",
            generation=1,
            provider="volcengine_s2s",
            error="provider transport failed",
        )
    )

    _wait_until(lambda: session.closed)

    snapshot = coordinator.status_snapshot()
    assert snapshot["active"] is False
    assert snapshot["quarantined"] is True
    assert snapshot["last_error"] == "provider transport failed"
    assert session.close_reasons == ["provider_error"]
    assert coordinator.offer_audio(
        VoiceMediaFrame(pcm=b"\x00\x00" * 160, sample_rate=16_000)
    ) is False


def test_recover_at_turn_boundary_uses_fresh_empty_context_and_epoch() -> None:
    session = _FakeSession()
    rendered: list[bytes] = []
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(
            session_id="session-1",
            dialog_id="dialog-old",
        ),
        mode="general_chat",
        audio_sink=lambda frame, _final: rendered.append(frame.pcm),
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=9))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 9)
    coordinator.close("provider_error")

    assert coordinator.recover_at_turn_boundary() is True
    snapshot = coordinator.status_snapshot()
    recovered_session_id = str(snapshot["session_id"])

    assert snapshot["active"] is True
    assert snapshot["recovery_count"] == 1
    assert snapshot["generation_epoch"] == 9
    assert recovered_session_id != "session-1"
    assert snapshot["dialog_id"] == ""
    assert session.start_contexts[-1].session_id == recovered_session_id
    assert session.start_contexts[-1].dialog_id == ""

    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
            session_id="session-1",
            generation=10,
        )
    )
    time.sleep(0.03)
    assert coordinator.status_snapshot()["generation"] == 0

    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
            session_id=recovered_session_id,
            generation=1,
        )
    )
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 10)
    session.emit(
        _event(
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            session_id=recovered_session_id,
            generation=1,
            transcript="你好",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.RESPONSE_STARTED,
            session_id=recovered_session_id,
            generation=1,
            text="你好。",
        )
    )
    session.emit(
        _event(
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            session_id="session-1",
            generation=10,
            pcm=b"old-audio",
        )
    )
    time.sleep(0.03)
    assert rendered == []
    assert coordinator.status_snapshot()["dropped_late_audio"] == 1
    coordinator.close()


def test_initial_start_failure_can_recover_at_the_next_turn_boundary() -> None:
    session = _FailsFirstStartSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(
            session_id="session-1",
            dialog_id="dialog-old",
        ),
        mode="general_chat",
    )

    assert coordinator.start() is False
    failed = coordinator.status_snapshot()
    assert failed["active"] is False
    assert failed["quarantined"] is True

    assert coordinator.recover_at_turn_boundary() is True
    recovered = coordinator.status_snapshot()
    assert recovered["active"] is True
    assert recovered["session_id"] != "session-1"
    assert recovered["dialog_id"] == ""
    assert len(session.start_contexts) == 2
    coordinator.close()


def test_recovery_waits_for_the_previous_event_consumer_to_exit() -> None:
    session = _BlockingCloseSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    assert coordinator.start() is True
    session.emit(_event(RealtimeVoiceEventType.SESSION_CLOSED))
    assert session.close_started.wait(timeout=1.0)

    recovered: list[bool] = []
    worker = threading.Thread(
        target=lambda: recovered.append(coordinator.recover_at_turn_boundary())
    )
    worker.start()
    time.sleep(0.05)

    assert len(session.start_contexts) == 1
    session.close_release.set()
    worker.join(timeout=1.0)

    assert recovered == [True]
    assert len(session.start_contexts) == 2
    coordinator.close()


def test_recovery_is_refused_while_active_or_rollback_is_pending() -> None:
    active_session = _FakeSession()
    active = RealtimeDialogueCoordinator(
        active_session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    active.start()
    assert active.recover_at_turn_boundary() is False
    active.close()

    session = _BlockingDeleteSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=1))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 1)
    coordinator.discard_current("reject_one", expected_generation=1)
    assert session.first_delete_started.wait(timeout=1.0)

    coordinator.close("provider_error")
    assert coordinator.recover_at_turn_boundary() is False
    session.release_first_delete.set()
    _wait_until(lambda: coordinator.status_snapshot()["rollback_queue_depth"] == 0)


def test_event_consumer_exception_closes_optional_lane() -> None:
    session = _ExplodingSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )

    assert coordinator.start() is True
    _wait_until(lambda: session.closed)

    snapshot = coordinator.status_snapshot()
    assert snapshot["active"] is False
    assert snapshot["last_error"] == "RuntimeError"
    assert session.close_reasons == ["event_consumer_failure"]


def test_unbound_rejected_turn_waits_for_provider_generation_then_deletes_it() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=2))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 2)

    coordinator.discard_current("local_general_fallback", after_generation=2)
    assert coordinator.offer_audio(
        VoiceMediaFrame(pcm=b"\x00\x00" * 160, sample_rate=16_000)
    ) is False
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=3))

    _wait_until(lambda: session.deleted_items == ["q-3"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    assert coordinator.status_snapshot()["rollback_count"] == 1
    coordinator.close()


def test_unbound_rejection_without_cloud_turn_closes_for_fresh_recovery() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=5))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 5)

    coordinator.discard_current("local_timeout", after_generation=5)
    assert coordinator.status_snapshot()["quarantined"] is True

    _wait_until(lambda: session.closed, timeout=2.5)

    snapshot = coordinator.status_snapshot()
    assert snapshot["active"] is False
    assert snapshot["rollback_failures"] == 1
    assert snapshot["last_error"] == "conversation_rollback_failed"
    assert session.deleted_items == []
    assert session.close_reasons == ["conversation_rollback_failed"]


def test_deferred_rollback_never_deletes_a_skipped_newer_generation() -> None:
    session = _FakeSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=2))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 2)

    coordinator.discard_current("local_fallback", after_generation=2)
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=4))

    _wait_until(lambda: session.closed)
    assert session.deleted_items == []
    assert coordinator.status_snapshot()["rollback_failures"] == 1
    assert session.close_reasons == ["conversation_rollback_failed"]


def test_consecutive_rollbacks_are_serialized_without_losing_generation() -> None:
    session = _BlockingDeleteSession()
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=1))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 1)

    coordinator.discard_current("reject_one", expected_generation=1)
    assert session.first_delete_started.wait(timeout=1.0)

    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=2))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 2)
    coordinator.discard_current("reject_two", expected_generation=2)
    session.release_first_delete.set()

    _wait_until(lambda: session.deleted_items == ["q-1", "q-2"])
    _wait_until(lambda: coordinator.status_snapshot()["quarantined"] is False)
    assert coordinator.status_snapshot()["rollback_count"] == 2
    assert session.closed is False
    coordinator.close()


def test_failed_history_rollback_closes_optional_session() -> None:
    session = _FakeSession()
    session.delete_success = False
    coordinator = RealtimeDialogueCoordinator(
        session,
        RealtimeVoiceSessionContext(session_id="session-1"),
        mode="general_chat",
    )
    coordinator.start()
    session.emit(_event(RealtimeVoiceEventType.INPUT_SPEECH_STARTED, generation=3))
    _wait_until(lambda: coordinator.status_snapshot()["generation"] == 3)

    coordinator.discard_current("unsafe_intent", expected_generation=3)

    _wait_until(lambda: session.closed)
    snapshot = coordinator.status_snapshot()
    assert session.deleted_items == ["q-3"]
    assert snapshot["active"] is False
    assert snapshot["rollback_failures"] == 1
    assert snapshot["last_error"] == "conversation_rollback_failed"
    assert session.close_reasons == ["conversation_rollback_failed"]
