from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.config import resolve_realtime_voice_config
from askme.voice.realtime.protocol import (
    MessageType,
    RealtimeEvent,
    decode_frame,
    encode_frame,
)
from askme.voice.realtime.volcengine import VolcengineRealtimeDialogue

CONNECTION_STARTED = 50
CONNECTION_FINISHED = 52
SESSION_STARTED = 150
SESSION_FINISHED = 152
USAGE_RESPONSE = 154
TTS_SENTENCE_START = 350
CHAT_RESPONSE = 550


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.incoming: queue.Queue[bytes] = queue.Queue()
        self.closed = False
        self.timeout = None
        self.task_send_gate: threading.Event | None = None
        self.auto_delete_ack = True
        self.auto_truncate_ack = True
        self.delete_ack_payload: dict | None = None
        self.truncate_ack_payload: dict | None = None
        self.recv_error: BaseException | None = None
        self.fail_send_events: set[int | RealtimeEvent] = set()
        self.handshake_response = SimpleNamespace(headers={"X-Tt-Logid": "provider-log-1"})

    def send_binary(self, payload: bytes) -> None:
        frame = decode_frame(payload)
        if frame.event == RealtimeEvent.TASK_REQUEST and self.task_send_gate is not None:
            self.task_send_gate.wait(timeout=1.0)
        if frame.event in self.fail_send_events:
            raise OSError("send failed")
        self.sent.append(payload)
        if frame.event == RealtimeEvent.START_CONNECTION:
            self.emit(CONNECTION_STARTED, {})
        elif frame.event == RealtimeEvent.START_SESSION:
            self.emit(SESSION_STARTED, {"dialog_id": "dialog-1"}, session_id=frame.session_id)
        elif frame.event == RealtimeEvent.FINISH_SESSION:
            self.emit(SESSION_FINISHED, {}, session_id=frame.session_id)
        elif frame.event == RealtimeEvent.FINISH_CONNECTION:
            self.emit(CONNECTION_FINISHED, {})
        elif frame.event == RealtimeEvent.CONVERSATION_DELETE and self.auto_delete_ack:
            self.emit(
                RealtimeEvent.CONVERSATION_DELETED,
                self.delete_ack_payload or {"items": frame.payload["items"]},
                session_id=frame.session_id,
            )
        elif frame.event == RealtimeEvent.CONVERSATION_TRUNCATE and self.auto_truncate_ack:
            self.emit(
                RealtimeEvent.CONVERSATION_TRUNCATED,
                self.truncate_ack_payload
                or {
                    "item_id": frame.payload["item_id"],
                    "audio_end_ms": frame.payload["audio_end_ms"],
                },
                session_id=frame.session_id,
            )

    def recv(self) -> bytes:
        if self.recv_error is not None:
            error = self.recv_error
            self.recv_error = None
            raise error
        try:
            return self.incoming.get(timeout=0.05)
        except queue.Empty as exc:
            raise TimeoutError("fake websocket timeout") from exc

    def settimeout(self, value) -> None:
        self.timeout = value

    def close(self) -> None:
        self.closed = True

    def emit(
        self,
        event: int | RealtimeEvent,
        payload,
        *,
        session_id: str | None = None,
        audio: bool = False,
    ) -> None:
        self.incoming.put(
            encode_frame(
                event,
                payload,
                session_id=session_id,
                message_type=(
                    MessageType.AUDIO_ONLY_RESPONSE if audio else MessageType.FULL_SERVER_RESPONSE
                ),
            )
        )


class _ConnectionFactory:
    def __init__(self, sockets: list[_FakeWebSocket] | None = None) -> None:
        self.sockets = [_FakeWebSocket()] if sockets is None else sockets
        self.calls: list[dict] = []

    def __call__(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        if not self.sockets:
            raise OSError("no socket")
        return self.sockets.pop(0)


def _config(**overrides):
    realtime = {
        "enabled": True,
        "mode": "general_chat",
        "app_id": "app-secret",
        "access_token": "access-secret",
        "close_timeout_s": 0.1,
        "max_reconnect_attempts": 0,
        **overrides,
    }
    return resolve_realtime_voice_config({"voice": {"realtime": realtime}})


def _context() -> RealtimeVoiceSessionContext:
    return RealtimeVoiceSessionContext(
        session_id="session-1",
        bot_name="小算",
        system_role="你是机器人的语音助手。",
        speaking_style="简洁自然。",
    )


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_session_authenticates_and_performs_official_two_stage_handshake() -> None:
    ws = _FakeWebSocket()
    factory = _ConnectionFactory([ws])
    session = VolcengineRealtimeDialogue(_config(), connection_factory=factory)

    assert session.start(_context()) is True

    call = factory.calls[0]
    assert call["endpoint"] == "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
    headers = "\n".join(call["header"])
    assert "X-Api-App-ID: app-secret" in headers
    assert "X-Api-Access-Key: access-secret" in headers
    assert "X-Api-Resource-Id: volc.speech.dialog" in headers
    assert "X-Api-App-Key: PlgvMymc7f3tQnJ6" in headers
    assert "X-Api-Connect-Id:" in headers

    sent = [decode_frame(item) for item in ws.sent]
    assert [item.event for item in sent[:2]] == [
        RealtimeEvent.START_CONNECTION,
        RealtimeEvent.START_SESSION,
    ]
    start_payload = sent[1].payload
    assert start_payload["dialog"]["extra"] == {
        "model": "1.2.1.1",
        "input_mod": "audio",
        "enable_conversation_truncate": True,
    }
    assert "不得调用任何工具" in start_payload["dialog"]["system_role"]
    assert "不得执行、控制或声称已经执行机器人" in start_payload["dialog"]["system_role"]
    assert start_payload["asr"]["extra"]["end_smooth_window_ms"] == 800
    assert start_payload["asr"]["extra"]["enable_custom_vad"] is True
    assert start_payload["tts"]["audio_config"] == {
        "channel": 1,
        "format": "pcm_s16le",
        "sample_rate": 24_000,
    }
    assert start_payload["tts"]["extra"] == {}

    snapshot = session.status_snapshot()
    assert snapshot["active"] is True
    assert snapshot["log_id"] == "provider-log-1"
    assert "app-secret" not in repr(snapshot)
    assert "access-secret" not in repr(snapshot)
    session.close()


def test_audio_is_repacked_into_nonblocking_official_20ms_frames() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(
        _config(audio_queue_ms=40),
        connection_factory=_ConnectionFactory([ws]),
    )
    assert session.start(_context()) is True

    pcm_40ms = b"\x01\x00" * 640
    assert session.offer_audio(VoiceMediaFrame(pcm=pcm_40ms, sample_rate=16_000, channels=1))
    _wait_until(
        lambda: (
            len(
                [item for item in ws.sent if decode_frame(item).event == RealtimeEvent.TASK_REQUEST]
            )
            == 2
        )
    )

    audio_frames = [
        decode_frame(item)
        for item in ws.sent
        if decode_frame(item).event == RealtimeEvent.TASK_REQUEST
    ]
    assert [len(item.payload) for item in audio_frames] == [640, 640]
    assert all(item.session_id == "session-1" for item in audio_frames)
    session.close()


def test_audio_offer_and_finish_propagate_outbound_enqueue_failure() -> None:
    session = VolcengineRealtimeDialogue(_config())
    session._active = True
    session._context = _context()
    session._send_event = lambda *_args, **_kwargs: False  # type: ignore[method-assign]

    assert session.offer_audio(
        VoiceMediaFrame(pcm=b"\x01\x00" * 320, sample_rate=16_000, channels=1)
    ) is False
    assert session.status_snapshot()["last_error"] == "input_audio_enqueue_failed"

    session._audio_buffer.extend(b"\x01\x00")
    assert session.finish_input() is False
    assert (
        session.status_snapshot()["last_error"]
        == "finish_input_audio_enqueue_failed"
    )


def test_ptt_finish_input_is_ordered_after_all_queued_audio() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(
        _config(audio_queue_ms=40),
        connection_factory=_ConnectionFactory([ws]),
    )
    assert session.start(replace(_context(), input_mode="push_to_talk")) is True
    ws.task_send_gate = threading.Event()

    assert session.offer_audio(
        VoiceMediaFrame(pcm=b"\x01\x00" * 640, sample_rate=16_000, channels=1)
    )
    session.finish_input()
    time.sleep(0.02)
    ws.task_send_gate.set()

    _wait_until(lambda: any(decode_frame(item).event == RealtimeEvent.END_ASR for item in ws.sent))
    runtime_events = [
        decode_frame(item).event
        for item in ws.sent
        if decode_frame(item).event in {RealtimeEvent.TASK_REQUEST, RealtimeEvent.END_ASR}
    ]
    assert runtime_events == [
        RealtimeEvent.TASK_REQUEST,
        RealtimeEvent.TASK_REQUEST,
        RealtimeEvent.END_ASR,
    ]
    session.close()


def test_control_frames_survive_a_saturated_media_queue() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(
        _config(audio_queue_ms=20),
        connection_factory=_ConnectionFactory([ws]),
    )
    assert session.start(replace(_context(), input_mode="push_to_talk")) is True
    ws.task_send_gate = threading.Event()

    assert session.offer_audio(
        VoiceMediaFrame(pcm=b"\x01\x00" * 3_200, sample_rate=16_000, channels=1)
    )
    session.finish_input()
    session.interrupt("barge_in")
    ws.task_send_gate.set()

    _wait_until(
        lambda: {
            RealtimeEvent.END_ASR,
            RealtimeEvent.CLIENT_INTERRUPT,
        }.issubset({decode_frame(item).event for item in ws.sent})
    )
    runtime_events = [
        decode_frame(item).event
        for item in ws.sent
        if decode_frame(item).event
        in {
            RealtimeEvent.TASK_REQUEST,
            RealtimeEvent.END_ASR,
            RealtimeEvent.CLIENT_INTERRUPT,
        }
    ]
    assert runtime_events.index(RealtimeEvent.END_ASR) > runtime_events.index(
        RealtimeEvent.TASK_REQUEST
    )
    assert RealtimeEvent.CLIENT_INTERRUPT in runtime_events
    assert session.status_snapshot()["dropped_input_frames"] > 0
    session.close()


def test_server_events_are_normalized_to_provider_neutral_contracts() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    ws.emit(RealtimeEvent.ASR_INFO, {"question_id": "q1"}, session_id="session-1")
    ws.emit(
        RealtimeEvent.ASR_RESPONSE,
        {"results": [{"text": "你好", "is_interim": False}]},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"text": "你好呀。", "question_id": "q1", "reply_id": "r1"},
        session_id="session-1",
    )
    ws.emit(
        CHAT_RESPONSE,
        {"content": "今天想聊什么？", "question_id": "q1", "reply_id": "r1"},
        session_id="session-1",
    )
    ws.emit(
        RealtimeEvent.TTS_RESPONSE,
        b"\x01\x00" * 480,
        session_id="session-1",
        audio=True,
    )
    ws.emit(
        USAGE_RESPONSE,
        {"usage": {"input_audio_tokens": 12, "output_audio_tokens": 8}},
        session_id="session-1",
    )
    ws.emit(
        RealtimeEvent.TTS_ENDED,
        {"question_id": "q1", "reply_id": "r1"},
        session_id="session-1",
    )

    events = [session.next_event(timeout=1.0) for _ in range(7)]
    assert [event.event_type for event in events if event is not None] == [
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
        RealtimeVoiceEventType.RESPONSE_STARTED,
        RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
        RealtimeVoiceEventType.OUTPUT_AUDIO,
        RealtimeVoiceEventType.USAGE,
        RealtimeVoiceEventType.RESPONSE_DONE,
    ]
    assert events[1].transcript == "你好"
    assert events[4].audio.pcm == b"\x01\x00" * 480
    assert events[4].audio.sample_rate == 24_000
    assert events[5].metadata["usage"]["input_audio_tokens"] == 12
    session.close()


def test_interrupt_generation_fence_drops_late_provider_audio() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True
    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "old-question"},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"text": "旧回答", "reply_id": "old"},
        session_id="session-1",
    )
    assert session.next_event(timeout=1.0).event_type is RealtimeVoiceEventType.INPUT_SPEECH_STARTED
    assert session.next_event(timeout=1.0).event_type is RealtimeVoiceEventType.RESPONSE_STARTED

    session.interrupt("barge_in")
    assert session.status_snapshot()["generation"] == 1
    ws.emit(
        RealtimeEvent.TTS_RESPONSE,
        b"late-audio",
        session_id="session-1",
        audio=True,
    )
    time.sleep(0.05)

    assert session.next_event(timeout=0.05) is None
    assert session.status_snapshot()["dropped_stale_audio_frames"] == 1
    session.close()


def test_question_ids_own_generations_and_tts_end_clears_active_response() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-1"},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"question_id": "question-1", "reply_id": "reply-1"},
        session_id="session-1",
    )
    assert session.next_event(timeout=1.0).generation == 1
    assert session.next_event(timeout=1.0).generation == 1

    session.interrupt("barge_in")
    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-1"},
        session_id="session-1",
    )
    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-2"},
        session_id="session-1",
    )
    ws.emit(
        CHAT_RESPONSE,
        {"question_id": "question-1", "reply_id": "reply-1", "content": "late"},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"question_id": "question-2", "reply_id": "reply-2"},
        session_id="session-1",
    )

    new_input = session.next_event(timeout=1.0)
    old_chat = session.next_event(timeout=1.0)
    new_response = session.next_event(timeout=1.0)
    assert new_input.generation == 2
    assert old_chat.generation == 1
    assert new_response.generation == 2
    assert session.status_snapshot()["active_response_generation"] == 2

    ws.emit(
        RealtimeEvent.TTS_ENDED,
        {"question_id": "question-2", "reply_id": "reply-2"},
        session_id="session-1",
    )
    assert session.next_event(timeout=1.0).generation == 2
    assert session.status_snapshot()["active_response_generation"] == 0
    session.close()


def test_invalid_media_shape_fails_closed_without_sending() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True
    before = len(ws.sent)

    assert (
        session.offer_audio(VoiceMediaFrame(pcm=b"\x00\x00" * 480, sample_rate=48_000, channels=1))
        is False
    )

    assert len(ws.sent) == before
    assert session.status_snapshot()["last_error"] == "invalid_input_audio_shape"
    session.close()


def test_conversation_delete_rolls_back_one_complete_provider_turn() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    assert session.delete_conversation_turn("question-7", timeout=0.5) is True

    delete_frame = next(
        decode_frame(item)
        for item in ws.sent
        if decode_frame(item).event == RealtimeEvent.CONVERSATION_DELETE
    )
    assert delete_frame.payload == {"items": [{"item_id": "question-7"}]}
    assert session.status_snapshot()["conversation_delete_count"] == 1
    session.close()


def test_conversation_delete_rejects_failed_or_empty_acknowledgement() -> None:
    for payload in (
        {"status_code": 500, "items": [{"item_id": "question-7"}]},
        {"status_code": 0, "items": []},
    ):
        ws = _FakeWebSocket()
        ws.delete_ack_payload = payload
        session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
        assert session.start(_context()) is True

        assert session.delete_conversation_turn("question-7", timeout=0.2) is False
        assert session.status_snapshot()["conversation_delete_failures"] == 1
        session.close()


def test_late_delete_ack_cannot_complete_a_new_delete_request() -> None:
    ws = _FakeWebSocket()
    ws.auto_delete_ack = False
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True
    assert session.delete_conversation_turn("old-question", timeout=0.05) is False

    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(session.delete_conversation_turn("new-question", timeout=0.5))
    )
    worker.start()
    _wait_until(
        lambda: any(
            decode_frame(item).event == RealtimeEvent.CONVERSATION_DELETE
            and decode_frame(item).payload["items"][0]["item_id"] == "new-question"
            for item in ws.sent
        )
    )

    ws.emit(
        RealtimeEvent.CONVERSATION_DELETED,
        {"status_code": 0, "items": [{"item_id": "old-question"}]},
        session_id="session-1",
    )
    time.sleep(0.03)
    assert worker.is_alive()
    ws.emit(
        RealtimeEvent.CONVERSATION_DELETED,
        {"status_code": 0, "items": [{"item_id": "new-question"}]},
        session_id="session-1",
    )
    worker.join(timeout=1.0)

    assert result == [True]
    session.close()


def test_late_unidentified_delete_failure_cannot_reject_a_new_request() -> None:
    ws = _FakeWebSocket()
    ws.auto_delete_ack = False
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True
    assert session.delete_conversation_turn("old-question", timeout=0.05) is False

    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(session.delete_conversation_turn("new-question", timeout=0.5))
    )
    worker.start()
    _wait_until(
        lambda: any(
            decode_frame(item).event == RealtimeEvent.CONVERSATION_DELETE
            and decode_frame(item).payload["items"][0]["item_id"] == "new-question"
            for item in ws.sent
        )
    )

    ws.emit(
        RealtimeEvent.CONVERSATION_DELETED,
        {"status_code": 500, "items": []},
        session_id="session-1",
    )
    time.sleep(0.03)
    assert worker.is_alive()
    ws.emit(
        RealtimeEvent.CONVERSATION_DELETED,
        {"status_code": 0, "items": [{"item_id": "new-question"}]},
        session_id="session-1",
    )
    worker.join(timeout=1.0)

    assert result == [True]
    session.close()


def test_ptt_interrupt_and_truncate_use_official_payloads() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(replace(_context(), input_mode="push_to_talk")) is True

    session.interrupt("barge_in")
    assert session.truncate_response(reply_id="reply-3", audio_end_ms=420) is True

    frames = [decode_frame(item) for item in ws.sent]
    interrupt = next(item for item in frames if item.event == RealtimeEvent.CLIENT_INTERRUPT)
    truncate = next(item for item in frames if item.event == RealtimeEvent.CONVERSATION_TRUNCATE)
    assert interrupt.payload == {}
    assert truncate.payload == {"item_id": "reply-3", "audio_end_ms": 420}
    session.close()


def test_truncate_requires_matching_successful_570_acknowledgement() -> None:
    for payload in (
        {"status_code": 500, "item_id": "reply-3"},
        {"status_code": 0, "item_id": "different-reply"},
        {"status_code": 0},
    ):
        ws = _FakeWebSocket()
        ws.truncate_ack_payload = payload
        session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
        assert session.start(_context()) is True

        assert (
            session.truncate_response(
                reply_id="reply-3",
                audio_end_ms=420,
                timeout=0.1,
            )
            is False
        )
        session.close()


def test_session_rejects_tool_or_hardware_capabilities() -> None:
    session = VolcengineRealtimeDialogue(
        _config(), connection_factory=_ConnectionFactory([_FakeWebSocket()])
    )

    assert session.start(replace(_context(), allow_tool_calls=True)) is False
    assert session.status_snapshot()["last_error"] == ("unsafe_session_capabilities_requested")


def test_duplicate_asr_info_for_same_question_does_not_advance_generation() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-1"},
        session_id="session-1",
    )
    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-1"},
        session_id="session-1",
    )
    first = session.next_event(timeout=1.0)
    time.sleep(0.05)

    assert first is not None
    assert first.generation == 1
    assert session.next_event(timeout=0.05) is None
    assert session.status_snapshot()["generation"] == 1
    session.close()


def test_only_new_question_id_asr_info_advances_generation() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    ws.emit(
        RealtimeEvent.ASR_RESPONSE,
        {"results": [{"text": "orphan transcript", "is_interim": False}]},
        session_id="session-1",
    )
    time.sleep(0.03)

    assert session.next_event(timeout=0.05) is None
    assert session.status_snapshot()["generation"] == 0
    session.close()


def test_runtime_transport_protocol_and_provider_errors_degrade_for_fallback() -> None:
    failures = (
        lambda ws: setattr(ws, "recv_error", OSError("connection reset")),
        lambda ws: ws.incoming.put("unexpected-text-frame"),
        lambda ws: ws.incoming.put(b"not-a-protocol-frame"),
        lambda ws: ws.incoming.put(
            encode_frame(None, {"message": "provider failed"}, error_code=45000001)
        ),
    )
    for inject in failures:
        ws = _FakeWebSocket()
        session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
        assert session.start(_context()) is True

        inject(ws)
        error = session.next_event(timeout=1.0)

        assert error is not None
        assert error.event_type is RealtimeVoiceEventType.ERROR
        snapshot = session.status_snapshot()
        assert snapshot["active"] is False
        assert snapshot["connected"] is False
        assert snapshot["state"] == "degraded"
        session.close()


def test_malformed_provider_event_payload_degrades_instead_of_killing_receiver() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    ws.emit(
        USAGE_RESPONSE,
        {"usage": "not-a-mapping"},
        session_id="session-1",
    )
    error = session.next_event(timeout=1.0)

    assert error is not None
    assert error.event_type is RealtimeVoiceEventType.ERROR
    assert error.error == "provider_event_error"
    assert session.status_snapshot()["state"] == "degraded"
    session.close()


def test_runtime_send_failure_degrades_for_local_fallback() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True
    ws.fail_send_events.add(RealtimeEvent.TASK_REQUEST)

    assert session.offer_audio(
        VoiceMediaFrame(pcm=b"\x01\x00" * 320, sample_rate=16_000, channels=1)
    )
    error = session.next_event(timeout=1.0)

    assert error is not None
    assert error.event_type is RealtimeVoiceEventType.ERROR
    snapshot = session.status_snapshot()
    assert snapshot["active"] is False
    assert snapshot["state"] == "degraded"
    session.close()


def test_critical_events_survive_an_event_queue_filled_by_streaming_deltas() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(
        _config(event_queue_size=8),
        connection_factory=_ConnectionFactory([ws]),
    )
    assert session.start(_context()) is True

    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "setup-question"},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"question_id": "setup-question", "reply_id": "setup-reply"},
        session_id="session-1",
    )
    assert session.next_event(timeout=1.0) is not None
    assert session.next_event(timeout=1.0) is not None

    for index in range(12):
        ws.emit(
            CHAT_RESPONSE,
            {"question_id": "old", "content": f"delta-{index}"},
            session_id="session-1",
        )
        ws.emit(
            RealtimeEvent.TTS_RESPONSE,
            b"\x01\x00" * 160,
            session_id="session-1",
            audio=True,
        )
    ws.emit(
        RealtimeEvent.ASR_INFO,
        {"question_id": "question-2"},
        session_id="session-1",
    )
    ws.emit(
        TTS_SENTENCE_START,
        {"question_id": "question-2", "reply_id": "reply-2"},
        session_id="session-1",
    )
    ws.emit(
        RealtimeEvent.TTS_ENDED,
        {"question_id": "question-2", "reply_id": "reply-2"},
        session_id="session-1",
    )
    ws.incoming.put(encode_frame(None, {"message": "provider failed"}, error_code=45000001))
    _wait_until(lambda: session.status_snapshot()["state"] == "degraded")

    observed = []
    while (event := session.next_event(timeout=0.02)) is not None:
        observed.append(event.event_type)
    assert RealtimeVoiceEventType.INPUT_SPEECH_STARTED in observed
    assert RealtimeVoiceEventType.RESPONSE_DONE in observed
    assert RealtimeVoiceEventType.ERROR in observed
    session.close()


def test_close_waits_for_finish_events_and_is_idempotent() -> None:
    ws = _FakeWebSocket()
    session = VolcengineRealtimeDialogue(_config(), connection_factory=_ConnectionFactory([ws]))
    assert session.start(_context()) is True

    session.close("test")
    session.close("test-again")

    sent_events = [decode_frame(item).event for item in ws.sent]
    assert sent_events[-2:] == [
        RealtimeEvent.FINISH_SESSION,
        RealtimeEvent.FINISH_CONNECTION,
    ]
    assert ws.closed is True
    assert session.status_snapshot()["active"] is False
    finish_frames = [
        decode_frame(item)
        for item in ws.sent
        if decode_frame(item).event
        in {RealtimeEvent.FINISH_SESSION, RealtimeEvent.FINISH_CONNECTION}
    ]
    assert [frame.payload for frame in finish_frames] == [{}, {}]


def test_start_failure_opens_circuit_and_never_exposes_credentials() -> None:
    factory = _ConnectionFactory([])
    config = _config(circuit_failure_threshold=1, max_reconnect_attempts=0)
    session = VolcengineRealtimeDialogue(config, connection_factory=factory)

    assert session.start(_context()) is False
    assert session.start(_context()) is False

    snapshot = session.status_snapshot()
    assert snapshot["circuit_open"] is True
    assert snapshot["last_error"] in {"OSError", "circuit_open"}
    assert "app-secret" not in repr(snapshot)
    assert "access-secret" not in repr(snapshot)
