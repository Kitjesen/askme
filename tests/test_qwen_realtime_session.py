from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import replace

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.qwen import QwenRealtimeConfig, QwenRealtimeDialogue


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: queue.Queue[str] = queue.Queue()
        self.closed = False
        self.timeout: float | None = None
        self.audio_send_gate: threading.Event | None = None
        self.audio_send_started = threading.Event()
        self.emit(
            {
                "type": "session.created",
                "event_id": "event-created",
                "session": {"id": "provider-session-1"},
            }
        )

    def send(self, payload: str) -> None:
        event = json.loads(payload)
        if event["type"] == "input_audio_buffer.append" and self.audio_send_gate is not None:
            self.audio_send_started.set()
            self.audio_send_gate.wait(timeout=1.0)
        self.sent.append(payload)
        if event["type"] == "session.update":
            self.emit(
                {
                    "type": "session.updated",
                    "event_id": "event-updated",
                    "session": {"id": "provider-session-1"},
                }
            )

    def recv(self) -> str:
        try:
            return self.incoming.get(timeout=0.05)
        except queue.Empty as exc:
            raise TimeoutError("fake websocket timeout") from exc

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def close(self) -> None:
        self.closed = True

    def emit(self, event: dict) -> None:
        self.incoming.put(json.dumps(event))


class _ConnectionFactory:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws
        self.calls: list[dict] = []

    def __call__(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        return self.ws


class _ClosingWebSocket(_FakeWebSocket):
    def send(self, payload: str) -> None:
        self.sent.append(payload)
        if json.loads(payload)["type"] == "session.update":
            self.incoming.put("")


def _config(**overrides) -> QwenRealtimeConfig:
    return QwenRealtimeConfig(
        enabled=True,
        api_key="dashscope-secret",
        close_timeout_s=0.1,
        max_reconnect_attempts=0,
        **overrides,
    )


def _context() -> RealtimeVoiceSessionContext:
    return RealtimeVoiceSessionContext(
        session_id="local-session-1",
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


def test_config_hides_api_key_and_rejects_non_official_websocket_endpoint() -> None:
    config = _config(endpoint="wss://attacker.example/realtime")
    workspace_config = _config(
        endpoint="wss://workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
    )

    assert "dashscope-secret" not in repr(config)
    assert config.available is False
    assert config.validation_errors() == [
        "qwen realtime endpoint must use the official DashScope realtime endpoint"
    ]
    assert workspace_config.available is True
    assert workspace_config.validation_errors() == []
    for endpoint in (
        "ws://workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        "wss://user@workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        "wss://workspace-123.cn-beijing.maas.aliyuncs.com:443/api-ws/v1/realtime",
        "wss://workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime#fragment",
        "wss://nested.workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
    ):
        assert _config(endpoint=endpoint).available is False


def test_session_authenticates_and_configures_official_realtime_protocol() -> None:
    ws = _FakeWebSocket()
    factory = _ConnectionFactory(ws)
    session = QwenRealtimeDialogue(_config(), connection_factory=factory)

    assert session.start(_context()) is True

    call = factory.calls[0]
    assert call["endpoint"].endswith("?model=qwen3.5-omni-flash-realtime")
    assert call["header"] == ["Authorization: Bearer dashscope-secret"]
    update = json.loads(ws.sent[0])
    assert update["type"] == "session.update"
    assert update["session"]["modalities"] == ["text", "audio"]
    assert update["session"]["voice"] == "Tina"
    assert update["session"]["input_audio_format"] == "pcm"
    assert update["session"]["output_audio_format"] == "pcm"
    assert update["session"]["input_audio_transcription"] == {"model": "qwen3-asr-flash-realtime"}
    assert update["session"]["turn_detection"] == {
        "type": "semantic_vad",
        "threshold": 0.1,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
    }
    assert "你的名字是小算" in update["session"]["instructions"]
    assert "不得调用任何工具" in update["session"]["instructions"]
    assert "不得执行、控制或声称已经执行机器人" in update["session"]["instructions"]
    assert "dashscope-secret" not in repr(session.status_snapshot())
    session.close()


def test_push_to_talk_streams_pcm_then_commits_and_creates_response() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(replace(_context(), input_mode="push_to_talk")) is True

    pcm_40_ms = b"\x01\x00" * 640
    assert session.offer_audio(VoiceMediaFrame(pcm=pcm_40_ms, sample_rate=16_000, channels=1))
    assert session.finish_input() is True

    events = [json.loads(item) for item in ws.sent]
    audio_events = [item for item in events if item["type"] == "input_audio_buffer.append"]
    assert len(audio_events) == 2
    assert all(len(base64.b64decode(item["audio"])) == 640 for item in audio_events)
    assert [item["type"] for item in events[-2:]] == [
        "input_audio_buffer.commit",
        "response.create",
    ]
    session.close()


def test_audio_capture_path_is_nonblocking_and_queue_overflow_fails_closed() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(
        _config(audio_queue_ms=20), connection_factory=_ConnectionFactory(ws)
    )
    assert session.start(_context()) is True
    ws.audio_send_gate = threading.Event()
    frame = VoiceMediaFrame(pcm=b"\x01\x00" * 320, sample_rate=16_000, channels=1)

    assert session.offer_audio(frame) is True
    assert ws.audio_send_started.wait(timeout=1.0)
    started_at = time.monotonic()
    assert session.offer_audio(frame) is True
    assert time.monotonic() - started_at < 0.05
    assert session.offer_audio(frame) is False

    snapshot = session.status_snapshot()
    assert snapshot["state"] == "degraded"
    assert snapshot["last_error"] == "input_audio_queue_overflow"
    assert snapshot["dropped_input_frames"] == 1
    ws.audio_send_gate.set()
    session.close()


def test_server_events_are_normalized_to_provider_neutral_contracts() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(_context()) is True

    ws.emit(
        {
            "type": "input_audio_buffer.speech_started",
            "event_id": "event-speech",
            "item_id": "question-1",
            "audio_start_ms": 25,
        }
    )
    ws.emit(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "event-asr-delta",
            "item_id": "question-1",
            "text": "你",
            "stash": "好",
        }
    )
    ws.emit(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "event-asr-final",
            "item_id": "question-1",
            "transcript": "你好",
        }
    )
    ws.emit(
        {
            "type": "response.created",
            "event_id": "event-response",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    ws.emit(
        {
            "type": "response.audio_transcript.delta",
            "event_id": "event-text",
            "response_id": "response-1",
            "item_id": "reply-1",
            "delta": "你好呀",
        }
    )
    audio = b"\x01\x00" * 480
    ws.emit(
        {
            "type": "response.audio.delta",
            "event_id": "event-audio",
            "response_id": "response-1",
            "item_id": "reply-1",
            "delta": base64.b64encode(audio).decode("ascii"),
        }
    )
    ws.emit(
        {
            "type": "response.done",
            "event_id": "event-done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "input_tokens_details": {"audio_tokens": 5},
                    "output_tokens_details": {"audio_tokens": 3},
                },
            },
        }
    )

    events = [session.next_event(timeout=1.0) for _ in range(8)]
    assert [event.event_type for event in events if event is not None] == [
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
        RealtimeVoiceEventType.RESPONSE_STARTED,
        RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
        RealtimeVoiceEventType.OUTPUT_AUDIO,
        RealtimeVoiceEventType.USAGE,
        RealtimeVoiceEventType.RESPONSE_DONE,
    ]
    assert events[1].transcript == "你好"
    assert events[2].transcript == "你好"
    assert events[5].audio is not None
    assert events[5].audio.pcm == audio
    assert events[5].audio.sample_rate == 24_000
    assert events[6].metadata["usage"]["input_tokens"] == 12
    assert all(event.generation == 1 for event in events if event is not None)
    session.close()


def test_final_audio_transcript_is_carried_by_response_done() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(_context()) is True

    ws.emit(
        {
            "type": "input_audio_buffer.speech_started",
            "item_id": "question-1",
        }
    )
    ws.emit(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    ws.emit(
        {
            "type": "response.audio_transcript.done",
            "response_id": "response-1",
            "item_id": "reply-1",
            "transcript": "这是完整回答。",
        }
    )
    ws.emit(
        {
            "type": "response.done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "output": [],
            },
        }
    )

    events = [session.next_event(timeout=1.0) for _ in range(4)]
    assert [event.event_type for event in events if event is not None] == [
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.RESPONSE_STARTED,
        RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
        RealtimeVoiceEventType.RESPONSE_DONE,
    ]
    assert events[2].text == "这是完整回答。"
    assert events[3].text == "这是完整回答。"
    session.close()


def test_response_done_recovers_nested_transcript_without_delta_events() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(_context()) is True

    ws.emit(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    ws.emit(
        {
            "type": "response.done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "audio", "transcript": "来自最终事件的回答。"}],
                    }
                ],
            },
        }
    )

    started = session.next_event(timeout=1.0)
    done = session.next_event(timeout=1.0)
    assert started.event_type is RealtimeVoiceEventType.RESPONSE_STARTED
    assert done.event_type is RealtimeVoiceEventType.RESPONSE_DONE
    assert done.text == "来自最终事件的回答。"
    session.close()


def test_bounded_event_queue_preserves_usage_completion_and_error() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(
        _config(event_queue_size=4), connection_factory=_ConnectionFactory(ws)
    )
    assert session.start(_context()) is True
    ws.emit({"type": "input_audio_buffer.speech_started", "item_id": "question-1"})
    ws.emit(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    assert session.next_event(timeout=1.0) is not None
    assert session.next_event(timeout=1.0) is not None

    for index in range(12):
        ws.emit(
            {
                "type": "response.audio_transcript.delta",
                "response_id": "response-1",
                "item_id": "reply-1",
                "delta": f"delta-{index}",
            }
        )
    ws.emit(
        {
            "type": "response.done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        }
    )
    ws.emit(
        {
            "type": "error",
            "error": {"code": "provider_failed", "message": "failed"},
        }
    )
    _wait_until(lambda: session.status_snapshot()["state"] == "degraded")

    observed = []
    while (event := session.next_event(timeout=0.01)) is not None:
        observed.append(event.event_type)
    assert len(observed) <= 4
    assert RealtimeVoiceEventType.USAGE in observed
    assert RealtimeVoiceEventType.RESPONSE_DONE in observed
    assert RealtimeVoiceEventType.ERROR in observed
    assert session.status_snapshot()["dropped_events"] >= 11
    session.close()


def test_interrupt_cancels_response_and_fences_late_audio() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(_context()) is True
    ws.emit(
        {
            "type": "input_audio_buffer.speech_started",
            "item_id": "question-1",
        }
    )
    ws.emit(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    assert session.next_event(timeout=1.0).event_type is RealtimeVoiceEventType.INPUT_SPEECH_STARTED
    assert session.next_event(timeout=1.0).event_type is RealtimeVoiceEventType.RESPONSE_STARTED

    session.interrupt("barge_in")
    ws.emit(
        {
            "type": "response.audio.delta",
            "response_id": "response-1",
            "item_id": "reply-1",
            "delta": base64.b64encode(b"\x01\x00" * 480).decode("ascii"),
        }
    )

    assert session.next_event(timeout=0.1) is None
    assert any(json.loads(item)["type"] == "response.cancel" for item in ws.sent)
    assert session.status_snapshot()["dropped_stale_audio_frames"] == 1
    session.close()


def test_provider_error_degrades_for_cascade_fallback_without_leaking_secret() -> None:
    ws = _FakeWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))
    assert session.start(_context()) is True

    ws.emit(
        {
            "type": "error",
            "event_id": "event-error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "bad session setting",
            },
        }
    )
    error = session.next_event(timeout=1.0)

    assert error is not None
    assert error.event_type is RealtimeVoiceEventType.ERROR
    assert error.error == "invalid_value"
    snapshot = session.status_snapshot()
    assert snapshot["state"] == "degraded"
    assert snapshot["active"] is False
    assert snapshot["connected"] is False
    assert "dashscope-secret" not in repr(snapshot)
    session.close()


def test_provider_close_during_handshake_has_a_stable_fallback_error() -> None:
    ws = _ClosingWebSocket()
    session = QwenRealtimeDialogue(_config(), connection_factory=_ConnectionFactory(ws))

    assert session.start(_context()) is False
    snapshot = session.status_snapshot()
    assert snapshot["state"] == "degraded"
    assert snapshot["last_error"] == "provider_connection_closed"
    assert "dashscope-secret" not in repr(snapshot)
