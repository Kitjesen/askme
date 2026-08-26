"""Behavior tests for the Volcengine Seeduplex 3.0 JSON session."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from typing import Any

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.volcengine_duplex import (
    VolcengineDuplexConfig,
    VolcengineDuplexDialogue,
)


class _NeverConnect:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.called = True
        raise AssertionError("invalid configuration must fail before network access")


class _HandshakeResponse:
    headers = {"X-Tt-Logid": "provider-log-1"}


class _FakeWebSocket:
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        self.incoming: queue.Queue[str] = queue.Queue()
        for event in incoming:
            self.push(event)
        self.sent: list[str] = []
        self._send_lock = threading.Lock()
        self.closed = False
        self.timeout = 0.1
        self.handshake_response = _HandshakeResponse()

    def send(self, payload: str) -> None:
        with self._send_lock:
            self.sent.append(payload)

    def recv(self) -> str:
        try:
            return self.incoming.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def close(self) -> None:
        self.closed = True

    def push(self, event: dict[str, Any]) -> None:
        self.incoming.put(json.dumps(event))


class _BlockingMuteWebSocket(_FakeWebSocket):
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        super().__init__(incoming)
        self.mute_started = threading.Event()
        self.release_mute = threading.Event()

    def send(self, payload: str) -> None:
        event = json.loads(payload)
        if event.get("type") == "input_audio_mute.commit":
            self.mute_started.set()
            self.release_mute.wait(timeout=1.0)
        super().send(payload)


class _ConnectionFactory:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws
        self.calls: list[dict[str, Any]] = []

    def __call__(self, endpoint: str, **kwargs: Any) -> _FakeWebSocket:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return self.ws


def _wait_until(predicate: Any, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_missing_api_key_fails_closed_without_network_or_secret_exposure() -> None:
    factory = _NeverConnect()
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key=""),
        connection_factory=factory,
    )

    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is False

    snapshot = session.status_snapshot()
    assert factory.called is False
    assert snapshot["state"] == "degraded"
    assert snapshot["last_error"] == "invalid_or_disabled_config"
    assert snapshot["credentials_configured"] is False
    assert "api_key" not in snapshot
    assert "secret" not in repr(snapshot).lower()


def test_start_uses_official_header_and_creates_a_safe_pcm_session() -> None:
    ws = _FakeWebSocket(
        [
            {
                "type": "session.created",
                "event_id": "server-event-1",
                "session": {"id": "dialog-1"},
            }
        ]
    )
    factory = _ConnectionFactory(ws)
    config = VolcengineDuplexConfig(enabled=True, api_key="top-secret")
    session = VolcengineDuplexDialogue(config, connection_factory=factory)

    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True

    assert factory.calls == [
        {
            "endpoint": config.endpoint,
            "header": ["X-Api-Key: top-secret"],
            "timeout": config.connect_timeout_s,
        }
    ]
    request = json.loads(ws.sent[0])
    assert request["type"] == "session.create"
    assert request["event_id"].startswith("event_")
    assert request["session"] == {
        "type": "realtime",
        "id": "local-1",
        "model": "1.2.6.1",
        "instructions": (
            "简洁、自然、口语化；不要声称已经执行机器人动作。\n"
            "安全边界：不得调用任何工具或外部服务；"
            "不得执行、控制或声称已经执行机器人、设备或现实世界动作。"
        ),
        "audio": {
            "input": {"format": {"type": "pcm", "rate": 16000}},
            "output": {
                "format": {"type": "pcm_s16le", "rate": 24000},
                "voice": "zh_male_xiaotian_jupiter_bigtts",
            },
        },
    }
    assert request["extension"] == {"asr": {}, "tts": {}, "dialog": {}}
    snapshot = session.status_snapshot()
    assert snapshot["state"] == "listening"
    assert snapshot["connected"] is True
    assert snapshot["active"] is True
    assert snapshot["dialog_id"] == "dialog-1"
    assert snapshot["provider_session_id"] == "dialog-1"
    assert snapshot["log_id"] == "provider-log-1"
    assert "top-secret" not in repr(snapshot)


def test_start_rejects_unsafe_capabilities_and_wrong_audio_shape_before_network() -> None:
    invalid_contexts = [
        RealtimeVoiceSessionContext(session_id="tools", allow_tool_calls=True),
        RealtimeVoiceSessionContext(session_id="hardware", allow_hardware_dispatch=True),
        RealtimeVoiceSessionContext(session_id="rate", input_sample_rate=48_000),
        RealtimeVoiceSessionContext(session_id="format", output_format="ogg_opus"),
    ]

    for context in invalid_contexts:
        factory = _NeverConnect()
        session = VolcengineDuplexDialogue(
            VolcengineDuplexConfig(enabled=True, api_key="configured"),
            connection_factory=factory,
        )

        assert session.start(context) is False
        assert factory.called is False
        assert session.status_snapshot()["last_error"] in {
            "unsafe_session_capabilities_requested",
            "session_audio_shape_mismatch",
        }


def test_provider_events_are_normalized_through_the_public_session_interface() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured"),
        connection_factory=_ConnectionFactory(ws),
    )
    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True

    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "event_id": "server-1",
            "item_id": "question-1",
        }
    )
    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "server-2",
            "item_id": "question-1",
            "delta": "你",
        }
    )
    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "server-3",
            "item_id": "question-1",
            "transcript": "你好",
        }
    )
    ws.push(
        {
            "type": "response.output_text.delta",
            "question_id": "question-1",
            "response_id": "response-1",
            "delta": "你好呀",
        }
    )
    ws.push(
        {
            "type": "response.output_audio.started",
            "question_id": "question-1",
            "response_id": "response-1",
        }
    )
    ws.push(
        {
            "type": "response.output_audio.delta",
            "question_id": "question-1",
            "response_id": "response-1",
            "delta": base64.b64encode(b"\x01\x02\x03\x04").decode("ascii"),
        }
    )
    ws.push(
        {
            "type": "response.done",
            "question_id": "question-1",
            "response_id": "response-1",
            "usage": {"audio_input_tokens": 12, "audio_output_tokens": 8},
        }
    )

    events = [session.next_event(timeout=1.0) for _ in range(8)]

    assert all(event is not None for event in events)
    assert [event.event_type for event in events if event is not None] == [
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
        RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
        RealtimeVoiceEventType.RESPONSE_STARTED,
        RealtimeVoiceEventType.OUTPUT_AUDIO,
        RealtimeVoiceEventType.USAGE,
        RealtimeVoiceEventType.RESPONSE_DONE,
    ]
    assert events[1].transcript == "你"
    assert events[2].transcript == "你好"
    assert events[2].is_final is True
    assert events[3].text == "你好呀"
    assert events[5].audio is not None
    assert events[5].audio.pcm == b"\x01\x02\x03\x04"
    assert events[5].audio.sample_rate == 24_000
    assert events[6].metadata["usage"] == {
        "audio_input_tokens": 12,
        "audio_output_tokens": 8,
    }
    assert {event.generation for event in events if event is not None} == {1}
    session.close("test")


def test_audio_commit_mute_and_next_turn_unmute_preserve_order() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured"),
        connection_factory=_ConnectionFactory(ws),
    )
    context = RealtimeVoiceSessionContext(session_id="local-1", input_mode="push_to_talk")
    assert session.start(context) is True

    assert (
        session.offer_audio(VoiceMediaFrame(pcm=b"A" * 960, sample_rate=16_000, channels=1)) is True
    )
    assert session.finish_input() is True
    assert (
        session.offer_audio(VoiceMediaFrame(pcm=b"B" * 640, sample_rate=16_000, channels=1)) is True
    )
    _wait_until(lambda: len(ws.sent) >= 7)

    events = [json.loads(payload) for payload in ws.sent[1:7]]
    assert [event["type"] for event in events] == [
        "input_audio_buffer.append",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_mute.commit",
        "input_audio_unmute.commit",
        "input_audio_buffer.append",
    ]
    assert all(event["event_id"].startswith("event_") for event in events)
    assert base64.b64decode(events[0]["audio"]) == b"A" * 640
    assert base64.b64decode(events[1]["audio"]) == b"A" * 320
    assert base64.b64decode(events[5]["audio"]) == b"B" * 640
    snapshot = session.status_snapshot()
    assert snapshot["sent_audio_frames"] == 3
    assert snapshot["input_muted"] is False
    assert snapshot["audio_buffer_bytes"] == 0
    session.close("test")


def test_mute_send_timeout_discards_the_unknown_provider_session() -> None:
    ws = _BlockingMuteWebSocket(
        [{"type": "session.created", "session": {"id": "dialog-1"}}]
    )
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(
            enabled=True,
            api_key="configured",
            close_timeout_s=0.05,
        ),
        connection_factory=_ConnectionFactory(ws),
    )
    context = RealtimeVoiceSessionContext(session_id="local-1", input_mode="push_to_talk")
    assert session.start(context) is True
    assert (
        session.offer_audio(VoiceMediaFrame(pcm=b"A" * 640, sample_rate=16_000, channels=1)) is True
    )

    try:
        assert session.finish_input() is False
        assert ws.mute_started.is_set() is True
        snapshot = session.status_snapshot()
        assert snapshot["state"] == "degraded"
        assert snapshot["connected"] is False
        assert snapshot["active"] is False
        assert snapshot["last_error"] == "finish_input_mute_timeout"
    finally:
        ws.release_mute.set()
        session.close("test")


def test_interrupt_sends_response_cancel_and_fences_late_audio() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured"),
        connection_factory=_ConnectionFactory(ws),
    )
    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True
    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "question-1",
        }
    )
    ws.push(
        {
            "type": "response.output_audio.started",
            "question_id": "question-1",
            "response_id": "response-1",
        }
    )
    assert session.next_event(timeout=1).event_type is RealtimeVoiceEventType.INPUT_SPEECH_STARTED
    assert session.next_event(timeout=1).event_type is RealtimeVoiceEventType.RESPONSE_STARTED

    session.interrupt("barge_in")
    _wait_until(lambda: len(ws.sent) >= 2)
    cancel = json.loads(ws.sent[1])
    assert cancel["type"] == "response.cancel"
    assert set(cancel) == {"type", "event_id"}

    ws.push(
        {
            "type": "response.output_audio.delta",
            "question_id": "question-1",
            "response_id": "response-1",
            "delta": base64.b64encode(b"late audio").decode("ascii"),
        }
    )
    ws.push({"type": "response.canceled", "event_id": cancel["event_id"]})

    interrupted = session.next_event(timeout=1)
    assert interrupted is not None
    assert interrupted.event_type is RealtimeVoiceEventType.INTERRUPTED
    assert interrupted.generation == 1
    assert session.next_event(timeout=0.1) is None
    assert session.status_snapshot()["dropped_stale_audio_frames"] == 1
    session.close("test")


def test_close_waits_for_session_closed_before_releasing_the_websocket() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured", close_timeout_s=0.5),
        connection_factory=_ConnectionFactory(ws),
    )
    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True

    closer = threading.Thread(target=session.close, args=("test",))
    closer.start()
    _wait_until(lambda: len(ws.sent) >= 2)
    close_event = json.loads(ws.sent[-1])
    assert close_event["type"] == "session.close"
    assert ws.closed is False

    ws.push({"type": "session.closed", "event_id": close_event["event_id"]})
    closer.join(timeout=1)

    assert closer.is_alive() is False
    assert ws.closed is True
    assert session.status_snapshot()["state"] == "closed"
    sent_count = len(ws.sent)
    session.close("again")
    assert len(ws.sent) == sent_count


def test_delete_conversation_turn_waits_for_the_correlated_provider_ack() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured"),
        connection_factory=_ConnectionFactory(ws),
    )
    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True
    outcome: list[bool] = []

    worker = threading.Thread(
        target=lambda: outcome.append(session.delete_conversation_turn("question-1", timeout=0.5))
    )
    worker.start()
    _wait_until(lambda: len(ws.sent) >= 2)
    request = json.loads(ws.sent[-1])
    assert request == {
        "type": "conversation.item.delete",
        "event_id": request["event_id"],
        "items": [{"id": "question-1"}],
    }
    ws.push(
        {
            "type": "conversation.item.deleted",
            "event_id": request["event_id"],
            "items": [{"id": "question-1"}],
        }
    )
    worker.join(timeout=1)

    assert outcome == [True]
    assert session.status_snapshot()["conversation_delete_count"] == 1
    session.close("test")


def test_low_value_audio_cannot_evict_terminal_or_fault_events() -> None:
    ws = _FakeWebSocket([{"type": "session.created", "session": {"id": "dialog-1"}}])
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(
            enabled=True,
            api_key="configured",
            event_queue_size=8,
        ),
        connection_factory=_ConnectionFactory(ws),
    )
    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True
    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "question-1",
        }
    )
    started = session.next_event(timeout=1)
    assert started is not None
    assert started.event_type is RealtimeVoiceEventType.INPUT_SPEECH_STARTED

    audio_delta = {
        "type": "response.output_audio.delta",
        "question_id": "question-1",
        "response_id": "response-1",
        "delta": base64.b64encode(b"audio").decode("ascii"),
    }
    for _ in range(8):
        ws.push(audio_delta)
    ws.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "question-1",
            "transcript": "最终问题",
        }
    )
    ws.push(
        {
            "type": "response.done",
            "question_id": "question-1",
            "response_id": "response-1",
            "usage": {"audio_input_tokens": 12},
        }
    )
    for _ in range(20):
        ws.push(audio_delta)
    ws.push(
        {
            "type": "error",
            "error": {"code": "52000022", "message": "model failure"},
        }
    )
    _wait_until(lambda: session.status_snapshot()["state"] == "degraded")

    queued = [session.next_event(timeout=0.1) for _ in range(8)]
    event_types = {event.event_type for event in queued if event is not None}

    assert RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL in event_types
    assert RealtimeVoiceEventType.USAGE in event_types
    assert RealtimeVoiceEventType.RESPONSE_DONE in event_types
    assert RealtimeVoiceEventType.ERROR in event_types
    assert session.status_snapshot()["dropped_events"] > 0
