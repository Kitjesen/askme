"""Thread-safe adapter for Volcengine Doubao Seeduplex 3.0.

The provider protocol is JSON-over-WebSocket.  This module owns only protocol
translation; microphones, speakers, tools, and robot actions stay outside this
adapter.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)

DEFAULT_VOLCENGINE_DUPLEX_ENDPOINT = (
    "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
)
SUPPORTED_VOLCENGINE_DUPLEX_MODEL = "1.2.6.1"

_TERMINAL_EVENT_PRIORITY = frozenset(
    {
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
        RealtimeVoiceEventType.RESPONSE_DONE,
        RealtimeVoiceEventType.INTERRUPTED,
    }
)
_FAULT_EVENT_PRIORITY = frozenset(
    {
        RealtimeVoiceEventType.ERROR,
        RealtimeVoiceEventType.SESSION_CLOSED,
    }
)
_STATE_EVENT_PRIORITY = frozenset(
    {
        RealtimeVoiceEventType.CONNECTION_READY,
        RealtimeVoiceEventType.SESSION_READY,
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.RESPONSE_STARTED,
    }
)

ConnectionFactory = Callable[..., Any]


@dataclass
class _OutboundItem:
    event: dict[str, Any]
    is_media: bool = False
    sent: threading.Event = field(default_factory=threading.Event)
    success: bool = False


@dataclass
class _PendingDelete:
    event_id: str
    item_id: str
    done: threading.Event = field(default_factory=threading.Event)
    success: bool = False


@dataclass(frozen=True)
class VolcengineDuplexConfig:
    """Provider-specific, secret-safe configuration for Seeduplex 3.0."""

    enabled: bool = False
    api_key: str = field(default="", repr=False)
    endpoint: str = DEFAULT_VOLCENGINE_DUPLEX_ENDPOINT
    model: str = SUPPORTED_VOLCENGINE_DUPLEX_MODEL
    speaker: str = "zh_male_xiaotian_jupiter_bigtts"
    bot_name: str = "小算"
    system_role: str = ""
    speaking_style: str = "简洁、自然、口语化；不要声称已经执行机器人动作。"
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    output_format: str = "pcm_s16le"
    chunk_ms: int = 20
    connect_timeout_s: float = 4.0
    close_timeout_s: float = 1.0
    audio_queue_ms: int = 400
    event_queue_size: int = 256

    @property
    def credentials_configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def available(self) -> bool:
        return self.enabled and not self.validation_errors()

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.enabled and not self.credentials_configured:
            errors.append("api_key_required")
        if self.endpoint != DEFAULT_VOLCENGINE_DUPLEX_ENDPOINT:
            errors.append("official_endpoint_required")
        if self.model != SUPPORTED_VOLCENGINE_DUPLEX_MODEL:
            errors.append("supported_model_required")
        if self.input_sample_rate != 16_000:
            errors.append("input_sample_rate_must_be_16000")
        if self.output_sample_rate != 24_000:
            errors.append("output_sample_rate_must_be_24000")
        if self.output_format != "pcm_s16le":
            errors.append("output_format_must_be_pcm_s16le")
        if self.chunk_ms != 20:
            errors.append("chunk_ms_must_be_20")
        return errors


class VolcengineDuplexDialogue:
    """Seeduplex adapter satisfying the provider-neutral realtime interface."""

    def __init__(
        self,
        config: VolcengineDuplexConfig,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self._config = config
        self._connection_factory = connection_factory
        self._lifecycle_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._ws: Any = None
        self._context: RealtimeVoiceSessionContext | None = None
        self._session_id = ""
        self._dialog_id = ""
        self._log_id = ""
        self._state = "disabled" if not config.enabled else "idle"
        self._last_error = ""
        self._connected = False
        self._active = False
        self._closing = False
        self._event_counter = 0
        self._input_order_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._audio_buffer = bytearray()
        self._input_muted = False
        audio_frames = max(1, config.audio_queue_ms // max(1, config.chunk_ms))
        self._outbound_media_capacity = audio_frames
        self._outbound_capacity = audio_frames + 8
        self._outbound_queue: deque[_OutboundItem] = deque()
        self._outbound_media_count = 0
        self._outbound_condition = threading.Condition()
        self._sender_stop = threading.Event()
        self._sender_thread: threading.Thread | None = None
        self._event_capacity = max(8, config.event_queue_size)
        self._event_queue: deque[RealtimeVoiceEvent] = deque()
        self._event_condition = threading.Condition()
        self._dropped_events = 0
        self._receiver_stop = threading.Event()
        self._receiver_thread: threading.Thread | None = None
        self._session_closed = threading.Event()
        self._conversation_delete_lock = threading.Lock()
        self._ack_lock = threading.Lock()
        self._pending_delete: _PendingDelete | None = None
        self._conversation_delete_count = 0
        self._conversation_delete_failures = 0
        self._generation = 0
        self._current_generation = 0
        self._active_response_generation = 0
        self._item_generations: dict[str, int] = {}
        self._response_generations: dict[str, int] = {}
        self._fenced_generations: set[int] = set()
        self._connected_at = 0.0
        self._first_output_audio_ms: float | None = None
        self._sent_audio_frames = 0
        self._dropped_input_frames = 0
        self._dropped_stale_audio_frames = 0
        self._received_audio_frames = 0

    @property
    def available(self) -> bool:
        return self._config.available

    def start(self, context: RealtimeVoiceSessionContext) -> bool:
        with self._lifecycle_lock:
            if self._active:
                return True
            if not self.available:
                self._state = "degraded"
                self._last_error = "invalid_or_disabled_config"
                return False
            if context.allow_tool_calls or context.allow_hardware_dispatch:
                self._state = "degraded"
                self._last_error = "unsafe_session_capabilities_requested"
                return False
            if (
                context.input_sample_rate != self._config.input_sample_rate
                or context.output_sample_rate != self._config.output_sample_rate
                or context.output_format != self._config.output_format
            ):
                self._state = "degraded"
                self._last_error = "session_audio_shape_mismatch"
                return False
            self._context = context
            self._session_id = context.session_id
            self._dialog_id = context.dialog_id
            self._input_muted = False
            self._state = "connecting"
            self._last_error = ""
            try:
                self._connect_and_create_session(context)
            except Exception as exc:
                self._state = "degraded"
                self._last_error = type(exc).__name__
                self._connected = False
                self._active = False
                self._close_socket_only()
                return False
            if context.input_mode != "push_to_talk":
                self._start_workers()
                return True
            with self._input_order_lock:
                self._start_workers()
                mute_item = _OutboundItem(
                    event={
                        "type": "input_audio_mute.commit",
                        "event_id": self._new_event_id(),
                    }
                )
                if not self._enqueue_outbound(mute_item):
                    self._transition_degraded("start_input_mute_enqueue_failed")
                    return False
                if not mute_item.sent.wait(
                    timeout=max(0.05, self._config.close_timeout_s)
                ):
                    self._transition_degraded("start_input_mute_timeout")
                    return False
                if not mute_item.success:
                    self._transition_degraded("start_input_mute_failed")
                    return False
                if not self._active or not self._connected:
                    return False
                self._input_muted = True
            return True

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        if (
            not self._active
            or frame.sample_rate != self._config.input_sample_rate
            or frame.channels != 1
            or not frame.pcm
            or len(frame.pcm) % 2
        ):
            if self._active:
                self._last_error = "invalid_input_audio_shape"
            return False
        packet_bytes = self._config.input_sample_rate * 2 * self._config.chunk_ms // 1000
        with self._input_order_lock:
            if not self._active or not self._connected:
                return False
            if self._input_muted:
                unmute_item = _OutboundItem(
                    event={
                        "type": "input_audio_unmute.commit",
                        "event_id": self._new_event_id(),
                    }
                )
                if not self._enqueue_outbound(unmute_item):
                    self._last_error = "input_audio_unmute_enqueue_failed"
                    return False
                self._input_muted = False
            chunks: list[bytes] = []
            with self._buffer_lock:
                self._audio_buffer.extend(frame.pcm)
                while len(self._audio_buffer) >= packet_bytes:
                    chunks.append(bytes(self._audio_buffer[:packet_bytes]))
                    del self._audio_buffer[:packet_bytes]
            for chunk in chunks:
                if not self._enqueue_audio(chunk):
                    self._last_error = "input_audio_enqueue_failed"
                    return False
        return True

    def finish_input(self) -> bool:
        if not self._active:
            self._last_error = "finish_input_inactive"
            return False
        with self._input_order_lock:
            with self._buffer_lock:
                remainder = bytes(self._audio_buffer)
                self._audio_buffer.clear()
            if remainder and not self._enqueue_audio(remainder):
                self._last_error = "finish_input_audio_enqueue_failed"
                return False
            if self._context is not None and self._context.input_mode == "push_to_talk":
                item = _OutboundItem(
                    event={
                        "type": "input_audio_buffer.commit",
                        "event_id": self._new_event_id(),
                    }
                )
                if not self._enqueue_outbound(item):
                    self._last_error = "finish_input_commit_enqueue_failed"
                    return False
                if not item.sent.wait(timeout=max(0.05, self._config.close_timeout_s)):
                    self._transition_degraded("finish_input_commit_timeout")
                    return False
                if not item.success:
                    self._last_error = "finish_input_commit_failed"
                    return False
                mute_item = _OutboundItem(
                    event={
                        "type": "input_audio_mute.commit",
                        "event_id": self._new_event_id(),
                    }
                )
                if not self._enqueue_outbound(mute_item):
                    self._last_error = "finish_input_mute_enqueue_failed"
                    return False
                if not mute_item.sent.wait(timeout=max(0.05, self._config.close_timeout_s)):
                    self._transition_degraded("finish_input_mute_timeout")
                    return False
                if not mute_item.success:
                    self._last_error = "finish_input_mute_failed"
                    return False
                self._input_muted = True
        return True

    def interrupt(self, reason: str) -> None:
        with self._lifecycle_lock:
            if not self._active:
                return
            generation = self._active_response_generation or self._current_generation
            if generation:
                self._fenced_generations.add(generation)
        item = _OutboundItem(event={"type": "response.cancel", "event_id": self._new_event_id()})
        if not self._enqueue_outbound(item):
            self._transition_degraded("response_cancel_enqueue_failed")

    def delete_conversation_turn(self, item_id: str, *, timeout: float = 1.0) -> bool:
        """Delete the paired provider turn and require a correlated ack."""

        clean_item_id = str(item_id or "").strip()
        if not self._active or not clean_item_id:
            return False
        with self._conversation_delete_lock:
            event_id = self._new_event_id()
            pending = _PendingDelete(event_id=event_id, item_id=clean_item_id)
            with self._ack_lock:
                self._pending_delete = pending
            outbound = _OutboundItem(
                event={
                    "type": "conversation.item.delete",
                    "event_id": event_id,
                    "items": [{"id": clean_item_id}],
                }
            )
            if not self._enqueue_outbound(outbound):
                with self._ack_lock:
                    if self._pending_delete is pending:
                        self._pending_delete = None
                self._conversation_delete_failures += 1
                return False
            if not outbound.sent.wait(timeout=max(0.05, timeout)) or not outbound.success:
                with self._ack_lock:
                    if self._pending_delete is pending:
                        self._pending_delete = None
                self._conversation_delete_failures += 1
                return False
            if not pending.done.wait(timeout=max(0.05, timeout)):
                with self._ack_lock:
                    if self._pending_delete is pending:
                        self._pending_delete = None
                self._conversation_delete_failures += 1
                self._last_error = "conversation_delete_timeout"
                return False
            with self._ack_lock:
                if self._pending_delete is pending:
                    self._pending_delete = None
            if not pending.success:
                self._conversation_delete_failures += 1
                self._last_error = "conversation_delete_rejected"
                return False
            self._conversation_delete_count += 1
            return True

    def next_event(self, timeout: float | None = None) -> RealtimeVoiceEvent | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._event_condition:
            while not self._event_queue:
                if deadline is None:
                    self._event_condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._event_condition.wait(timeout=remaining)
            return self._event_queue.popleft()

    def events(self) -> Iterator[RealtimeVoiceEvent]:
        while self._active or self._event_queue:
            event = self.next_event(timeout=0.1)
            if event is not None:
                yield event

    def close(self, reason: str = "shutdown") -> None:
        with self._lifecycle_lock:
            if self._closing:
                return
            if self._ws is None and not self._active:
                self._state = "closed"
                return
            self._closing = True
            self._active = False
            connected = self._connected and self._ws is not None
            sender = self._sender_thread
        self._sender_stop.set()
        self._discard_outbound_queue()
        with self._outbound_condition:
            self._outbound_condition.notify_all()
        if sender is not None and sender is not threading.current_thread():
            sender.join(timeout=self._config.close_timeout_s)
        try:
            if connected:
                try:
                    self._send_json({"type": "session.close", "event_id": self._new_event_id()})
                    if not self._session_closed.wait(
                        timeout=max(0.05, self._config.close_timeout_s)
                    ):
                        self._last_error = "session_close_timeout"
                except Exception:
                    self._last_error = "session_close_failed"
        finally:
            self._receiver_stop.set()
            self._close_socket_only()
            receiver = self._receiver_thread
            if receiver is not None and receiver is not threading.current_thread():
                receiver.join(timeout=self._config.close_timeout_s)
            self._receiver_thread = None
            self._sender_thread = None
            self._connected = False
            self._input_muted = False
            self._closing = False
            self._state = "closed"

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "provider": "volcengine_duplex",
            "enabled": self._config.enabled,
            "available": self.available,
            "endpoint": self._config.endpoint,
            "model": self._config.model,
            "speaker": self._config.speaker,
            "credentials_configured": self._config.credentials_configured,
            "state": self._state,
            "connected": self._connected,
            "active": self._active,
            "session_id": self._session_id,
            "dialog_id": self._dialog_id,
            "provider_session_id": self._dialog_id,
            "log_id": self._log_id,
            "generation": self._generation,
            "audio_queue_depth": self._outbound_media_count,
            "outbound_queue_depth": len(self._outbound_queue),
            "audio_buffer_bytes": len(self._audio_buffer),
            "input_muted": self._input_muted,
            "sent_audio_frames": self._sent_audio_frames,
            "dropped_input_frames": self._dropped_input_frames,
            "dropped_stale_audio_frames": self._dropped_stale_audio_frames,
            "received_audio_frames": self._received_audio_frames,
            "event_queue_depth": len(self._event_queue),
            "dropped_events": self._dropped_events,
            "first_output_audio_ms": self._first_output_audio_ms,
            "conversation_delete_count": self._conversation_delete_count,
            "conversation_delete_failures": self._conversation_delete_failures,
            "last_error": self._last_error,
        }

    def _connect_and_create_session(self, context: RealtimeVoiceSessionContext) -> None:
        factory = self._connection_factory
        if factory is None:
            try:
                import websocket
            except ImportError as exc:  # pragma: no cover - dependency failure
                raise RuntimeError("websocket_client_unavailable") from exc
            factory = websocket.create_connection

        self._ws = factory(
            self._config.endpoint,
            header=[f"X-Api-Key: {self._config.api_key}"],
            timeout=self._config.connect_timeout_s,
        )
        self._capture_log_id()
        self._send_json(self._session_create_event(context))
        event = self._receive_session_created()
        provider_session = event.get("session")
        if isinstance(provider_session, dict):
            self._dialog_id = str(provider_session.get("id") or self._dialog_id)
        settimeout = getattr(self._ws, "settimeout", None)
        if callable(settimeout):
            settimeout(0.1)
        self._connected = True
        self._active = True
        self._state = "listening"
        self._connected_at = time.monotonic()

    def _session_create_event(self, context: RealtimeVoiceSessionContext) -> dict[str, Any]:
        instructions = context.system_role or self._config.system_role
        style = context.speaking_style or self._config.speaking_style
        guard = (
            "安全边界：不得调用任何工具或外部服务；"
            "不得执行、控制或声称已经执行机器人、设备或现实世界动作。"
        )
        instructions = "\n".join(part for part in (instructions, style, guard) if part.strip())
        return {
            "type": "session.create",
            "event_id": self._new_event_id(),
            "session": {
                "type": "realtime",
                "id": context.dialog_id or context.session_id,
                "model": self._config.model,
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {
                            "type": "pcm",
                            "rate": self._config.input_sample_rate,
                        }
                    },
                    "output": {
                        "format": {
                            "type": self._config.output_format,
                            "rate": self._config.output_sample_rate,
                        },
                        "voice": self._config.speaker,
                    },
                },
            },
            "extension": {"asr": {}, "tts": {}, "dialog": {}},
        }

    def _receive_session_created(self) -> dict[str, Any]:
        deadline = time.monotonic() + self._config.connect_timeout_s
        while time.monotonic() < deadline:
            try:
                event = self._receive_json()
            except Exception as exc:
                if self._is_timeout(exc):
                    continue
                raise
            event_type = str(event.get("type") or "")
            if event_type == "session.created":
                return event
            if event_type == "error":
                raise RuntimeError("provider_session_error")
        raise TimeoutError("provider_session_timeout")

    def _send_json(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._send_lock:
            if self._ws is None:
                raise RuntimeError("websocket_not_connected")
            self._ws.send(payload)

    def _receive_json(self) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("websocket_not_connected")
        frame = self._ws.recv()
        if isinstance(frame, bytes):
            frame = frame.decode("utf-8")
        if not isinstance(frame, str):
            raise TypeError("provider_frame_must_be_text")
        if not frame:
            raise ConnectionError("provider_connection_closed")
        event = json.loads(frame)
        if not isinstance(event, dict):
            raise TypeError("provider_event_must_be_object")
        return event

    def _new_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{uuid.uuid4().hex}"

    def _start_workers(self) -> None:
        self._sender_stop.clear()
        self._receiver_stop.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="volcengine-duplex-send",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name="volcengine-duplex-recv",
            daemon=True,
        )
        self._sender_thread.start()
        self._receiver_thread.start()

    def _sender_loop(self) -> None:
        while True:
            with self._outbound_condition:
                while not self._outbound_queue and not self._sender_stop.is_set():
                    self._outbound_condition.wait(timeout=0.1)
                if self._sender_stop.is_set() and not self._outbound_queue:
                    return
                item = self._outbound_queue.popleft()
                if item.is_media:
                    self._outbound_media_count = max(0, self._outbound_media_count - 1)
            try:
                self._send_json(item.event)
                item.success = True
                if item.is_media:
                    self._sent_audio_frames += 1
            except Exception as exc:
                item.success = False
                item.sent.set()
                if not self._closing and self._active:
                    self._transition_degraded(self._send_error_name(exc))
                return
            item.sent.set()

    def _receiver_loop(self) -> None:
        while not self._receiver_stop.is_set():
            try:
                event = self._receive_json()
            except Exception as exc:
                if self._is_timeout(exc):
                    continue
                if not self._closing and not self._receiver_stop.is_set():
                    self._transition_degraded(self._receive_error_name(exc))
                return
            try:
                self._handle_provider_event(event)
            except (KeyError, TypeError, ValueError, binascii.Error):
                self._transition_degraded("provider_event_error")
                return

    def _handle_provider_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        metadata = self._public_metadata(event)
        if event_type == "session.closed":
            self._session_closed.set()
            self._active = False
            self._connected = False
            self._input_muted = False
            self._receiver_stop.set()
            self._sender_stop.set()
            self._discard_outbound_queue()
            with self._outbound_condition:
                self._outbound_condition.notify_all()
            self._state = "closed"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.SESSION_CLOSED,
                    session_id=self._session_id,
                    generation=self._current_generation,
                    provider="volcengine_duplex",
                    metadata=metadata,
                )
            )
            return
        if event_type == "conversation.item.deleted":
            self._handle_conversation_deleted(event)
            return
        if event_type == "conversation.item.input_audio_transcription.started":
            item_id = str(event.get("item_id") or "").strip()
            if not item_id or item_id in self._item_generations:
                return
            self._generation += 1
            self._current_generation = self._generation
            self._item_generations[item_id] = self._generation
            self._state = "user_speaking"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
                    session_id=self._session_id,
                    generation=self._generation,
                    provider="volcengine_duplex",
                    metadata=metadata,
                )
            )
            return
        if event_type == "conversation.item.input_audio_transcription.delta":
            generation = self._generation_for(event)
            if not generation:
                return
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_duplex",
                    transcript=str(event.get("delta") or ""),
                    metadata=metadata,
                )
            )
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            generation = self._generation_for(event)
            if not generation:
                return
            self._state = "thinking"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_duplex",
                    transcript=str(event.get("transcript") or event.get("text") or ""),
                    is_final=True,
                    metadata=metadata,
                )
            )
            return
        if event_type == "conversation.item.input_audio_transcription.failed":
            self._emit_error("provider_transcription_error", metadata=metadata)
            return
        if event_type == "response.output_text.delta":
            generation = self._generation_for(event) or self._current_generation or 1
            self._remember_response(event, generation)
            if generation in self._fenced_generations:
                return
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_duplex",
                    text=str(event.get("delta") or ""),
                    metadata=metadata,
                )
            )
            return
        if event_type == "response.output_audio.started":
            generation = self._generation_for(event) or self._current_generation or 1
            self._remember_response(event, generation)
            self._active_response_generation = generation
            self._state = "responding"
            if generation in self._fenced_generations:
                return
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.RESPONSE_STARTED,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_duplex",
                    metadata=metadata,
                )
            )
            return
        if event_type == "response.output_audio.delta":
            self._handle_audio_delta(event, metadata=metadata)
            return
        if event_type == "response.output_audio.done":
            return
        if event_type == "response.done":
            generation = self._generation_for(event) or self._active_response_generation or 1
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._put_event(
                    RealtimeVoiceEvent(
                        event_type=RealtimeVoiceEventType.USAGE,
                        session_id=self._session_id,
                        generation=generation,
                        provider="volcengine_duplex",
                        metadata={**metadata, "usage": dict(usage)},
                    )
                )
            if generation not in self._fenced_generations:
                self._put_event(
                    RealtimeVoiceEvent(
                        event_type=RealtimeVoiceEventType.RESPONSE_DONE,
                        session_id=self._session_id,
                        generation=generation,
                        provider="volcengine_duplex",
                        is_final=True,
                        metadata=metadata,
                    )
                )
            self._active_response_generation = 0
            self._state = "listening"
            return
        if event_type == "response.canceled":
            generation = self._active_response_generation or self._current_generation
            if generation:
                self._fenced_generations.add(generation)
            self._active_response_generation = 0
            self._state = "listening"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.INTERRUPTED,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_duplex",
                    metadata=metadata,
                )
            )
            return
        if event_type == "response.function_call_arguments.done":
            self._transition_degraded("unexpected_tool_call")
            return
        if event_type == "error":
            error_metadata = dict(metadata)
            detail = event.get("error")
            if isinstance(detail, dict):
                message = detail.get("message")
                if isinstance(message, str):
                    safe_message = " ".join(message.splitlines()).strip()[:240]
                    if safe_message:
                        error_metadata["provider_message"] = safe_message
            self._transition_degraded(
                self._safe_provider_error(event),
                metadata=error_metadata,
            )

    def _handle_audio_delta(self, event: dict[str, Any], *, metadata: dict[str, str]) -> None:
        generation = self._generation_for(event) or self._active_response_generation or 1
        self._remember_response(event, generation)
        if generation in self._fenced_generations:
            self._dropped_stale_audio_frames += 1
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            raise ValueError("missing_audio_delta")
        pcm = base64.b64decode(delta, validate=True)
        now = time.monotonic()
        if self._first_output_audio_ms is None and self._connected_at:
            self._first_output_audio_ms = round((now - self._connected_at) * 1000.0, 2)
        self._received_audio_frames += 1
        self._put_event(
            RealtimeVoiceEvent(
                event_type=RealtimeVoiceEventType.OUTPUT_AUDIO,
                session_id=self._session_id,
                generation=generation,
                provider="volcengine_duplex",
                audio=VoiceMediaFrame(
                    pcm=pcm,
                    sample_rate=self._config.output_sample_rate,
                    channels=1,
                    metadata=metadata,
                ),
                metadata=metadata,
            )
        )

    def _generation_for(self, event: dict[str, Any]) -> int:
        item_id = str(event.get("item_id") or event.get("question_id") or "").strip()
        response_id = str(event.get("response_id") or "").strip()
        return self._item_generations.get(item_id, 0) or self._response_generations.get(
            response_id, 0
        )

    def _remember_response(self, event: dict[str, Any], generation: int) -> None:
        response_id = str(event.get("response_id") or "").strip()
        if response_id:
            self._response_generations[response_id] = generation

    def _handle_conversation_deleted(self, event: dict[str, Any]) -> None:
        event_id = str(event.get("event_id") or "").strip()
        raw_items = event.get("items")
        item_ids = (
            {
                str(item.get("id") or "").strip()
                for item in raw_items
                if isinstance(item, dict) and item.get("id")
            }
            if isinstance(raw_items, list)
            else set()
        )
        with self._ack_lock:
            pending = self._pending_delete
            if pending is None:
                return
            if event_id == pending.event_id or pending.item_id in item_ids:
                pending.success = True
                pending.done.set()

    def _put_event(self, event: RealtimeVoiceEvent) -> None:
        with self._event_condition:
            if len(self._event_queue) >= self._event_capacity:
                incoming_priority = self._event_priority(event.event_type)
                evict_index = self._lower_priority_event_index(incoming_priority)
                if evict_index is None and incoming_priority == 0:
                    evict_index = self._same_low_priority_event_index()
                if evict_index is None:
                    self._dropped_events += 1
                    return
                del self._event_queue[evict_index]
                self._dropped_events += 1
            self._event_queue.append(event)
            self._event_condition.notify()

    def _lower_priority_event_index(self, incoming_priority: int) -> int | None:
        candidate_index: int | None = None
        candidate_priority = incoming_priority
        for index, queued in enumerate(self._event_queue):
            queued_priority = self._event_priority(queued.event_type)
            if queued_priority >= candidate_priority:
                continue
            candidate_index = index
            candidate_priority = queued_priority
            if queued_priority == 0:
                break
        return candidate_index

    def _same_low_priority_event_index(self) -> int | None:
        for index, queued in enumerate(self._event_queue):
            if self._event_priority(queued.event_type) == 0:
                return index
        return None

    @staticmethod
    def _event_priority(event_type: RealtimeVoiceEventType) -> int:
        if event_type in _FAULT_EVENT_PRIORITY:
            return 4
        if event_type in _TERMINAL_EVENT_PRIORITY:
            return 3
        if event_type is RealtimeVoiceEventType.USAGE:
            return 2
        if event_type in _STATE_EVENT_PRIORITY:
            return 1
        return 0

    def _emit_error(self, error: str, *, metadata: dict[str, str] | None = None) -> None:
        self._put_event(
            RealtimeVoiceEvent(
                event_type=RealtimeVoiceEventType.ERROR,
                session_id=self._session_id,
                generation=self._current_generation,
                provider="volcengine_duplex",
                error=error,
                metadata=metadata or {},
            )
        )

    def _transition_degraded(
        self,
        error: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        already_reported = self._state == "degraded" and self._last_error == error
        self._state = "degraded"
        self._last_error = error
        self._active = False
        self._connected = False
        self._sender_stop.set()
        self._receiver_stop.set()
        with self._ack_lock:
            pending = self._pending_delete
            if pending is not None:
                pending.done.set()
        self._discard_outbound_queue()
        with self._outbound_condition:
            self._outbound_condition.notify_all()
        self._close_socket_only()
        if not already_reported and not self._closing:
            self._emit_error(error, metadata=metadata)

    @staticmethod
    def _public_metadata(event: dict[str, Any]) -> dict[str, str]:
        keys = ("event_id", "item_id", "question_id", "response_id", "call_id")
        return {key: str(event[key]) for key in keys if event.get(key) not in (None, "")}

    @staticmethod
    def _safe_provider_error(event: dict[str, Any]) -> str:
        detail = event.get("error")
        if not isinstance(detail, dict):
            return "provider_error"
        code = str(detail.get("code") or "").strip()
        error_type = str(detail.get("type") or "").strip()
        safe = code or error_type
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in "_-.")[:80]
        return f"provider_error_{safe}" if safe else "provider_error"

    def _capture_log_id(self) -> None:
        response = getattr(self._ws, "handshake_response", None)
        headers = getattr(response, "headers", {})
        if not hasattr(headers, "items"):
            return
        for key, value in headers.items():
            if str(key).lower() == "x-tt-logid":
                self._log_id = str(value)
                return

    def _close_socket_only(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _enqueue_audio(self, audio: bytes) -> bool:
        return self._enqueue_outbound(
            _OutboundItem(
                event={
                    "type": "input_audio_buffer.append",
                    "event_id": self._new_event_id(),
                    "audio": base64.b64encode(audio).decode("ascii"),
                },
                is_media=True,
            )
        )

    def _enqueue_outbound(self, item: _OutboundItem) -> bool:
        with self._outbound_condition:
            if not self._active or self._sender_stop.is_set():
                return False
            if item.is_media and self._outbound_media_count >= self._outbound_media_capacity:
                if not self._evict_oldest_media_locked():
                    return False
            if len(self._outbound_queue) >= self._outbound_capacity:
                if not self._evict_oldest_media_locked():
                    return False
            self._outbound_queue.append(item)
            if item.is_media:
                self._outbound_media_count += 1
            self._outbound_condition.notify()
            return True

    def _evict_oldest_media_locked(self) -> bool:
        for index, queued in enumerate(self._outbound_queue):
            if not queued.is_media:
                continue
            del self._outbound_queue[index]
            self._outbound_media_count = max(0, self._outbound_media_count - 1)
            self._dropped_input_frames += 1
            queued.sent.set()
            return True
        return False

    def _discard_outbound_queue(self) -> None:
        with self._outbound_condition:
            while self._outbound_queue:
                item = self._outbound_queue.popleft()
                item.sent.set()
            self._outbound_media_count = 0

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()

    @staticmethod
    def _receive_error_name(exc: BaseException) -> str:
        name = type(exc).__name__.lower()
        if isinstance(exc, (ConnectionError, EOFError)) or any(
            marker in name
            for marker in ("connectionclosed", "connectionreset", "brokenpipe")
        ):
            return "provider_connection_closed"
        if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError, TypeError)):
            return "provider_frame_error"
        return "provider_receive_error"

    @classmethod
    def _send_error_name(cls, exc: BaseException) -> str:
        if cls._is_timeout(exc):
            return "provider_send_timeout"
        name = type(exc).__name__.lower()
        if isinstance(exc, (ConnectionError, EOFError)) or any(
            marker in name
            for marker in ("connectionclosed", "connectionreset", "brokenpipe")
        ):
            return "provider_connection_closed"
        if isinstance(exc, (TypeError, ValueError)):
            return "provider_send_payload_error"
        return "provider_send_error"
