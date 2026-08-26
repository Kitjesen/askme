"""Threaded adapter for Volcengine Doubao end-to-end RealtimeAPI."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.config import RealtimeVoiceConfig
from askme.voice.realtime.protocol import (
    Compression,
    MessageType,
    ProtocolError,
    RealtimeEvent,
    RealtimeFrame,
    decode_frame,
    encode_frame,
)

logger = logging.getLogger(__name__)

_CONNECTION_STARTED = 50
_CONNECTION_FAILED = 51
_CONNECTION_FINISHED = 52
_SESSION_STARTED = 150
_SESSION_FINISHED = 152
_SESSION_FAILED = 153
_USAGE_RESPONSE = 154
_CONFIG_UPDATED = 251
_TTS_SENTENCE_START = 350
_TTS_SENTENCE_END = 351
_END_ASR = 400
_ASR_ENDED = 459
_CHAT_RESPONSE = 550
_CHAT_ENDED = 559
_CLIENT_INTERRUPT = 515
_CONVERSATION_TRUNCATE = 513
_CONVERSATION_DELETE = 514
_CONVERSATION_DELETED = 571

_LOSSY_EVENTS = frozenset(
    {
        RealtimeVoiceEventType.OUTPUT_AUDIO,
        RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
        RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
        RealtimeVoiceEventType.USAGE,
    }
)
_ESSENTIAL_EVENTS = frozenset(
    {
        RealtimeVoiceEventType.ERROR,
        RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
        RealtimeVoiceEventType.RESPONSE_DONE,
    }
)

ConnectionFactory = Callable[..., Any]


@dataclass
class _OutboundItem:
    """One serialized runtime write on the provider connection."""

    frame: bytes
    event_id: int
    is_media: bool = False
    sent: threading.Event = field(default_factory=threading.Event)
    success: bool = False


@dataclass
class _PendingAck:
    """One synchronous control request waiting for its provider result."""

    item_id: str
    done: threading.Event = field(default_factory=threading.Event)
    success: bool = False


class VolcengineRealtimeDialogue:
    """One reusable bidirectional S2S connection with bounded queues.

    The adapter never opens a microphone, writes to a speaker, calls tools, or
    mutates robot state.  It only translates between PCM/events and the cloud
    session contract.
    """

    def __init__(
        self,
        config: RealtimeVoiceConfig,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self._config = config
        self._connection_factory = connection_factory
        self._lifecycle_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._input_order_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._ws: Any = None
        self._context: RealtimeVoiceSessionContext | None = None
        self._connect_id = ""
        self._session_id = ""
        self._dialog_id = ""
        self._log_id = ""
        self._state = "disabled" if not config.enabled else "idle"
        self._active = False
        self._connected = False
        self._closing = False
        self._last_error = ""

        self._audio_buffer = bytearray()
        audio_queue_frames = max(1, config.audio_queue_ms // max(1, config.chunk_ms))
        self._outbound_media_capacity = audio_queue_frames
        self._outbound_capacity = audio_queue_frames + 8
        self._outbound_queue: deque[_OutboundItem] = deque()
        self._outbound_media_count = 0
        self._outbound_condition = threading.Condition()
        self._event_capacity = max(8, config.event_queue_size)
        self._event_control_reserve = min(4, max(1, self._event_capacity // 2))
        self._event_queue: deque[RealtimeVoiceEvent] = deque()
        self._event_condition = threading.Condition()
        self._sender_stop = threading.Event()
        self._receiver_stop = threading.Event()
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._session_finished = threading.Event()
        self._connection_finished = threading.Event()
        self._conversation_delete_lock = threading.Lock()
        self._conversation_truncate_lock = threading.Lock()
        self._ack_lock = threading.Lock()
        self._pending_delete: _PendingAck | None = None
        self._pending_truncate: _PendingAck | None = None
        self._stale_delete_acks: Counter[str] = Counter()
        self._stale_truncate_acks: Counter[str] = Counter()

        self._generation = 0
        self._active_response_generation = 0
        self._current_question_id = ""
        self._current_reply_id = ""
        self._active_response_question_id = ""
        self._active_response_reply_id = ""
        self._question_generations: dict[str, int] = {}
        self._reply_generations: dict[str, int] = {}
        self._fenced_generations: set[int] = set()
        self._dropped_input_frames = 0
        self._dropped_events = 0
        self._dropped_stale_audio_frames = 0
        self._sent_audio_frames = 0
        self._received_audio_frames = 0
        self._last_input_audio_at = 0.0
        self._last_output_audio_at = 0.0
        self._connected_at = 0.0
        self._first_output_audio_ms: float | None = None

        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._reconnect_count = 0
        self._conversation_delete_count = 0
        self._conversation_delete_failures = 0

    @property
    def available(self) -> bool:
        return self._config.available

    def start(self, context: RealtimeVoiceSessionContext) -> bool:
        """Open the WebSocket and complete StartConnection/StartSession."""

        with self._lifecycle_lock:
            if self._active:
                return True
            if not self.available:
                self._state = "degraded"
                self._last_error = "invalid_or_disabled_config"
                return False
            if time.monotonic() < self._circuit_open_until:
                self._state = "circuit_open"
                self._last_error = "circuit_open"
                return False
            if (
                context.input_sample_rate != self._config.input_sample_rate
                or context.output_sample_rate != self._config.output_sample_rate
                or context.output_format != self._config.output_format
            ):
                self._state = "degraded"
                self._last_error = "session_audio_shape_mismatch"
                return False
            if context.allow_tool_calls or context.allow_hardware_dispatch:
                self._state = "degraded"
                self._last_error = "unsafe_session_capabilities_requested"
                return False
            self._reset_runtime_state(context)
            attempts = max(1, self._config.max_reconnect_attempts + 1)
            last_exc: BaseException | None = None
            for attempt in range(attempts):
                try:
                    self._connect_and_handshake(context)
                    self._start_workers()
                    self._consecutive_failures = 0
                    self._circuit_open_until = 0.0
                    self._last_error = ""
                    return True
                except Exception as exc:
                    last_exc = exc
                    self._close_socket_only()
                    if attempt + 1 < attempts:
                        self._reconnect_count += 1
                        continue
            self._record_start_failure(last_exc or RuntimeError("connection failed"))
            return False

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        """Repack clean PCM into bounded 20 ms provider frames without blocking."""

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
        self._last_input_audio_at = time.time()
        return True

    def finish_input(self) -> bool:
        """Flush a partial packet and commit only in push-to-talk mode."""

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
                if not self._send_event(_END_ASR, {}):
                    self._last_error = "finish_input_event_enqueue_failed"
                    return False
        return True

    def interrupt(self, reason: str) -> None:
        """Fence the old response; server-VAD is interrupted by new input audio."""

        with self._lifecycle_lock:
            if not self._active:
                return
            generation = self._active_response_generation or self._generation
            if generation > 0:
                self._fenced_generations.add(generation)
            context = self._context
        if context is not None and context.input_mode == "push_to_talk":
            self._send_event(_CLIENT_INTERRUPT, {})

    def truncate_response(
        self,
        *,
        reply_id: str,
        audio_end_ms: int,
        timeout: float = 0.25,
    ) -> bool:
        """Tell the provider how much of an interrupted reply was actually played."""

        clean_reply_id = str(reply_id or "").strip()
        if not self._active or not clean_reply_id or audio_end_ms < 0:
            return False
        with self._conversation_truncate_lock:
            pending = _PendingAck(item_id=clean_reply_id)
            with self._ack_lock:
                self._pending_truncate = pending
            if not self._send_event(
                _CONVERSATION_TRUNCATE,
                {"item_id": clean_reply_id, "audio_end_ms": int(audio_end_ms)},
                wait_sent=True,
            ):
                with self._ack_lock:
                    if self._pending_truncate is pending:
                        self._pending_truncate = None
                return False
            if not pending.done.wait(timeout=max(0.05, timeout)):
                with self._ack_lock:
                    if self._pending_truncate is pending:
                        self._pending_truncate = None
                        self._stale_truncate_acks[clean_reply_id] += 1
                self._last_error = "conversation_truncate_timeout"
                return False
            with self._ack_lock:
                if self._pending_truncate is pending:
                    self._pending_truncate = None
            if not pending.success:
                if self._active:
                    self._last_error = "conversation_truncate_rejected"
            return pending.success

    def delete_conversation_turn(
        self,
        item_id: str,
        *,
        timeout: float = 1.0,
    ) -> bool:
        """Delete one complete question/answer pair and wait for event 571."""

        clean_item_id = str(item_id or "").strip()
        if not self._active or not clean_item_id:
            return False
        with self._conversation_delete_lock:
            pending = _PendingAck(item_id=clean_item_id)
            with self._ack_lock:
                self._pending_delete = pending
            if not self._send_event(
                _CONVERSATION_DELETE,
                {"items": [{"item_id": clean_item_id}]},
                wait_sent=True,
            ):
                with self._ack_lock:
                    if self._pending_delete is pending:
                        self._pending_delete = None
                self._conversation_delete_failures += 1
                return False
            if not pending.done.wait(timeout=max(0.05, timeout)):
                with self._ack_lock:
                    if self._pending_delete is pending:
                        self._pending_delete = None
                        self._stale_delete_acks[clean_item_id] += 1
                self._conversation_delete_failures += 1
                self._last_error = "conversation_delete_timeout"
                return False
            with self._ack_lock:
                if self._pending_delete is pending:
                    self._pending_delete = None
            if not pending.success:
                self._conversation_delete_failures += 1
                if self._active:
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
        """Finish session and connection in official order, then release threads."""

        with self._lifecycle_lock:
            if self._closing:
                return
            if self._ws is None and not self._active:
                self._state = "closed"
                return
            self._closing = True
            sender = self._sender_thread
            receiver = self._receiver_thread
            was_connected = self._connected
            self._active = False
            self._discard_outbound_media()
        try:
            if was_connected and self._session_id:
                self._send_event(
                    RealtimeEvent.FINISH_SESSION,
                    {},
                    wait_sent=True,
                )
                self._session_finished.wait(timeout=self._config.close_timeout_s)
            if was_connected:
                self._send_event(
                    RealtimeEvent.FINISH_CONNECTION,
                    {},
                    wait_sent=True,
                )
                self._connection_finished.wait(timeout=self._config.close_timeout_s)
        finally:
            self._sender_stop.set()
            with self._outbound_condition:
                self._outbound_condition.notify_all()
            if sender is not None and sender is not threading.current_thread():
                sender.join(timeout=self._config.close_timeout_s)
            self._receiver_stop.set()
            self._close_socket_only()
            if receiver is not None and receiver is not threading.current_thread():
                receiver.join(timeout=self._config.close_timeout_s)
            with self._lifecycle_lock:
                self._active = False
                self._connected = False
                self._closing = False
                self._sender_thread = None
                self._receiver_thread = None
                self._state = "closed"

    def status_snapshot(self) -> dict[str, Any]:
        now_mono = time.monotonic()
        return {
            "provider": "volcengine_s2s",
            "enabled": self._config.enabled,
            "available": self.available,
            "mode": self._config.mode.value,
            "fallback": self._config.fallback,
            "endpoint": self._config.endpoint,
            "resource_id": self._config.resource_id,
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
            "active_response_generation": self._active_response_generation,
            "input_audio_active": bool(self._last_input_audio_at),
            "output_audio_active": bool(self._last_output_audio_at),
            "audio_queue_depth": self._outbound_media_count,
            "outbound_queue_depth": len(self._outbound_queue),
            "event_queue_depth": len(self._event_queue),
            "audio_buffer_bytes": len(self._audio_buffer),
            "sent_audio_frames": self._sent_audio_frames,
            "received_audio_frames": self._received_audio_frames,
            "dropped_input_frames": self._dropped_input_frames,
            "dropped_events": self._dropped_events,
            "dropped_stale_audio_frames": self._dropped_stale_audio_frames,
            "first_output_audio_ms": self._first_output_audio_ms,
            "reconnect_count": self._reconnect_count,
            "consecutive_failures": self._consecutive_failures,
            "conversation_delete_count": self._conversation_delete_count,
            "conversation_delete_failures": self._conversation_delete_failures,
            "circuit_open": now_mono < self._circuit_open_until,
            "circuit_retry_after_s": round(max(0.0, self._circuit_open_until - now_mono), 2),
            "last_error": self._last_error,
        }

    def _connect_and_handshake(self, context: RealtimeVoiceSessionContext) -> None:
        factory = self._connection_factory
        if factory is None:
            try:
                import websocket
            except ImportError as exc:
                raise RuntimeError("websocket-client is not installed") from exc
            factory = websocket.create_connection

        self._connect_id = str(uuid.uuid4())
        headers = [
            f"X-Api-App-ID: {self._config.app_id}",
            f"X-Api-Access-Key: {self._config.access_token}",
            f"X-Api-Resource-Id: {self._config.resource_id}",
            f"X-Api-App-Key: {self._config.app_key}",
            f"X-Api-Connect-Id: {self._connect_id}",
        ]
        self._state = "connecting"
        self._ws = factory(
            self._config.endpoint,
            header=headers,
            timeout=self._config.connect_timeout_s,
        )
        self._capture_log_id()
        self._send_raw(
            encode_frame(
                RealtimeEvent.START_CONNECTION,
                {},
                compression=Compression.NONE,
            )
        )
        connection_ack = self._receive_handshake({_CONNECTION_STARTED, _CONNECTION_FAILED})
        if int(connection_ack.event or 0) != _CONNECTION_STARTED:
            raise RuntimeError("provider_connection_failed")
        self._connected = True

        self._send_raw(
            encode_frame(
                RealtimeEvent.START_SESSION,
                self._start_session_payload(context),
                session_id=context.session_id,
                compression=Compression.NONE,
            )
        )
        session_ack = self._receive_handshake({_SESSION_STARTED, _SESSION_FAILED})
        if int(session_ack.event or 0) != _SESSION_STARTED:
            raise RuntimeError("provider_session_failed")
        payload = session_ack.payload if isinstance(session_ack.payload, dict) else {}
        self._dialog_id = str(payload.get("dialog_id", context.dialog_id or ""))
        settimeout = getattr(self._ws, "settimeout", None)
        if callable(settimeout):
            settimeout(0.1)
        self._active = True
        self._state = "listening"
        self._connected_at = time.monotonic()

    def _start_session_payload(self, context: RealtimeVoiceSessionContext) -> dict[str, Any]:
        system_role = context.system_role or self._config.system_role
        restrictions: list[str] = []
        if not context.allow_tool_calls:
            restrictions.append("不得调用任何工具或外部服务")
        if not context.allow_hardware_dispatch:
            restrictions.append("不得执行、控制或声称已经执行机器人、设备或现实世界动作")
        if restrictions:
            guard = "安全边界：" + "；".join(restrictions) + "。"
            system_role = "\n".join(part for part in (system_role, guard) if part)
        dialog: dict[str, Any] = {
            "bot_name": context.bot_name or self._config.bot_name,
            "system_role": system_role,
            "speaking_style": (context.speaking_style or self._config.speaking_style),
            "extra": {
                "model": self._config.model,
                "input_mod": context.input_mode or self._config.input_mode,
                "enable_conversation_truncate": True,
            },
        }
        if context.dialog_id:
            dialog["dialog_id"] = context.dialog_id
        return {
            "asr": {
                "audio_info": {
                    "format": "pcm",
                    "sample_rate": self._config.input_sample_rate,
                    "channel": 1,
                },
                "extra": {
                    "end_smooth_window_ms": self._config.end_smooth_window_ms,
                    "enable_custom_vad": True,
                },
            },
            "tts": {
                "speaker": self._config.speaker,
                "audio_config": {
                    "channel": 1,
                    "format": self._config.output_format,
                    "sample_rate": self._config.output_sample_rate,
                },
                "extra": {},
            },
            "dialog": dialog,
        }

    def _receive_handshake(self, allowed_events: set[int]) -> RealtimeFrame:
        deadline = time.monotonic() + self._config.connect_timeout_s
        while time.monotonic() < deadline:
            try:
                raw = self._ws.recv()
            except Exception as exc:
                if self._is_timeout(exc):
                    continue
                raise
            if not isinstance(raw, bytes):
                raise ProtocolError("provider handshake frame must be binary")
            frame = decode_frame(raw)
            if frame.message_type == MessageType.ERROR:
                raise RuntimeError(f"provider_error_{frame.error_code}")
            if frame.event is not None and int(frame.event) in allowed_events:
                return frame
        raise TimeoutError("provider_handshake_timeout")

    def _start_workers(self) -> None:
        self._sender_stop.clear()
        self._receiver_stop.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="volcengine-s2s-send",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name="volcengine-s2s-recv",
            daemon=True,
        )
        self._sender_thread.start()
        self._receiver_thread.start()

    def _sender_loop(self) -> None:
        while True:
            with self._outbound_condition:
                while not self._outbound_queue and not self._sender_stop.is_set():
                    self._outbound_condition.wait(timeout=0.1)
                if not self._outbound_queue:
                    return
                item = self._outbound_queue.popleft()
                if item.is_media:
                    self._outbound_media_count -= 1
                self._outbound_condition.notify_all()
            try:
                self._send_raw(item.frame)
                item.success = True
                if item.is_media:
                    self._sent_audio_frames += 1
            except Exception as exc:
                self._transition_degraded(
                    type(exc).__name__,
                    provider_event_id=item.event_id,
                )
            finally:
                item.sent.set()

    def _receiver_loop(self) -> None:
        while not self._receiver_stop.is_set():
            try:
                raw = self._ws.recv()
            except Exception as exc:
                if self._is_timeout(exc):
                    continue
                if not self._closing:
                    self._transition_degraded(type(exc).__name__)
                return
            if not isinstance(raw, bytes):
                self._transition_degraded("non_binary_provider_frame")
                return
            try:
                frame = decode_frame(raw)
            except ProtocolError:
                self._transition_degraded("protocol_error")
                return
            try:
                self._handle_server_frame(frame)
            except (KeyError, TypeError, ValueError):
                self._transition_degraded("provider_event_error")
                return

    def _handle_server_frame(self, frame: RealtimeFrame) -> None:
        event_id = int(frame.event) if frame.event is not None else 0
        payload = frame.payload if isinstance(frame.payload, dict) else {}
        if frame.message_type == MessageType.ERROR:
            self._transition_degraded(
                f"provider_error_{frame.error_code}",
                provider_event_id=event_id or None,
            )
            return
        if event_id == _SESSION_FINISHED:
            self._session_finished.set()
            return
        if event_id == _CONNECTION_FINISHED:
            self._connection_finished.set()
            return
        if event_id == _CONVERSATION_DELETED:
            self._handle_conversation_deleted(payload)
            return
        if event_id == int(RealtimeEvent.CONVERSATION_TRUNCATED):
            self._handle_conversation_truncated(payload)
            return
        if event_id in {
            _CONNECTION_FAILED,
            _SESSION_FAILED,
            int(RealtimeEvent.DIALOG_COMMON_ERROR),
        }:
            self._transition_degraded(
                str(payload.get("status_code") or "provider_dialog_error"),
                provider_event_id=event_id,
            )
            return
        if event_id == int(RealtimeEvent.ASR_INFO):
            question_id = str(payload.get("question_id", "")).strip()
            if not question_id or question_id in self._question_generations:
                return
            self._generation += 1
            self._question_generations[question_id] = self._generation
            self._current_question_id = question_id
            self._state = "user_speaking"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
                    session_id=self._session_id,
                    generation=self._generation,
                    provider="volcengine_s2s",
                    provider_event_id=event_id,
                    metadata=self._public_ids(payload),
                )
            )
            return
        if event_id == int(RealtimeEvent.ASR_RESPONSE):
            self._handle_asr_response(payload, provider_event_id=event_id)
            return
        if event_id == _ASR_ENDED:
            self._state = "thinking"
            return
        if event_id == _TTS_SENTENCE_START:
            question_id = str(payload.get("question_id", "")).strip()
            reply_id = str(payload.get("reply_id", "")).strip()
            generation = (
                self._question_generations.get(question_id)
                or self._reply_generations.get(reply_id)
                or self._generation
                or 1
            )
            self._active_response_generation = generation
            self._active_response_question_id = question_id or self._current_question_id
            self._active_response_reply_id = reply_id
            if reply_id:
                self._reply_generations[reply_id] = generation
            self._current_reply_id = reply_id
            self._state = "responding"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.RESPONSE_STARTED,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_s2s",
                    provider_event_id=event_id,
                    text=str(payload.get("text", "")),
                    metadata=self._public_ids(payload),
                )
            )
            return
        if event_id == _CHAT_RESPONSE:
            question_id = str(payload.get("question_id", "")).strip()
            reply_id = str(payload.get("reply_id", "")).strip()
            generation = (
                self._question_generations.get(question_id)
                or self._reply_generations.get(reply_id)
                or self._active_response_generation
                or self._generation
                or 1
            )
            if reply_id:
                self._reply_generations[reply_id] = generation
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_s2s",
                    provider_event_id=event_id,
                    text=str(payload.get("content", "")),
                    metadata=self._public_ids(payload),
                )
            )
            return
        if event_id == int(RealtimeEvent.TTS_RESPONSE):
            self._handle_audio_response(frame)
            return
        if event_id == _USAGE_RESPONSE:
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.USAGE,
                    session_id=self._session_id,
                    generation=self._generation,
                    provider="volcengine_s2s",
                    provider_event_id=event_id,
                    metadata={"usage": dict(payload.get("usage", {}) or {})},
                )
            )
            return
        if event_id == int(RealtimeEvent.TTS_ENDED):
            question_id = str(payload.get("question_id", "")).strip()
            reply_id = str(payload.get("reply_id", "")).strip()
            generation = (
                self._question_generations.get(question_id)
                or self._reply_generations.get(reply_id)
                or self._active_response_generation
                or self._generation
                or 1
            )
            active_matches = (
                not question_id
                or not self._active_response_question_id
                or question_id == self._active_response_question_id
            ) and (
                not reply_id
                or not self._active_response_reply_id
                or reply_id == self._active_response_reply_id
            )
            if active_matches:
                if self._current_reply_id == self._active_response_reply_id:
                    self._current_reply_id = ""
                self._active_response_generation = 0
                self._active_response_question_id = ""
                self._active_response_reply_id = ""
            if generation in self._fenced_generations or generation < self._generation:
                return
            self._state = "listening"
            self._put_event(
                RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.RESPONSE_DONE,
                    session_id=self._session_id,
                    generation=generation,
                    provider="volcengine_s2s",
                    provider_event_id=event_id,
                    metadata=self._public_ids(payload),
                )
            )
            return
        if event_id in {_TTS_SENTENCE_END, _CHAT_ENDED, _CONFIG_UPDATED}:
            return

    def _handle_asr_response(self, payload: dict[str, Any], *, provider_event_id: int) -> None:
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            return
        result = results[-1]
        if not isinstance(result, dict):
            return
        question_id = str(payload.get("question_id", "")).strip()
        generation = (
            self._question_generations.get(question_id)
            if question_id
            else self._question_generations.get(self._current_question_id)
        )
        if not generation:
            return
        transcript = str(result.get("text", "")).strip()
        interim = bool(result.get("is_interim", False))
        self._put_event(
            RealtimeVoiceEvent(
                event_type=(
                    RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA
                    if interim
                    else RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL
                ),
                session_id=self._session_id,
                generation=generation,
                provider="volcengine_s2s",
                provider_event_id=provider_event_id,
                transcript=transcript,
                is_final=not interim,
                metadata=self._public_ids(payload),
            )
        )

    def _handle_conversation_deleted(self, payload: dict[str, Any]) -> None:
        item_ids = self._ack_item_ids(payload)
        with self._ack_lock:
            if not item_ids and self._consume_stale_ack_locked(self._stale_delete_acks):
                return
            for item_id in item_ids:
                if self._stale_delete_acks[item_id] <= 0:
                    continue
                self._stale_delete_acks[item_id] -= 1
                if self._stale_delete_acks[item_id] <= 0:
                    del self._stale_delete_acks[item_id]
                return
            pending = self._pending_delete
            if pending is None:
                return
            if not self._successful_status(payload) or not item_ids:
                pending.success = False
                pending.done.set()
                return
            if pending.item_id in item_ids:
                pending.success = True
                pending.done.set()

    def _handle_conversation_truncated(self, payload: dict[str, Any]) -> None:
        item_ids = self._ack_item_ids(payload)
        with self._ack_lock:
            if not item_ids and self._consume_stale_ack_locked(self._stale_truncate_acks):
                return
            for item_id in item_ids:
                if self._stale_truncate_acks[item_id] <= 0:
                    continue
                self._stale_truncate_acks[item_id] -= 1
                if self._stale_truncate_acks[item_id] <= 0:
                    del self._stale_truncate_acks[item_id]
                return
            pending = self._pending_truncate
            if pending is None:
                return
            if not self._successful_status(payload) or not item_ids:
                pending.success = False
                pending.done.set()
                return
            if pending.item_id in item_ids:
                pending.success = True
                pending.done.set()

    def _handle_audio_response(self, frame: RealtimeFrame) -> None:
        generation = self._active_response_generation or self._generation or 1
        if generation in self._fenced_generations or generation < self._generation:
            self._dropped_stale_audio_frames += 1
            return
        if not isinstance(frame.payload, bytes):
            self._transition_degraded(
                "invalid_output_audio_payload",
                provider_event_id=int(RealtimeEvent.TTS_RESPONSE),
            )
            return
        now = time.monotonic()
        if self._first_output_audio_ms is None and self._connected_at:
            self._first_output_audio_ms = round((now - self._connected_at) * 1000.0, 2)
        self._last_output_audio_at = time.time()
        self._received_audio_frames += 1
        self._put_event(
            RealtimeVoiceEvent(
                event_type=RealtimeVoiceEventType.OUTPUT_AUDIO,
                session_id=self._session_id,
                generation=generation,
                provider="volcengine_s2s",
                provider_event_id=int(RealtimeEvent.TTS_RESPONSE),
                audio=VoiceMediaFrame(
                    pcm=frame.payload,
                    sample_rate=self._config.output_sample_rate,
                    channels=1,
                    metadata={
                        "question_id": (
                            self._active_response_question_id or self._current_question_id
                        ),
                        "reply_id": (self._active_response_reply_id or self._current_reply_id),
                    },
                ),
            )
        )

    def _send_event(
        self,
        event: RealtimeEvent | int,
        payload: Any,
        *,
        wait_sent: bool = False,
    ) -> bool:
        ws = self._ws
        if ws is None:
            return False
        event_id = int(event)
        try:
            frame = encode_frame(
                event,
                payload,
                session_id=self._session_id if event_id >= 100 else None,
                compression=Compression.NONE,
            )
            sender = self._sender_thread
            if sender is None or not sender.is_alive():
                self._send_raw(frame)
                return True
            item = _OutboundItem(
                frame=frame,
                event_id=event_id,
                is_media=event_id == int(RealtimeEvent.TASK_REQUEST),
            )
            if not self._enqueue_outbound(item):
                return False
            if not wait_sent:
                return True
            if not item.sent.wait(timeout=self._config.close_timeout_s):
                self._transition_degraded(
                    "provider_send_timeout",
                    provider_event_id=event_id,
                )
                return False
            return item.success
        except Exception as exc:
            if not self._closing:
                self._transition_degraded(
                    type(exc).__name__,
                    provider_event_id=event_id,
                )
            return False

    def _send_raw(self, frame: bytes) -> None:
        with self._send_lock:
            if self._ws is None:
                raise RuntimeError("websocket_not_connected")
            self._ws.send_binary(frame)

    def _enqueue_audio(self, audio: bytes) -> bool:
        return bool(self._send_event(RealtimeEvent.TASK_REQUEST, audio))

    def _enqueue_outbound(self, item: _OutboundItem) -> bool:
        control_timed_out = False
        with self._outbound_condition:
            if item.is_media:
                while (
                    self._outbound_media_count >= self._outbound_media_capacity
                    or len(self._outbound_queue) >= self._outbound_capacity
                ):
                    if not self._evict_oldest_media_locked():
                        self._dropped_input_frames += 1
                        return False
                self._outbound_media_count += 1
            else:
                deadline = time.monotonic() + self._config.close_timeout_s
                while len(self._outbound_queue) >= self._outbound_capacity:
                    if self._evict_oldest_media_locked():
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        control_timed_out = True
                        break
                    self._outbound_condition.wait(timeout=remaining)
            if not control_timed_out:
                self._outbound_queue.append(item)
                self._outbound_condition.notify()
                return True
        self._transition_degraded(
            "outbound_control_queue_timeout",
            provider_event_id=item.event_id,
        )
        return False

    def _evict_oldest_media_locked(self) -> bool:
        for index, queued in enumerate(self._outbound_queue):
            if not queued.is_media:
                continue
            del self._outbound_queue[index]
            self._outbound_media_count -= 1
            self._dropped_input_frames += 1
            queued.sent.set()
            return True
        return False

    def _put_event(self, event: RealtimeVoiceEvent) -> None:
        with self._event_condition:
            lossy_limit = self._event_capacity - self._event_control_reserve
            if event.event_type in _LOSSY_EVENTS and len(self._event_queue) >= lossy_limit:
                evict_index = next(
                    (
                        index
                        for index, queued in enumerate(self._event_queue)
                        if queued.event_type in _LOSSY_EVENTS
                    ),
                    None,
                )
                if evict_index is None:
                    self._dropped_events += 1
                    return
                del self._event_queue[evict_index]
                self._dropped_events += 1
            if len(self._event_queue) >= self._event_capacity:
                evict_index = next(
                    (
                        index
                        for index, queued in enumerate(self._event_queue)
                        if queued.event_type in _LOSSY_EVENTS
                    ),
                    None,
                )
                if evict_index is None and event.event_type in _ESSENTIAL_EVENTS:
                    evict_index = next(
                        (
                            index
                            for index, queued in enumerate(self._event_queue)
                            if queued.event_type not in _ESSENTIAL_EVENTS
                        ),
                        None,
                    )
                if evict_index is None:
                    self._dropped_events += 1
                    return
                del self._event_queue[evict_index]
                self._dropped_events += 1
            self._event_queue.append(event)
            self._event_condition.notify()

    def _emit_error(self, error: str, *, provider_event_id: int | None = None) -> None:
        self._put_event(
            RealtimeVoiceEvent(
                event_type=RealtimeVoiceEventType.ERROR,
                session_id=self._session_id,
                generation=self._generation,
                provider="volcengine_s2s",
                provider_event_id=provider_event_id,
                error=str(error)[:160],
            )
        )

    def _transition_degraded(
        self,
        error: str,
        *,
        provider_event_id: int | None = None,
    ) -> None:
        """Atomically make provider failure visible so callers can fall back."""

        with self._lifecycle_lock:
            already_reported = (
                not self._active
                and not self._connected
                and self._state == "degraded"
                and self._last_error == error
            )
            self._active = False
            self._connected = False
            self._state = "degraded"
            self._last_error = str(error)
            self._consecutive_failures += 1
            self._sender_stop.set()
            self._receiver_stop.set()
        with self._ack_lock:
            for pending in (self._pending_delete, self._pending_truncate):
                if pending is not None:
                    pending.success = False
                    pending.done.set()
        self._discard_outbound_queue()
        if not already_reported and not self._closing:
            self._emit_error(error, provider_event_id=provider_event_id)

    def _reset_runtime_state(self, context: RealtimeVoiceSessionContext) -> None:
        self._context = context
        self._session_id = context.session_id
        self._dialog_id = context.dialog_id
        self._state = "connecting"
        self._active = False
        self._connected = False
        self._closing = False
        self._session_finished.clear()
        self._connection_finished.clear()
        self._sender_stop.clear()
        self._receiver_stop.clear()
        self._generation = 0
        self._active_response_generation = 0
        self._current_question_id = ""
        self._current_reply_id = ""
        self._active_response_question_id = ""
        self._active_response_reply_id = ""
        self._question_generations.clear()
        self._reply_generations.clear()
        self._fenced_generations.clear()
        with self._ack_lock:
            self._pending_delete = None
            self._pending_truncate = None
            self._stale_delete_acks.clear()
            self._stale_truncate_acks.clear()
        self._first_output_audio_ms = None
        self._connected_at = 0.0
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._discard_outbound_queue()
        self._discard_event_queue()

    def _record_start_failure(self, exc: BaseException) -> None:
        self._active = False
        self._connected = False
        self._state = "degraded"
        self._last_error = type(exc).__name__
        self._consecutive_failures += 1
        if self._consecutive_failures >= max(1, self._config.circuit_failure_threshold):
            self._circuit_open_until = time.monotonic() + self._config.circuit_reset_seconds

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

    def _discard_outbound_media(self) -> None:
        with self._outbound_condition:
            retained: deque[_OutboundItem] = deque()
            for item in self._outbound_queue:
                if item.is_media:
                    self._dropped_input_frames += 1
                    item.sent.set()
                else:
                    retained.append(item)
            self._outbound_queue = retained
            self._outbound_media_count = 0
            self._outbound_condition.notify_all()

    def _discard_outbound_queue(self) -> None:
        with self._outbound_condition:
            while self._outbound_queue:
                self._outbound_queue.popleft().sent.set()
            self._outbound_media_count = 0
            self._outbound_condition.notify_all()

    def _discard_event_queue(self) -> None:
        with self._event_condition:
            self._event_queue.clear()
            self._event_condition.notify_all()

    @staticmethod
    def _public_ids(payload: dict[str, Any]) -> dict[str, str]:
        return {
            key: str(payload.get(key, ""))
            for key in ("question_id", "reply_id", "tts_type", "status_code")
            if payload.get(key) not in (None, "")
        }

    @staticmethod
    def _ack_item_ids(payload: dict[str, Any]) -> set[str]:
        direct = str(payload.get("item_id") or "").strip()
        item_ids = {direct} if direct else set()
        items = payload.get("items")
        if not isinstance(items, list):
            return item_ids
        for item in items:
            value = item.get("item_id") if isinstance(item, dict) else item
            clean = str(value or "").strip()
            if clean:
                item_ids.add(clean)
        return item_ids

    @staticmethod
    def _successful_status(payload: dict[str, Any]) -> bool:
        status = payload.get("status_code")
        if status in (None, ""):
            return True
        clean_status = str(status).strip().lower()
        if clean_status in {"ok", "success"}:
            return True
        try:
            return int(clean_status) == 0
        except ValueError:
            return False

    @staticmethod
    def _consume_stale_ack_locked(stale_acks: Counter[str]) -> bool:
        for item_id in list(stale_acks):
            if stale_acks[item_id] <= 0:
                del stale_acks[item_id]
                continue
            stale_acks[item_id] -= 1
            if stale_acks[item_id] <= 0:
                del stale_acks[item_id]
            return True
        return False

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
