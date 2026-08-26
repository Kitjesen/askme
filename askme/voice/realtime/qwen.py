"""Threaded adapter for Qwen3.5-Omni Realtime WebSocket sessions."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.config import RealtimeVoiceMode

DEFAULT_QWEN_REALTIME_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_QWEN_REALTIME_MODEL = "qwen3.5-omni-flash-realtime"

ConnectionFactory = Callable[..., Any]

_QWEN_WORKSPACE_HOST_SUFFIXES = (
    ".cn-beijing.maas.aliyuncs.com",
    ".ap-southeast-1.maas.aliyuncs.com",
)


def _is_official_qwen_endpoint(endpoint: str) -> bool:
    if endpoint == DEFAULT_QWEN_REALTIME_ENDPOINT:
        return True
    try:
        parts = urlsplit(endpoint)
        port = parts.port
    except ValueError:
        return False
    if (
        parts.scheme != "wss"
        or parts.path != "/api-ws/v1/realtime"
        or parts.username is not None
        or parts.password is not None
        or port is not None
        or bool(parts.fragment)
    ):
        return False
    host = (parts.hostname or "").lower()
    for suffix in _QWEN_WORKSPACE_HOST_SUFFIXES:
        if host.endswith(suffix):
            workspace = host[: -len(suffix)]
            return bool(workspace and "." not in workspace)
    return False


class _ProviderConnectionClosed(ValueError):
    """The peer sent a WebSocket close frame instead of a JSON event."""


@dataclass
class _OutboundItem:
    """One serialized runtime event waiting for the sender thread."""

    payload: str
    is_media: bool = False
    sent: threading.Event = field(default_factory=threading.Event)
    success: bool = False


@dataclass(frozen=True)
class QwenRealtimeConfig:
    """Provider-specific configuration kept independent of secret storage."""

    enabled: bool = False
    mode: RealtimeVoiceMode = RealtimeVoiceMode.GENERAL_CHAT
    fallback: str = "cascade"
    endpoint: str = DEFAULT_QWEN_REALTIME_ENDPOINT
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_QWEN_REALTIME_MODEL
    voice: str = "Tina"
    bot_name: str = "小算"
    system_role: str = ""
    speaking_style: str = "简洁、自然、口语化；不要声称已经执行机器人动作。"
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    output_format: str = "pcm_s16le"
    chunk_ms: int = 20
    vad_threshold: float = 0.1
    vad_prefix_padding_ms: int = 500
    vad_silence_duration_ms: int = 900
    connect_timeout_s: float = 4.0
    close_timeout_s: float = 1.0
    audio_queue_ms: int = 400
    event_queue_size: int = 256
    max_reconnect_attempts: int = 1

    @property
    def credentials_configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def available(self) -> bool:
        return self.enabled and not self.validation_errors()

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.enabled and not self.credentials_configured:
            errors.append("qwen realtime requires api_key when enabled")
        if not _is_official_qwen_endpoint(self.endpoint):
            errors.append(
                "qwen realtime endpoint must use the official DashScope realtime endpoint"
            )
        if self.fallback != "cascade":
            errors.append("qwen realtime fallback must stay cascade")
        if self.input_sample_rate != 16_000:
            errors.append("qwen realtime input sample rate must be 16000")
        if self.output_sample_rate != 24_000:
            errors.append("qwen realtime output sample rate must be 24000")
        if self.output_format != "pcm_s16le":
            errors.append("qwen realtime output format must be pcm_s16le")
        if self.chunk_ms != 20:
            errors.append("qwen realtime chunk size must be 20 ms")
        if not -1.0 <= self.vad_threshold <= 1.0:
            errors.append("qwen realtime VAD threshold must be between -1 and 1")
        if self.vad_prefix_padding_ms < 0:
            errors.append("qwen realtime VAD prefix padding must not be negative")
        if not 200 <= self.vad_silence_duration_ms <= 6_000:
            errors.append("qwen realtime VAD silence must be between 200 and 6000 ms")
        if self.audio_queue_ms < self.chunk_ms:
            errors.append("qwen realtime audio queue must hold at least one chunk")
        if self.event_queue_size < 3:
            errors.append("qwen realtime event queue must hold at least three events")
        return errors


class QwenRealtimeDialogue:
    """Translate AskMe's provider-neutral session interface to Qwen events."""

    def __init__(
        self,
        config: QwenRealtimeConfig,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self._config = config
        self._connection_factory = connection_factory
        self._lifecycle_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._input_order_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._event_condition = threading.Condition()
        self._event_capacity = max(3, config.event_queue_size)
        self._event_queue: deque[RealtimeVoiceEvent] = deque()
        audio_queue_frames = max(1, config.audio_queue_ms // max(1, config.chunk_ms))
        self._outbound_media_capacity = audio_queue_frames
        self._outbound_capacity = audio_queue_frames + 8
        self._outbound_queue: deque[_OutboundItem] = deque()
        self._outbound_media_count = 0
        self._outbound_condition = threading.Condition()
        self._sender_stop = threading.Event()
        self._sender_thread: threading.Thread | None = None
        self._receiver_stop = threading.Event()
        self._receiver_thread: threading.Thread | None = None
        self._ws: Any = None
        self._context: RealtimeVoiceSessionContext | None = None
        self._session_id = ""
        self._provider_session_id = ""
        self._state = "disabled" if not config.enabled else "idle"
        self._active = False
        self._connected = False
        self._closing = False
        self._last_error = ""
        self._audio_buffer = bytearray()
        self._sent_audio_frames = 0
        self._dropped_input_frames = 0
        self._dropped_events = 0
        self._received_audio_frames = 0
        self._last_input_audio_at = 0.0
        self._last_output_audio_at = 0.0
        self._first_output_audio_ms: float | None = None
        self._generation = 0
        self._active_response_generation = 0
        self._item_generations: dict[str, int] = {}
        self._response_generations: dict[str, int] = {}
        self._response_final_text: dict[str, str] = {}
        self._fenced_generations: set[int] = set()
        self._dropped_stale_audio_frames = 0
        self._connected_at = 0.0
        self._reconnect_count = 0

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

            self._context = context
            self._session_id = context.session_id
            attempts = max(1, self._config.max_reconnect_attempts + 1)
            last_error: BaseException | None = None
            for attempt in range(attempts):
                try:
                    self._connect_and_handshake(context)
                    self._start_workers()
                    self._last_error = ""
                    return True
                except Exception as exc:
                    last_error = exc
                    self._close_socket_only()
                    if attempt + 1 < attempts:
                        self._reconnect_count += 1
            self._state = "degraded"
            self._last_error = (
                "provider_connection_closed"
                if isinstance(last_error, _ProviderConnectionClosed)
                else type(last_error).__name__
                if last_error
                else "connection_failed"
            )
            return False

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        """Packetize 16 kHz mono PCM into provider-sized Base64 events."""

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
            with self._buffer_lock:
                self._audio_buffer.extend(frame.pcm)
                chunks: list[bytes] = []
                while len(self._audio_buffer) >= packet_bytes:
                    chunks.append(bytes(self._audio_buffer[:packet_bytes]))
                    del self._audio_buffer[:packet_bytes]
            try:
                for chunk in chunks:
                    if not self._send_audio(chunk):
                        self._dropped_input_frames += 1
                        self._transition_degraded("input_audio_queue_overflow")
                        return False
            except Exception as exc:
                self._transition_degraded(type(exc).__name__)
                return False
        self._last_input_audio_at = time.time()
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

    def finish_input(self) -> bool:
        """Flush trailing PCM and explicitly submit a manual/PTT turn."""

        if not self._active:
            self._last_error = "finish_input_inactive"
            return False
        with self._input_order_lock:
            with self._buffer_lock:
                remainder = bytes(self._audio_buffer)
                self._audio_buffer.clear()
            try:
                if remainder:
                    if not self._send_audio(remainder):
                        self._dropped_input_frames += 1
                        self._transition_degraded("finish_input_audio_queue_overflow")
                        return False
                if self._context is not None and self._context.input_mode == "push_to_talk":
                    if not self._send_client_event(
                        "input_audio_buffer.commit",
                        wait_sent=True,
                    ):
                        self._transition_degraded("finish_input_commit_failed")
                        return False
                    if not self._send_client_event("response.create", wait_sent=True):
                        self._transition_degraded("finish_input_response_create_failed")
                        return False
            except Exception as exc:
                self._transition_degraded(type(exc).__name__)
                return False
        return True

    def interrupt(self, reason: str) -> None:
        """Cancel and fence the active provider response generation."""

        del reason
        with self._lifecycle_lock:
            if not self._active:
                return
            generation = self._active_response_generation
            if generation <= 0:
                return
            self._fenced_generations.add(generation)
        try:
            if not self._send_client_event("response.cancel"):
                self._transition_degraded("response_cancel_enqueue_failed")
        except Exception as exc:
            self._transition_degraded(type(exc).__name__)

    def close(self, reason: str = "shutdown") -> None:
        del reason
        with self._lifecycle_lock:
            if self._closing:
                return
            self._closing = True
            self._active = False
            sender = self._sender_thread
            receiver = self._receiver_thread
            self._sender_stop.set()
            self._discard_outbound()
            self._receiver_stop.set()
            self._close_socket_only()
        if sender is not None and sender is not threading.current_thread():
            sender.join(timeout=self._config.close_timeout_s)
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=self._config.close_timeout_s)
        with self._lifecycle_lock:
            self._sender_thread = None
            self._receiver_thread = None
            self._state = "closed"
            self._closing = False

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "provider": "qwen3_5_omni",
            "enabled": self._config.enabled,
            "available": self.available,
            "mode": self._config.mode.value,
            "fallback": self._config.fallback,
            "endpoint": self._config.endpoint,
            "model": self._config.model,
            "voice": self._config.voice,
            "credentials_configured": self._config.credentials_configured,
            "state": self._state,
            "connected": self._connected,
            "active": self._active,
            "session_id": self._session_id,
            "provider_session_id": self._provider_session_id,
            "reconnect_count": self._reconnect_count,
            "audio_buffer_bytes": len(self._audio_buffer),
            "audio_queue_depth": self._outbound_media_count,
            "outbound_queue_depth": len(self._outbound_queue),
            "sent_audio_frames": self._sent_audio_frames,
            "dropped_input_frames": self._dropped_input_frames,
            "received_audio_frames": self._received_audio_frames,
            "input_audio_active": bool(self._last_input_audio_at),
            "output_audio_active": bool(self._last_output_audio_at),
            "event_queue_depth": len(self._event_queue),
            "dropped_events": self._dropped_events,
            "generation": self._generation,
            "active_response_generation": self._active_response_generation,
            "dropped_stale_audio_frames": self._dropped_stale_audio_frames,
            "first_output_audio_ms": self._first_output_audio_ms,
            "last_error": self._last_error,
        }

    def _connect_and_handshake(self, context: RealtimeVoiceSessionContext) -> None:
        factory = self._connection_factory
        if factory is None:
            try:
                import websocket
            except ImportError as exc:  # pragma: no cover - project dependency
                raise RuntimeError("websocket-client is not installed") from exc
            factory = websocket.create_connection

        self._state = "connecting"
        self._ws = factory(
            self._endpoint_with_model(),
            header=[f"Authorization: Bearer {self._config.api_key}"],
            timeout=self._config.connect_timeout_s,
        )
        created = self._receive_handshake("session.created")
        created_session = created.get("session")
        if not isinstance(created_session, dict):
            raise ValueError("provider_session_created_missing_session")
        self._provider_session_id = str(created_session.get("id", ""))
        self._send_json(
            {
                "event_id": f"event_{uuid.uuid4().hex}",
                "type": "session.update",
                "session": self._session_payload(context),
            }
        )
        updated = self._receive_handshake("session.updated")
        updated_session = updated.get("session")
        if isinstance(updated_session, dict) and updated_session.get("id"):
            self._provider_session_id = str(updated_session["id"])
        settimeout = getattr(self._ws, "settimeout", None)
        if callable(settimeout):
            settimeout(0.1)
        self._connected = True
        self._active = True
        self._state = "listening"
        self._connected_at = time.monotonic()

    def _session_payload(self, context: RealtimeVoiceSessionContext) -> dict[str, Any]:
        system_role = context.system_role or self._config.system_role
        restrictions = [
            "不得调用任何工具或外部服务",
            "不得执行、控制或声称已经执行机器人、设备或现实世界动作",
        ]
        instructions = "\n".join(
            part
            for part in (
                f"你的名字是{context.bot_name or self._config.bot_name}。",
                system_role,
                context.speaking_style or self._config.speaking_style,
                "安全边界：" + "；".join(restrictions) + "。",
            )
            if part
        )
        turn_detection: dict[str, Any] | None = {
            "type": "semantic_vad",
            "threshold": self._config.vad_threshold,
            "prefix_padding_ms": self._config.vad_prefix_padding_ms,
            "silence_duration_ms": self._config.vad_silence_duration_ms,
        }
        if context.input_mode == "push_to_talk":
            turn_detection = None
        return {
            "modalities": ["text", "audio"],
            "voice": self._config.voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
            "instructions": instructions,
            "turn_detection": turn_detection,
        }

    def _endpoint_with_model(self) -> str:
        parts = urlsplit(self._config.endpoint)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["model"] = self._config.model
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )

    def _receive_handshake(self, expected_type: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._config.connect_timeout_s
        while time.monotonic() < deadline:
            try:
                raw = self._ws.recv()
            except Exception as exc:
                if self._is_timeout(exc):
                    continue
                raise
            event = self._decode_event(raw)
            if event.get("type") == "error":
                raise RuntimeError("provider_handshake_error")
            if event.get("type") == expected_type:
                return event
        raise TimeoutError("provider_handshake_timeout")

    def _send_json(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._send_lock:
            self._ws.send(payload)

    def _send_client_event(
        self,
        event_type: str,
        *,
        is_media: bool = False,
        wait_sent: bool = False,
        **payload: Any,
    ) -> bool:
        serialized = json.dumps(
            {
                "event_id": f"event_{uuid.uuid4().hex}",
                "type": event_type,
                **payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        item = _OutboundItem(payload=serialized, is_media=is_media)
        with self._outbound_condition:
            if not self._active or self._sender_stop.is_set():
                return False
            if is_media and (
                self._outbound_media_count >= self._outbound_media_capacity
                or len(self._outbound_queue) >= self._outbound_capacity
            ):
                return False
            if not is_media and len(self._outbound_queue) >= self._outbound_capacity:
                return False
            self._outbound_queue.append(item)
            if is_media:
                self._outbound_media_count += 1
            self._outbound_condition.notify()
        if not wait_sent:
            return True
        if not item.sent.wait(timeout=max(0.05, self._config.close_timeout_s)):
            return False
        return item.success

    def _send_audio(self, pcm: bytes) -> bool:
        return self._send_client_event(
            "input_audio_buffer.append",
            is_media=True,
            audio=base64.b64encode(pcm).decode("ascii"),
        )

    def _start_workers(self) -> None:
        self._sender_stop.clear()
        self._receiver_stop.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="qwen-realtime-send",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name="qwen-realtime-recv",
            daemon=True,
        )
        self._sender_thread.start()
        self._receiver_thread.start()

    def _sender_loop(self) -> None:
        while not self._sender_stop.is_set():
            with self._outbound_condition:
                while not self._outbound_queue and not self._sender_stop.is_set():
                    self._outbound_condition.wait(timeout=0.1)
                if self._sender_stop.is_set():
                    return
                item = self._outbound_queue.popleft()
                if item.is_media:
                    self._outbound_media_count -= 1
                self._outbound_condition.notify_all()
            try:
                with self._send_lock:
                    self._ws.send(item.payload)
                item.success = True
                if item.is_media:
                    self._sent_audio_frames += 1
            except Exception as exc:
                if not self._closing:
                    self._transition_degraded(type(exc).__name__)
            finally:
                item.sent.set()

    def _discard_outbound(self) -> None:
        with self._outbound_condition:
            while self._outbound_queue:
                item = self._outbound_queue.popleft()
                item.sent.set()
            self._outbound_media_count = 0
            self._outbound_condition.notify_all()

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
            try:
                self._handle_server_event(self._decode_event(raw))
            except _ProviderConnectionClosed:
                self._transition_degraded("provider_connection_closed")
                return
            except (KeyError, TypeError, ValueError):
                self._transition_degraded("provider_event_error")
                return

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "input_audio_buffer.speech_started":
            item_id = self._required_id(event, "item_id")
            if item_id in self._item_generations:
                return
            self._generation += 1
            self._item_generations[item_id] = self._generation
            self._state = "user_speaking"
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.INPUT_SPEECH_STARTED,
                    generation=self._generation,
                    event=event,
                    metadata={"question_id": item_id},
                )
            )
            return
        if event_type == "input_audio_buffer.committed":
            item_id = self._required_id(event, "item_id")
            if item_id not in self._item_generations:
                self._generation += 1
                self._item_generations[item_id] = self._generation
            return
        if event_type == "conversation.item.input_audio_transcription.delta":
            item_id = self._required_id(event, "item_id")
            generation = self._item_generations.get(item_id, self._generation)
            if generation <= 0:
                return
            transcript = str(event.get("text", "")) + str(event.get("stash", ""))
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
                    generation=generation,
                    event=event,
                    transcript=transcript,
                    metadata={"question_id": item_id},
                )
            )
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            item_id = self._required_id(event, "item_id")
            generation = self._item_generations.get(item_id, self._generation)
            if generation <= 0:
                return
            self._state = "thinking"
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
                    generation=generation,
                    event=event,
                    transcript=str(event.get("transcript", "")),
                    is_final=True,
                    metadata={"question_id": item_id},
                )
            )
            return
        if event_type == "response.created":
            response = self._required_mapping(event, "response")
            response_id = self._required_id(response, "id")
            generation = self._generation
            if generation <= 0:
                generation = 1
                self._generation = generation
            self._response_generations[response_id] = generation
            self._response_final_text.pop(response_id, None)
            self._active_response_generation = generation
            self._state = "responding"
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.RESPONSE_STARTED,
                    generation=generation,
                    event=event,
                    metadata={"reply_id": response_id},
                )
            )
            return
        if event_type in {"response.audio_transcript.done", "response.text.done"}:
            response_id = self._required_id(event, "response_id")
            generation = self._response_generations.get(response_id, self._generation)
            reply_id = str(event.get("item_id", "")) or response_id
            field_name = "transcript" if event_type == "response.audio_transcript.done" else "text"
            final_text = str(event.get(field_name, ""))
            self._response_final_text[response_id] = final_text
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
                    generation=generation,
                    event=event,
                    text=final_text,
                    metadata={"reply_id": reply_id, "authoritative_final": True},
                )
            )
            return
        if event_type in {"response.audio_transcript.delta", "response.text.delta"}:
            response_id = self._required_id(event, "response_id")
            generation = self._response_generations.get(response_id, self._generation)
            reply_id = str(event.get("item_id", "")) or response_id
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
                    generation=generation,
                    event=event,
                    text=str(event.get("delta", "")),
                    metadata={"reply_id": reply_id},
                )
            )
            return
        if event_type == "response.audio.delta":
            response_id = self._required_id(event, "response_id")
            generation = self._response_generations.get(response_id, self._generation)
            reply_id = str(event.get("item_id", "")) or response_id
            if generation in self._fenced_generations:
                self._dropped_stale_audio_frames += 1
                return
            raw_audio = base64.b64decode(str(event.get("delta", "")), validate=True)
            if not raw_audio or len(raw_audio) % 2:
                raise ValueError("invalid_output_audio")
            now = time.monotonic()
            if self._first_output_audio_ms is None and self._connected_at:
                self._first_output_audio_ms = round((now - self._connected_at) * 1000.0, 2)
            self._last_output_audio_at = time.time()
            self._received_audio_frames += 1
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.OUTPUT_AUDIO,
                    generation=generation,
                    event=event,
                    audio=VoiceMediaFrame(
                        pcm=raw_audio,
                        sample_rate=self._config.output_sample_rate,
                        channels=1,
                    ),
                    metadata={"reply_id": reply_id},
                )
            )
            return
        if event_type == "response.done":
            response = self._required_mapping(event, "response")
            response_id = self._required_id(response, "id")
            generation = self._response_generations.get(response_id, self._generation)
            streamed_final_text = self._response_final_text.pop(response_id, "")
            final_text = self._extract_response_text(response) or streamed_final_text
            usage = response.get("usage")
            if isinstance(usage, dict):
                self._put_event(
                    self._voice_event(
                        RealtimeVoiceEventType.USAGE,
                        generation=generation,
                        event=event,
                        metadata={"usage": usage, "reply_id": response_id},
                    )
                )
            self._put_event(
                self._voice_event(
                    RealtimeVoiceEventType.RESPONSE_DONE,
                    generation=generation,
                    event=event,
                    text=final_text,
                    is_final=True,
                    metadata={
                        "reply_id": response_id,
                        "status": str(response.get("status", "")),
                    },
                )
            )
            self._active_response_generation = 0
            self._state = "listening"
            return
        if event_type == "error":
            error = event.get("error")
            code = (
                str(error.get("code", "provider_error"))
                if isinstance(error, dict)
                else "provider_error"
            )
            self._transition_degraded(code, event=event)

    def _voice_event(
        self,
        event_type: RealtimeVoiceEventType,
        *,
        generation: int,
        event: dict[str, Any],
        transcript: str = "",
        text: str = "",
        is_final: bool = False,
        audio: VoiceMediaFrame | None = None,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RealtimeVoiceEvent:
        public_metadata = dict(metadata or {})
        provider_event_id = str(event.get("event_id", ""))
        if provider_event_id:
            public_metadata["provider_event_id"] = provider_event_id
        return RealtimeVoiceEvent(
            event_type=event_type,
            session_id=self._session_id,
            generation=generation,
            provider="qwen3_5_omni",
            transcript=transcript,
            text=text,
            is_final=is_final,
            audio=audio,
            error=error,
            metadata=public_metadata,
        )

    def _put_event(self, event: RealtimeVoiceEvent) -> None:
        with self._event_condition:
            if len(self._event_queue) >= self._event_capacity:
                incoming_priority = self._event_priority(event.event_type)
                eviction_index = next(
                    (
                        index
                        for index, queued in enumerate(self._event_queue)
                        if self._event_priority(queued.event_type) < incoming_priority
                    ),
                    None,
                )
                if eviction_index is None:
                    self._dropped_events += 1
                    return
                del self._event_queue[eviction_index]
                self._dropped_events += 1
            self._event_queue.append(event)
            self._event_condition.notify()

    @staticmethod
    def _event_priority(event_type: RealtimeVoiceEventType) -> int:
        if event_type is RealtimeVoiceEventType.ERROR:
            return 3
        if event_type in {
            RealtimeVoiceEventType.RESPONSE_DONE,
            RealtimeVoiceEventType.USAGE,
        }:
            return 2
        if event_type in {
            RealtimeVoiceEventType.OUTPUT_AUDIO,
            RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
            RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA,
        }:
            return 0
        return 1

    def _transition_degraded(
        self,
        error: str,
        *,
        event: dict[str, Any] | None = None,
    ) -> None:
        self._last_error = str(error or "provider_error")[:160]
        self._state = "degraded"
        self._active = False
        self._connected = False
        self._sender_stop.set()
        self._discard_outbound()
        self._receiver_stop.set()
        self._put_event(
            self._voice_event(
                RealtimeVoiceEventType.ERROR,
                generation=self._active_response_generation or self._generation,
                event=event or {},
                error=self._last_error,
                metadata={},
            )
        )

    @staticmethod
    def _required_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
        value = container.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
        return value

    @staticmethod
    def _extract_response_text(response: dict[str, Any]) -> str:
        output = response.get("output")
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                value = part.get("transcript") or part.get("text")
                if isinstance(value, str) and value:
                    parts.append(value)
        return "".join(parts)

    @staticmethod
    def _required_id(container: dict[str, Any], key: str) -> str:
        value = str(container.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} is required")
        return value

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()

    @staticmethod
    def _decode_event(raw: Any) -> dict[str, Any]:
        if raw in ("", b""):
            raise _ProviderConnectionClosed("provider_connection_closed")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise ValueError("provider_event_must_be_json_text")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ValueError("provider_event_must_be_object")
        return event

    def _close_socket_only(self) -> None:
        ws = self._ws
        self._ws = None
        self._active = False
        self._connected = False
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
