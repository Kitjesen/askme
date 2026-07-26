"""Provider-only Volcengine/Doubao bidirectional TTS WebSocket client.

This module deliberately stays below ``TTSEngine``: it opens a provider
connection, runs one text synthesis session at a time, and emits provider audio
bytes through a callback.  It does not play audio, select voices globally, or
retry partially-audible sessions.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from askme.voice.output.volcengine_tts_protocol import (
    EventType,
    MessageType,
    VolcProtocolError,
    VolcTTSFrame,
    decode_server_frame,
    encode_cancel_session,
    encode_client_event_frame,
    encode_finish_session,
    encode_start_connection,
    encode_start_session,
    encode_task_request,
)

AudioCallback = Callable[[bytes], None]
ContinuePredicate = Callable[[], bool]
ConnectionFactory = Callable[..., Any]


class VolcengineTTSClientError(RuntimeError):
    """Provider failure that must not expose credentials."""


@dataclass(frozen=True)
class VolcengineTTSConfig:
    """Runtime settings for the Volcengine bidirectional TTS provider."""

    endpoint: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
    api_key: str = ""
    app_id: str = ""
    access_key: str = ""
    resource_id: str = "seed-tts-2.0"
    speaker: str = ""
    sample_rate: int = 24000
    audio_format: str = "pcm"
    timeout: float = 10.0
    connect_timeout: float | None = None
    session_timeout: float | None = None
    extra_req_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.endpoint:
            raise VolcengineTTSClientError("Volcengine TTS endpoint is required")
        if not self.api_key and not (self.app_id and self.access_key):
            raise VolcengineTTSClientError(
                "Volcengine TTS credentials are missing"
            )
        if not self.resource_id:
            raise VolcengineTTSClientError("Volcengine TTS resource_id is required")
        if not self.speaker:
            raise VolcengineTTSClientError("Volcengine TTS speaker is required")
        if self.sample_rate <= 0:
            raise VolcengineTTSClientError("Volcengine TTS sample_rate must be > 0")
        for name, value in (
            ("timeout", self.timeout),
            ("connect_timeout", self.connect_timeout),
            ("session_timeout", self.session_timeout),
        ):
            if value is not None and value <= 0:
                raise VolcengineTTSClientError(
                    f"Volcengine TTS {name} must be > 0"
                )

    @property
    def effective_connect_timeout(self) -> float:
        return self.connect_timeout if self.connect_timeout is not None else self.timeout

    @property
    def effective_session_timeout(self) -> float:
        return self.session_timeout if self.session_timeout is not None else self.timeout


@dataclass(frozen=True)
class VolcengineTTSSynthesisResult:
    session_id: str
    audio_chunks: int
    audio_bytes: int
    status: str


class VolcengineTTSClient:
    """Synchronous websocket-client adapter for Volcengine TTS V3."""

    def __init__(
        self,
        config: VolcengineTTSConfig,
        *,
        connection_factory: ConnectionFactory | None = None,
        connect_id_factory: Callable[[], str] | None = None,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._connection_factory = connection_factory
        self._connect_id_factory = connect_id_factory or (lambda: uuid.uuid4().hex)
        self._session_id_factory = session_id_factory or (lambda: uuid.uuid4().hex)
        self._connect_id: str | None = None
        self._ws: Any | None = None
        self._lock = threading.Lock()
        self._prewarm_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_session_id: str | None = None
        self._connecting_sockets: list[Any] = []
        self._closing_sockets: list[Any] = []
        self._operation_epoch = 0

    def build_headers(self, connect_id: str | None = None) -> list[str]:
        """Build provider headers without returning credential snapshots."""

        connect_id = connect_id or self._connect_id_factory()
        headers = [
            f"X-Api-Resource-Id: {self._config.resource_id}",
            f"X-Api-Connect-Id: {connect_id}",
        ]
        if self._config.api_key:
            headers.append(f"X-Api-Key: {self._config.api_key}")
        else:
            headers.extend(
                [
                    f"X-Api-App-ID: {self._config.app_id}",
                    f"X-Api-Access-Key: {self._config.access_key}",
                ]
            )
        return headers

    def prewarm(self) -> dict[str, Any]:
        """Open/reuse the provider connection without starting a session."""

        if not self._prewarm_lock.acquire(blocking=False):
            return {"ok": False, "status": "skipped", "reason": "already_running"}
        try:
            operation_epoch = self._begin_operation()
            with self._state_lock:
                existing = self._ws
            if existing is not None and getattr(existing, "connected", True) is not False:
                return {"ok": True, "status": "reused"}

            try:
                candidate = self._open_started_connection(operation_epoch)
            except Exception as exc:
                if self._operation_cancelled(operation_epoch):
                    return {
                        "ok": False,
                        "status": "cancelled",
                        "reason": "interrupted",
                    }
                return {
                    "ok": False,
                    "status": "failed",
                    "reason": exc.__class__.__name__,
                }
            if self._operation_cancelled(operation_epoch):
                self._discard_connecting(candidate)
                return {"ok": False, "status": "cancelled", "reason": "interrupted"}

            if not self._lock.acquire(blocking=False):
                self._discard_connecting(candidate)
                return {
                    "ok": False,
                    "status": "superseded",
                    "reason": "synthesis_started",
                }
            close_candidate = False
            try:
                with self._state_lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or not self._is_connecting_locked(candidate)
                    ):
                        return {
                            "ok": False,
                            "status": "cancelled",
                            "reason": "interrupted",
                        }
                    current = self._ws
                    if current is not None and getattr(current, "connected", True) is not False:
                        close_candidate = self._retire_connecting_locked(candidate)
                    else:
                        self._remove_connecting_locked(candidate)
                        self._ws = candidate
                        return {"ok": True, "status": "opened"}
            finally:
                self._lock.release()
            if close_candidate:
                self._close_retired(candidate)
            return {"ok": True, "status": "superseded_by_live_session"}
        finally:
            self._prewarm_lock.release()

    def interrupt(self) -> None:
        """Best-effort cross-thread cancellation that can unblock ``recv``."""

        sockets, session_id = self._cancel_and_detach_connections()
        if not sockets:
            return
        if session_id:
            try:
                sockets[0].send_binary(encode_cancel_session(session_id))
            except Exception:
                pass
        for ws in sockets:
            _safe_close(ws)

    def synthesize(
        self,
        text: str,
        *,
        on_audio: AudioCallback,
        should_continue: ContinuePredicate | None = None,
    ) -> VolcengineTTSSynthesisResult:
        """Synthesize one complete text request over a serialized session."""

        if not text:
            raise VolcengineTTSClientError("Volcengine TTS text is required")
        if not callable(on_audio):
            raise TypeError("on_audio must be callable")
        should_continue = should_continue or (lambda: True)

        with self._lock:
            operation_epoch = self._begin_operation()
            session_id = self._session_id_factory()
            audio_chunks = 0
            audio_bytes = 0
            received_audio = False
            try:
                ws: Any = self._ensure_connection(operation_epoch)
                with self._state_lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or self._ws is not ws
                    ):
                        raise VolcengineTTSClientError(
                            "Volcengine TTS synthesis was interrupted"
                        )
                    self._active_session_id = session_id
                assert ws is not None
                if not should_continue():
                    self._cancel_and_drop(ws, session_id)
                    return VolcengineTTSSynthesisResult(session_id, 0, 0, "cancelled")
                self._start_session(ws, session_id)
                if not should_continue():
                    self._cancel_and_drop(ws, session_id)
                    return VolcengineTTSSynthesisResult(session_id, 0, 0, "cancelled")
                ws.send_binary(
                    encode_task_request(
                        session_id,
                        self._session_payload(EventType.TASK_REQUEST, text=text),
                    )
                )
                ws.send_binary(encode_finish_session(session_id))

                while True:
                    if self._operation_cancelled(operation_epoch) or not should_continue():
                        self._cancel_and_drop(ws, session_id)
                        return VolcengineTTSSynthesisResult(
                            session_id,
                            audio_chunks,
                            audio_bytes,
                            "cancelled",
                        )
                    frame = self._recv_frame(ws)
                    if self._operation_cancelled(operation_epoch) or not should_continue():
                        self._cancel_and_drop(ws, session_id)
                        return VolcengineTTSSynthesisResult(
                            session_id,
                            audio_chunks,
                            audio_bytes,
                            "cancelled",
                        )
                    event = _event(frame)
                    self._require_session_match(frame, session_id)
                    if self._is_audio_frame(frame):
                        payload = bytes(frame.payload)
                        if payload:
                            received_audio = True
                            on_audio(payload)
                            audio_chunks += 1
                            audio_bytes += len(payload)
                        continue
                    if event == EventType.SESSION_FINISHED:
                        return VolcengineTTSSynthesisResult(
                            session_id,
                            audio_chunks,
                            audio_bytes,
                            "finished",
                        )
                    if event == EventType.SESSION_CANCELED:
                        return VolcengineTTSSynthesisResult(
                            session_id,
                            audio_chunks,
                            audio_bytes,
                            "cancelled",
                        )
                    event_code = _failed_event_code(event)
                    if event_code is not None:
                        raise VolcengineTTSClientError(
                            f"Volcengine TTS provider event failed: {event_code}"
                        )
                    if frame.message_type == MessageType.ERROR:
                        raise VolcengineTTSClientError(
                            f"Volcengine TTS provider error: {frame.error_code}"
                        )
            except Exception as exc:
                if self._operation_cancelled(operation_epoch):
                    self._close_unlocked(graceful=False)
                    return VolcengineTTSSynthesisResult(
                        session_id,
                        audio_chunks,
                        audio_bytes,
                        "cancelled",
                    )
                self._close_unlocked(graceful=False)
                suffix = " after audio" if received_audio else ""
                raise VolcengineTTSClientError(
                    f"Volcengine TTS synthesis failed{suffix}: "
                    f"{self._safe_error(exc)}"
                ) from exc
            finally:
                with self._state_lock:
                    if self._active_session_id == session_id:
                        self._active_session_id = None

    def close(self) -> None:
        if self._lock.acquire(blocking=False):
            try:
                sockets, _session_id = self._cancel_and_detach_connections()
                self._close_sockets(sockets, graceful=True)
            finally:
                self._lock.release()
            return
        sockets, _session_id = self._cancel_and_detach_connections()
        self._close_sockets(sockets, graceful=False)

    def _ensure_connection(self, operation_epoch: int | None = None) -> Any:
        if operation_epoch is None:
            operation_epoch = self._begin_operation()
        with self._state_lock:
            current = self._ws
            if current is not None and getattr(current, "connected", True) is not False:
                return current

        ws = self._open_started_connection(operation_epoch)
        reusable_connection: Any | None = None
        with self._state_lock:
            if (
                operation_epoch != self._operation_epoch
                or not self._is_connecting_locked(ws)
            ):
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection was interrupted before publication"
                )
            current = self._ws
            if current is not None and getattr(current, "connected", True) is not False:
                self._retire_connecting_locked(ws)
                reusable_connection = current
            else:
                self._remove_connecting_locked(ws)
                self._ws = ws
                return ws
        self._close_retired(ws)
        assert reusable_connection is not None
        return reusable_connection

    def _open_started_connection(self, operation_epoch: int) -> Any:
        ws = self._open_connection_cancellable(operation_epoch)
        self._track_connecting(ws)
        try:
            if self._operation_cancelled(operation_epoch):
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection was interrupted"
                )
            ws.send_binary(encode_start_connection())
            frame = self._recv_frame(ws)
            if self._operation_cancelled(operation_epoch):
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection was interrupted"
                )
            if _event(frame) != EventType.CONNECTION_STARTED:
                raise VolcengineTTSClientError("Volcengine TTS connection was not accepted")
            settimeout = getattr(ws, "settimeout", None)
            if callable(settimeout):
                settimeout(self._config.effective_session_timeout)
            return ws
        except Exception:
            self._discard_connecting(ws)
            raise

    def _open_connection_cancellable(self, operation_epoch: int) -> Any:
        """Open a socket without making barge-in wait for DNS/TCP timeout.

        ``websocket.create_connection`` exposes a timeout but no cross-thread
        cancellation handle before it returns a socket.  Run only that connect
        phase in a daemon worker, poll the operation epoch, and close any late
        socket after the caller has cancelled or timed out.
        """

        results: queue.Queue[tuple[Any | None, BaseException | None]] = queue.Queue(
            maxsize=1
        )
        ownership_lock = threading.Lock()
        abandoned = False

        def publish_result(ws: Any | None, exc: BaseException | None) -> None:
            nonlocal abandoned
            close_late_socket = False
            with ownership_lock:
                if abandoned:
                    close_late_socket = ws is not None
                else:
                    results.put_nowait((ws, exc))
            if close_late_socket:
                _safe_close(ws)

        def connect_worker() -> None:
            try:
                publish_result(self._open_connection(), None)
            except BaseException as exc:  # forwarded to the owning operation
                publish_result(None, exc)

        def abandon() -> Any | None:
            nonlocal abandoned
            with ownership_lock:
                abandoned = True
                try:
                    ws, _exc = results.get_nowait()
                except queue.Empty:
                    return None
                return ws

        worker = threading.Thread(
            target=connect_worker,
            name="volcengine-tts-connect",
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + self._config.effective_connect_timeout
        while True:
            if self._operation_cancelled(operation_epoch):
                late_socket = abandon()
                if late_socket is not None:
                    _safe_close(late_socket)
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection was interrupted"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                late_socket = abandon()
                if late_socket is not None:
                    _safe_close(late_socket)
                raise VolcengineTTSClientError("Volcengine TTS connect timed out")
            try:
                ws, exc = results.get(timeout=min(0.01, remaining))
            except queue.Empty:
                continue
            if self._operation_cancelled(operation_epoch):
                with ownership_lock:
                    abandoned = True
                if ws is not None:
                    _safe_close(ws)
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection was interrupted"
                )
            if exc is not None:
                raise exc
            if ws is None:
                raise VolcengineTTSClientError(
                    "Volcengine TTS connection returned no socket"
                )
            return ws

    def _open_connection(self) -> Any:
        connect_id = self._connect_id_factory()
        headers = self.build_headers(connect_id)
        try:
            if self._connection_factory is not None:
                ws = self._connection_factory(
                    self._config.endpoint,
                    header=headers,
                    timeout=self._config.effective_connect_timeout,
                )
            else:
                import websocket

                ws = websocket.create_connection(
                    self._config.endpoint,
                    header=headers,
                    timeout=self._config.effective_connect_timeout,
                )
        except Exception as exc:
            raise VolcengineTTSClientError(
                f"Volcengine TTS connect failed: {self._safe_error(exc)}"
            ) from exc
        self._connect_id = connect_id
        return ws

    def _start_session(self, ws: Any, session_id: str) -> None:
        ws.send_binary(
            encode_start_session(
                session_id,
                self._session_payload(EventType.START_SESSION),
            )
        )
        while True:
            frame = self._recv_frame(ws)
            event = _event(frame)
            self._require_session_match(frame, session_id)
            if event == EventType.SESSION_STARTED:
                return
            event_code = _failed_event_code(event)
            if event_code is not None:
                raise VolcengineTTSClientError(
                    f"Volcengine TTS session start failed: {event_code}"
                )
            if frame.message_type == MessageType.ERROR:
                raise VolcengineTTSClientError(
                    f"Volcengine TTS provider error: {frame.error_code}"
                )

    def _session_payload(self, event: EventType, *, text: str | None = None) -> dict[str, Any]:
        req_params = {
            **self._config.extra_req_params,
            "speaker": self._config.speaker,
            "audio_params": {
                "format": self._config.audio_format,
                "sample_rate": self._config.sample_rate,
            },
        }
        if text is not None:
            req_params["text"] = text
        return {"event": int(event), "req_params": req_params}

    def _recv_frame(self, ws: Any) -> VolcTTSFrame:
        raw = ws.recv()
        if isinstance(raw, str):
            raise VolcengineTTSClientError(
                "Volcengine TTS expected binary websocket frame"
            )
        try:
            return decode_server_frame(bytes(raw))
        except VolcProtocolError as exc:
            raise VolcengineTTSClientError(
                f"Volcengine TTS protocol error: {exc}"
            ) from exc

    @staticmethod
    def _is_audio_frame(frame: VolcTTSFrame) -> bool:
        return _event(frame) == EventType.TTS_RESPONSE and bool(frame.payload)

    @staticmethod
    def _require_session_match(frame: VolcTTSFrame, session_id: str) -> None:
        event = _event(frame)
        if frame.message_type == MessageType.AUDIO_ONLY_SERVER and (
            event != EventType.TTS_RESPONSE or not frame.session_id
        ):
            raise VolcengineTTSClientError(
                "Volcengine TTS audio requires a session-scoped TTS_RESPONSE"
            )
        if int(event or 0) < 100:
            return
        if frame.session_id != session_id:
            raise VolcengineTTSClientError(
                "Volcengine TTS provider returned mismatched session_id"
            )

    def _cancel_and_drop(self, ws: Any, session_id: str) -> None:
        try:
            ws.send_binary(encode_cancel_session(session_id))
        except Exception:
            self._close_unlocked(graceful=False)
            raise
        self._close_unlocked(graceful=False)

    def _close_unlocked(self, *, graceful: bool) -> None:
        sockets, _session_id = self._detach_connections()
        self._close_sockets(sockets, graceful=graceful)

    @staticmethod
    def _close_sockets(sockets: list[Any], *, graceful: bool) -> None:
        if not sockets:
            return
        ws = sockets[0]
        if graceful:
            try:
                ws.send_binary(encode_client_event_frame(EventType.FINISH_CONNECTION, {}))
            except Exception:
                pass
        for socket in sockets:
            _safe_close(socket)

    def _detach_connections(self) -> tuple[list[Any], str | None]:
        with self._state_lock:
            return self._detach_connections_locked()

    def _cancel_and_detach_connections(self) -> tuple[list[Any], str | None]:
        with self._state_lock:
            self._operation_epoch += 1
            return self._detach_connections_locked()

    def _detach_connections_locked(self) -> tuple[list[Any], str | None]:
        ws = self._ws
        session_id = self._active_session_id
        connecting = list(self._connecting_sockets)
        self._connecting_sockets.clear()
        self._ws = None
        self._active_session_id = None
        sockets = ([ws] if ws is not None else []) + [
            candidate for candidate in connecting if candidate is not ws
        ]
        return sockets, session_id

    def _begin_operation(self) -> int:
        with self._state_lock:
            return self._operation_epoch

    def _operation_cancelled(self, operation_epoch: int) -> bool:
        with self._state_lock:
            return operation_epoch != self._operation_epoch

    def _track_connecting(self, ws: Any) -> None:
        with self._state_lock:
            self._connecting_sockets.append(ws)

    def _discard_connecting(self, ws: Any) -> None:
        with self._state_lock:
            should_close = self._retire_connecting_locked(ws)
        if should_close:
            self._close_retired(ws)

    def _is_connecting_locked(self, ws: Any) -> bool:
        return any(candidate is ws for candidate in self._connecting_sockets)

    def _remove_connecting_locked(self, ws: Any) -> None:
        self._connecting_sockets = [
            candidate for candidate in self._connecting_sockets if candidate is not ws
        ]

    def _retire_connecting_locked(self, ws: Any) -> bool:
        if not self._is_connecting_locked(ws):
            return False
        self._remove_connecting_locked(ws)
        if not any(candidate is ws for candidate in self._closing_sockets):
            self._closing_sockets.append(ws)
        return True

    def _close_retired(self, ws: Any) -> None:
        try:
            _safe_close(ws)
        finally:
            with self._state_lock:
                self._closing_sockets = [
                    candidate
                    for candidate in self._closing_sockets
                    if candidate is not ws
                ]

    def _safe_error(self, exc: BaseException) -> str:
        message = str(exc)
        for secret in (
            self._config.api_key,
            self._config.app_id,
            self._config.access_key,
        ):
            if secret:
                message = message.replace(secret, "[redacted]")
        return message[:300]


def _event(frame: VolcTTSFrame) -> EventType | int | None:
    if frame.event is None:
        return None
    try:
        return EventType(frame.event)
    except ValueError:
        return frame.event


def _failed_event_code(event: EventType | int | None) -> int | None:
    if event is None:
        return None
    event_code = int(event)
    if event_code in {
        int(EventType.SESSION_FAILED),
        int(EventType.CONNECTION_FAILED),
    }:
        return event_code
    return None


def _safe_close(ws: Any) -> None:
    for name in ("abort", "shutdown"):
        method = getattr(ws, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    close = getattr(ws, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
