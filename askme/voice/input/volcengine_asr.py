"""Volcengine Doubao streaming ASR WebSocket backend.

The provider uses Volcengine's version-3 binary protocol. Audio remains raw
PCM16 inside Gzip-compressed WebSocket frames; recognition results arrive as
Gzip-compressed JSON frames.
"""

from __future__ import annotations

import gzip
import json
import logging
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from askme.interfaces.asr import ASRBackend

logger = logging.getLogger(__name__)

_DEFAULT_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
_DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
_DEFAULT_MODEL = "bigmodel"

_FULL_CLIENT_REQUEST = 0x1
_AUDIO_ONLY_REQUEST = 0x2
_FULL_SERVER_RESPONSE = 0x9
_ERROR_RESPONSE = 0xF

_NO_SEQUENCE = 0x0
_POSITIVE_SEQUENCE = 0x1
_LAST_PACKET = 0x2
_NEGATIVE_SEQUENCE = 0x3

_NO_SERIALIZATION = 0x0
_JSON_SERIALIZATION = 0x1
_NO_COMPRESSION = 0x0
_GZIP_COMPRESSION = 0x1


@dataclass(frozen=True)
class VolcengineServerFrame:
    """Decoded server frame from the Volcengine binary protocol."""

    message_type: int
    flags: int
    payload: dict[str, Any]
    sequence: int | None = None
    error_code: int | None = None
    is_final: bool = False


class VolcengineASR(ASRBackend):
    """Doubao Seed ASR 2.0 via Volcengine streaming WebSocket.

    Authentication supports both console generations:

    * legacy: ``app_id`` plus ``access_token``
    * current: ``api_key``

    ``secret_key`` is deliberately unsupported because this API does not use
    it for WebSocket authentication.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._enabled = bool(cfg.get("enabled", False))
        self._api_key = str(cfg.get("api_key", "")).strip()
        self._app_id = str(cfg.get("app_id", "")).strip()
        self._access_token = str(cfg.get("access_token", "")).strip()
        self._resource_id = str(
            cfg.get("resource_id", _DEFAULT_RESOURCE_ID)
        ).strip()
        self._endpoint = str(cfg.get("endpoint", _DEFAULT_WS_URL)).strip()
        self._model = str(cfg.get("model", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
        self._sample_rate = int(cfg.get("sample_rate", 16000))
        self._bits = int(cfg.get("bits", 16))
        self._channels = int(cfg.get("channels", 1))
        self._chunk_ms = max(20, int(cfg.get("chunk_ms", 200)))
        self._connect_timeout = float(cfg.get("connect_timeout", 5.0))
        self._user_id = str(cfg.get("user_id", "askme-voice")).strip()
        self._device_id = str(cfg.get("device_id", "xiaosuan-1")).strip()
        self._hotwords = self._normalized_hotwords(cfg.get("hotwords", []))
        self._enable_nonstream = bool(cfg.get("enable_nonstream", True))
        self._enable_itn = bool(cfg.get("enable_itn", True))
        self._enable_punc = bool(cfg.get("enable_punc", True))
        self._enable_ddc = bool(cfg.get("enable_ddc", False))
        self._show_utterances = bool(cfg.get("show_utterances", True))
        self._end_window_size = max(200, int(cfg.get("end_window_size", 800)))

        self._ws: Any = None
        self._request_id = ""
        self._connect_id = ""
        self._log_id = ""
        self._result_text = ""
        self._interim_text = ""
        self._result_ready = threading.Event()
        self._error: str | None = None
        self._recv_thread: threading.Thread | None = None
        self._session_active = False
        self._send_lock = threading.Lock()
        self._audio_buffer = bytearray()

        self._session_start = 0.0
        self._last_ttft = 0.0
        self._last_partial_at_epoch_s = 0.0
        self._last_final_at_epoch_s = 0.0
        self._last_session_error = ""

        if self._enabled and not self._credentials_present():
            logger.warning(
                "VolcengineASR: enabled but credentials are incomplete; using local ASR"
            )
            self._enabled = False
        elif self._enabled and not self._resource_id:
            logger.warning(
                "VolcengineASR: enabled but resource_id is empty; using local ASR"
            )
            self._enabled = False

    @property
    def available(self) -> bool:
        return self._enabled and self._credentials_present() and bool(self._resource_id)

    def start_session(self) -> bool:
        """Connect, authenticate, and send the full client request frame."""
        if not self.available:
            return False

        try:
            import websocket
        except ImportError:
            logger.warning(
                "VolcengineASR: websocket-client not installed; using local ASR"
            )
            self._enabled = False
            return False

        create_connection = getattr(websocket, "create_connection", None)
        if create_connection is None:
            logger.warning(
                "VolcengineASR: incompatible websocket package; install websocket-client"
            )
            self._enabled = False
            return False

        self._reset_session_state()

        try:
            self._ws = create_connection(
                self._endpoint,
                header=self._connection_headers(),
                timeout=self._connect_timeout,
            )
            self._capture_log_id()
            self._ws.send_binary(self._build_full_request_frame())

            ack_raw = self._ws.recv()
            if not isinstance(ack_raw, bytes):
                raise RuntimeError("unexpected non-binary acknowledgement")
            acknowledgement = self._parse_server_frame(ack_raw)
            if acknowledgement.error_code is not None:
                self._record_provider_error(acknowledgement)
                self._cleanup()
                return False
            self._handle_server_frame(acknowledgement)

            settimeout = getattr(self._ws, "settimeout", None)
            if callable(settimeout):
                settimeout(None)

            self._session_active = True
            self._recv_thread = threading.Thread(
                target=self._receive_loop,
                name="volcengine-asr-recv",
                daemon=True,
            )
            self._recv_thread.start()
            logger.info(
                "VolcengineASR: session started resource=%s request_id=%s",
                self._resource_id,
                self._request_id[:8],
            )
            return True
        except Exception as exc:
            self._last_session_error = str(exc)
            logger.error("VolcengineASR: start_session failed: %s", exc)
            self._cleanup()
            return False

    def feed(self, pcm16_bytes: bytes) -> None:
        """Buffer PCM16 and send provider-recommended 200 ms packets."""
        if not self._session_active or self._ws is None or not pcm16_bytes:
            return

        bytes_per_sample = self._bits // 8
        packet_bytes = (
            self._sample_rate
            * self._channels
            * bytes_per_sample
            * self._chunk_ms
            // 1000
        )
        with self._send_lock:
            self._audio_buffer.extend(pcm16_bytes)
            while len(self._audio_buffer) >= packet_bytes:
                packet = bytes(self._audio_buffer[:packet_bytes])
                del self._audio_buffer[:packet_bytes]
                if not self._send_audio_packet(packet, final=False):
                    break

    def finish_session(self, timeout: float = 5.0) -> str:
        """Send the final audio packet and return the best transcript."""
        if not self._session_active or self._ws is None:
            return self._best_text()

        with self._send_lock:
            final_audio = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            self._send_audio_packet(final_audio, final=True)

        self._result_ready.wait(timeout=timeout)
        text = self._best_text()
        total_ms = (time.monotonic() - self._session_start) * 1000.0
        logger.info(
            "VolcengineASR: result='%s' total=%.0fms ttft=%.0fms",
            text[:50],
            total_ms,
            self._last_ttft,
        )
        self._cleanup()
        return text

    def cancel_session(self) -> None:
        # Release a concurrent finish_session() immediately even when no
        # receiver thread was created or the WebSocket close is slow.
        self._result_ready.set()
        self._cleanup()

    def status_snapshot(self) -> dict[str, Any]:
        """Return provider state without exposing authentication material."""
        now = time.time()
        return {
            "provider": "volcengine_seed_asr",
            "endpoint": self._endpoint,
            "enabled": self._enabled,
            "available": self.available,
            "active": self._session_active,
            "model": self._model,
            "resource_id": self._resource_id,
            "sample_rate": self._sample_rate,
            "chunk_ms": self._chunk_ms,
            "request_id": self._request_id[:8] if self._request_id else "",
            "log_id": self._log_id,
            "partial_text": self._interim_text,
            "final_text": self._result_text,
            "partial_age_ms": self._age_ms(now, self._last_partial_at_epoch_s),
            "final_age_ms": self._age_ms(now, self._last_final_at_epoch_s),
            "first_partial_ms": round(self._last_ttft, 2) if self._last_ttft else None,
            "last_error": self._last_session_error or self._error or "",
        }

    def _connection_headers(self) -> list[str]:
        self._request_id = self._request_id or str(uuid.uuid4())
        self._connect_id = self._connect_id or str(uuid.uuid4())
        if self._api_key:
            headers = [f"X-Api-Key: {self._api_key}"]
        else:
            headers = [
                f"X-Api-App-Key: {self._app_id}",
                f"X-Api-Access-Key: {self._access_token}",
            ]
        headers.extend(
            [
                f"X-Api-Resource-Id: {self._resource_id}",
                f"X-Api-Request-Id: {self._request_id}",
                f"X-Api-Connect-Id: {self._connect_id}",
                "X-Api-Sequence: -1",
            ]
        )
        return headers

    def _build_full_request_frame(self) -> bytes:
        payload = gzip.compress(
            json.dumps(
                self._request_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        header = self._header(
            message_type=_FULL_CLIENT_REQUEST,
            flags=_NO_SEQUENCE,
            serialization=_JSON_SERIALIZATION,
            compression=_GZIP_COMPRESSION,
        )
        return header + struct.pack(">I", len(payload)) + payload

    def _build_audio_frame(self, audio: bytes, *, final: bool) -> bytes:
        payload = gzip.compress(audio)
        header = self._header(
            message_type=_AUDIO_ONLY_REQUEST,
            flags=_LAST_PACKET if final else _NO_SEQUENCE,
            serialization=_NO_SERIALIZATION,
            compression=_GZIP_COMPRESSION,
        )
        return header + struct.pack(">I", len(payload)) + payload

    def _parse_server_frame(self, frame: bytes) -> VolcengineServerFrame:
        if len(frame) < 8:
            raise ValueError("Volcengine ASR frame is shorter than 8 bytes")

        version = frame[0] >> 4
        header_size = (frame[0] & 0x0F) * 4
        if version != 1 or header_size < 4 or len(frame) < header_size + 4:
            raise ValueError("Volcengine ASR frame has an invalid header")

        message_type = frame[1] >> 4
        flags = frame[1] & 0x0F
        serialization = frame[2] >> 4
        compression = frame[2] & 0x0F
        offset = header_size
        sequence: int | None = None
        error_code: int | None = None

        if message_type == _FULL_SERVER_RESPONSE and flags in {
            _POSITIVE_SEQUENCE,
            _NEGATIVE_SEQUENCE,
        }:
            sequence = struct.unpack(">i", self._slice(frame, offset, 4))[0]
            offset += 4
        elif message_type == _ERROR_RESPONSE:
            error_code = struct.unpack(">I", self._slice(frame, offset, 4))[0]
            offset += 4

        payload_size = struct.unpack(">I", self._slice(frame, offset, 4))[0]
        offset += 4
        payload_bytes = self._slice(frame, offset, payload_size)

        if compression == _GZIP_COMPRESSION and payload_bytes:
            payload_bytes = gzip.decompress(payload_bytes)
        elif compression not in {_NO_COMPRESSION, _GZIP_COMPRESSION}:
            raise ValueError(f"Unsupported Volcengine compression: {compression}")

        payload = self._decode_payload(payload_bytes, serialization)
        return VolcengineServerFrame(
            message_type=message_type,
            flags=flags,
            payload=payload,
            sequence=sequence,
            error_code=error_code,
            is_final=(
                message_type == _FULL_SERVER_RESPONSE
                and (flags in {_LAST_PACKET, _NEGATIVE_SEQUENCE} or (sequence or 0) < 0)
            ),
        )

    def _request_payload(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model_name": self._model,
            "enable_nonstream": self._enable_nonstream,
            "enable_itn": self._enable_itn,
            "enable_punc": self._enable_punc,
            "enable_ddc": self._enable_ddc,
            "show_utterances": self._show_utterances,
            "end_window_size": self._end_window_size,
        }
        if self._hotwords:
            request["corpus"] = {
                "context": json.dumps(
                    {"hotwords": [{"word": word} for word in self._hotwords]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            }
        return {
            "user": {
                "uid": self._user_id,
                "did": self._device_id,
                "platform": sys.platform,
                "sdk_version": "askme",
            },
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": self._sample_rate,
                "bits": self._bits,
                "channel": self._channels,
            },
            "request": request,
        }

    def _send_audio_packet(self, packet: bytes, *, final: bool) -> bool:
        if self._ws is None:
            return False
        try:
            self._ws.send_binary(self._build_audio_frame(packet, final=final))
            return True
        except Exception as exc:
            self._error = str(exc)
            self._last_session_error = str(exc)
            logger.warning("VolcengineASR: audio send failed: %s", exc)
            self._result_ready.set()
            return False

    def _receive_loop(self) -> None:
        try:
            while self._session_active and self._ws is not None:
                raw = self._ws.recv()
                if not isinstance(raw, bytes):
                    continue
                response = self._parse_server_frame(raw)
                if response.error_code is not None:
                    self._record_provider_error(response)
                    break
                self._handle_server_frame(response)
                if response.is_final:
                    break
        except Exception as exc:
            if self._session_active:
                self._error = str(exc)
                self._last_session_error = str(exc)
                logger.warning("VolcengineASR: receive loop ended: %s", exc)
        finally:
            self._result_ready.set()

    def _handle_server_frame(self, response: VolcengineServerFrame) -> None:
        result = response.payload.get("result")
        if not isinstance(result, dict):
            if response.is_final:
                self._result_ready.set()
            return

        text = str(result.get("text", "")).strip()
        if text:
            if not self._last_ttft:
                self._last_ttft = (
                    time.monotonic() - self._session_start
                ) * 1000.0
            self._interim_text = text
            self._last_partial_at_epoch_s = time.time()
            if response.is_final:
                self._result_text = text
                self._last_final_at_epoch_s = time.time()
        if response.is_final:
            self._result_ready.set()

    def _record_provider_error(self, response: VolcengineServerFrame) -> None:
        message = str(
            response.payload.get("message")
            or response.payload.get("error")
            or response.payload
        )
        self._error = f"{response.error_code}: {message}"
        self._last_session_error = self._error
        logger.error("VolcengineASR: provider error %s", self._error)
        self._result_ready.set()

    def _capture_log_id(self) -> None:
        response = getattr(self._ws, "handshake_response", None)
        headers = getattr(response, "headers", {})
        if not isinstance(headers, dict):
            return
        for key, value in headers.items():
            if str(key).lower() == "x-tt-logid":
                self._log_id = str(value)
                return

    def _reset_session_state(self) -> None:
        self._request_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._log_id = ""
        self._result_text = ""
        self._interim_text = ""
        self._result_ready.clear()
        self._error = None
        self._last_session_error = ""
        self._session_start = time.monotonic()
        self._last_ttft = 0.0
        self._last_partial_at_epoch_s = 0.0
        self._last_final_at_epoch_s = 0.0
        self._audio_buffer.clear()

    def _best_text(self) -> str:
        return (self._result_text or self._interim_text).strip()

    def _cleanup(self) -> None:
        self._session_active = False
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        thread = self._recv_thread
        self._recv_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._audio_buffer.clear()

    def _credentials_present(self) -> bool:
        return bool(self._api_key or (self._app_id and self._access_token))

    @staticmethod
    def _normalized_hotwords(raw: Any) -> list[str]:
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(word).strip() for word in raw if str(word).strip()]

    @staticmethod
    def _header(
        *,
        message_type: int,
        flags: int,
        serialization: int,
        compression: int,
    ) -> bytes:
        return bytes(
            (
                0x11,
                (message_type << 4) | flags,
                (serialization << 4) | compression,
                0x00,
            )
        )

    @staticmethod
    def _slice(frame: bytes, offset: int, length: int) -> bytes:
        end = offset + length
        if length < 0 or end > len(frame):
            raise ValueError("Volcengine ASR frame payload is truncated")
        return frame[offset:end]

    @staticmethod
    def _decode_payload(payload: bytes, serialization: int) -> dict[str, Any]:
        if not payload:
            return {}
        text = payload.decode("utf-8", errors="replace")
        if serialization == _JSON_SERIALIZATION:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        if serialization != _NO_SERIALIZATION:
            raise ValueError(f"Unsupported Volcengine serialization: {serialization}")
        return {"message": text}

    @staticmethod
    def _age_ms(now: float, timestamp: float) -> float | None:
        if timestamp <= 0:
            return None
        return round((now - timestamp) * 1000.0, 2)


__all__ = ["VolcengineASR", "VolcengineServerFrame"]
