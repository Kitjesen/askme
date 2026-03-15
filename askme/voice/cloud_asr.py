"""Cloud ASR backend — Alibaba DashScope Paraformer real-time.

Streams audio via WebSocket, returns transcription results.
Falls back to local sherpa-onnx ASR when API key is not set or network fails.

Usage::

    cloud = CloudASR(config)
    if cloud.available:
        cloud.start_session()
        cloud.feed(pcm16_bytes)  # call repeatedly during speech
        text = cloud.finish_session()  # returns final transcription
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# DashScope WebSocket endpoint
_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
_DEFAULT_MODEL = "paraformer-realtime-v2"


class CloudASR:
    """Alibaba DashScope Paraformer real-time ASR via WebSocket.

    Config keys (under ``voice.cloud_asr``)::

        enabled: bool           - Enable cloud ASR (default False)
        api_key: str            - DashScope API key (sk-xxx) or ${DASHSCOPE_API_KEY}
        model: str              - Model ID (default paraformer-realtime-v2)
        sample_rate: int        - Audio sample rate (default 16000)
        language_hints: list    - Language hints (default ["zh", "en"])
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._api_key: str = cfg.get("api_key", "")
        self._model: str = cfg.get("model", _DEFAULT_MODEL)
        self._sample_rate: int = int(cfg.get("sample_rate", 16000))
        self._language_hints: list[str] = cfg.get("language_hints", ["zh", "en"])

        # Session state
        self._ws = None
        self._task_id: str = ""
        self._result_text: str = ""
        self._result_ready = threading.Event()
        self._error: str | None = None
        self._recv_thread: threading.Thread | None = None
        self._session_active: bool = False

        # Performance tracking
        self._last_ttft: float = 0.0
        self._session_start: float = 0.0

        if self._enabled and self._api_key:
            logger.info(
                "CloudASR: enabled model=%s sr=%d",
                self._model, self._sample_rate,
            )
        elif self._enabled:
            logger.warning("CloudASR: enabled but no API key — will fall back to local")
            self._enabled = False

    @property
    def available(self) -> bool:
        return self._enabled and bool(self._api_key)

    def start_session(self) -> bool:
        """Open a WebSocket connection and start recognition.

        Returns True if session started successfully, False on error
        (caller should fall back to local ASR).
        """
        if not self.available:
            return False

        try:
            import websocket  # websocket-client library (sync)
        except ImportError:
            logger.warning("CloudASR: websocket-client not installed, falling back")
            self._enabled = False
            return False

        self._task_id = str(uuid.uuid4())
        self._result_text = ""
        self._result_ready.clear()
        self._error = None
        self._session_start = time.monotonic()
        self._last_ttft = 0.0

        try:
            self._ws = websocket.WebSocket()
            self._ws.connect(
                _WS_URL,
                header=[f"Authorization: bearer {self._api_key}"],
                timeout=5,
            )

            # Send run-task message
            start_msg = {
                "header": {
                    "action": "run-task",
                    "task_id": self._task_id,
                    "streaming": "duplex",
                },
                "payload": {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": self._model,
                    "parameters": {
                        "sample_rate": self._sample_rate,
                        "format": "pcm",
                        "language_hints": self._language_hints,
                    },
                    "input": {},
                },
            }
            self._ws.send(json.dumps(start_msg))

            # Wait for task-started ack
            ack_raw = self._ws.recv()
            ack = json.loads(ack_raw)
            event = ack.get("header", {}).get("event", "")
            if event != "task-started":
                logger.error("CloudASR: unexpected ack event: %s", event)
                self._ws.close()
                return False

            # Start receiver thread
            self._session_active = True
            self._recv_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self._recv_thread.start()

            logger.info("CloudASR: session started (task_id=%s)", self._task_id[:8])
            return True

        except Exception as exc:
            logger.error("CloudASR: start_session failed: %s", exc)
            self._cleanup()
            return False

    def feed(self, pcm16_bytes: bytes) -> None:
        """Send a chunk of PCM16 audio to the cloud ASR.

        Args:
            pcm16_bytes: Raw PCM16 little-endian bytes (16kHz mono).
        """
        if not self._session_active or self._ws is None:
            return
        try:
            self._ws.send_binary(pcm16_bytes)
        except Exception as exc:
            logger.warning("CloudASR: feed error: %s", exc)
            self._error = str(exc)

    def finish_session(self, timeout: float = 5.0) -> str:
        """Signal end of audio and wait for final result.

        Returns the transcribed text, or empty string on error/timeout.
        """
        if not self._session_active or self._ws is None:
            return self._result_text

        try:
            # Send finish-task
            finish_msg = {
                "header": {
                    "action": "finish-task",
                    "task_id": self._task_id,
                    "streaming": "duplex",
                },
                "payload": {"input": {}},
            }
            self._ws.send(json.dumps(finish_msg))
        except Exception as exc:
            logger.warning("CloudASR: finish send error: %s", exc)

        # Wait for final result
        self._result_ready.wait(timeout=timeout)

        text = self._result_text.strip()
        total_ms = (time.monotonic() - self._session_start) * 1000
        logger.info(
            "CloudASR: result='%s' total=%.0fms ttft=%.0fms",
            text[:50], total_ms, self._last_ttft,
        )

        self._cleanup()
        return text

    def cancel_session(self) -> None:
        """Cancel the current session without waiting for results."""
        self._cleanup()

    def _receive_loop(self) -> None:
        """Background thread: receive transcription results from WebSocket."""
        first_result = True
        try:
            while self._session_active and self._ws is not None:
                try:
                    raw = self._ws.recv()
                except Exception:
                    break

                if isinstance(raw, bytes):
                    continue

                msg = json.loads(raw)
                event = msg.get("header", {}).get("event", "")

                if event == "result-generated":
                    sentence = msg.get("payload", {}).get("output", {}).get("sentence", {})
                    text = sentence.get("text", "")
                    is_final = sentence.get("sentence_end", False)

                    if first_result and text:
                        first_result = False
                        self._last_ttft = (time.monotonic() - self._session_start) * 1000

                    if is_final and text:
                        self._result_text = text
                        logger.debug("CloudASR: final sentence: '%s'", text[:50])

                    elif text:
                        # Interim — keep updating
                        self._result_text = text
                        logger.debug("CloudASR: interim: '%s'", text[:50])

                elif event == "task-finished":
                    logger.debug("CloudASR: task finished")
                    break

                elif event == "task-failed":
                    err = msg.get("header", {}).get("error_message", "unknown")
                    logger.error("CloudASR: task failed: %s", err)
                    self._error = err
                    break

        except Exception as exc:
            logger.error("CloudASR: receive error: %s", exc)
            self._error = str(exc)
        finally:
            self._result_ready.set()

    def _cleanup(self) -> None:
        """Close WebSocket and reset session state."""
        self._session_active = False
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
