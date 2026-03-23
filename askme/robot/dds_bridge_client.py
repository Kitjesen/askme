"""Async client for the DDS bridge Unix socket.

Replaces /tmp/ file polling with a persistent Unix socket connection
to ``poc_dds_bridge.py`` (or its production successor).  The bridge
publishes newline-delimited JSON messages of the form::

    {"topic": "/thunder/detections", "data": {...}, "ts": 1234567890.123}

This client connects, auto-reconnects, caches the latest message per
topic, and dispatches registered callbacks in the asyncio event loop.

Configuration (``config.yaml`` under ``dds_bridge``)::

    dds_bridge:
      enabled: true
      socket_path: /tmp/askme_dds_bridge.sock
      reconnect_interval: 2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_SOCKET_PATH = "/tmp/askme_dds_bridge.sock"
_DEFAULT_RECONNECT_INTERVAL = 2.0

# Well-known topic names
_TOPIC_ESTOP = "/thunder/estop"
_TOPIC_DETECTIONS = "/thunder/detections"

# Callback signature: (topic: str, data: dict, ts: float) -> None
TopicCallback = Callable[[str, dict, float], Any]


class DdsBridgeClient:
    """Async client that reads DDS bridge messages via Unix socket.

    Thread-safe: ``get_latest()`` and ``is_estop_active()`` can be called
    from any thread.  Callbacks fire in the asyncio event loop that called
    ``start()``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._socket_path: str = cfg.get("socket_path", _DEFAULT_SOCKET_PATH)
        self._reconnect_interval: float = float(
            cfg.get("reconnect_interval", _DEFAULT_RECONNECT_INTERVAL)
        )
        self._enabled: bool = cfg.get("enabled", False)

        # Latest message cache per topic — guarded by _lock
        self._latest: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Callbacks per topic
        self._callbacks: dict[str, list[TopicCallback]] = {}

        # Background task handle
        self._task: asyncio.Task[None] | None = None
        self._connected = False
        self._msg_count = 0

    # ── Public API ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to the socket and spawn a background read task.

        No-op if ``dds_bridge.enabled`` is ``false`` in config.
        """
        if not self._enabled:
            logger.debug("DdsBridgeClient: disabled in config")
            return
        if self._task is not None:
            return  # already running
        self._task = asyncio.get_running_loop().create_task(
            self._read_loop(), name="dds-bridge-reader"
        )
        logger.info(
            "DdsBridgeClient: started (socket=%s, reconnect=%.1fs)",
            self._socket_path,
            self._reconnect_interval,
        )

    async def stop(self) -> None:
        """Cancel the background task and clean up."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._connected = False
        logger.info("DdsBridgeClient: stopped")

    def on(self, topic: str, callback: TopicCallback) -> None:
        """Register a callback for *topic*.

        Callbacks are invoked in the asyncio event loop with
        ``(topic, data, ts)`` arguments.  Both sync and async callbacks
        are supported.
        """
        self._callbacks.setdefault(topic, []).append(callback)

    def get_latest(self, topic: str) -> dict[str, Any] | None:
        """Return the most recent message for *topic*, or ``None``.

        Thread-safe, non-blocking.  The returned dict contains
        ``{"topic": ..., "data": ..., "ts": ...}``.
        """
        with self._lock:
            cached = self._latest.get(topic)
            return dict(cached) if cached is not None else None

    # ── Convenience methods ────────────────────────────────────────────

    def is_estop_active(self) -> bool:
        """Return ``True`` if the latest estop message indicates active.

        Returns ``False`` when no estop data has been received (safe default).
        """
        with self._lock:
            cached = self._latest.get(_TOPIC_ESTOP)
        if cached is None:
            return False
        return bool(cached.get("data", {}).get("active", False))

    def get_detections(self) -> dict[str, Any] | None:
        """Return the latest detections payload, or ``None``."""
        msg = self.get_latest(_TOPIC_DETECTIONS)
        if msg is None:
            return None
        return msg.get("data")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def message_count(self) -> int:
        return self._msg_count

    # ── Internal ───────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Connect, read, and auto-reconnect forever."""
        while True:
            writer = None
            try:
                reader, writer = await asyncio.open_unix_connection(
                    self._socket_path
                )
                self._connected = True
                logger.info(
                    "DdsBridgeClient: connected to %s", self._socket_path
                )

                buffer = b""
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        logger.warning(
                            "DdsBridgeClient: connection closed by bridge"
                        )
                        break

                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        await self._handle_message(line)

            except asyncio.CancelledError:
                if writer is not None:
                    writer.close()
                raise
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                self._connected = False
                logger.debug(
                    "DdsBridgeClient: bridge not available (%s), "
                    "retrying in %.1fs",
                    exc,
                    self._reconnect_interval,
                )
            except OSError as exc:
                self._connected = False
                logger.warning(
                    "DdsBridgeClient: connection error (%s), "
                    "retrying in %.1fs",
                    exc,
                    self._reconnect_interval,
                )

            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

            self._connected = False
            await asyncio.sleep(self._reconnect_interval)

    async def _handle_message(self, raw: bytes) -> None:
        """Parse one newline-delimited JSON message and dispatch."""
        if not raw:
            return
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("DdsBridgeClient: malformed message, skipping")
            return

        topic: str = msg.get("topic", "")
        data: dict = msg.get("data", {})
        ts: float = msg.get("ts", 0.0)

        self._msg_count += 1

        # Cache latest
        with self._lock:
            self._latest[topic] = msg

        # Dispatch callbacks
        for cb in self._callbacks.get(topic, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(topic, data, ts)
                else:
                    cb(topic, data, ts)
            except Exception:
                logger.exception(
                    "DdsBridgeClient: callback error for topic %s", topic
                )
