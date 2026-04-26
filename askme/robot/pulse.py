"""Pulse — 脉搏数据总线.

穹沛机器人统一数据总线。底层用 CycloneDDS，不依赖 ROS2。
只需要 libddsc.so + CYCLONEDDS_HOME。

Usage::

    bus = Pulse(cfg)
    bus.on("/thunder/detections", my_callback)
    await bus.start()
"""

# NOTE: Do NOT use "from __future__ import annotations" here.
# CycloneDDS IdlStruct needs real type objects (str, bool), not string literals.

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from askme.interfaces.bus import BusBackend

logger = logging.getLogger(__name__)

try:
    from dataclasses import dataclass as _dc

    from cyclonedds.core import Listener, ReadCondition, WaitSet
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.idl import IdlStruct
    from cyclonedds.pub import DataWriter
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from cyclonedds.util import duration

    _CYCLONE_AVAILABLE = True
except (ImportError, Exception):
    # ImportError: cyclonedds not installed
    # Other exceptions: cyclonedds installed but libddsc.so not found (CycloneDDSLoaderException)
    _CYCLONE_AVAILABLE = False


# ── IDL message type for string topics ───────────────

if _CYCLONE_AVAILABLE:
    @_dc
    class StringMsg(IdlStruct):
        """Generic string message for JSON-encoded topics."""
        data: str

    @_dc
    class BoolMsg(IdlStruct):
        """Generic bool message for ESTOP/heartbeat topics."""
        data: bool


# ── Topic config ─────────────────────────────────────

_TOPIC_CONFIG = {
    "/thunder/detections": ("string", lambda m: json.loads(m.data)),
    "/thunder/estop": ("bool", lambda m: {"active": m.data}),
    "/thunder/heartbeat": ("bool", lambda m: {"alive": m.data}),
    "/thunder/joint_states": ("string", lambda m: json.loads(m.data)),
    "/thunder/imu": ("string", lambda m: json.loads(m.data)),
    "/thunder/cms_state": ("string", lambda m: json.loads(m.data)),
}


class Pulse(BusBackend):
    """脉搏数据总线 — CycloneDDS 直连.

    Same API as previous Pulse implementation, now backed by CycloneDDS.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self._enabled = cfg.get("enabled", _CYCLONE_AVAILABLE)
        self._node_name = cfg.get("node_name", "askme")

        self._dp: Any = None  # DomainParticipant
        self._readers: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        self._topics: dict[str, Any] = {}
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._latest: dict[str, dict] = {}
        self._latest_lock = threading.Lock()
        self._callbacks: dict[str, list[Callable]] = {}
        self._msg_count = 0
        self._started = False

        self._init_topic_tracking()

    @property
    def available(self) -> bool:
        return _CYCLONE_AVAILABLE and self._enabled

    @property
    def connected(self) -> bool:
        return self._started and self._dp is not None

    @property
    def msg_count(self) -> int:
        return self._msg_count

    async def start(self) -> None:
        if self._started:
            return
        if not self.available:
            logger.info("Pulse: disabled (cyclonedds not available)")
            return

        self._loop = asyncio.get_running_loop()
        self._dp = DomainParticipant()

        # Create readers for all registered topics
        for topic_name, (msg_kind, parser) in _TOPIC_CONFIG.items():
            msg_type = BoolMsg if msg_kind == "bool" else StringMsg
            topic = Topic(self._dp, topic_name, msg_type)
            reader = DataReader(self._dp, topic)
            self._readers[topic_name] = (reader, parser)
            self._topics[topic_name] = topic

        # Poll in background thread
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="pulse_poll", daemon=True,
        )
        self._poll_thread.start()
        self._started = True
        logger.info("Pulse: started (node=%s, topics=%d, NO ROS2)",
                     self._node_name, len(self._readers))

    async def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        self._readers.clear()
        self._writers.clear()
        self._topics.clear()
        self._dp = None
        self._started = False
        logger.info("Pulse: stopped (total_msgs=%d)", self._msg_count)

    def _poll_loop(self) -> None:
        """Background thread: poll all readers at ~100Hz."""
        while not self._stop_event.is_set():
            for topic_name, (reader, parser) in self._readers.items():
                try:
                    samples = reader.take(N=10)
                    for sample in samples:
                        self._on_message(topic_name, parser, sample)
                except Exception:
                    pass
            self._stop_event.wait(timeout=0.01)  # ~100Hz

    def _on_message(self, topic: str, parser: Callable, raw_msg: Any) -> None:
        try:
            data = parser(raw_msg)
        except Exception:
            return

        data["_ts"] = time.time()
        self._msg_count += 1

        with self._latest_lock:
            self._latest[topic] = data

        self._record_topic_msg(topic, data["_ts"])

        cbs = self._callbacks.get(topic)
        if cbs and self._loop is not None:
            for cb in cbs:
                if asyncio.iscoroutinefunction(cb):
                    self._loop.call_soon_threadsafe(
                        lambda c=cb, t=topic, d=data: asyncio.ensure_future(c(t, d)),
                    )
                else:
                    self._loop.call_soon_threadsafe(cb, topic, data)

    def on(self, topic: str, callback: Callable) -> None:
        self._callbacks.setdefault(topic, []).append(callback)

    def get_latest(self, topic: str) -> dict | None:
        with self._latest_lock:
            data = self._latest.get(topic)
            return dict(data) if data else None

    # Topics that askme is allowed to publish to. Reject anything else to
    # prevent accidental writes to sensor/safety topics owned by other processes.
    _PUBLISH_WHITELIST: set[str] = {
        "/thunder/cms_state",
        "/thunder/heartbeat",
    }

    def publish(self, topic: str, data: dict) -> None:
        if self._dp is None:
            return
        if topic not in self._PUBLISH_WHITELIST:
            logger.warning("Pulse: publish rejected — topic %r not in whitelist", topic)
            return
        if topic not in self._writers:
            msg_type = StringMsg
            t = Topic(self._dp, topic, msg_type)
            self._writers[topic] = DataWriter(self._dp, t)
        self._writers[topic].write(StringMsg(data=json.dumps(data, ensure_ascii=False)))

    def health(self) -> dict[str, Any]:
        h = {
            "status": "ok" if self.connected else ("disabled" if not self.available else "disconnected"),
            "available": self.available,
            "connected": self.connected,
            "msg_count": self._msg_count,
            "backend": "cyclonedds",
            "topics": self._build_topics_health(list(self._latest.keys())),
        }
        return h
