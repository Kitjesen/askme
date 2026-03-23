"""PubSubBase — abstract pub/sub interface for all transport backends.

Any transport (DDS/rclpy, MQTT, in-memory mock) implements this ABC.
Convenience methods for common Thunder topics are provided in the base class.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)

# Default staleness thresholds (ms) per topic pattern.
# Event-driven topics (estop) have a long threshold; high-rate sensor topics shorter.
_DEFAULT_STALE_THRESHOLDS_MS: dict[str, float] = {
    "/thunder/estop": 60_000.0,
    "/thunder/heartbeat": 10_000.0,
    "/thunder/detections": 2_000.0,
    "/thunder/joint_states": 2_000.0,
    "/thunder/imu": 2_000.0,
    "/thunder/cms_state": 10_000.0,
}
_DEFAULT_STALE_MS = 5_000.0  # fallback for unknown topics

# Sliding window length for rate estimation (seconds).
_RATE_WINDOW_S = 10.0

# Message type names for well-known topics.
_TOPIC_MSG_TYPE_NAMES: dict[str, str] = {
    "/thunder/estop": "EstopState",
    "/thunder/detections": "DetectionFrame",
    "/thunder/joint_states": "JointStateSnapshot",
    "/thunder/imu": "ImuSnapshot",
    "/thunder/cms_state": "CmsState",
    "/thunder/heartbeat": "Bool",
}


class PubSubBase(ABC):
    """Abstract pub/sub interface -- all transport backends implement this."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport backend."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport backend and release resources."""

    @abstractmethod
    def on(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic with a callback."""

    @abstractmethod
    def get_latest(self, topic: str) -> dict | None:
        """Get most recent message for a topic (non-blocking, thread-safe)."""

    @abstractmethod
    def publish(self, topic: str, data: dict) -> None:
        """Publish a message to a topic."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the transport is currently connected and running."""

    # ── Topic tracking (subclasses populate via _record_topic_msg) ──

    def _init_topic_tracking(self) -> None:
        """Initialize per-topic tracking structures. Call from __init__."""
        self._topic_msg_count: dict[str, int] = {}
        self._topic_last_ts: dict[str, float] = {}
        # Sliding window: list of timestamps for msgs in last _RATE_WINDOW_S seconds
        self._topic_window_ts: dict[str, list[float]] = {}

    def _record_topic_msg(self, topic: str, ts: float) -> None:
        """Record a message arrival for per-topic freshness tracking."""
        self._topic_msg_count[topic] = self._topic_msg_count.get(topic, 0) + 1
        self._topic_last_ts[topic] = ts

        # Sliding window for rate calculation
        window = self._topic_window_ts.get(topic)
        if window is None:
            window = []
            self._topic_window_ts[topic] = window
        window.append(ts)
        # Prune entries older than window
        cutoff = ts - _RATE_WINDOW_S
        while window and window[0] < cutoff:
            window.pop(0)

    def _build_topic_info(self, topic: str, now: float) -> dict[str, Any]:
        """Build per-topic freshness dict."""
        last_ts = self._topic_last_ts.get(topic, 0.0)
        age_ms = (now - last_ts) * 1000.0 if last_ts else 0.0
        threshold = _DEFAULT_STALE_THRESHOLDS_MS.get(topic, _DEFAULT_STALE_MS)
        window = self._topic_window_ts.get(topic, [])
        # Rate: msgs in window / window duration (use _RATE_WINDOW_S)
        rate_hz = len(window) / _RATE_WINDOW_S if window else 0.0
        info: dict[str, Any] = {
            "last_ts": last_ts,
            "age_ms": round(age_ms, 1),
            "stale": age_ms > threshold if last_ts else True,
            "msg_count": self._topic_msg_count.get(topic, 0),
            "rate_hz": round(rate_hz, 1),
        }
        msg_type_name = _TOPIC_MSG_TYPE_NAMES.get(topic)
        if msg_type_name:
            info["last_message_type"] = msg_type_name
        return info

    def _build_topics_health(self, latest_keys: list[str]) -> dict[str, dict[str, Any]]:
        """Build the topics dict for health()."""
        now = time.time()
        return {topic: self._build_topic_info(topic, now) for topic in latest_keys}

    def health(self) -> dict[str, Any]:
        """Health snapshot for runtime introspection."""
        return {
            "status": "ok" if self.connected else "disconnected",
            "connected": self.connected,
            "topics": {},
        }

    # ── Convenience methods (use get_latest) ─────────

    def is_estop_active(self) -> bool:
        """Read cached ESTOP state. Returns False if no data."""
        estop = self.get_estop()
        if estop is None:
            return False
        return estop.active

    def get_detections(self) -> dict | None:
        """Get latest YOLO detections."""
        return self.get_latest("/thunder/detections")

    def get_joint_states(self) -> dict | None:
        """Get latest joint states."""
        return self.get_latest("/thunder/joint_states")

    def get_imu(self) -> dict | None:
        """Get latest IMU data."""
        return self.get_latest("/thunder/imu")

    # ── Typed convenience methods ────────────────────

    def get_estop(self) -> EstopState | None:
        """Get latest ESTOP state as a typed dataclass."""
        data = self.get_latest("/thunder/estop")
        if data is None:
            return None
        return EstopState.from_dict(data)

    def get_detection_frame(self) -> DetectionFrame | None:
        """Get latest detection frame as a typed dataclass."""
        data = self.get_latest("/thunder/detections")
        if data is None:
            return None
        return DetectionFrame.from_dict(data)

    def get_joints(self) -> JointStateSnapshot | None:
        """Get latest joint state as a typed dataclass."""
        data = self.get_latest("/thunder/joint_states")
        if data is None:
            return None
        return JointStateSnapshot.from_dict(data)

    def get_imu_snapshot(self) -> ImuSnapshot | None:
        """Get latest IMU reading as a typed dataclass."""
        data = self.get_latest("/thunder/imu")
        if data is None:
            return None
        return ImuSnapshot.from_dict(data)

    def get_cms_state(self) -> CmsState | None:
        """Get latest CMS state as a typed dataclass."""
        data = self.get_latest("/thunder/cms_state")
        if data is None:
            return None
        return CmsState.from_dict(data)
