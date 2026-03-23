"""PubSubBase — abstract pub/sub interface for all transport backends.

Any transport (DDS/rclpy, MQTT, in-memory mock) implements this ABC.
Convenience methods for common Thunder topics are provided in the base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)


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

    def health(self) -> dict[str, Any]:
        """Health snapshot for runtime introspection."""
        return {
            "status": "ok" if self.connected else "disconnected",
            "connected": self.connected,
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
