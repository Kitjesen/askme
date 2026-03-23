"""MockPulse — pure in-memory pub/sub for unit tests.

No rclpy, no threads, no I/O.  Dict-based storage with synchronous callbacks.
"""

from __future__ import annotations

from typing import Any, Callable

from askme.robot.pubsub import PubSubBase


class MockPulse(PubSubBase):
    """Pure in-memory pub/sub -- for unit tests, no rclpy needed."""

    def __init__(self) -> None:
        self._latest: dict[str, dict] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._started = False

    @property
    def connected(self) -> bool:
        return self._started

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def on(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic with a callback."""
        self._callbacks.setdefault(topic, []).append(callback)

    def get_latest(self, topic: str) -> dict | None:
        """Get most recent message for a topic."""
        data = self._latest.get(topic)
        return dict(data) if data else None

    def publish(self, topic: str, data: dict) -> None:
        """Publish a message: store in cache and fire callbacks synchronously."""
        self._latest[topic] = data
        for cb in self._callbacks.get(topic, []):
            cb(topic, data)

    def health(self) -> dict[str, Any]:
        """Health snapshot."""
        return {
            "status": "ok" if self.connected else "disconnected",
            "connected": self.connected,
            "topics": list(self._latest.keys()),
        }
