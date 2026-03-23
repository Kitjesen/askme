"""Data bus backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from askme.runtime.registry import BackendRegistry


class BusBackend(ABC):
    """Abstract data bus — subscribe/publish typed topics.

    Implementations: Pulse (rclpy DDS), MockPulse (memory), future ZeroMQ/LCM.
    """

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    async def start(self) -> None:
        """Start the bus connection."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the bus and release resources."""

    @abstractmethod
    def on(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic with a callback."""

    @abstractmethod
    def get_latest(self, topic: str) -> dict | None:
        """Get most recent message for a topic (non-blocking)."""

    @abstractmethod
    def publish(self, topic: str, data: dict) -> None:
        """Publish a message to a topic."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the bus is connected."""

    def health(self) -> dict[str, Any]:
        """Health snapshot."""
        return {"status": "ok" if self.connected else "disconnected"}


bus_registry = BackendRegistry("bus", BusBackend, default="pulse")
