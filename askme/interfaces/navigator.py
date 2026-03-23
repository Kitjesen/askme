"""Navigation backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from askme.runtime.registry import BackendRegistry


class NavigatorBackend(ABC):
    """Abstract navigator — goal to motion."""

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    async def navigate_to(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """Navigate to a position. Returns True on success."""

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel current navigation task."""

    @abstractmethod
    def is_navigating(self) -> bool:
        """Whether a navigation task is active."""

    async def follow_person(self, target_id: str) -> bool:
        """Follow a detected person. Override if supported."""
        return False

    async def start_mapping(self) -> bool:
        """Start SLAM mapping. Override if supported."""
        return False


navigator_registry = BackendRegistry("navigator", NavigatorBackend, default="lingtu")
