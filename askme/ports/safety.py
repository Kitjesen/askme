"""Robot safety application port."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SafetyPort(Protocol):
    """Minimal safety-state contract consumed above hardware clients."""

    def is_configured(self) -> bool:
        """Return whether the downstream safety service is configured."""

    def is_estop_active(self) -> bool:
        """Return whether emergency stop is currently active."""

    def query_estop_state(self) -> dict[str, Any] | None:
        """Return the latest structured estop state when available."""

    def notify_estop(self) -> None:
        """Notify the downstream safety service about an emergency stop."""


__all__ = ["SafetyPort"]
