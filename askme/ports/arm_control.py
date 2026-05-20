"""Mechanical arm application port."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArmControlPort(Protocol):
    """Contract consumed by upper layers for direct arm operations."""

    async def execute(
        self,
        action_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a named arm action."""

    def emergency_stop(self) -> None:
        """Immediately stop arm motion."""

    def reset(self) -> None:
        """Clear the local arm emergency-stop state."""

    def get_state(self) -> dict[str, Any]:
        """Return the current arm state."""

    def is_connected(self) -> bool:
        """Return whether the arm transport is connected."""

    def close(self) -> None:
        """Release arm resources."""


__all__ = ["ArmControlPort"]
