"""Status LED application ports."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LedControllerPort(Protocol):
    """Minimal LED controller contract used by upper layers."""

    def set_state(self, state: Any) -> None:
        """Drive the status indicator to the given state."""


@runtime_checkable
class LedBridgePort(Protocol):
    """Background LED bridge contract owned by the provider layer."""

    async def run(self) -> None:
        """Run the LED state polling loop until cancelled."""


__all__ = ["LedBridgePort", "LedControllerPort"]
