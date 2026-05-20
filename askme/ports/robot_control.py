"""Robot control application port."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotControlPort(Protocol):
    """Minimal motion-control contract consumed by upper layers.

    Concrete hardware clients such as ``DogControlClient`` can satisfy this
    structurally without becoming a required dependency of the voice gateway or
    interaction layers.
    """

    def is_configured(self) -> bool:
        """Return whether the downstream control service is configured."""

    def dispatch_capability(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a named robot capability through the control plane."""


__all__ = ["RobotControlPort"]
