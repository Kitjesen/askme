"""Navigation application port."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NavigationPort(Protocol):
    """Minimal navigation contract consumed by upper layers."""

    def is_configured(self) -> bool:
        """Return whether the downstream navigation service is configured."""

    def dispatch_navigation(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
        *,
        mission_type: str = "voice_command",
        mission_id: str = "",
    ) -> dict[str, Any]:
        """Dispatch a navigation capability through the navigation plane."""

    def status(self) -> dict[str, Any]:
        """Return current navigation status."""


__all__ = ["NavigationPort"]
