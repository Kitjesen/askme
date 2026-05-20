"""Spatial/temporal memory application ports."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TemporalMemoryPort(Protocol):
    """Query contract for time-indexed robot scene memory."""

    def is_configured(self) -> bool:
        """Return whether the downstream memory service is configured."""

    def query_temporal_observations(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return temporal observations matching the provided query params."""


__all__ = ["TemporalMemoryPort"]
