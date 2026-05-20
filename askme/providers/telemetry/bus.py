"""Data-bus provider adapters."""

from __future__ import annotations

from typing import Any

from askme.interfaces.bus import BusBackend


def build_bus(config: dict[str, Any] | None = None) -> BusBackend:
    """Build the configured robot telemetry bus implementation."""

    cfg = dict(config or {})
    backend = str(cfg.get("backend", "pulse")).lower()
    if backend == "mock":
        from askme.robot.telemetry.mock_pulse import MockPulse

        return MockPulse(cfg)
    if backend == "pulse":
        from askme.robot.telemetry.pulse import Pulse

        return Pulse(cfg)
    raise ValueError(f"Unsupported bus backend: {backend!r}")


__all__ = ["build_bus"]
