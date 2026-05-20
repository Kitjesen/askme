"""Robot pulse bus and telemetry compatibility modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "MockPulse": ("askme.robot.telemetry.mock_pulse", "MockPulse"),
    "OTABridge": ("askme.telemetry.ota_bridge", "OTABridge"),
    "OTABridgeAuthError": ("askme.telemetry.ota_bridge", "OTABridgeAuthError"),
    "OTABridgeMetrics": ("askme.telemetry.ota_bridge", "OTABridgeMetrics"),
    "PubSubBase": ("askme.robot.telemetry.pubsub", "PubSubBase"),
    "Pulse": ("askme.robot.telemetry.pulse", "Pulse"),
    "get_ota_runtime_metrics": (
        "askme.telemetry.ota_bridge",
        "get_ota_runtime_metrics",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve robot telemetry contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
