"""Runtime telemetry and OTA integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "OTABridge": ("askme.telemetry.ota_bridge", "OTABridge"),
    "OTABridgeAuthError": ("askme.telemetry.ota_bridge", "OTABridgeAuthError"),
    "OTABridgeMetrics": ("askme.telemetry.ota_bridge", "OTABridgeMetrics"),
    "get_ota_runtime_metrics": (
        "askme.telemetry.ota_bridge",
        "get_ota_runtime_metrics",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
