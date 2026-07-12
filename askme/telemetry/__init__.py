"""Runtime telemetry, structured logging, tracing, and OTA integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    # OTA bridge
    "OTABridge": ("askme.telemetry.ota_bridge", "OTABridge"),
    "OTABridgeAuthError": ("askme.telemetry.ota_bridge", "OTABridgeAuthError"),
    "OTABridgeMetrics": ("askme.telemetry.ota_bridge", "OTABridgeMetrics"),
    "get_ota_runtime_metrics": (
        "askme.telemetry.ota_bridge",
        "get_ota_runtime_metrics",
    ),
    # Structured logging
    "JsonFormatter": ("askme.telemetry.logging", "JsonFormatter"),
    "setup_structured_logging": (
        "askme.telemetry.logging",
        "setup_structured_logging",
    ),
    "generate_trace_id": ("askme.telemetry.logging", "generate_trace_id"),
    # Tracing
    "TraceContext": ("askme.telemetry.tracing", "TraceContext"),
    "get_trace": ("askme.telemetry.tracing", "get_trace"),
    "set_trace": ("askme.telemetry.tracing", "set_trace"),
    "trace_id_var": ("askme.telemetry.tracing", "trace_id_var"),
    # Metrics
    "METRICS": ("askme.telemetry.metrics", "METRICS"),
    "increment_counter": ("askme.telemetry.metrics", "increment_counter"),
    "record_metric": ("askme.telemetry.metrics", "record_metric"),
    "reset_metrics": ("askme.telemetry.metrics", "reset_metrics"),
    "snapshot_metrics": ("askme.telemetry.metrics", "snapshot_metrics"),
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
