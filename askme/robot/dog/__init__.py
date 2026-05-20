"""Thunder dog runtime clients, commands and health modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DogControlClient": ("askme.robot.dog.control_client", "DogControlClient"),
    "DogSafetyClient": ("askme.robot.dog.safety_client", "DogSafetyClient"),
    "RuntimeHealthSnapshot": (
        "askme.robot.dog.runtime_health",
        "RuntimeHealthSnapshot",
    ),
    "get_service_summary": ("askme.robot.dog.runtime_health", "get_service_summary"),
    "log_startup_service_status": (
        "askme.robot.dog.runtime_health",
        "log_startup_service_status",
    ),
    "merge_voice_pipeline_status": (
        "askme.robot.dog.runtime_health",
        "merge_voice_pipeline_status",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve Thunder dog runtime contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
