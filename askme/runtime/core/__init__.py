"""Runtime graph, profiles and backend registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "Alias": ("askme.runtime.core.module", "Alias"),
    "AmbiguousPortError": ("askme.runtime.core.module", "AmbiguousPortError"),
    "BackendRegistry": ("askme.interfaces.core.registry", "BackendRegistry"),
    "EDGE_ROBOT_PROFILE": ("askme.runtime.core.profiles", "EDGE_ROBOT_PROFILE"),
    "In": ("askme.runtime.core.module", "In"),
    "MCP_PROFILE": ("askme.runtime.core.profiles", "MCP_PROFILE"),
    "Module": ("askme.runtime.core.module", "Module"),
    "ModuleRegistry": ("askme.runtime.core.module", "ModuleRegistry"),
    "Out": ("askme.runtime.core.module", "Out"),
    "PortInfo": ("askme.runtime.core.module", "PortInfo"),
    "Required": ("askme.runtime.core.module", "Required"),
    "Runtime": ("askme.runtime.core.module", "Runtime"),
    "RuntimeApp": ("askme.runtime.core.module", "RuntimeApp"),
    "RuntimeProfile": ("askme.runtime.core.profiles", "RuntimeProfile"),
    "TEXT_PROFILE": ("askme.runtime.core.profiles", "TEXT_PROFILE"),
    "VOICE_PROFILE": ("askme.runtime.core.profiles", "VOICE_PROFILE"),
    "WireResult": ("askme.runtime.core.module", "WireResult"),
    "legacy_profile_for": ("askme.runtime.core.profiles", "legacy_profile_for"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve runtime core contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
