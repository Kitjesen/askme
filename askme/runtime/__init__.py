"""Runtime package.

Owner subpackages:
- ``core``: runtime module graph, profiles and backend registry.
- ``task``: TaskHandoff, arbiter clients, mission service and runtime audit.
- ``modules``: runtime modules grouped around concrete product capabilities.
- ``diagnostics``: runtime diagnostic smoke checks for dialogue and memory retrieval.

Historical imports such as ``askme.runtime.module`` remain available for
compatibility. New code should import from the owner subpackage.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.runtime.arbiter_client": "askme.runtime.task.arbiter_client",
    "askme.runtime.audit": "askme.runtime.task.audit",
    "askme.runtime.field_callbacks": "askme.runtime.task.field_callbacks",
    "askme.runtime.handoff": "askme.runtime.task.handoff",
    "askme.runtime.mission": "askme.runtime.task.mission",
    "askme.runtime.module": "askme.runtime.core.module",
    "askme.runtime.profiles": "askme.runtime.core.profiles",
    "askme.runtime.registry": "askme.interfaces.core.registry",
}

_LAZY_EXPORTS = {
    "EDGE_ROBOT_PROFILE": ("askme.runtime.core.profiles", "EDGE_ROBOT_PROFILE"),
    "MCP_PROFILE": ("askme.runtime.core.profiles", "MCP_PROFILE"),
    "RuntimeProfile": ("askme.runtime.core.profiles", "RuntimeProfile"),
    "TEXT_PROFILE": ("askme.runtime.core.profiles", "TEXT_PROFILE"),
    "VOICE_PROFILE": ("askme.runtime.core.profiles", "VOICE_PROFILE"),
    "legacy_profile_for": ("askme.runtime.core.profiles", "legacy_profile_for"),
}

__all__ = sorted(_LAZY_EXPORTS)


install_legacy_aliases(__name__, _LEGACY_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    legacy_module = _LEGACY_MODULE_ALIASES.get(f"{__name__}.{name}")
    if legacy_module:
        value = import_module(legacy_module)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
