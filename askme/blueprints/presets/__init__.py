"""Concrete product runtime blueprint presets.

Import concrete runtime objects from their owning submodule, for example::

    from askme.blueprints.presets.edge_robot import edge_robot

The package also exposes non-conflicting ``*_runtime`` convenience exports for
callers that want lazy runtime objects without shadowing normal submodule
imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_RUNTIME_EXPORTS = {
    "edge_robot_runtime": ("askme.blueprints.presets.edge_robot", "edge_robot"),
    "lingtu_voice_runtime": ("askme.blueprints.presets.lingtu_voice", "lingtu_voice"),
    "mcp_runtime": ("askme.blueprints.presets.mcp", "mcp"),
    "text_runtime": ("askme.blueprints.presets.text", "text"),
    "voice_runtime": ("askme.blueprints.presets.voice", "voice"),
    "voice_perception_runtime": (
        "askme.blueprints.presets.voice_perception",
        "voice_perception",
    ),
}
_LAZY_EXPORTS = _RUNTIME_EXPORTS

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
