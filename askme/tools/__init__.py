"""Tool package.

The root package preserves historical imports such as
``askme.tools.tool_registry``. New code should import from the owner
subpackage, for example ``askme.tools.core.tool_registry`` or
``askme.tools.robot.move_tool``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.tools.builtin_tools": "askme.tools.core.builtin_tools",
    "askme.tools.execution_control": "askme.tools.core.execution_control",
    "askme.tools.field_event_tool": "askme.tools.field.field_event_tool",
    "askme.tools.move_tool": "askme.tools.robot.move_tool",
    "askme.tools.robot_api_tool": "askme.tools.robot.robot_api_tool",
    "askme.tools.robot_tools": "askme.tools.robot.robot_tools",
    "askme.tools.scan_tool": "askme.tools.spatial.scan_tool",
    "askme.tools.skill_tools": "askme.tools.skills.skill_tools",
    "askme.tools.space_tool": "askme.tools.spatial.space_tool",
    "askme.tools.temporal_query_tool": "askme.tools.spatial.temporal_query_tool",
    "askme.tools.tool_registry": "askme.tools.core.tool_registry",
    "askme.tools.vision_tool": "askme.tools.spatial.vision_tool",
    "askme.tools.voice_tools": "askme.tools.voice.voice_tools",
}

_LAZY_EXPORTS = {
    "BaseTool": ("askme.tools.core.tool_registry", "BaseTool"),
    "ToolRegistry": ("askme.tools.core.tool_registry", "ToolRegistry"),
    "register_builtin_tools": ("askme.tools.core.builtin_tools", "register_builtin_tools"),
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
