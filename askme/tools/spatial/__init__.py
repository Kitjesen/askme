"""Spatial, vision, scan, route, and temporal query tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "FindTargetTool": ("askme.tools.spatial.vision_tool", "FindTargetTool"),
    "LookAroundTool": ("askme.tools.spatial.vision_tool", "LookAroundTool"),
    "ScanAroundTool": ("askme.tools.spatial.scan_tool", "ScanAroundTool"),
    "SpaceLookupPlaceTool": (
        "askme.tools.spatial.space_tool",
        "SpaceLookupPlaceTool",
    ),
    "SpaceRecommendRouteTool": (
        "askme.tools.spatial.space_tool",
        "SpaceRecommendRouteTool",
    ),
    "TemporalQueryTool": (
        "askme.tools.spatial.temporal_query_tool",
        "TemporalQueryTool",
    ),
    "register_scan_tools": (
        "askme.tools.spatial.scan_tool",
        "register_scan_tools",
    ),
    "register_temporal_tools": (
        "askme.tools.spatial.temporal_query_tool",
        "register_temporal_tools",
    ),
    "register_vision_tools": (
        "askme.tools.spatial.vision_tool",
        "register_vision_tools",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve spatial tools on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
