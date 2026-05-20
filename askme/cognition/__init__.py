"""Cognitive adapter primitives for robot-aware interaction."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ActivePerceptionRequest": (
        "askme.cognition.perception",
        "ActivePerceptionRequest",
    ),
    "ActivePerceptionResolver": (
        "askme.cognition.perception",
        "ActivePerceptionResolver",
    ),
    "CognitivePlan": ("askme.cognition.planning", "CognitivePlan"),
    "CognitionPerceptionSync": (
        "askme.cognition.perception",
        "CognitionPerceptionSync",
    ),
    "normalize_scene_snapshot": (
        "askme.cognition.perception",
        "normalize_scene_snapshot",
    ),
    "CognitivePlanner": ("askme.cognition.planning", "CognitivePlanner"),
    "PlanningSession": ("askme.cognition.planning", "PlanningSession"),
    "WorkingMemory": ("askme.cognition.memory", "WorkingMemory"),
    "WorkingMemoryItem": ("askme.cognition.memory", "WorkingMemoryItem"),
    "WorldFact": ("askme.cognition.world", "WorldFact"),
    "WorldStateService": ("askme.cognition.world", "WorldStateService"),
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
