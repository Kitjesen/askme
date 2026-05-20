"""Skill dispatch, planning, and gatekeeping pipeline modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MissionContext": ("askme.pipeline.skills.skill_dispatcher", "MissionContext"),
    "MissionState": ("askme.pipeline.skills.skill_dispatcher", "MissionState"),
    "MissionStep": ("askme.pipeline.skills.skill_dispatcher", "MissionStep"),
    "PlanStep": ("askme.pipeline.skills.planner_agent", "PlanStep"),
    "PlannerAgent": ("askme.pipeline.skills.planner_agent", "PlannerAgent"),
    "SkillDispatcher": ("askme.pipeline.skills.skill_dispatcher", "SkillDispatcher"),
    "SkillGate": ("askme.pipeline.skills.skill_gate", "SkillGate"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
