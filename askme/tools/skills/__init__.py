"""Skill-management tools exposed to agents."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CreateAgentProfileTool": (
        "askme.tools.skills.skill_tools",
        "CreateAgentProfileTool",
    ),
    "CreateSkillTool": ("askme.tools.skills.skill_tools", "CreateSkillTool"),
    "register_skill_tools": (
        "askme.tools.skills.skill_tools",
        "register_skill_tools",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve skill-management tools on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
