"""Skill model, execution, management and validation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "SkillDefinition": ("askme.skills.core.skill_model", "SkillDefinition"),
    "SkillExecutor": ("askme.skills.core.skill_executor", "SkillExecutor"),
    "SkillManager": ("askme.skills.core.skill_manager", "SkillManager"),
    "SkillValidationIssue": (
        "askme.skills.core.validation",
        "SkillValidationIssue",
    ),
    "SlotSpec": ("askme.skills.core.skill_model", "SlotSpec"),
    "validate_generated_skill": (
        "askme.skills.core.validation",
        "validate_generated_skill",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve skill core contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
