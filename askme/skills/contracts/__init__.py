"""Typed skill contracts and field capability contract routes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "SkillContract": ("askme.skills.contracts.contracts", "SkillContract"),
    "SkillContractRegistry": (
        "askme.skills.contracts.contracts",
        "SkillContractRegistry",
    ),
    "SkillParameter": ("askme.skills.contracts.contracts", "SkillParameter"),
    "build_skills_openapi": (
        "askme.skills.contracts.contracts",
        "build_skills_openapi",
    ),
    "field_capability_route": (
        "askme.skills.contracts.field_capability_contracts",
        "field_capability_route",
    ),
    "field_capability_routes": (
        "askme.skills.contracts.field_capability_contracts",
        "field_capability_routes",
    ),
    "normalize_capability_package": (
        "askme.skills.contracts.field_capability_contracts",
        "normalize_capability_package",
    ),
    "register_skill_contract": (
        "askme.skills.contracts.contracts",
        "register_skill_contract",
    ),
    "registered_skill_contracts": (
        "askme.skills.contracts.contracts",
        "registered_skill_contracts",
    ),
    "skill_contract": ("askme.skills.contracts.contracts", "skill_contract"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public skill contract helpers on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
