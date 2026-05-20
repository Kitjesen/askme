"""Skill audit, approval, package and growth governance."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "APPROVED": ("askme.skills.governance.governance", "APPROVED"),
    "DISABLED": ("askme.skills.governance.governance", "DISABLED"),
    "DEFAULT_PACKAGE_ID": ("askme.skills.governance.packages", "DEFAULT_PACKAGE_ID"),
    "PENDING": ("askme.skills.governance.governance", "PENDING"),
    "REJECTED": ("askme.skills.governance.governance", "REJECTED"),
    "SkillAuditLog": ("askme.skills.governance.audit", "SkillAuditLog"),
    "SkillGovernanceRecord": (
        "askme.skills.governance.governance",
        "SkillGovernanceRecord",
    ),
    "SkillGovernanceStore": (
        "askme.skills.governance.governance",
        "SkillGovernanceStore",
    ),
    "SkillGrowthBacklog": (
        "askme.skills.governance.growth_backlog",
        "SkillGrowthBacklog",
    ),
    "SkillGrowthCandidate": (
        "askme.skills.governance.growth_backlog",
        "SkillGrowthCandidate",
    ),
    "SkillPackageRecord": (
        "askme.skills.governance.packages",
        "SkillPackageRecord",
    ),
    "SkillPackageStore": ("askme.skills.governance.packages", "SkillPackageStore"),
    "default_skill_audit_path": (
        "askme.skills.governance.audit",
        "default_skill_audit_path",
    ),
    "default_skill_growth_state_path": (
        "askme.skills.governance.growth_backlog",
        "default_skill_growth_state_path",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve skill governance contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
