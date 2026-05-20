"""Skill package.

Owner subpackages:
- ``core``: skill model, manager, executor and validation.
- ``contracts``: typed skill contracts and field capability routes.
- ``governance``: audit log, generated-skill review and package policy.
- ``catalog``: customer-visible capability center projections.
- ``builtin``: built-in customer and robot skill definitions.

Historical imports such as ``askme.skills.skill_manager`` remain available for
compatibility. New code should import from the owner subpackage.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.skills.audit": "askme.skills.governance.audit",
    "askme.skills.capability_center": "askme.skills.catalog.capability_center",
    "askme.skills.contracts_builtin": "askme.skills.contracts.contracts_builtin",
    "askme.skills.field_capability_contracts": "askme.skills.contracts.field_capability_contracts",
    "askme.skills.growth_backlog": "askme.skills.governance.growth_backlog",
    "askme.skills.packages": "askme.skills.governance.packages",
    "askme.skills.skill_executor": "askme.skills.core.skill_executor",
    "askme.skills.skill_manager": "askme.skills.core.skill_manager",
    "askme.skills.skill_model": "askme.skills.core.skill_model",
    "askme.skills.validation": "askme.skills.core.validation",
}

_LAZY_EXPORTS = {
    "SkillDefinition": ("askme.skills.core.skill_model", "SkillDefinition"),
    "SkillExecutor": ("askme.skills.core.skill_executor", "SkillExecutor"),
    "SkillManager": ("askme.skills.core.skill_manager", "SkillManager"),
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
