"""Human-robot interaction gate and perception context package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "InteractionAction": ("askme.robot_interaction.interaction_gate", "InteractionAction"),
    "InteractionDecision": ("askme.robot_interaction.interaction_gate", "InteractionDecision"),
    "InteractionGate": ("askme.robot_interaction.interaction_gate", "InteractionGate"),
    "InteractionPerceptionSnapshot": (
        "askme.robot_interaction.perception_context",
        "InteractionPerceptionSnapshot",
    ),
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
