"""Reaction engines and proactive alert agents."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "HybridReaction": ("askme.pipeline.reactions.reaction_engine", "HybridReaction"),
    "LLMReaction": ("askme.pipeline.reactions.reaction_engine", "LLMReaction"),
    "ProactiveAgent": ("askme.pipeline.reactions.proactive_agent", "ProactiveAgent"),
    "RuleBasedReaction": (
        "askme.pipeline.reactions.reaction_engine",
        "RuleBasedReaction",
    ),
    "StateLedBridge": ("askme.pipeline.reactions.state_led_bridge", "StateLedBridge"),
    "evaluate_rules": ("askme.pipeline.reactions.reaction_engine", "evaluate_rules"),
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
