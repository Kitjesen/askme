"""Memory-derived intelligence and analysis modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AssociationGraph": ("askme.memory.intelligence.association", "AssociationGraph"),
    "ExtractionAdapter": (
        "askme.memory.intelligence.extraction_adapter",
        "ExtractionAdapter",
    ),
    "StrategyGenerator": ("askme.memory.intelligence.strategy", "StrategyGenerator"),
    "Suggestion": ("askme.memory.intelligence.strategy", "Suggestion"),
    "Trend": ("askme.memory.intelligence.trend_analyzer", "Trend"),
    "TrendAnalyzer": ("askme.memory.intelligence.trend_analyzer", "TrendAnalyzer"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public memory intelligence contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
