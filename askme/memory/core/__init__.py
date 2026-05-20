"""Core runtime memory services and memory types.

Import from this package when product code needs memory state, episode,
session, admission, procedure, policy, or service contracts. Exports are lazy
so importing the package does not initialize storage or external adapters.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AdmissionScore": ("askme.memory.core.admission", "AdmissionScore"),
    "ConversationManager": ("askme.memory.core.conversation", "ConversationManager"),
    "Episode": ("askme.memory.core.episode", "Episode"),
    "EpisodicMemory": ("askme.memory.core.episodic_memory", "EpisodicMemory"),
    "MemoryAdmissionControl": (
        "askme.memory.core.admission",
        "MemoryAdmissionControl",
    ),
    "MemoryService": ("askme.memory.core.service", "MemoryService"),
    "MemorySystem": ("askme.memory.core.system", "MemorySystem"),
    "PolicyStore": ("askme.memory.core.policies", "PolicyStore"),
    "Procedure": ("askme.memory.core.procedural", "Procedure"),
    "ProceduralMemory": ("askme.memory.core.procedural", "ProceduralMemory"),
    "SessionMemory": ("askme.memory.core.session", "SessionMemory"),
    "classify_event_type": ("askme.memory.core.episodic_memory", "classify_event_type"),
    "get_memory_service": ("askme.memory.core.service", "get_memory_service"),
    "recency_boost": ("askme.memory.core.episode", "recency_boost"),
    "reset_memory_service": ("askme.memory.core.service", "reset_memory_service"),
    "score_importance": ("askme.memory.core.episode", "score_importance"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public memory core contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
