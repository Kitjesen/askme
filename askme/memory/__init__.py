"""Memory package.

Owner subpackages:
- ``core``: runtime memory services and memory types.
- ``retrieval``: knowledge catalog, RAG bridge, vector and semantic indexes.
- ``backends``: optional external memory backends.
- ``intelligence``: trends, associations, extraction and suggestions.

Historical imports such as ``askme.memory.bridge`` are kept as compatibility
aliases. New code should import from the owning subpackage.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.memory.admission": "askme.memory.core.admission",
    "askme.memory.association": "askme.memory.intelligence.association",
    "askme.memory.bridge": "askme.memory.retrieval.bridge",
    "askme.memory.catalog": "askme.memory.retrieval.catalog",
    "askme.memory.conversation": "askme.memory.core.conversation",
    "askme.memory.episode": "askme.memory.core.episode",
    "askme.memory.episodic_memory": "askme.memory.core.episodic_memory",
    "askme.memory.extraction_adapter": "askme.memory.intelligence.extraction_adapter",
    "askme.memory.importer": "askme.memory.retrieval.importer",
    "askme.memory.index_jobs": "askme.memory.retrieval.index_jobs",
    "askme.memory.map_adapter": "askme.memory.retrieval.map_adapter",
    "askme.memory.mempalace_backend": "askme.memory.backends.mempalace_backend",
    "askme.memory.policies": "askme.memory.core.policies",
    "askme.memory.procedural": "askme.memory.core.procedural",
    "askme.memory.robotmem_backend": "askme.memory.backends.robotmem_backend",
    "askme.memory.semantic_index": "askme.memory.retrieval.semantic_index",
    "askme.memory.service": "askme.memory.core.service",
    "askme.memory.session": "askme.memory.core.session",
    "askme.memory.site_knowledge": "askme.memory.retrieval.site_knowledge",
    "askme.memory.strategy": "askme.memory.intelligence.strategy",
    "askme.memory.system": "askme.memory.core.system",
    "askme.memory.taxonomy": "askme.memory.retrieval.taxonomy",
    "askme.memory.trend_analyzer": "askme.memory.intelligence.trend_analyzer",
    "askme.memory.vector_store": "askme.memory.retrieval.vector_store",
}

_LAZY_EXPORTS = {
    "AssociationGraph": ("askme.memory.intelligence.association", "AssociationGraph"),
    "ConversationManager": ("askme.memory.core.conversation", "ConversationManager"),
    "Episode": ("askme.memory.core.episode", "Episode"),
    "EpisodicMemory": ("askme.memory.core.episodic_memory", "EpisodicMemory"),
    "MemoryBridge": ("askme.memory.retrieval.bridge", "MemoryBridge"),
    "MemPalaceBackend": ("askme.memory.backends.mempalace_backend", "MemPalaceBackend"),
    "RobotMemBackend": ("askme.memory.backends.robotmem_backend", "RobotMemBackend"),
    "SessionMemory": ("askme.memory.core.session", "SessionMemory"),
    "StrategyGenerator": ("askme.memory.intelligence.strategy", "StrategyGenerator"),
    "Suggestion": ("askme.memory.intelligence.strategy", "Suggestion"),
    "MemorySystem": ("askme.memory.core.system", "MemorySystem"),
    "Trend": ("askme.memory.intelligence.trend_analyzer", "Trend"),
    "TrendAnalyzer": ("askme.memory.intelligence.trend_analyzer", "TrendAnalyzer"),
    "VectorStore": ("askme.memory.retrieval.vector_store", "VectorStore"),
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
