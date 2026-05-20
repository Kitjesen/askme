"""Knowledge retrieval, catalog and index modules.

This package is the stable entrypoint for RAG catalog, import, semantic index,
vector store, site knowledge, and map adapter contracts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ImportResult": ("askme.memory.retrieval.importer", "ImportResult"),
    "KnowledgeCatalog": ("askme.memory.retrieval.catalog", "KnowledgeCatalog"),
    "KnowledgeIndexJobStore": (
        "askme.memory.retrieval.index_jobs",
        "KnowledgeIndexJobStore",
    ),
    "KnowledgeRecord": ("askme.memory.retrieval.importer", "KnowledgeRecord"),
    "knowledge_category_metadata": (
        "askme.memory.retrieval.taxonomy",
        "knowledge_category_metadata",
    ),
    "knowledge_category_taxonomy_payload": (
        "askme.memory.retrieval.taxonomy",
        "knowledge_category_taxonomy_payload",
    ),
    "Location": ("askme.memory.retrieval.site_knowledge", "Location"),
    "MapAdapter": ("askme.memory.retrieval.map_adapter", "MapAdapter"),
    "MemoryBridge": ("askme.memory.retrieval.bridge", "MemoryBridge"),
    "SemanticIndex": ("askme.memory.retrieval.semantic_index", "SemanticIndex"),
    "SiteKnowledge": ("askme.memory.retrieval.site_knowledge", "SiteKnowledge"),
    "SpatialEvent": ("askme.memory.retrieval.site_knowledge", "SpatialEvent"),
    "VectorStore": ("askme.memory.retrieval.vector_store", "VectorStore"),
    "parse_knowledge_file": (
        "askme.memory.retrieval.importer",
        "parse_knowledge_file",
    ),
    "parse_knowledge_text": (
        "askme.memory.retrieval.importer",
        "parse_knowledge_text",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public retrieval contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
