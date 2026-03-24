"""Association graph — vector-similarity-based memory association.

Uses VectorStore semantic search as the association mechanism.
No graph database needed: "find similar situations" = vector similarity.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AssociationGraph:
    """Find historically similar events and entity-related memories via vector search."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    @property
    def available(self) -> bool:
        """Whether the underlying vector store is usable."""
        return self._store.available

    def find_similar_situations(self, description: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Find historically similar events via vector search.

        Returns a list of dicts with keys: text, score, metadata.
        """
        if not description.strip():
            return []
        return self._store.search(description, top_k=top_k)

    def find_related_to_entity(self, entity: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find memories related to a specific entity (person, location, object).

        Returns a list of dicts with keys: text, score, metadata.
        """
        if not entity.strip():
            return []
        return self._store.search(entity, top_k=top_k)

    def get_associations_text(self, description: str, top_k: int = 3) -> str:
        """Return formatted Chinese text of similar historical situations.

        Suitable for direct injection into LLM prompts.
        Returns empty string when no associations found.
        """
        results = self.find_similar_situations(description, top_k=top_k)
        if not results:
            return ""
        lines: list[str] = []
        for r in results:
            score = r.get("score", 0.0)
            text = r.get("text", "")
            if score > 0.3:  # filter low-relevance noise
                lines.append(f"- (相关度{score:.0%}) {text[:120]}")
        return "\n".join(lines)
