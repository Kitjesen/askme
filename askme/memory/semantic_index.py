"""L5 Semantic Index — unified semantic search across all memory layers.

Indexes content from L2 (session digests), L3 (episodic knowledge),
and L4 (conversation history) into a single RobotMem collection with
source tagging.  A single ``search(query)`` returns ranked results
with layer attribution.

Usage::

    from askme.memory.semantic_index import SemanticIndex

    idx = SemanticIndex(mem_cfg)
    await idx.sync(episodic, session)   # re-index L2+L3 content
    results = await idx.search("温度异常")
    # [{"text": "...", "source": "knowledge", "score": 0.95}, ...]
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.session import SessionMemory

logger = logging.getLogger(__name__)


class SemanticIndex:
    """Unified semantic search across L2/L3/L4 memory layers.

    Backed by a dedicated RobotMem collection (``askme_semantic``).
    Content from each layer is tagged with ``source`` in context so
    results can be attributed back to their origin.
    """

    def __init__(self, mem_cfg: dict[str, Any] | None = None) -> None:
        self._mem_cfg = mem_cfg or {}
        self._rm: Any = None
        self._rm_failed: bool = False
        self._indexed_hashes: set[str] = set()

    def _ensure_rm(self) -> bool:
        """Lazy-init RobotMem for the semantic index collection."""
        if self._rm is not None:
            return True
        if self._rm_failed:
            return False
        try:
            from robotmem.sdk import RobotMemory

            self._rm = RobotMemory(
                collection="askme_semantic",
                embed_backend="onnx",
            )
            logger.info("[SemanticIndex] Initialised.")
            return True
        except Exception as e:
            logger.warning("[SemanticIndex] Init failed: %s", e)
            self._rm_failed = True
            return False

    @property
    def available(self) -> bool:
        return self._rm is not None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _content_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    async def _index_text(self, text: str, source: str, category: str = "") -> bool:
        """Index a single text entry (skip if already indexed)."""
        if not text.strip():
            return False
        h = self._content_hash(text)
        if h in self._indexed_hashes:
            return False
        if not self._ensure_rm():
            return False
        try:
            context = {"source": source}
            if category:
                context["category"] = category
            await asyncio.to_thread(self._rm.learn, text, context=context)
            self._indexed_hashes.add(h)
            return True
        except Exception as e:
            logger.debug("[SemanticIndex] Index failed: %s", e)
            return False

    async def index_knowledge(self, knowledge_dir: Path) -> int:
        """Index L3 episodic knowledge files (knowledge/*.md)."""
        if not knowledge_dir.is_dir():
            return 0
        indexed = 0
        for md_file in sorted(knowledge_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("- "):
                        fact = line[2:].strip()
                        if await self._index_text(
                            fact,
                            source="knowledge",
                            category=md_file.stem,
                        ):
                            indexed += 1
            except Exception as e:
                logger.debug("[SemanticIndex] Read %s failed: %s", md_file.name, e)
        if indexed:
            logger.info("[SemanticIndex] Indexed %d knowledge facts.", indexed)
        return indexed

    async def index_digests(self, digests_dir: Path) -> int:
        """Index L2/L3 session digests (digests/*.md)."""
        if not digests_dir.is_dir():
            return 0
        indexed = 0
        for md_file in sorted(digests_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content and await self._index_text(
                    content[:500],
                    source="digest",
                    category=md_file.stem,
                ):
                    indexed += 1
            except Exception as e:
                logger.debug("[SemanticIndex] Read %s failed: %s", md_file.name, e)
        if indexed:
            logger.info("[SemanticIndex] Indexed %d digests.", indexed)
        return indexed

    async def sync(
        self,
        episodic: EpisodicMemory | None = None,
        session: SessionMemory | None = None,
    ) -> int:
        """Re-index L2+L3 content into the semantic store.

        Idempotent — skips content already indexed (hash-based dedup).
        """
        total = 0
        if episodic:
            total += await self.index_knowledge(episodic._knowledge_dir)
            total += await self.index_digests(episodic._digests_dir)
        if session:
            session_dir = getattr(session, "_session_dir", None)
            if session_dir and Path(session_dir).is_dir():
                total += await self.index_digests(Path(session_dir))
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        n: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Unified semantic search across all indexed memory layers.

        Args:
            query: Search query text.
            n: Max results to return.
            source_filter: Optional — only return results from this source
                (``"knowledge"``, ``"digest"``, ``"conversation"``).

        Returns:
            List of dicts with ``text``, ``source``, ``category``, ``score``.
        """
        if not self._ensure_rm():
            return []
        try:
            raw = await asyncio.to_thread(
                self._rm.recall, query, n=n, min_confidence=0.0,
            )
            if not raw:
                return []

            results = []
            for m in raw:
                content = m.get("content", "")
                if not content:
                    continue
                ctx_str = m.get("context", "{}")
                if isinstance(ctx_str, str):
                    import json
                    try:
                        ctx = json.loads(ctx_str)
                    except (json.JSONDecodeError, ValueError):
                        ctx = {}
                else:
                    ctx = ctx_str or {}
                source = ctx.get("source", "unknown")
                if source_filter and source != source_filter:
                    continue
                results.append({
                    "text": content,
                    "source": source,
                    "category": ctx.get("category", ""),
                    "score": m.get("_rrf_score", m.get("confidence", 0)),
                })
            return results
        except Exception as e:
            logger.debug("[SemanticIndex] Search failed: %s", e)
            return []

    def close(self) -> None:
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
