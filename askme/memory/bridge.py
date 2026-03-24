"""
Memory bridge — L4 vector memory via local sentence-transformers.

Replaced the MemU / external embedding server approach with a local
``VectorStore`` that runs sentence-transformers in-process.

Graceful degradation: when sentence-transformers is not installed,
``available`` returns False and all operations no-op.

Usage::

    from askme.memory.bridge import MemoryBridge

    mem = MemoryBridge()            # does NOT block or crash
    context = await mem.retrieve("今天天气怎么样")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from askme.config import get_config, project_root
from askme.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryBridge:
    """Thin wrapper around ``VectorStore`` for L4 vector memory."""

    def __init__(self) -> None:
        cfg = get_config()
        self._mem_cfg: dict[str, Any] = cfg.get("memory", {})

        self._enabled: bool = self._mem_cfg.get("enabled", True)
        self._embed_model: str = self._mem_cfg.get(
            "embed_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._retrieve_timeout: float = self._mem_cfg.get("retrieve_timeout", 2.0)

        # Resolve store path
        data_dir = cfg.get("app", {}).get("data_dir", "data")
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        store_path = resolved / "memory" / "vectors" / "store.json"

        self._store = VectorStore(
            model_name=self._embed_model,
            store_path=store_path,
        )

        if not self._enabled:
            logger.info("[Memory] Memory disabled in config.")
        elif not self._store.available:
            logger.info("[Memory] sentence-transformers not installed — L4 vector memory disabled.")
        else:
            logger.info("[Memory] VectorStore ready (model=%s, entries=%d)",
                        self._embed_model, self._store.size)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Pre-load the embedding model in a background thread."""
        if not self._enabled or not self._store.available:
            return
        try:
            await asyncio.to_thread(self._store.search, "warmup", 1)
            logger.info("[Memory] Warmup complete.")
        except Exception:
            logger.debug("[Memory] Warmup triggered model load (expected).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, text: str) -> str:
        """Retrieve relevant memory context for *text*.

        Returns a formatted context string (one ``- item`` per line),
        or an empty string if memory is unavailable / finds nothing.
        """
        if not self._enabled or not self._store.available:
            return ""

        try:
            logger.debug("[Memory] Searching for: %s", text[:60])
            results = await asyncio.wait_for(
                asyncio.to_thread(self._store.search, text, 5),
                timeout=self._retrieve_timeout,
            )
            if results:
                logger.info("[Memory] Found %d items.", len(results))
                return "\n".join(
                    f"- {item['text']}" for item in results if item.get("score", 0) > 0.3
                )
            logger.debug("[Memory] No relevant memories found.")
            return ""

        except asyncio.TimeoutError:
            logger.warning("[Memory] Retrieval timed out (%.1fs).", self._retrieve_timeout)
            return ""
        except Exception as exc:
            logger.warning("[Memory] Retrieval error: %s", exc)
            return ""

    async def save(self, user_text: str, assistant_text: str) -> None:
        """Persist a conversation exchange to the L4 vector store.

        Silently no-ops when the embedding model is unavailable.
        """
        if not self._enabled or not self._store.available:
            return
        content = f"用户: {user_text}\n助手: {assistant_text[:200]}"
        try:
            await asyncio.to_thread(self._store.add, content, {
                "type": "conversation",
                "ts": __import__("time").time(),
            })
            # Periodic save (every 10 new entries)
            if self._store.size % 10 == 0:
                await asyncio.to_thread(self._store.save)
            logger.debug("[Memory] Saved conversation turn.")
        except Exception as exc:
            logger.warning("[Memory] Save failed: %s", exc)

    def import_existing_data(self) -> int:
        """Scan L3 knowledge/digests and import into vector store.

        Returns the number of entries imported. Runs synchronously.
        No-ops when unavailable.
        """
        if not self._enabled or not self._store.available:
            return 0

        cfg = get_config()
        data_dir = cfg.get("app", {}).get("data_dir", "data")
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        memory_dir = resolved / "memory"

        imported = 0

        # Import knowledge .md files (line by line)
        knowledge_dir = memory_dir / "knowledge"
        if knowledge_dir.exists():
            for md_file in knowledge_dir.glob("*.md"):
                try:
                    for line in md_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and line.startswith("- "):
                            self._store.add(line[2:], {"type": "knowledge", "source": md_file.name})
                            imported += 1
                except Exception as exc:
                    logger.warning("[Memory] Import %s failed: %s", md_file.name, exc)

        # Import digest .md files (whole file)
        digest_dir = memory_dir / "digests"
        if digest_dir.exists():
            for md_file in digest_dir.glob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8").strip()
                    if content:
                        self._store.add(content, {"type": "digest", "source": md_file.name})
                        imported += 1
                except Exception as exc:
                    logger.warning("[Memory] Import %s failed: %s", md_file.name, exc)

        if imported:
            self._store.save()
            logger.info("[Memory] Imported %d entries from existing L3 data.", imported)
        return imported

    @property
    def available(self) -> bool:
        """Whether the memory service is initialised and usable."""
        return self._enabled and self._store.available

    @property
    def vector_store(self) -> VectorStore:
        """Direct access to the underlying VectorStore."""
        return self._store
