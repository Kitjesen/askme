"""
Memory bridge — L4 vector memory with pluggable backends.

Supported backends (``memory.backend`` in config.yaml):

- ``"mem0"``      — Mem0 (default, backward-compatible)
- ``"robotmem"``  — robotmem SDK (pip install robotmem)
- ``"vector"``    — local VectorStore (sentence-transformers, no server)

Lazy initialization: the selected backend is only instantiated on first
use.  If it is unavailable (import error, config error, etc.), the bridge
falls back to the local ``VectorStore``.

Graceful degradation: all operations return empty / no-op on failure.

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
    """L4 vector memory — pluggable backend with VectorStore fallback."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        data_dir: "str | Path | None" = None,
    ) -> None:
        """Create a MemoryBridge.

        Args:
            config: Full config dict (e.g. from get_config()).  If None, the
                global config.yaml is read via get_config().  Pass a dict in
                tests to avoid filesystem reads.
            data_dir: Override the data directory for the VectorStore fallback.
                If None, read from ``config["app"]["data_dir"]``.
        """
        cfg = config if config is not None else get_config()
        self._mem_cfg: dict[str, Any] = cfg.get("memory", {})

        self._enabled: bool = self._mem_cfg.get("enabled", True)
        self._embed_model: str = self._mem_cfg.get(
            "embed_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._retrieve_timeout: float = self._mem_cfg.get("retrieve_timeout", 2.0)

        # Backend selection: "mem0" (default), "robotmem", "vector"
        self._backend: str = self._mem_cfg.get("backend", "mem0")

        # Mem0 instance — lazy init via _ensure_mem0()
        self._mem0: Any = None
        self._mem0_failed: bool = False  # True after init failure, skip retries

        # RobotMem backend — lazy init via _ensure_robotmem()
        self._robotmem: Any = None  # RobotMemBackend instance
        self._robotmem_failed: bool = False

        # Fallback: local VectorStore (lazy — only init when actually needed)
        if data_dir is not None:
            resolved = Path(data_dir)
        else:
            raw = cfg.get("app", {}).get("data_dir", "data")
            resolved = Path(raw)
            if not resolved.is_absolute():
                resolved = project_root() / resolved
        self._store_path = resolved / "memory" / "vectors" / "store.json"
        self._store: VectorStore | None = None

        if not self._enabled:
            logger.info("[Memory] Memory disabled in config.")
        else:
            logger.info(
                "[Memory] MemoryBridge ready (backend=%s, fallback VectorStore).",
                self._backend,
            )

    # ------------------------------------------------------------------
    # VectorStore lazy initialization (fallback)
    # ------------------------------------------------------------------

    def _ensure_store(self) -> VectorStore | None:
        """Lazy-init VectorStore only when actually needed as fallback."""
        if self._store is not None:
            return self._store
        try:
            self._store = VectorStore(
                model_name=self._embed_model,
                store_path=self._store_path,
            )
            return self._store
        except Exception as e:
            logger.debug("[Memory] VectorStore init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Mem0 lazy initialization
    # ------------------------------------------------------------------

    def _ensure_mem0(self) -> bool:
        """Try to initialise the Mem0 instance. Returns True if ready."""
        if self._mem0 is not None:
            return True
        if not self._enabled or self._mem0_failed:
            return False
        try:
            from mem0 import Memory

            brain_cfg = get_config().get("brain", {})
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "askme",
                        "path": str(project_root() / "data" / "memory" / "mem0_store"),
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": brain_cfg.get("api_key", ""),
                        "openai_base_url": brain_cfg.get("base_url", ""),
                        "model": brain_cfg.get("model", "MiniMax-M2.7-highspeed"),
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": self._embed_model,
                    },
                },
            }
            self._mem0 = Memory.from_config(config)
            logger.info("[Memory] Mem0 initialised successfully.")
            return True
        except Exception as e:
            logger.warning("[Memory] Mem0 init failed, using VectorStore fallback: %s", e)
            self._mem0_failed = True
            return False

    # ------------------------------------------------------------------
    # RobotMem lazy initialization
    # ------------------------------------------------------------------

    def _ensure_robotmem(self) -> bool:
        """Try to initialise the RobotMem backend. Returns True if ready."""
        if self._robotmem is not None and self._robotmem.available:
            return True
        if not self._enabled or self._robotmem_failed:
            return False
        try:
            from askme.memory.robotmem_backend import RobotMemBackend

            brain_cfg = get_config().get("brain", {})
            self._robotmem = RobotMemBackend(self._mem_cfg, brain_cfg)
            inited = self._robotmem._ensure_robotmem()
            if not inited:
                self._robotmem_failed = True
                self._robotmem = None
                return False
            logger.info("[Memory] RobotMem backend ready.")
            return True
        except Exception as e:
            logger.warning("[Memory] RobotMem init failed, falling back: %s", e)
            self._robotmem_failed = True
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Pre-load the embedding model in a background thread."""
        if not self._enabled:
            return

        # Warm up the configured backend
        if self._backend == "robotmem":
            try:
                inited = await asyncio.to_thread(self._ensure_robotmem)
                if inited:
                    await self._robotmem.warmup()
                    logger.info("[Memory] RobotMem warmup complete.")
                    return
            except Exception:
                logger.debug("[Memory] RobotMem warmup failed, trying fallback.")

        if self._backend in ("mem0", "robotmem"):
            # Try Mem0 (primary for mem0 backend, fallback for robotmem)
            try:
                inited = await asyncio.to_thread(self._ensure_mem0)
                if inited:
                    logger.info("[Memory] Mem0 warmup complete.")
                    return
            except Exception:
                logger.debug("[Memory] Mem0 warmup failed, trying VectorStore.")

        # Fallback: warm up VectorStore
        store = self._ensure_store()
        if store and store.available:
            try:
                await asyncio.to_thread(store.search, "warmup", 1)
                logger.info("[Memory] VectorStore warmup complete.")
            except Exception:
                logger.debug("[Memory] VectorStore warmup triggered model load (expected).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, text: str) -> str:
        """Retrieve relevant memory context for *text*.

        Returns a formatted context string (one ``- item`` per line),
        or an empty string if memory is unavailable / finds nothing.
        """
        if not self._enabled:
            return ""

        # Try configured backend first — use to_thread so lazy init (ONNX load)
        # never blocks the event loop on cold start.
        if self._backend == "robotmem" and await asyncio.to_thread(self._ensure_robotmem):
            return await self._robotmem.retrieve(text)

        if self._backend in ("mem0", "robotmem") and await asyncio.to_thread(self._ensure_mem0):
            return await self._retrieve_mem0(text)

        # Fallback to VectorStore
        return await self._retrieve_vector_store(text)

    async def save(self, user_text: str, assistant_text: str) -> None:
        """Persist a conversation exchange to L4 memory.

        Silently no-ops when the backend is unavailable.
        """
        if not self._enabled:
            return

        # Try configured backend first — use to_thread so lazy init never blocks.
        if self._backend == "robotmem" and await asyncio.to_thread(self._ensure_robotmem):
            await self._robotmem.save(user_text, assistant_text)
            return

        if self._backend in ("mem0", "robotmem") and await asyncio.to_thread(self._ensure_mem0):
            await self._save_mem0(user_text, assistant_text)
            return

        # Fallback to VectorStore
        await self._save_vector_store(user_text, assistant_text)

    def import_existing_data(self) -> int:
        """Scan L3 knowledge/digests and import into vector store.

        Returns the number of entries imported. Runs synchronously.
        No-ops when unavailable.
        """
        store = self._ensure_store()
        if not self._enabled or not store or not store.available:
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
                            store.add(line[2:], {"type": "knowledge", "source": md_file.name})
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
                        store.add(content, {"type": "digest", "source": md_file.name})
                        imported += 1
                except Exception as exc:
                    logger.warning("[Memory] Import %s failed: %s", md_file.name, exc)

        if imported:
            store.save()
            logger.info("[Memory] Imported %d entries from existing L3 data.", imported)
        return imported

    @property
    def available(self) -> bool:
        """Whether the memory service is initialised and usable."""
        if not self._enabled:
            return False
        if self._robotmem is not None and self._robotmem.available:
            return True
        if self._mem0 is not None:
            return True
        store = self._ensure_store()
        return store.available if store else False

    @property
    def vector_store(self) -> VectorStore | None:
        """Direct access to the underlying VectorStore (for AssociationGraph)."""
        return self._ensure_store()

    # ------------------------------------------------------------------
    # Mem0 backend
    # ------------------------------------------------------------------

    async def _retrieve_mem0(self, text: str) -> str:
        """Search Mem0 for relevant memories."""
        try:
            logger.debug("[Memory] Mem0 searching for: %s", text[:60])
            results = await asyncio.wait_for(
                asyncio.to_thread(self._mem0.search, text, user_id="robot"),
                timeout=self._retrieve_timeout,
            )
            if not results or not results.get("results"):
                logger.debug("[Memory] Mem0 no relevant memories found.")
                return ""
            memories = [r.get("memory", "") for r in results["results"][:5]]
            items = [m for m in memories if m]
            if items:
                logger.info("[Memory] Mem0 found %d items.", len(items))
                return "\n".join(f"- {m}" for m in items)
            return ""
        except asyncio.TimeoutError:
            logger.warning("[Memory] Mem0 retrieval timed out (%.1fs).", self._retrieve_timeout)
            return ""
        except Exception as exc:
            logger.debug("[Memory] Mem0 retrieve failed: %s", exc)
            return ""

    async def _save_mem0(self, user_text: str, assistant_text: str) -> None:
        """Add conversation turn to Mem0 (auto-extracts facts)."""
        try:
            text = f"用户: {user_text}\n回复: {assistant_text[:200]}"
            await asyncio.to_thread(self._mem0.add, text, user_id="robot")
            logger.debug("[Memory] Mem0 saved conversation turn.")
        except Exception as exc:
            logger.debug("[Memory] Mem0 save failed: %s", exc)

    # ------------------------------------------------------------------
    # VectorStore fallback
    # ------------------------------------------------------------------

    async def _retrieve_vector_store(self, text: str) -> str:
        """Retrieve from local VectorStore."""
        store = self._ensure_store()
        if not store or not store.available:
            return ""
        try:
            logger.debug("[Memory] VectorStore searching for: %s", text[:60])
            results = await asyncio.wait_for(
                asyncio.to_thread(store.search, text, 5),
                timeout=self._retrieve_timeout,
            )
            if results:
                logger.info("[Memory] VectorStore found %d items.", len(results))
                return "\n".join(
                    f"- {item['text']}" for item in results if item.get("score", 0) > 0.3
                )
            logger.debug("[Memory] VectorStore no relevant memories found.")
            return ""
        except asyncio.TimeoutError:
            logger.warning("[Memory] VectorStore retrieval timed out (%.1fs).", self._retrieve_timeout)
            return ""
        except Exception as exc:
            logger.warning("[Memory] VectorStore retrieval error: %s", exc)
            return ""

    async def _save_vector_store(self, user_text: str, assistant_text: str) -> None:
        """Persist to local VectorStore."""
        store = self._ensure_store()
        if not store or not store.available:
            return
        content = f"用户: {user_text}\n助手: {assistant_text[:200]}"
        try:
            await asyncio.to_thread(store.add, content, {
                "type": "conversation",
                "ts": __import__("time").time(),
            })
            # Periodic save (every 10 new entries)
            if store.size % 10 == 0:
                await asyncio.to_thread(store.save)
            logger.debug("[Memory] VectorStore saved conversation turn.")
        except Exception as exc:
            logger.warning("[Memory] VectorStore save failed: %s", exc)
