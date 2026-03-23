"""
Memory bridge to the MemU memory service.

Handles lazy initialisation: checks the embedding server's health before
attempting to create the heavy ``MemoryService`` object, so the rest of
the application starts cleanly even when the embedding server is offline.

Usage::

    from askme.brain import MemoryBridge

    mem = MemoryBridge()            # does NOT block or crash
    context = await mem.retrieve("今天天气怎么样")
"""

from __future__ import annotations

import asyncio
import logging
import sys
import urllib.request
from typing import Any

from askme.config import get_config

logger = logging.getLogger(__name__)


class MemoryBridge:
    """Thin wrapper around ``memu.app.MemoryService`` with smart init."""

    def __init__(self) -> None:
        cfg = get_config()
        self._mem_cfg: dict[str, Any] = cfg.get("memory", {})
        self._brain_cfg: dict[str, Any] = cfg.get("brain", {})

        self._enabled: bool = self._mem_cfg.get("enabled", True)
        self._embed_url: str = self._mem_cfg.get("embed_url", "http://localhost:8000/v1")
        self._embed_model: str = self._mem_cfg.get("embed_model", "all-MiniLM-L6-v2")
        self._retrieve_timeout: float = self._mem_cfg.get("retrieve_timeout", 2.0)
        self._retrieve_method: str = self._mem_cfg.get("retrieve_method", "rag")
        self._user_id: str = self._mem_cfg.get("user_id", "user_1")
        self._health_timeout: int = self._mem_cfg.get("health_check_timeout", 1)

        # Lazily initialised MemoryService instance
        self._service: Any | None = None
        self._init_attempted: bool = False
        self._init_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _check_embedding_server(self) -> bool:
        """Return ``True`` if the local embedding server is reachable."""
        try:
            url = self._embed_url.rstrip("/") + "/models"
            urllib.request.urlopen(url, timeout=self._health_timeout)
            return True
        except Exception:
            return False

    async def _ensure_service(self) -> bool:
        """Attempt to create the ``MemoryService`` (once).

        Returns ``True`` if the service is ready.
        """
        if self._service is not None:
            return True
        if self._init_attempted:
            return self._service is not None

        # Lazy-create lock inside an async context (Lock must be created in the loop)
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            # Double-check inside lock — concurrent callers wait here
            if self._init_attempted:
                return self._service is not None
            self._init_attempted = True

            if not self._enabled:
                logger.info("[Memory] Memory disabled in config.")
                return False

            # Health check via thread to avoid blocking the event loop
            reachable = await asyncio.to_thread(self._check_embedding_server)
            if not reachable:
                logger.warning("[Memory] Embedding server not reachable -- memory disabled.")
                return False

            # Import MemU lazily so the rest of askme works without it installed
            try:
                # Add memU/src to path if not already present
                import os
                from askme.config import project_root
                memu_src = str(project_root() / "memU" / "src")
                if memu_src not in sys.path:
                    sys.path.insert(0, memu_src)

                from memu.app import MemoryService  # type: ignore[import-untyped]

                self._service = MemoryService(
                    llm_profiles={
                        "default": {
                            "api_key": self._brain_cfg.get("api_key", ""),
                            "base_url": self._brain_cfg.get("base_url", ""),
                            "chat_model": self._brain_cfg.get("model", "MiniMax-M2.7-highspeed"),
                            "client_backend": "sdk",
                        },
                        "embedding": {
                            "api_key": "sk-dummy",
                            "base_url": self._embed_url,
                            "embed_model": self._embed_model,
                        },
                    },
                    retrieve_config={
                        "method": self._retrieve_method,
                        "route_intention": False,
                        "sufficiency_check": False,
                        "resource": {"enabled": False},
                        "category": {"top_k": 3},
                        "item": {"top_k": 5},
                    },
                )
                logger.info("[Memory] MemoryService initialised successfully.")
                return True

            except Exception as exc:
                logger.warning("[Memory] MemoryService init failed: %s", exc)
                self._service = None
                return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Pre-initialize MemU service in background to avoid cold-start on first query."""
        if not await self._ensure_service():
            return
        try:
            await asyncio.wait_for(
                self._service.retrieve(
                    queries=[{"role": "user", "content": {"text": "warmup"}}],
                    where={"user_id": self._user_id},
                ),
                timeout=10.0,
            )
            logger.info("[Memory] Warmup complete.")
        except Exception:
            logger.debug("[Memory] Warmup query returned no results (expected).")

    async def retrieve(self, text: str) -> str:
        """Retrieve relevant memory context for *text*.

        Returns a formatted context string (one ``- item`` per line),
        or an empty string if memory is unavailable / times out / finds
        nothing.
        """
        # Fast path: if init was already attempted and failed, return immediately
        # without logging — the WARNING was already emitted once at startup.
        if self._init_attempted and self._service is None:
            return ""

        if not await self._ensure_service():
            return ""

        try:
            logger.debug("[Memory] Searching for: %s", text[:60])
            retrieval = await asyncio.wait_for(
                self._service.retrieve(
                    queries=[{"role": "user", "content": {"text": text}}],
                    where={"user_id": self._user_id},
                ),
                timeout=self._retrieve_timeout,
            )
            items = retrieval.get("items", [])
            if items:
                logger.info("[Memory] Found %d items.", len(items))
                return "\n".join(f"- {item['content']}" for item in items)
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

        Silently no-ops when the embedding server is unavailable.
        """
        if self._init_attempted and self._service is None:
            return
        if not await self._ensure_service():
            return
        content = f"用户: {user_text}\n助手: {assistant_text[:200]}"
        try:
            await asyncio.wait_for(
                self._service.create_memory_item(
                    memory_type="event",
                    memory_content=content,
                    memory_categories=[],
                    user={"user_id": self._user_id},
                ),
                timeout=5.0,
            )
            logger.debug("[Memory] Saved conversation turn.")
        except Exception as exc:
            logger.warning("[Memory] Save failed: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the memory service is initialised and usable."""
        return self._service is not None
