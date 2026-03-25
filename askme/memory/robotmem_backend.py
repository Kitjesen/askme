"""RobotMem backend — pluggable L4 memory via robotmem SDK.

Wraps ``robotmem.sdk.RobotMemory`` to match the MemoryBridge
retrieve/save contract.  Falls back gracefully when the ``robotmem``
package is not installed.

Usage::

    from askme.memory.robotmem_backend import RobotMemBackend

    backend = RobotMemBackend(config)
    context = await backend.retrieve("仓库A温度")
    await backend.save("用户问题", "助手回复")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RobotMemBackend:
    """RobotMem wrapper with the same API as MemoryBridge backends.

    Lazy-initialises the SDK on first use.  All public methods are
    async and offload blocking SDK calls to a thread.
    """

    def __init__(self, mem_cfg: dict[str, Any], brain_cfg: dict[str, Any]) -> None:
        self._mem_cfg = mem_cfg
        self._brain_cfg = brain_cfg

        # robotmem SDK instance — lazy init via _ensure_robotmem()
        self._rm: Any = None
        self._rm_failed: bool = False

        # Config knobs
        self._collection: str = mem_cfg.get("robotmem_collection", "askme")
        self._retrieve_timeout: float = mem_cfg.get("retrieve_timeout", 2.0)
        self._db_path: str | None = mem_cfg.get("robotmem_db_path", None)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_robotmem(self) -> bool:
        """Try to initialise the RobotMemory SDK.  Returns True if ready."""
        if self._rm is not None:
            return True
        if self._rm_failed:
            return False
        try:
            from robotmem.sdk import RobotMemory

            kwargs: dict[str, Any] = {
                "collection": self._collection,
                "embed_backend": "onnx",
            }
            if self._db_path:
                kwargs["db_path"] = self._db_path

            self._rm = RobotMemory(**kwargs)
            logger.info("[Memory] RobotMem initialised (collection=%s).", self._collection)
            return True
        except Exception as e:
            logger.warning("[Memory] RobotMem init failed: %s", e)
            self._rm_failed = True
            return False

    # ------------------------------------------------------------------
    # Public API (matches MemoryBridge contract)
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the robotmem backend is initialised and usable."""
        return self._rm is not None

    async def warmup(self) -> None:
        """Pre-load the robotmem DB + embedder in a background thread."""
        try:
            inited = await asyncio.to_thread(self._ensure_robotmem)
            if inited:
                # Trigger embedder warmup via a dummy recall
                await asyncio.to_thread(self._rm.recall, "warmup", n=1)
                logger.info("[Memory] RobotMem warmup complete.")
        except Exception as e:
            logger.debug("[Memory] RobotMem warmup failed: %s", e)

    async def retrieve(self, text: str) -> str:
        """Retrieve relevant memories for *text*.

        Returns a formatted context string (``- item`` per line),
        or empty string on failure / no results.
        """
        if not self._ensure_robotmem():
            return ""
        try:
            logger.debug("[Memory] RobotMem searching for: %s", text[:60])
            memories = await asyncio.wait_for(
                asyncio.to_thread(self._rm.recall, text, n=5),
                timeout=self._retrieve_timeout,
            )
            if not memories:
                logger.debug("[Memory] RobotMem no relevant memories found.")
                return ""

            items = []
            for m in memories:
                content = m.get("content", "")
                if content:
                    items.append(content)
            if items:
                logger.info("[Memory] RobotMem found %d items.", len(items))
                return "\n".join(f"- {item}" for item in items)
            return ""
        except asyncio.TimeoutError:
            logger.warning(
                "[Memory] RobotMem retrieval timed out (%.1fs).",
                self._retrieve_timeout,
            )
            return ""
        except Exception as exc:
            logger.debug("[Memory] RobotMem retrieve failed: %s", exc)
            return ""

    async def save(self, user_text: str, assistant_text: str) -> None:
        """Persist a conversation exchange to robotmem.

        Uses ``learn()`` to store the exchange as an insight that
        robotmem can later retrieve via hybrid search.
        """
        if not self._ensure_robotmem():
            return
        try:
            text = f"用户: {user_text}\n回复: {assistant_text[:200]}"
            context = {
                "source": "conversation",
                "robot": "thunder",
            }
            await asyncio.to_thread(
                self._rm.learn,
                text,
                context=context,
            )
            logger.debug("[Memory] RobotMem saved conversation turn.")
        except Exception as exc:
            logger.debug("[Memory] RobotMem save failed: %s", exc)

    def close(self) -> None:
        """Close the robotmem SDK (release DB connection)."""
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
