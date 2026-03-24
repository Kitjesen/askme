"""MemoryModule — wraps the four-layer memory stack as a declarative module.

Mirrors the memory construction logic from ``assembly.py`` lines 416-430::

    session_memory = SessionMemory(llm=llm)
    conversation = ConversationManager(session_memory=session_memory, metrics=ota_metrics)
    memory = MemoryBridge()
    episodic = EpisodicMemory(llm=llm)
    memory_system = MemorySystem(llm=llm, conversation=conversation, ...)

The LLMClient is obtained from LLMModule via ``In[LLMClient]`` auto-wiring.
After wiring, ``self.llm_client`` is the LLMModule instance (not the client
directly). Access the client via ``self.llm_client.client``.
"""

from __future__ import annotations

import logging
from typing import Any

from askme.llm.client import LLMClient
from askme.llm.conversation import ConversationManager
from askme.memory.bridge import MemoryBridge
from askme.memory.episodic_memory import EpisodicMemory
from askme.memory.session import SessionMemory
from askme.memory.system import MemorySystem
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.schemas.messages import MemoryContext

logger = logging.getLogger(__name__)


class MemoryModule(Module):
    """Provides the four-layer memory stack to the runtime."""

    name = "memory"
    provides = ("conversation", "episodic", "vector_memory", "session_memory")

    llm_client: In[LLMClient]
    memory_context: Out[MemoryContext]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # Get LLMClient from LLMModule (auto-wired In port gives module instance)
        llm_mod = self.llm_client
        llm: LLMClient | None = getattr(llm_mod, "client", None) if llm_mod else None

        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None

        self._session_memory = SessionMemory(llm=llm)
        self._conversation = ConversationManager(
            session_memory=self._session_memory,
            metrics=ota_metrics,
        )
        self._memory_bridge = MemoryBridge()
        self._episodic = EpisodicMemory(llm=llm)
        self._memory_system = MemorySystem(
            llm=llm,
            conversation=self._conversation,
            session_memory=self._session_memory,
            episodic=self._episodic,
            vector_memory=self._memory_bridge,
        )
        logger.info("MemoryModule: built (llm=%s)", "wired" if llm else "none")

    # -- typed accessors ------------------------------------------------
    @property
    def conversation(self) -> ConversationManager:
        """L1 conversation manager."""
        return self._conversation

    @property
    def session_memory(self) -> SessionMemory:
        """L2 session memory."""
        return self._session_memory

    @property
    def episodic(self) -> EpisodicMemory:
        """L3 episodic memory."""
        return self._episodic

    @property
    def memory_bridge(self) -> MemoryBridge:
        """L4 vector memory bridge."""
        return self._memory_bridge

    @property
    def memory_system(self) -> MemorySystem:
        """Unified memory system."""
        return self._memory_system

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "conversation_len": len(self._conversation.history),
            "episodic_buffer_len": len(self._episodic._buffer),
        }
