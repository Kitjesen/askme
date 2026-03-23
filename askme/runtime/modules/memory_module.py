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

logger = logging.getLogger(__name__)


class MemoryContext:
    """Marker type for the assembled memory subsystem."""
    pass


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

        self.session_memory = SessionMemory(llm=llm)
        self.conversation = ConversationManager(
            session_memory=self.session_memory,
            metrics=ota_metrics,
        )
        self.memory_bridge = MemoryBridge()
        self.episodic = EpisodicMemory(llm=llm)
        self.memory_system = MemorySystem(
            llm=llm,
            conversation=self.conversation,
            session_memory=self.session_memory,
            episodic=self.episodic,
            vector_memory=self.memory_bridge,
        )
        logger.info("MemoryModule: built (llm=%s)", "wired" if llm else "none")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "conversation_len": len(self.conversation.history),
            "episodic_buffer_len": len(self.episodic._buffer),
        }
