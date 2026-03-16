"""Unified facade over L1-L4 memory layers.

Single injection point for BrainPipeline. Eliminates scattered guard blocks
like ``if self._episodic: self._episodic.log(...)`` throughout the pipeline.

Usage::

    from askme.brain.memory_system import MemorySystem

    mem = MemorySystem(
        llm=llm,
        conversation=conversation,
        session_memory=session_memory,
        episodic=episodic,
        vector_memory=memory_bridge,
    )
    mem.log_event("command", "user said hello")
    ctx = mem.get_memory_context("hello")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.brain.conversation import ConversationManager
    from askme.brain.episodic_memory import EpisodicMemory
    from askme.brain.llm_client import LLMClient
    from askme.brain.memory_bridge import MemoryBridge
    from askme.brain.procedural_memory import ProceduralMemory
    from askme.brain.session_memory import SessionMemory
    from askme.brain.site_knowledge import SiteKnowledge

logger = logging.getLogger(__name__)


class MemorySystem:
    """Unified facade over L1-L4 memory layers.

    Single injection point for BrainPipeline. Eliminates 15+ guard blocks.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        conversation: ConversationManager,
        session_memory: SessionMemory | None,
        episodic: EpisodicMemory | None,
        vector_memory: MemoryBridge | None,
        site_knowledge: SiteKnowledge | None = None,
        procedural: ProceduralMemory | None = None,
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._session = session_memory
        self._episodic = episodic
        self._vector = vector_memory
        self._site = site_knowledge
        self._procedural = procedural

    # -- Record --

    def log_event(self, kind: str, text: str, context: dict[str, Any] | None = None) -> None:
        """Log event to L3 episodic (safe no-op if episodic is None)."""
        if self._episodic:
            self._episodic.log(kind, text, context)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Record user/assistant turn to L1 conversation."""
        self._conversation.add_user_message(user_text)
        self._conversation.add_assistant_message(assistant_text)

    async def save_to_vector(self, user_text: str, assistant_text: str) -> None:
        """Fire-and-forget save to L4 vector memory."""
        if self._vector:
            await self._vector.save(user_text, assistant_text)

    # -- Retrieve --

    def get_memory_context(self, user_text: str = "") -> str:
        """Assemble memory context for system prompt from L2+L3.

        Replaces the scattered context assembly in _build_system_prompt.
        """
        parts: list[str] = []
        if self._episodic:
            world = self._episodic.get_knowledge_context()
            if world:
                parts.append(world)
            digest = self._episodic.get_recent_digest()
            if digest:
                parts.append(digest)
            if user_text:
                relevant = self._episodic.get_relevant_context(user_text)
                if relevant:
                    parts.append(relevant)
        if self._session:
            session_ctx = self._session.get_recent_summaries()
            if session_ctx:
                parts.append(session_ctx)
        return "\n".join(parts)

    def start_prefetch(self, user_text: str) -> asyncio.Task[str] | None:
        """Start L4 vector retrieval as background task."""
        if not self._vector:
            return None
        return asyncio.create_task(self._vector.retrieve(user_text))

    # -- Maintenance --

    async def compress(self) -> None:
        """L1 sliding window compression (fire-and-forget safe)."""
        try:
            await self._conversation.maybe_compress(self._llm)
        except Exception as e:
            logger.warning("Memory compression failed: %s", e)

    def should_reflect(self) -> bool:
        """Check if L3 reflection should be triggered."""
        return bool(self._episodic and self._episodic.should_reflect())

    async def reflect(self) -> str | None:
        """Run L3 reflection if conditions met."""
        if not self._episodic or not self._episodic.should_reflect():
            return None
        try:
            result = await self._episodic.reflect()
            self._episodic.cleanup_old_episodes()
            return result
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            return None

    # -- Properties for direct access when needed --

    @property
    def conversation(self) -> ConversationManager:
        """L1 conversation manager."""
        return self._conversation

    @property
    def episodic(self) -> EpisodicMemory | None:
        """L3 episodic memory (may be None)."""
        return self._episodic

    @property
    def has_episodic(self) -> bool:
        """Whether episodic memory is available."""
        return self._episodic is not None

    @property
    def site_knowledge(self) -> SiteKnowledge | None:
        """Spatial memory (may be None)."""
        return self._site

    @property
    def procedural(self) -> ProceduralMemory | None:
        """Procedural memory (may be None)."""
        return self._procedural
