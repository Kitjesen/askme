"""Unified facade over L1-L4 memory layers.

Single injection point for BrainPipeline. Eliminates scattered guard blocks
like ``if self._episodic: self._episodic.log(...)`` throughout the pipeline.

Usage::

    from askme.memory.system import MemorySystem

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
from typing import TYPE_CHECKING, Any

from askme.memory.core.admission import MemoryAdmissionControl
from askme.memory.core.policies import PolicyStore
from askme.memory.intelligence.association import AssociationGraph
from askme.memory.intelligence.strategy import StrategyGenerator, Suggestion
from askme.memory.intelligence.trend_analyzer import TrendAnalyzer
from askme.memory.retrieval.semantic_index import SemanticIndex

if TYPE_CHECKING:
    from askme.llm.core.client import LLMClient
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.procedural import ProceduralMemory
    from askme.memory.core.session import SessionMemory
    from askme.memory.retrieval.bridge import MemoryBridge
    from askme.memory.retrieval.site_knowledge import SiteKnowledge

logger = logging.getLogger(__name__)

_BEHAVIOR_FACT_MARKERS = ("记住", "以后请", "我喜欢", "我不喜欢", "叫我")
_UNSAFE_BEHAVIOR_MARKERS = (
    "忽略安全",
    "忽略之前",
    "系统提示",
    "绕过权限",
    "绕过安全",
    "泄露密钥",
    "执行任意命令",
    "ignore previous",
    "system prompt",
    "bypass safety",
    "reveal secret",
)


def _is_unsafe_behavior_fact(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in _UNSAFE_BEHAVIOR_MARKERS)


def _is_explicit_behavior_fact(text: str) -> bool:
    if any(marker in text for marker in _BEHAVIOR_FACT_MARKERS):
        return True
    correction_start = text.find("不是")
    return correction_start >= 0 and "是" in text[correction_start + 2 :]


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
        behavior_memory: Any | None = None,
        site_knowledge: SiteKnowledge | None = None,
        procedural: ProceduralMemory | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._session = session_memory
        self._episodic = episodic
        self._vector = vector_memory
        self._site = site_knowledge
        self._procedural = procedural

        # Resolve config once. It is injectable for tests, with a global fallback.
        if config is None:
            try:
                from askme.config import get_config
                config = get_config()
            except (ImportError, TypeError, AttributeError):
                config = {}

        # L5: Semantic Index (unified search across L2+L3+L4)
        mem_cfg = config.get("memory", {})
        self._behavior = behavior_memory
        self._behavior_enabled = bool(
            mem_cfg.get("robot_behavior_memory_enabled", False)
        )
        self._behavior_admission = MemoryAdmissionControl(threshold=0.7)
        self._semantic = SemanticIndex(mem_cfg)

        # L6: Policies & Templates
        try:
            self._policies = PolicyStore(config=config)
        except Exception as e:
            logger.debug("PolicyStore init failed: %s", e)
            self._policies = None

        # Barrier capabilities: trend analysis, association, strategy
        self._trend_analyzer = TrendAnalyzer()
        vs = vector_memory.vector_store if vector_memory else None
        self._association = AssociationGraph(vs) if vs else None
        self._strategy = StrategyGenerator(llm) if llm else None

    # -- Record --

    def log_event(self, kind: str, text: str, context: dict[str, Any] | None = None) -> None:
        """Log event to L3 episodic (safe no-op if episodic is None)."""
        if self._episodic:
            self._episodic.log(kind, text, context)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Record user/assistant turn to L1 conversation."""
        self._conversation.add_user_message(user_text)
        self._conversation.add_assistant_message(assistant_text)

    async def save_behavior_memory(
        self,
        user_text: str,
        assistant_text: str = "",
    ) -> bool:
        """Persist an explicit long-lived user preference or instruction."""
        clean = str(user_text or "").strip()
        if not (
            self._behavior_enabled
            and self._behavior
            and not _is_unsafe_behavior_fact(clean)
            and _is_explicit_behavior_fact(clean)
        ):
            return False

        admitted, _ = self._behavior_admission.should_admit(
            "command", clean, importance=0.8
        )
        if not admitted:
            return False

        _ = assistant_text
        try:
            result = await self._behavior.save_fact(
                clean,
                {
                    "memory_type": "robot_behavior",
                    "source": "conversation",
                },
            )
        except Exception as exc:
            self._behavior_admission.discard(clean)
            logger.warning("Behavior memory save failed: %s", exc)
            return False
        saved = bool(result)
        if not saved:
            self._behavior_admission.discard(clean)
        return saved

    async def admit_turn(self, user_text: str, **kwargs: Any) -> Any:
        """Apply governed long-term admission to user-authored text only."""
        if not self._vector:
            return None
        return await self._vector.admit_turn(user_text, **kwargs)

    async def save_to_vector(self, user_text: str, assistant_text: str) -> Any:
        """Deprecated compatibility alias; assistant text is intentionally ignored."""
        del assistant_text
        result = await self.admit_turn(user_text)
        return False if result is None else result

    # -- Retrieve --

    async def retrieve_behavior(self, user_text: str) -> str:
        """Retrieve non-authoritative personalization context."""
        clean = str(user_text or "").strip()
        if not (self._behavior_enabled and self._behavior and clean):
            return ""
        try:
            result = await self._behavior.retrieve(clean)
        except Exception as exc:
            logger.warning("Behavior memory retrieve failed: %s", exc)
            return ""
        return str(result or "").strip()

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
        # L6: inject policy rules into context
        if self._policies:
            policy_ctx = self._policies.get_policy_prompt()
            if policy_ctx:
                parts.append(f"行为规则:\n{policy_ctx}")
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
        """Run L3 reflection if conditions met, then sync L5 index."""
        if not self._episodic or not self._episodic.should_reflect():
            return None
        try:
            result = await self._episodic.reflect()
            self._episodic.cleanup_old_episodes()
            # Sync new knowledge into L5 semantic index
            if result:
                await self.sync_semantic_index()
            return result
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            return None

    # -- L5: Semantic Index --

    async def semantic_search(
        self,
        query: str,
        n: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Unified semantic search across all memory layers (L5)."""
        return await self._semantic.search(query, n=n, source_filter=source_filter)

    async def sync_semantic_index(self) -> int:
        """Re-index L2+L3 content into L5 semantic store."""
        return await self._semantic.sync(
            episodic=self._episodic,
            session=self._session,
        )

    # -- Barrier capabilities: trends, associations, strategy --

    def get_trends(self) -> str:
        """Return Chinese text summary of active trends from episodic data."""
        if not self._episodic:
            return ""
        episodes = list(self._episodic._buffer)
        return self._trend_analyzer.get_summary(episodes)

    def has_trends(self) -> bool:
        """Whether there are active trends to report."""
        return bool(self.get_trends())

    def find_similar(self, description: str) -> str:
        """Find historically similar situations via vector association."""
        if not self._association:
            return ""
        return self._association.get_associations_text(description)

    async def suggest_actions(
        self,
        *,
        world_state: str = "",
    ) -> list[Suggestion]:
        """Generate action suggestions from current context."""
        if not self._strategy:
            return []
        trends = self.get_trends()
        procedures = ""
        if self._procedural:
            try:
                procedures = self._procedural.get_context()
            except Exception:
                logger.exception("[MemorySystem] Procedural context fetch failed")
                pass
        return await self._strategy.suggest(
            trends=trends,
            world_state=world_state,
            procedures=procedures,
        )

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
    def policies(self) -> PolicyStore | None:
        """L6 policies and templates."""
        return self._policies

    @property
    def semantic(self) -> SemanticIndex:
        """L5 semantic index."""
        return self._semantic

    @property
    def site_knowledge(self) -> SiteKnowledge | None:
        """Spatial memory (may be None)."""
        return self._site

    @property
    def procedural(self) -> ProceduralMemory | None:
        """Procedural memory (may be None)."""
        return self._procedural
