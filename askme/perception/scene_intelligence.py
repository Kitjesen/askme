"""
SceneIntelligence — unified scene-awareness API.

Wraps the memory layers (EpisodicMemory, SessionMemory) and surfaces them
as human-readable, queryable scene knowledge.  This is the core of the
"场景之王" positioning: a single entry point for everything the AI
knows about the physical space it lives in.

Usage::

    from askme.perception.scene_intelligence import SceneIntelligence

    scene = SceneIntelligence(episodic=episodic_mem, session=session_mem)
    print(await scene.briefing())           # today's scene summary
    print(scene.who_is_around())            # known entities
    print(scene.anomalies())                # error / unusual events
    print(await scene.today_summary(llm))  # LLM-generated narrative
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.llm.client import LLMClient
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.session import SessionMemory

logger = logging.getLogger(__name__)

# Importance threshold above which an event counts as anomalous
_ANOMALY_IMPORTANCE = 0.5
_ANOMALY_TYPES = {"error", "outcome"}


class SceneIntelligence:
    """Unified read API over the scene's accumulated knowledge.

    Does NOT write to memory — it only reads and synthesises what is
    already stored in EpisodicMemory and SessionMemory.
    """

    def __init__(
        self,
        *,
        episodic: EpisodicMemory,
        session: SessionMemory | None = None,
    ) -> None:
        self._episodic = episodic
        self._session = session

    # ── Core queries ──────────────────────────────────────────

    def who_is_around(self) -> list[str]:
        """Return the names / labels of entities recorded in scene knowledge."""
        entities: Any = self._episodic._world_knowledge.get("entities", {})  # noqa: SLF001
        if isinstance(entities, dict):
            return sorted(entities.keys())
        return []

    def anomalies(self) -> list[dict[str, Any]]:
        """Return recent high-importance error or unusual outcome events."""
        results = []
        for ep in self._episodic._buffer:  # noqa: SLF001
            if (
                ep.get("event_type") in _ANOMALY_TYPES
                and ep.get("importance", 0.0) >= _ANOMALY_IMPORTANCE
            ):
                results.append({
                    "time": ep.get("timestamp", ""),
                    "type": ep.get("event_type", ""),
                    "description": ep.get("content", ""),
                    "importance": round(ep.get("importance", 0.0), 2),
                })
        # Most important first
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    async def briefing(self) -> str:
        """Assemble a structured scene briefing from all memory layers.

        Returns a multi-section text covering world knowledge, recent
        episode digest, and session summaries.
        """
        parts: list[str] = []

        knowledge = self._episodic.get_knowledge_context()
        if knowledge:
            parts.append(knowledge)

        digest = self._episodic.get_recent_digest()
        if digest:
            parts.append(digest)

        if self._session:
            session_ctx = self._session.get_recent_summaries()
            if session_ctx:
                parts.append(session_ctx)

        anomalies = self.anomalies()
        if anomalies:
            lines = ["近期异常事件:"]
            for a in anomalies[:5]:
                lines.append(f"  [{a['type']}] {a['description'][:80]} (重要度 {a['importance']})")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "暂无场景记录。"

    async def today_summary(self, llm: LLMClient) -> str:
        """Generate a concise narrative summary of today's scene activity.

        Uses the LLM to synthesise the raw briefing into a human-readable
        2-3 sentence digest.
        """
        raw = await self.briefing()
        if raw == "暂无场景记录。":
            return raw

        try:
            result = await llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是场景分析助手。用2-3句简洁的中文口语总结场景情况，"
                            "突出异常和重要事件。不要用markdown。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"请总结以下场景记录：\n{raw[:2000]}",
                    },
                ],
                temperature=0.3,
            )
            return result.strip() if result else raw
        except Exception as exc:
            logger.warning("[Scene] LLM summary failed: %s", exc)
            return raw

    def status(self) -> dict[str, Any]:
        """Return a machine-readable scene status snapshot."""
        return {
            "known_entities": self.who_is_around(),
            "anomaly_count": len(self.anomalies()),
            "episode_buffer_size": len(self._episodic._buffer),  # noqa: SLF001
            "has_world_knowledge": bool(self._episodic._world_knowledge),  # noqa: SLF001
        }
