"""Strategy generator — LLM-based action suggestions from trends and associations.

Consumes trend summaries, association context, and world state to produce
ranked action suggestions via a single LLM call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A single action suggestion from the strategy generator."""

    action: str
    reason: str
    confidence: float


class StrategyGenerator:
    """Generate action suggestions from memory context via LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def suggest(
        self,
        *,
        trends: str = "",
        associations: str = "",
        world_state: str = "",
        procedures: str = "",
    ) -> list[Suggestion]:
        """Generate 1-3 action suggestions based on context.

        Returns empty list on LLM failure or empty context.
        """
        # Skip when there's no context to reason about
        if not any([trends, associations, world_state, procedures]):
            return []

        prompt = (
            "你是巡检机器人的决策系统。根据以下信息，给出1-3个行动建议。\n"
            "每个建议格式严格按照：行动|原因|信心(0-1)\n\n"
        )
        if trends:
            prompt += f"当前趋势:\n{trends}\n\n"
        if associations:
            prompt += f"相关历史:\n{associations}\n\n"
        if world_state:
            prompt += f"当前场景:\n{world_state}\n\n"
        if procedures:
            prompt += f"程序记录:\n{procedures}\n"

        try:
            result = await self._llm.chat([
                {"role": "system", "content": "你是决策助手。输出格式严格按照：行动|原因|信心(0-1)。每行一个建议。"},
                {"role": "user", "content": prompt},
            ])
            return self._parse(result)
        except Exception as exc:
            logger.warning("[StrategyGenerator] LLM call failed: %s", exc)
            return []

    @staticmethod
    def _parse(text: str) -> list[Suggestion]:
        """Parse LLM output into Suggestion objects.

        Expected format per line: ``action|reason|confidence``
        """
        suggestions: list[Suggestion] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            action = parts[0].strip()
            reason = parts[1].strip()
            try:
                confidence = float(parts[2].strip())
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                confidence = 0.5
            if action:
                suggestions.append(Suggestion(
                    action=action,
                    reason=reason,
                    confidence=confidence,
                ))
        return suggestions[:3]  # cap at 3
