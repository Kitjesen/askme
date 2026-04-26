"""ProactiveOrchestrator — chains proactive agents before skill dispatch.

Default chain (in order):
  1. SlotCollectorAgent  — ask for missing required parameters
  2. ConfirmationAgent   — confirm dangerous / irreversible actions

Enriched user_text flows from agent to agent.
If any agent returns proceed=False, the chain short-circuits.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import ProactiveAgent, ProactiveContext, ProactiveResult
from .clarification_agent import ClarificationPlannerAgent
from .confirm_agent import ConfirmationAgent
from .session_state import ClarificationSession
from .slot_agent import SlotCollectorAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ProactiveOrchestrator:
    """Runs the proactive agent chain for a voice skill dispatch."""

    def __init__(
        self,
        agents: list[ProactiveAgent],
        *,
        pipeline: Any = None,
        dispatcher: Any = None,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline
        self._dispatcher = dispatcher

    @classmethod
    def default(
        cls, *, pipeline: Any = None, dispatcher: Any = None
    ) -> ProactiveOrchestrator:
        """Build the standard three-agent chain: Clarify → Slot(legacy) → Confirm."""
        return cls(
            agents=[
                ClarificationPlannerAgent(),  # typed slots (required_slots)
                SlotCollectorAgent(),          # legacy required_prompt only
                ConfirmationAgent(),           # confirm before dangerous actions
            ],
            pipeline=pipeline,
            dispatcher=dispatcher,
        )

    async def run(
        self,
        skill_name: str,
        user_text: str,
        audio: Any,
        *,
        source: str = "voice",
    ) -> ProactiveResult:
        """Run the proactive chain and return the final enriched result.

        If no dispatcher is configured, or the skill is not found, the chain
        is skipped and the call proceeds immediately.
        """
        if not self._dispatcher:
            return ProactiveResult(enriched_text=user_text, proceed=True)

        skill = self._dispatcher.get_skill(skill_name)
        if skill is None:
            return ProactiveResult(enriched_text=user_text, proceed=True)

        session = ClarificationSession(skill_name=skill_name)
        context = ProactiveContext(
            pipeline=self._pipeline,
            dispatcher=self._dispatcher,
            source=source,
            session=session,
        )
        current_text = user_text

        for agent in self._agents:
            if not agent.should_activate(skill, current_text, context):
                continue
            result = await agent.interact(skill, current_text, audio, context)
            current_text = result.enriched_text
            if not result.proceed:
                logger.info(
                    "Proactive chain cancelled by %s for skill '%s'",
                    type(agent).__name__, skill_name,
                )
                return result

        return ProactiveResult(enriched_text=current_text, proceed=True)
