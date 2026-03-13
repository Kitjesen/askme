"""SlotCollectorAgent — gathers missing required parameters before skill dispatch.

Supports:
- Multi-turn retry (up to MAX_ATTEMPTS) if the first answer is too short
- Memory hint: surfaces the previously-used value for the same skill within
  the current mission, so the user can simply confirm or provide a new one
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import ProactiveAgent, ProactiveContext, ProactiveResult, ask_and_listen
from .slot_utils import slot_present

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition

logger = logging.getLogger(__name__)

_MIN_SLOT_CHARS = 2   # minimum answer length to accept
_MAX_ATTEMPTS   = 2   # times to ask before giving up


class SlotCollectorAgent(ProactiveAgent):
    """Ask for a missing required slot, with retry and memory-hint support."""

    def should_activate(
        self, skill: "SkillDefinition", user_text: str, context: ProactiveContext
    ) -> bool:
        if not skill.required_prompt:
            return False
        # Skills with required_slots use ClarificationPlannerAgent instead
        if skill.required_slots:
            return False
        return not slot_present(skill, user_text, context.pipeline)

    async def interact(
        self,
        skill: "SkillDefinition",
        user_text: str,
        audio: Any,
        context: ProactiveContext,
    ) -> ProactiveResult:
        hint    = self._get_hint(skill, context)
        question = self._build_question(skill, hint)

        for attempt in range(_MAX_ATTEMPTS):
            answer = await ask_and_listen(question, audio)

            if answer and len(answer.strip()) >= _MIN_SLOT_CHARS:
                enriched = user_text.rstrip() + " " + answer.strip()
                logger.info(
                    "SlotCollector: collected slot for %s (attempt %d): %r → %r",
                    skill.name, attempt + 1, user_text, enriched,
                )
                return ProactiveResult(enriched_text=enriched, proceed=True)

            # Answer too short / empty — retry with plain required_prompt
            question = skill.required_prompt
            logger.debug(
                "SlotCollector: answer too short for %s (attempt %d), retrying",
                skill.name, attempt + 1,
            )

        # All attempts exhausted — proceed with original text anyway so we
        # don't silently drop the user's intent.
        logger.warning(
            "SlotCollector: gave up collecting slot for %s after %d attempts",
            skill.name, _MAX_ATTEMPTS,
        )
        return ProactiveResult(enriched_text=user_text, proceed=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_question(skill: "SkillDefinition", hint: str) -> str:
        if hint:
            return f"上次是{hint}，这次还是吗？或者{skill.required_prompt}"
        return skill.required_prompt

    def _get_hint(self, skill: "SkillDefinition", context: ProactiveContext) -> str:
        """Extract a slot hint from the current mission's step history.

        For ``navigate``: uses semantic extraction to surface just the
        destination name (e.g. "仓库B"), not the full utterance.
        For other skills: returns the trimmed previous user_text.
        """
        if not context.dispatcher:
            return ""
        mission = getattr(context.dispatcher, "current_mission", None)
        if mission is None:
            return ""

        for step in reversed(mission.steps):
            if step.skill_name != skill.name or not step.user_text:
                continue

            if skill.name == "navigate" and context.pipeline:
                target = context.pipeline.extract_semantic_target(step.user_text)
                if target and target.strip() != step.user_text.strip():
                    return target
            else:
                txt = step.user_text.strip()
                if len(txt) > 2:
                    return txt

        return ""
