"""ClarificationPlannerAgent — typed multi-slot clarification with smart questions.

Handles skills that declare ``required_slots`` (vs the legacy ``required_prompt``).

Key behaviours:
- Analyzes each slot individually (type-aware, vague-word detection)
- Combines multiple missing slots into a single efficient question
  e.g. "抓什么物体，放到哪里？" instead of two separate turns
- Retries once if the first answer still leaves slots empty
- Memory hint: surfaces the last-used value from the current mission
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .base import ESTOP_SIGNALS, ProactiveAgent, ProactiveContext, ProactiveResult, ask_and_listen
from .session_state import ClarificationState
from .slot_analyst import analyze_slots
from .slot_types import SlotAnalysis, SlotFill

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition

logger = logging.getLogger(__name__)

_MAX_TURNS = 2    # maximum clarification rounds per dispatch

# Words that signal the user is abandoning this clarification entirely.
# Exact-match only: "算了" is a bail-out, "去仓库B算了吧" is an answer.
_INTERRUPT_SIGNALS: frozenset[str] = frozenset([
    "算了", "算了吧", "不了", "不用了", "不想了",
    "先不了", "先不要", "先不", "不问了", "取消吧",
    "换一个", "换个",
])

# Characters that separate an interrupt prefix from a reroute payload.
_SEPARATORS: frozenset[str] = frozenset("，,。.！! 、")

# Signals sorted longest-first so that "先不了" is tried before "先不".
# This prevents a shorter prefix from consuming chars that belong to a longer
# signal (e.g. "先不了，去仓库" must match "先不了" not "先不" + "了，去仓库").
#
# COUPLING NOTE: built once at import time from _INTERRUPT_SIGNALS.
# If _INTERRUPT_SIGNALS is modified at runtime (e.g. monkeypatched in tests),
# _INTERRUPT_SIGNALS_BY_LEN will NOT reflect the change.
# Always patch _INTERRUPT_SIGNALS_BY_LEN directly if you need to override in tests.
_INTERRUPT_SIGNALS_BY_LEN: tuple[str, ...] = tuple(
    sorted(_INTERRUPT_SIGNALS, key=len, reverse=True)
)


def parse_interrupt_payload(answer: str | None) -> tuple[bool, str]:
    """Parse an interrupt, extracting any reroute payload.

    Returns ``(is_interrupt, payload)`` where *payload* is the new intent
    (e.g. ``"去仓库B"`` from ``"算了，去仓库B"``), or ``""`` for a pure bail-out.

    Algorithm (longest signal first to avoid partial prefix conflicts):
    1. Exact match against ``_INTERRUPT_SIGNALS``                → (True, "")
    2. Text starts with a known signal + separator                → (True, payload)
    3. Anything else                                              → (False, "")

    False-positive protection:
    - "不了解" starts with "不了" but '解' is NOT a separator      → (False, "")
    - "先不了，去仓库" uses "先不了" (checked before "先不")        → (True, "去仓库")
    """
    if not answer:
        return False, ""
    t = answer.strip()
    if t in _INTERRUPT_SIGNALS:
        return True, ""
    sep_chars = "".join(_SEPARATORS)
    for signal in _INTERRUPT_SIGNALS_BY_LEN:
        if t.startswith(signal):
            rest = t[len(signal):]
            if not rest:
                return True, ""
            if rest[0] in _SEPARATORS:
                payload = rest.lstrip(sep_chars).strip()
                return True, payload
    return False, ""


def _is_interrupt(answer: str | None) -> bool:
    """True if the answer is any form of bail-out (with or without payload)."""
    is_int, _ = parse_interrupt_payload(answer)
    return is_int


def _is_estop(answer: str | None) -> bool:
    """True if the answer is an emergency-stop command."""
    if not answer:
        return False
    t = answer.strip().lower()
    return any(sig in t for sig in ESTOP_SIGNALS)


class ClarificationPlannerAgent(ProactiveAgent):
    """Ask for missing typed slots in the fewest possible turns.

    Activates when the skill declares ``required_slots``.
    Skips cleanly if all required slots are already filled.
    """

    def should_activate(
        self, skill: "SkillDefinition", user_text: str, context: ProactiveContext
    ) -> bool:
        if not skill.required_slots:
            return False
        analysis = analyze_slots(skill, user_text, context.pipeline)
        return not analysis.ready

    async def interact(
        self,
        skill: "SkillDefinition",
        user_text: str,
        audio: Any,
        context: ProactiveContext,
    ) -> ProactiveResult:
        session = context.session
        current_text = user_text

        for turn in range(_MAX_TURNS):
            analysis = analyze_slots(skill, current_text, context.pipeline)
            if analysis.ready:
                break

            # Transition session state before asking
            if session is not None:
                slot_name = (
                    analysis.missing_required[0].spec.name
                    if analysis.missing_required else ""
                )
                session.turn_count += 1
                if session.state == ClarificationState.IDLE:
                    session.transition(
                        ClarificationState.AWAITING_SLOT, current_slot=slot_name
                    )
                elif session.state == ClarificationState.AWAITING_SLOT:
                    session.transition(
                        ClarificationState.AWAITING_SLOT, current_slot=slot_name
                    )

            question = self._build_question(analysis, skill, context, turn)
            answer = await ask_and_listen(question, audio)

            # ESTOP — highest priority, abort everything
            if _is_estop(answer):
                logger.warning(
                    "ClarificationPlanner: ESTOP detected for %s: %r", skill.name, answer
                )
                if session is not None:
                    session.transition(
                        ClarificationState.CANCELED, cancel_reason="estop"
                    )
                return ProactiveResult(
                    enriched_text=user_text, proceed=False, cancelled_by="estop"
                )

            # Interrupt — user bailing out (possibly with a new intent)
            is_int, payload = parse_interrupt_payload(answer)
            if is_int:
                logger.info(
                    "ClarificationPlanner: interrupted for %s: %r (payload=%r)",
                    skill.name, answer, payload,
                )
                if session is not None:
                    session.transition(
                        ClarificationState.INTERRUPTED, interrupted_by=answer
                    )
                return ProactiveResult(
                    enriched_text=user_text,
                    proceed=False,
                    cancelled_by="interrupted",
                    interrupt_payload=payload,
                )

            if not answer or len(answer.strip()) < 2:
                logger.debug(
                    "ClarificationPlanner: empty/short answer for %s (turn %d)",
                    skill.name, turn + 1,
                )
                if turn < _MAX_TURNS - 1:
                    try:
                        audio.drain_buffers()
                        audio.speak("没有听到，请再说一遍。")
                        audio.start_playback()
                        await asyncio.to_thread(audio.wait_speaking_done)
                        audio.stop_playback()
                    except Exception:  # noqa: BLE001
                        pass
                continue

            current_text = current_text.rstrip() + " " + answer.strip()
            logger.info(
                "ClarificationPlanner: enriched %s (turn %d): %r",
                skill.name, turn + 1, current_text,
            )

        # Return session to IDLE so ConfirmationAgent can take over if needed
        if session is not None and session.state == ClarificationState.AWAITING_SLOT:
            session.transition(ClarificationState.IDLE)

        # Proceed regardless — let the LLM handle whatever context we have
        return ProactiveResult(enriched_text=current_text, proceed=True)

    # ── question builders ─────────────────────────────────────────────────────

    def _build_question(
        self,
        analysis: SlotAnalysis,
        skill: "SkillDefinition",
        context: ProactiveContext,
        turn: int,
    ) -> str:
        missing = analysis.missing_required
        if not missing:
            return ""

        # On the first turn, check for a memory hint
        if turn == 0:
            hint = self._get_hint(skill, context)
            if hint:
                first_fill = missing[0]
                return (
                    f"上次是{hint}，这次还是吗？"
                    f"或者{first_fill.spec.prompt or '请提供' + first_fill.spec.name}"
                )

        # Build a combined question for all missing slots
        prompts = [
            f.spec.prompt or f"请提供{f.spec.name}"
            for f in missing
            if not f.is_ok
        ]
        return _combine_prompts(prompts)

    # ── memory hint ───────────────────────────────────────────────────────────

    def _get_hint(self, skill: "SkillDefinition", context: ProactiveContext) -> str:
        """Return the most recent slot value from the current mission, if any."""
        if not context.dispatcher:
            return ""
        mission = getattr(context.dispatcher, "current_mission", None)
        if not mission:
            return ""

        for step in reversed(mission.steps):
            if step.skill_name != skill.name or not step.user_text:
                continue
            # For location slots, use semantic extraction
            location_specs = [s for s in skill.required_slots if s.type == "location"]
            if location_specs and context.pipeline:
                target = context.pipeline.extract_semantic_target(step.user_text)
                if target and target.strip() != step.user_text.strip():
                    return target.strip()
            # Fall back to trimmed user_text
            txt = step.user_text.strip()
            if len(txt) > 2:
                return txt

        return ""


# ── helpers ───────────────────────────────────────────────────────────────────


def _combine_prompts(prompts: list[str]) -> str:
    """Merge multiple slot prompts into one natural question.

    Examples:
      ["搜索什么内容？"]               → "搜索什么内容？"
      ["抓取什么物体？", "放到哪里？"]  → "抓取什么物体，放到哪里？"
      ["A？", "B？", "C？"]            → "A，B，C？"
    """
    if not prompts:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    # Strip trailing punctuation from all but the last, then join
    parts = [p.rstrip("？?。，,") for p in prompts[:-1]]
    return "，".join(parts) + "，" + prompts[-1]
