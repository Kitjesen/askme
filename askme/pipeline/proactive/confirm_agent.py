"""ConfirmationAgent — requires explicit voice confirmation before dangerous actions.

Activates when ``skill.confirm_before_execute`` is True.
Safe default: cancels on ambiguous, empty, or timed-out answers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .base import ESTOP_SIGNALS, ProactiveAgent, ProactiveContext, ProactiveResult, ask_and_listen
from .clarification_agent import parse_interrupt_payload
from .session_state import ClarificationState

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition

logger = logging.getLogger(__name__)

_YES_WORDS: frozenset[str] = frozenset(
    ["确认", "确定", "是", "对", "好", "好的", "执行", "继续", "嗯", "yes", "ok"]
)
_NO_WORDS: frozenset[str] = frozenset(
    ["取消", "不", "不要", "算了", "停", "停止", "不用", "不行", "no", "cancel"]
)
# Single-char words that could be substrings of unrelated text
_SINGLE_CHAR: frozenset[str] = frozenset(w for w in _YES_WORDS | _NO_WORDS if len(w) == 1)

# _ESTOP_SIGNALS is now ESTOP_SIGNALS imported from base — single source of truth.


class ConfirmationAgent(ProactiveAgent):
    """Prompt for yes/no confirmation before executing a dangerous skill.

    Order in the proactive chain: runs *after* SlotCollectorAgent so the
    confirmation message can reference the fully-enriched user request.
    """

    def should_activate(
        self, skill: SkillDefinition, user_text: str, context: ProactiveContext
    ) -> bool:
        return bool(getattr(skill, "confirm_before_execute", False))

    async def interact(
        self,
        skill: SkillDefinition,
        user_text: str,
        audio: Any,
        context: ProactiveContext,
    ) -> ProactiveResult:
        session = context.session
        if session is not None:
            session.transition(ClarificationState.AWAITING_CONFIRMATION)

        prompt = f"即将执行{skill.description}，说'确认'继续，说'取消'停止。"
        answer = await ask_and_listen(prompt, audio)

        # ESTOP — abort immediately, no cancellation message needed
        if self._is_estop(answer):
            logger.warning("ConfirmationAgent: ESTOP during %s: %r", skill.name, answer)
            if session is not None:
                session.transition(ClarificationState.CANCELED, cancel_reason="estop")
            return ProactiveResult(
                enriched_text=user_text, proceed=False, cancelled_by="estop"
            )

        # Interrupt with optional reroute payload ("算了，去手动模式").
        # Uses INTERRUPTED state (not CANCELED) — interrupt is recoverable;
        # the session can reset to IDLE and the caller can reroute using
        # interrupt_payload. CANCELED is reserved for non-recoverable stops.
        is_int, payload = parse_interrupt_payload(answer)
        if is_int:
            logger.info(
                "ConfirmationAgent: interrupted during %s: %r (payload=%r)",
                skill.name, answer, payload,
            )
            if session is not None:
                session.transition(
                    ClarificationState.INTERRUPTED, interrupted_by=str(answer or "")
                )
            return ProactiveResult(
                enriched_text=user_text,
                proceed=False,
                cancelled_by="interrupted",
                interrupt_payload=payload,
            )

        if self._is_yes(answer):
            logger.info("ConfirmationAgent: confirmed %s", skill.name)
            if session is not None:
                session.transition(ClarificationState.IDLE)
            return ProactiveResult(enriched_text=user_text, proceed=True)

        # No / ambiguous / empty / timeout → cancel (safe default)
        if answer is None:
            reason = "未确认"
            cancel_msg = "没有听到您的回答，操作已取消。"
        elif self._is_no(answer):
            reason = "用户取消"
            cancel_msg = "好的，已取消。"
        else:
            reason = "未确认"
            cancel_msg = "好的，已取消。"
        logger.info(
            "ConfirmationAgent: cancelled %s (%s, answer=%r)",
            skill.name, reason, answer,
        )
        if session is not None:
            session.transition(ClarificationState.CANCELED, cancel_reason=reason)
        # Tell the user we cancelled before returning
        try:
            audio.drain_buffers()
            audio.speak(cancel_msg)
            audio.start_playback()
            await asyncio.to_thread(audio.wait_speaking_done)
            audio.stop_playback()
        except Exception:  # noqa: BLE001
            pass
        return ProactiveResult(
            enriched_text=user_text, proceed=False, cancelled_by=reason
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_estop(text: str | None) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        return any(sig in t for sig in ESTOP_SIGNALS)

    @staticmethod
    def _is_yes(text: str | None) -> bool:
        if not text:
            return False
        t = text.strip()
        for w in _YES_WORDS:
            if w in _SINGLE_CHAR:
                # Single-char words: only match if the answer IS exactly that char
                if t == w:
                    return True
            else:
                if w in t:
                    return True
        return False

    @staticmethod
    def _is_no(text: str | None) -> bool:
        if not text:
            return False
        t = text.strip()
        for w in _NO_WORDS:
            if w in _SINGLE_CHAR:
                if t == w:
                    return True
            else:
                if w in t:
                    return True
        return False
