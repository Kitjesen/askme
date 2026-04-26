"""Base types for the proactive interaction system."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition

    from .session_state import ClarificationSession

logger = logging.getLogger(__name__)

# Shared emergency-stop signals — imported by all proactive agents so that
# the set stays in one place. Adding a new word here automatically applies
# to both ClarificationPlannerAgent and ConfirmationAgent.
ESTOP_SIGNALS: frozenset[str] = frozenset([
    "急停", "紧急停止", "estop", "e-stop", "emergency stop",
])


@dataclass
class ProactiveResult:
    """Result returned by a proactive agent after its interaction turn."""

    enriched_text: str
    proceed: bool = True
    cancelled_by: str = ""  # agent name or reason, for logging
    interrupt_payload: str = ""  # new intent extracted from "算了，去仓库B" → "去仓库B"


@dataclass
class ProactiveContext:
    """Shared context passed to all proactive agents in the chain."""

    pipeline: Any = None    # BrainPipeline — for semantic extraction
    dispatcher: Any = None  # SkillDispatcher — for mission history
    source: str = "voice"
    session: ClarificationSession | None = None  # state machine (one per run)


class ProactiveAgent(ABC):
    """Base class for a single proactive interaction step."""

    @abstractmethod
    def should_activate(
        self, skill: SkillDefinition, user_text: str, context: ProactiveContext
    ) -> bool:
        """Return True if this agent should run for the given skill + text."""

    @abstractmethod
    async def interact(
        self,
        skill: SkillDefinition,
        user_text: str,
        audio: Any,
        context: ProactiveContext,
    ) -> ProactiveResult:
        """Run the interaction and return an enriched result."""


async def ask_and_listen(question: str, audio: Any) -> str | None:
    """Say *question* via TTS, wait, then listen for one user response.

    Returns the transcribed text, or None if nothing was captured.

    Audio failures (device errors, speaker/mic exceptions) are caught and
    logged rather than propagated — the caller treats None as an empty answer
    so the system degrades gracefully instead of crashing.
    """
    try:
        audio.drain_buffers()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask_and_listen: drain_buffers failed: %s", exc)

    try:
        audio.speak(question)
        audio.start_playback()
        await asyncio.to_thread(audio.wait_speaking_done)
        audio.stop_playback()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask_and_listen: TTS/playback failed: %s", exc)

    try:
        # Enable confirmation word pass-through so "好的"/"确认" etc.
        # are not silently eaten by the noise utterance filter.
        audio.awaiting_confirmation = True
        try:
            return await asyncio.to_thread(audio.listen_loop)
        finally:
            audio.awaiting_confirmation = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask_and_listen: listen_loop failed: %s", exc)
        return None
