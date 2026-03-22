"""StateLedBridge — non-invasive LED state driver.

Polls AudioAgent.state and SkillDispatcher.has_active_agent_task every
200 ms, computes the highest-priority LED state, and calls
LedController.set_state() only when the state changes.

Design goals
------------
- Zero invasive changes to AudioAgent or SkillDispatcher.
- Runs as a long-lived asyncio.Task; cancellable cleanly.
- LED failures are swallowed — never affect the robot control path.
- Priority table matches the docstring in LedStateKind.

Priority (highest first):
  ESTOP > MUTED > AGENT_TASK > SPEAKING > PROCESSING > LISTENING > IDLE
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from askme.robot.led_controller import LedController, LedStateKind, NullLedController
from askme.voice.audio_agent import AgentState

if TYPE_CHECKING:
    from askme.robot.safety_client import DogSafetyClient
    from askme.pipeline.skill_dispatcher import SkillDispatcher
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)

# Map AudioAgent states to LED states (used when no higher-priority condition holds)
_AGENT_STATE_TO_LED: dict[AgentState, LedStateKind] = {
    AgentState.IDLE:       LedStateKind.IDLE,
    AgentState.LISTENING:  LedStateKind.LISTENING,
    AgentState.PROCESSING: LedStateKind.PROCESSING,
    AgentState.SPEAKING:   LedStateKind.SPEAKING,
    AgentState.MUTED:      LedStateKind.MUTED,
}

_POLL_INTERVAL = 0.2  # seconds — 5 Hz is more than enough for LED transitions


class StateLedBridge:
    """Background task that keeps the LED in sync with system state.

    Parameters
    ----------
    audio:
        AudioAgent instance — provides current AgentState.
    dispatcher:
        SkillDispatcher — provides has_active_agent_task flag.
    safety:
        Optional DogSafetyClient — provides ESTOP state.
    led:
        LedController implementation.  Defaults to NullLedController.
    """

    def __init__(
        self,
        *,
        audio: AudioAgent,
        dispatcher: SkillDispatcher | None = None,
        safety: DogSafetyClient | None = None,
        led: LedController | None = None,
    ) -> None:
        self._audio = audio
        self._dispatcher = dispatcher
        self._safety = safety
        self._led = led or NullLedController()
        self._last_kind: LedStateKind | None = None

    def resolve(self) -> LedStateKind:
        """Compute the current LED state from live system state."""
        # ESTOP overrides everything (non-blocking cache read only)
        if self._safety is not None:
            try:
                if self._safety.is_estop_active():
                    return LedStateKind.ESTOP
            except Exception:
                pass  # treat safety read failure as non-ESTOP

        agent_state = self._audio.state

        # MUTED overrides task/speaking/processing
        if agent_state == AgentState.MUTED:
            return LedStateKind.MUTED

        # Long-running background agent task
        if self._dispatcher is not None and self._dispatcher.has_active_agent_task:
            return LedStateKind.AGENT_TASK

        return _AGENT_STATE_TO_LED.get(agent_state, LedStateKind.IDLE)

    async def run(self) -> None:
        """Poll loop — call as asyncio.create_task(bridge.run())."""
        logger.info("[StateLedBridge] Started (poll=%.0fms)", _POLL_INTERVAL * 1000)
        while True:
            try:
                kind = self.resolve()
                if kind != self._last_kind:
                    logger.debug("[StateLedBridge] %s → %s", self._last_kind, kind.value)
                    self._led.set_state(kind)
                    self._last_kind = kind
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[StateLedBridge] poll error (non-critical): %s", exc)
            await asyncio.sleep(_POLL_INTERVAL)
