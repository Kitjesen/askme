"""Unified skill orchestration — voice, text, and runtime share one dispatcher.

The SkillDispatcher sits between input loops (VoiceLoop / TextLoop) and
BrainPipeline.  It provides:

1. **Mission tracking** — multi-step skill sequences share context.
2. **dispatch_skill meta-tool** — LLM can invoke skills by name during
   general conversation, enabling natural language → skill composition.
3. **Source-agnostic API** — voice and text loops call the same methods.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.skills.skill_manager import SkillManager
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


# ── Mission Context ───────────────────────────────────────────────


@dataclass
class MissionStep:
    """One skill execution within a mission."""

    skill_name: str
    user_text: str
    result: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MissionContext:
    """Tracks a multi-step skill mission.

    A mission starts when the first skill is dispatched and ends when
    the next general (non-skill) turn arrives or ``complete()`` is called.
    """

    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "voice"  # voice | text | runtime
    steps: list[MissionStep] = field(default_factory=list)
    shared_context: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, skill_name: str, user_text: str, result: str) -> None:
        self.steps.append(MissionStep(
            skill_name=skill_name,
            user_text=user_text,
            result=result,
        ))

    def summary(self) -> str:
        """One-line summary for logging."""
        names = [s.skill_name for s in self.steps]
        elapsed = time.time() - self.started_at
        return (
            f"mission={self.mission_id} source={self.source} "
            f"steps={names} elapsed={elapsed:.1f}s"
        )

    def history_for_context(self) -> str:
        """Format previous steps as context for the next skill."""
        if not self.steps:
            return ""
        lines: list[str] = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"[步骤{i}] {step.skill_name}: {step.result[:200]}")
        return "\n".join(lines)


# ── Skill Dispatcher ──────────────────────────────────────────────


class SkillDispatcher:
    """Unified skill orchestration for all input channels.

    Both VoiceLoop and TextLoop should delegate skill dispatch here
    instead of calling ``pipeline.execute_skill()`` directly.

    The dispatcher also registers a ``dispatch_skill`` meta-tool so the
    LLM can compose skills during general conversation.
    """

    # Per-step timeout (seconds). Skills call the LLM so give them time.
    _STEP_TIMEOUT = 60.0

    def __init__(
        self,
        *,
        pipeline: BrainPipeline,
        skill_manager: SkillManager,
        audio: AudioAgent,
    ) -> None:
        self._pipeline = pipeline
        self._skill_manager = skill_manager
        self._audio = audio
        self._current_mission: MissionContext | None = None
        # Captured lazily on first async dispatch — used by execute_skill_sync
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Public API (called by loops) ──────────────────────────────

    async def dispatch(
        self,
        skill_name: str,
        user_text: str,
        *,
        source: str = "voice",
        extra_context: str = "",
    ) -> str:
        """Execute a skill and track it in the current mission.

        If no mission is active, one is created automatically.
        Returns the skill result string.
        """
        # Capture the running event loop for use by execute_skill_sync
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        # Fail fast for unknown skills — avoids a silent error step in mission history
        if skill_name and not self._skill_manager.get(skill_name):
            available = self._skill_manager.get_skill_catalog()
            logger.warning("Dispatch called with unknown skill: '%s'", skill_name)
            return f"[Error] 技能不存在: {skill_name}。可用技能: {available}"

        # Start or continue mission
        if self._current_mission is None:
            self._current_mission = MissionContext(source=source)
            logger.info("Mission started: %s", self._current_mission.mission_id)

        # Build combined context: prior mission steps + caller-supplied context
        mission_history = self._current_mission.history_for_context()
        combined_context = "\n".join(filter(None, [mission_history, extra_context]))
        if mission_history:
            logger.info(
                "Mission context injected (%d prior steps)",
                self._current_mission.step_count,
            )

        # Use skill's own timeout + a 10s safety margin for the dispatcher guard.
        # The skill executor has its own inner timeout that fires first; the outer
        # wait_for here is a last-resort catch for hangs that don't raise TimeoutError.
        skill_def = self._skill_manager.get(skill_name)
        step_timeout = (
            float(skill_def.timeout) + 10.0 if skill_def else self._STEP_TIMEOUT
        )
        try:
            result = await asyncio.wait_for(
                self._pipeline.execute_skill(skill_name, user_text, combined_context),
                timeout=step_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Skill '%s' timed out after %.0fs", skill_name, step_timeout)
            result = f"[超时] 技能 {skill_name} 执行超过 {int(step_timeout)} 秒，已跳过。"

        # Track step
        self._current_mission.add_step(skill_name, user_text, result)
        logger.info(
            "Mission step %d: %s → %s",
            self._current_mission.step_count,
            skill_name,
            result[:60],
        )

        return result

    async def handle_general(
        self,
        user_text: str,
        *,
        source: str = "voice",
        memory_task: asyncio.Task[str] | None = None,
    ) -> str:
        """Handle a general (non-skill) turn via the LLM pipeline.

        Completes any active mission first (the turn breaks the skill chain).
        """
        self.complete_mission()
        return await self._pipeline.process(user_text, memory_task=memory_task)

    def complete_mission(self) -> MissionContext | None:
        """End the current mission and return it for logging."""
        mission = self._current_mission
        if mission and mission.steps:
            logger.info("Mission completed: %s", mission.summary())
        self._current_mission = None
        return mission

    @property
    def has_active_mission(self) -> bool:
        return self._current_mission is not None

    @property
    def current_mission(self) -> MissionContext | None:
        return self._current_mission

    # ── dispatch_skill tool (for LLM-driven composition) ──────────

    def execute_skill_sync(self, skill_name: str, reason: str = "") -> str:
        """Synchronous skill execution — called by the dispatch_skill tool.

        Bridges async ``dispatch`` into the sync tool execution context.
        Tools run inside ``asyncio.to_thread()`` — a worker thread separate
        from the event loop.  ``asyncio.run_coroutine_threadsafe`` is the
        correct cross-thread bridge; the loop reference is captured lazily
        on the first async ``dispatch`` call.
        """
        skill = self._skill_manager.get(skill_name)
        if not skill:
            available = self._skill_manager.get_skill_catalog()
            return f"[Error] 技能不存在: {skill_name}。可用技能: {available}"

        if reason:
            logger.info("LLM dispatching skill '%s': %s", skill_name, reason)

        user_text = reason or f"执行技能: {skill_name}"

        loop = self._loop
        if loop is None or not loop.is_running():
            return "[Error] 事件循环未就绪，无法执行技能"

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.dispatch(skill_name, user_text, source="llm"),
                loop,
            )
            sync_timeout = float(skill.timeout) + 15.0  # skill + dispatch margin + thread overhead
            return future.result(timeout=sync_timeout)
        except Exception as exc:
            logger.error("dispatch_skill tool error: %s", exc)
            return f"[Error] 技能执行失败: {exc}"

    def get_skill_catalog_for_prompt(self) -> str:
        """Return skill catalog formatted for system prompt injection."""
        skills = self._skill_manager.get_enabled()
        if not skills:
            return ""
        lines = []
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)
