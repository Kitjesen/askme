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
import dataclasses
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from askme.config import project_root

if TYPE_CHECKING:
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.planner_agent import PlannerAgent, PlanStep
    from askme.skills.skill_manager import SkillManager
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


# ── Mission Context ───────────────────────────────────────────────


class MissionState(Enum):
    """Explicit lifecycle state for a mission.

    Transitions: RUNNING → SUCCEEDED | FAILED | CANCELED
    """
    RUNNING = "running"      # Active — steps are being dispatched
    SUCCEEDED = "succeeded"  # complete_mission() was called normally
    FAILED = "failed"        # A step raised an unrecoverable error
    CANCELED = "canceled"    # Explicitly canceled by user or system


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
    state: MissionState = MissionState.RUNNING
    # Optional binding to an external runtime mission (mission-orchestrator).
    # Set via SkillDispatcher.bind_runtime_mission() when the runtime creates a
    # mission before asking askme to execute skills.  Stored in the persisted
    # JSON so the mission log is traceable back to the runtime mission system.
    runtime_mission_id: str | None = None

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
        rt = f" rt={self.runtime_mission_id}" if self.runtime_mission_id else ""
        return (
            f"mission={self.mission_id}{rt} source={self.source} state={self.state.value} "
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
        planner: "PlannerAgent | None" = None,
    ) -> None:
        self._pipeline = pipeline
        self._skill_manager = skill_manager
        self._audio = audio
        self._planner = planner
        self._current_mission: MissionContext | None = None
        # Captured lazily on first async dispatch — used by execute_skill_sync
        self._loop: asyncio.AbstractEventLoop | None = None
        # Per-thread dispatch depth — guards against LLM-driven infinite recursion
        self._dispatch_depth = threading.local()

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
        else:
            # Continuing an existing multi-step mission — let the user know
            step_num = self._current_mission.step_count + 1
            _step_def = self._skill_manager.get(skill_name)
            step_label = (_step_def.description if _step_def and _step_def.description else skill_name)
            self._audio.speak(f"继续执行第{step_num}步：{step_label}")
            self._audio.start_playback()
            await asyncio.to_thread(self._audio.wait_speaking_done)
            self._audio.stop_playback()

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
            self._current_mission.state = MissionState.FAILED

        # Track step, store result in shared_context for cross-step data passing
        self._current_mission.add_step(skill_name, user_text, result)
        self._current_mission.shared_context[skill_name] = result[:500]
        self._persist_mission()
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
        """Handle a general (non-skill) turn.

        Tries PlannerAgent first: if the intent decomposes into a multi-step
        skill sequence, executes all steps and returns a combined result.
        Falls back to the LLM pipeline for single-step / conversational input.

        Completes any active mission first (this turn breaks the skill chain).
        """
        self.complete_mission()

        # Attempt multi-step planning before handing off to the LLM
        if self._planner is not None:
            try:
                steps = await self._planner.plan(user_text)
            except Exception as exc:
                logger.warning("PlannerAgent raised unexpectedly: %s", exc)
                steps = None

            if steps:
                results: list[str] = []
                for step in steps:
                    result = await self.dispatch(step.skill_name, step.intent, source=source)
                    results.append(result)
                    # Stop plan execution if a step failed
                    if (
                        self._current_mission is not None
                        and self._current_mission.state == MissionState.FAILED
                    ):
                        logger.warning("Plan aborted at step '%s' due to FAILED state", step.skill_name)
                        break
                self.complete_mission()
                return "\n".join(results)

        # Single-step or conversational: delegate to LLM pipeline
        return await self._pipeline.process(
            user_text, memory_task=memory_task, source=source,
        )

    def get_skill(self, skill_name: str):
        """Return the SkillDefinition for a skill name, or None if not found."""
        return self._skill_manager.get(skill_name)

    def complete_mission(self) -> MissionContext | None:
        """End the current mission and return it for logging."""
        mission = self._current_mission
        if mission and mission.steps:
            logger.info("Mission completed: %s", mission.summary())
            if len(mission.steps) > 1:
                names = "、".join(
                    (self._skill_manager.get(s.skill_name).description
                     if self._skill_manager.get(s.skill_name) and self._skill_manager.get(s.skill_name).description
                     else s.skill_name)
                    for s in mission.steps
                )
                self._audio.speak(f"多步任务完成：{names}")
                self._audio.start_playback()
        if mission:
            if mission.state == MissionState.RUNNING:
                mission.state = MissionState.SUCCEEDED
        self._persist_mission()
        self._current_mission = None
        return mission

    @property
    def has_active_mission(self) -> bool:
        return self._current_mission is not None

    @property
    def current_mission(self) -> MissionContext | None:
        return self._current_mission

    def bind_runtime_mission(self, runtime_mission_id: str) -> None:
        """Bind the current askme mission to an external runtime mission ID.

        Call this when the runtime (mission-orchestrator) has already created a
        mission and wants the askme skill sequence to be recorded as a sub-entity
        of that mission.  The ID is persisted in the mission JSON for audit trails.

        No-op when no mission is active.
        """
        if not self._current_mission:
            return
        self._current_mission.runtime_mission_id = runtime_mission_id
        self._persist_mission()
        logger.info(
            "Mission %s bound to runtime_mission_id=%s",
            self._current_mission.mission_id,
            runtime_mission_id,
        )

    # ── dispatch_skill tool (for LLM-driven composition) ──────────

    def execute_skill_sync(self, skill_name: str, reason: str = "") -> str:
        """Synchronous skill execution — called by the dispatch_skill tool.

        Bridges async ``dispatch`` into the sync tool execution context.
        Tools run inside ``asyncio.to_thread()`` — a worker thread separate
        from the event loop.  ``asyncio.run_coroutine_threadsafe`` is the
        correct cross-thread bridge; the loop reference is captured lazily
        on the first async ``dispatch`` call.
        """
        # Recursion guard: LLM can call dispatch_skill inside execute_skill,
        # which calls process(), which can call dispatch_skill again.
        # Limit nesting to 3 levels to prevent runaway chains.
        _MAX_DEPTH = 3
        depth = getattr(self._dispatch_depth, "value", 0)
        if depth >= _MAX_DEPTH:
            logger.warning("dispatch_skill depth=%d >= max=%d, rejecting '%s'", depth, _MAX_DEPTH, skill_name)
            return f"[Error] 技能调用嵌套过深（最大{_MAX_DEPTH}层），已拒绝: {skill_name}"

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

        self._dispatch_depth.value = depth + 1
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
        finally:
            self._dispatch_depth.value = depth

    def _persist_mission(self) -> None:
        """Serialize current mission to disk for auditability and crash recovery.

        Silently no-ops on any error — persistence is best-effort, not critical path.
        """
        if self._current_mission is None:
            return
        try:
            missions_dir = project_root() / "data" / "missions"
            missions_dir.mkdir(parents=True, exist_ok=True)
            path = missions_dir / f"{self._current_mission.mission_id}.json"
            d = dataclasses.asdict(self._current_mission)
            d["state"] = self._current_mission.state.value
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Mission persist failed: %s", exc)

    def get_skill_catalog_for_prompt(self) -> str:
        """Return skill catalog formatted for system prompt injection."""
        skills = self._skill_manager.get_enabled()
        if not skills:
            return ""
        lines = []
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)
