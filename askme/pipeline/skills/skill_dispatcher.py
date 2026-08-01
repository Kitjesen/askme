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
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

from askme.config import project_root
from askme.conversation import ConversationLedgerError, TurnOutcome, TurnStatus
from askme.llm.core.contracts import LLMCallContext
from askme.pipeline.core.protocols import SkillExecutionDisposition
from askme.pipeline.skills.outcome import (
    GENERIC_SKILL_FAILURE_MESSAGE,
    GENERIC_SKILL_TIMEOUT_MESSAGE,
    GENERIC_SKILL_UNAVAILABLE_MESSAGE,
    NAV_LOCATION_UNAVAILABLE_MESSAGE,
    SkillOutcome,
    is_internal_skill_text,
)

if TYPE_CHECKING:
    from askme.pipeline.core.brain_pipeline import BrainPipeline
    from askme.pipeline.core.protocols import CancellationToken
    from askme.pipeline.skills.planner_agent import PlannerAgent
    from askme.ports import AudioFrontendPort
    from askme.skills.core.skill_manager import SkillManager

logger = logging.getLogger(__name__)


# ── Mission Context ───────────────────────────────────────────────


class MissionState(Enum):
    """Explicit lifecycle state for a mission.

    Transitions: RUNNING → SUCCEEDED | FAILED | CANCELED
    """

    RUNNING = "running"  # Active — steps are being dispatched
    SUCCEEDED = "succeeded"  # complete_mission() was called normally
    FAILED = "failed"  # A step raised an unrecoverable error
    CANCELED = "canceled"  # Explicitly canceled by user or system


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
        self.steps.append(
            MissionStep(
                skill_name=skill_name,
                user_text=user_text,
                result=result,
            )
        )

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


@dataclass(slots=True)
class _AgentTaskTurn:
    """Canonical lifecycle handle for a detached agent_task execution."""

    context: Any | None
    settled: bool = False


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
        audio: AudioFrontendPort,
        planner: PlannerAgent | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._skill_manager = skill_manager
        self._audio = audio
        self._planner = planner
        self._current_mission: MissionContext | None = None
        self._last_mission: MissionContext | None = None  # preserved after complete_mission()
        # Captured lazily on first async dispatch — used by execute_skill_sync
        self._loop: asyncio.AbstractEventLoop | None = None
        # Per-thread dispatch depth — guards against LLM-driven infinite recursion
        self._dispatch_depth = threading.local()
        # Background agent task (agent_task skill runs here so VoiceLoop stays responsive)
        self._active_agent_task: asyncio.Task[None] | None = None

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (set by VoiceModule after build)."""
        self._audio = audio

    # ── Helpers ────────────────────────────────────────────────────

    async def _speak_voice(self, text: str, source: str) -> None:
        """Speak text. In voice mode, also wait for playback to finish."""
        if source != "voice":
            return
        if is_internal_skill_text(text):
            logger.error("Suppressed internal skill marker from TTS: %s", text[:120])
            text = GENERIC_SKILL_FAILURE_MESSAGE
        self._audio.speak(text)
        self._audio.start_playback()
        await asyncio.to_thread(self._audio.wait_speaking_done)
        self._audio.stop_playback()

    async def can_execute(
        self,
        skill_name: str,
        user_text: str = "",
        *,
        source: str = "voice",
    ) -> SkillOutcome:
        """Preflight a skill without creating a mission or speaking."""

        skill = self._skill_manager.get(skill_name)
        if skill is None:
            available = self._skill_manager.get_skill_catalog()
            return SkillOutcome.blocked(
                code="not_found",
                result=f"[Error] 技能不存在: {skill_name}。可用技能: {available}",
                user_message=GENERIC_SKILL_UNAVAILABLE_MESSAGE,
            )
        if getattr(skill, "enabled", True) is False:
            return SkillOutcome.blocked(
                code="disabled",
                result=f"[Skill] Disabled: {skill_name}",
                user_message=(
                    NAV_LOCATION_UNAVAILABLE_MESSAGE
                    if skill_name == "nav_query"
                    else GENERIC_SKILL_UNAVAILABLE_MESSAGE
                ),
            )

        pipeline_state = getattr(self._pipeline, "__dict__", {})
        gate = pipeline_state.get("_skill_gate") if isinstance(pipeline_state, dict) else None
        gate_preflight = getattr(gate, "can_execute", None)
        if callable(gate_preflight):
            outcome = await gate_preflight(
                skill_name,
                user_text,
                source=source,
            )
            if not outcome.can_execute:
                return outcome

        if skill_name == "agent_task":
            opener = getattr(type(self._pipeline), "_open_direct_interaction", None)
            settler = getattr(type(self._pipeline), "_settle_direct_interaction", None)
            if not callable(opener) or not callable(settler):
                return SkillOutcome.blocked(
                    code="canonical_interaction_unavailable",
                    result="[Skill] Unavailable: agent_task (canonical_interaction_unavailable)",
                    user_message=GENERIC_SKILL_UNAVAILABLE_MESSAGE,
                )

        return SkillOutcome.ready()

    async def _speak_outcome(self, outcome: SkillOutcome, *, source: str) -> None:
        if outcome.should_speak and outcome.user_message:
            await self._speak_voice(outcome.user_message, source)

    @staticmethod
    def _supported_kwargs(callback: Any, values: dict[str, Any]) -> dict[str, Any]:
        """Filter optional context for older pipeline implementations."""

        signature_target = callback
        side_effect = getattr(callback, "side_effect", None)
        if callable(side_effect):
            signature_target = side_effect
        try:
            parameters = signature(signature_target).parameters
            if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
                return values
        except (TypeError, ValueError):
            return {}
        return {name: value for name, value in values.items() if name in parameters}

    async def _execute_pipeline_skill(
        self,
        skill_name: str,
        user_text: str,
        extra_context: str,
        *,
        source: str,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
        record_turn: bool = True,
    ) -> str:
        callback = self._pipeline.execute_skill
        values: dict[str, Any] = {"source": source}
        if conversation_session_id is not None:
            values["conversation_session_id"] = conversation_session_id
        if voice_turn_id is not None:
            values["voice_turn_id"] = voice_turn_id
        if turn_cancel_token is not None:
            values["turn_cancel_token"] = turn_cancel_token
        if not record_turn:
            values["record_turn"] = False
        return await callback(
            skill_name,
            user_text,
            extra_context,
            **self._supported_kwargs(callback, values),
        )

    async def _record_agent_task_ack(
        self,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: CancellationToken | None,
    ) -> _AgentTaskTurn:
        opener = getattr(type(self._pipeline), "_open_direct_interaction", None)
        if callable(opener):
            metadata = {
                "interaction": "agent_task",
                "interaction_phase": "ack",
                "skill_name": "agent_task",
                "acknowledgement": assistant_text,
            }
            context = self._pipeline._open_direct_interaction(
                user_text=user_text,
                source=source,
                conversation_session_id=conversation_session_id,
                voice_turn_id=voice_turn_id,
                turn_cancel_token=turn_cancel_token,
                metadata=metadata,
            )
            if context is None:
                raise ConversationLedgerError(
                    "agent_task requires a canonical direct interaction context"
                )
            self._mark_agent_task_acknowledged(
                context,
                acknowledgement=assistant_text,
                metadata=metadata,
            )
            return _AgentTaskTurn(context=context)

        raise ConversationLedgerError("agent_task requires canonical direct interaction support")

    def _mark_agent_task_acknowledged(
        self,
        context: Any | None,
        *,
        acknowledgement: str,
        metadata: dict[str, Any],
    ) -> None:
        if context is None:
            return
        ledger = getattr(self._pipeline, "_turn_ledger", None)
        transition_turn = getattr(ledger, "transition_turn", None)
        if not callable(transition_turn):
            return
        try:
            transition_turn(
                context.turn_id,
                TurnStatus.SPEAKING,
                assistant_text=acknowledgement,
                heard_text=acknowledgement,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Unable to mark agent_task acknowledgement on canonical turn")

    def _classify_agent_task_result(self, result: str) -> SkillExecutionDisposition:
        classifier = getattr(type(self._pipeline), "_classify_skill_execution_result", None)
        if callable(classifier):
            try:
                disposition = self._pipeline._classify_skill_execution_result(
                    result,
                    "agent_task",
                )
                if isinstance(disposition, SkillExecutionDisposition):
                    return disposition
            except Exception:
                logger.exception("Unable to classify agent_task result")
        text = str(result or "").lstrip()
        if not text:
            return SkillExecutionDisposition(status="failed", code="empty_skill_result")
        if text.startswith(("[Timeout]", "[超时]")):
            return SkillExecutionDisposition(status="failed", code="execution_timeout")
        if text.startswith(
            (
                "[Skill]",
                "[Skill Error]",
                "[AgentShell Error]",
                "[Error]",
                "[错误]",
            )
        ):
            return SkillExecutionDisposition(status="failed", code="internal_error_result")
        return SkillExecutionDisposition(status="succeeded", code="succeeded")

    def _settle_agent_task_turn(
        self,
        lifecycle: _AgentTaskTurn,
        outcome: TurnOutcome,
    ) -> None:
        if lifecycle.settled or lifecycle.context is None:
            return
        settler = getattr(type(self._pipeline), "_settle_direct_interaction", None)
        if not callable(settler):
            raise ConversationLedgerError(
                "agent_task requires canonical direct interaction settlement"
            )
        try:
            self._pipeline._settle_direct_interaction(lifecycle.context, outcome)
            if self._agent_task_turn_status(lifecycle) != outcome.status:
                self._fail_unsettled_agent_task_turn(lifecycle)
                raise ConversationLedgerError(
                    f"agent_task turn did not settle as {outcome.status.value}"
                )
        except Exception as exc:
            if outcome.status is not TurnStatus.FAILED:
                self._fail_unsettled_agent_task_turn(lifecycle)
            logger.exception("Unable to settle canonical agent_task turn")
            if isinstance(exc, ConversationLedgerError):
                raise
            raise ConversationLedgerError("Unable to settle canonical agent_task turn") from exc
        lifecycle.settled = True

    def _agent_task_turn_status(self, lifecycle: _AgentTaskTurn) -> TurnStatus | None:
        ledger = getattr(self._pipeline, "_turn_ledger", None)
        get_turn = getattr(ledger, "get_turn", None)
        if not callable(get_turn) or lifecycle.context is None:
            return None
        turn = get_turn(lifecycle.context.turn_id)
        return getattr(turn, "status", None)

    def _fail_unsettled_agent_task_turn(self, lifecycle: _AgentTaskTurn) -> None:
        ledger = getattr(self._pipeline, "_turn_ledger", None)
        fail_turn = getattr(ledger, "fail_turn", None)
        if not callable(fail_turn) or lifecycle.context is None:
            return
        status = self._agent_task_turn_status(lifecycle)
        if status in {TurnStatus.COMMITTED, TurnStatus.CANCELLED, TurnStatus.FAILED}:
            return
        fail_turn(
            lifecycle.context.turn_id,
            reason="agent_task_settlement_failed",
            metadata={
                "interaction": "agent_task",
                "interaction_phase": "final",
                "skill_name": "agent_task",
                "result_code": "agent_task_settlement_failed",
            },
        )

    def _settle_agent_task_result(
        self,
        lifecycle: _AgentTaskTurn,
        result: str,
    ) -> None:
        disposition = self._classify_agent_task_result(result)
        metadata = {
            "interaction": "agent_task",
            "interaction_phase": "final",
            "skill_name": "agent_task",
            "result_code": disposition.code,
        }
        if disposition.status == "succeeded" and result:
            outcome = TurnOutcome.commit(
                assistant_text=result,
                heard_text=result,
                metadata=metadata,
            )
        elif disposition.status == "cancelled":
            outcome = TurnOutcome.cancel(
                reason=disposition.code,
                heard_text="",
                metadata=metadata,
            )
        else:
            outcome = TurnOutcome.fail(
                reason=disposition.code if result else "empty_skill_result",
                metadata=metadata,
            )
        self._settle_agent_task_turn(lifecycle, outcome)

    # ── Public API (called by loops) ──────────────────────────────

    async def dispatch(
        self,
        skill_name: str,
        user_text: str,
        *,
        source: str = "voice",
        extra_context: str = "",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        """Execute a skill and track it in the current mission.

        If no mission is active, one is created automatically.
        Returns the skill result string.
        """
        # Capture the running event loop for use by execute_skill_sync
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        preflight = await self.can_execute(
            skill_name,
            user_text,
            source=source,
        )
        if not preflight.can_execute:
            logger.warning(
                "Dispatch blocked by skill preflight: skill=%s reason=%s",
                skill_name,
                preflight.code,
            )
            await self._speak_outcome(preflight, source=source)
            return preflight.legacy_result

        # Start or continue mission
        if self._current_mission is None:
            self._current_mission = MissionContext(source=source)
            logger.info("Mission started: %s", self._current_mission.mission_id)
        else:
            # Continuing an existing multi-step mission — let the user know
            step_num = self._current_mission.step_count + 1
            _step_def = self._skill_manager.get(skill_name)
            step_label = (
                _step_def.description if _step_def and _step_def.description else skill_name
            )
            await self._speak_voice(f"继续执行第{step_num}步：{step_label}", source)

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
        step_timeout = float(skill_def.timeout) + 10.0 if skill_def else self._STEP_TIMEOUT

        # ── Background execution for agent_task ───────────────────────────────
        # agent_task is a legacy background lane; active agent execution is MCP/ZeroClaw.
        # Running it as a background asyncio.Task keeps VoiceLoop responsive so
        # the user can issue ESTOP or "停下" at any point during execution.
        if skill_name == "agent_task":
            _mission = self._current_mission
            acknowledgement = "好的，我开始处理，会边做边播报进度。"
            try:
                agent_task_turn = await self._record_agent_task_ack(
                    user_text=user_text,
                    assistant_text=acknowledgement,
                    source=source,
                    conversation_session_id=conversation_session_id,
                    voice_turn_id=voice_turn_id,
                    turn_cancel_token=turn_cancel_token,
                )
            except ConversationLedgerError as exc:
                logger.error("agent_task canonical acknowledgement failed: %s", exc)
                _mission.state = MissionState.FAILED
                if self._current_mission is _mission:
                    self._current_mission = None
                return "[Skill] Unavailable: agent_task (canonical_interaction_unavailable)"

            async def _run_agent() -> None:
                try:
                    res = await asyncio.wait_for(
                        self._execute_pipeline_skill(
                            skill_name,
                            user_text,
                            combined_context,
                            source=source,
                            conversation_session_id=conversation_session_id,
                            voice_turn_id=voice_turn_id,
                            turn_cancel_token=turn_cancel_token,
                            record_turn=False,
                        ),
                        timeout=step_timeout,
                    )
                    _mission.add_step(skill_name, user_text, res)
                    _mission.shared_context[skill_name] = res[:500]
                    self._settle_agent_task_result(agent_task_turn, res)
                    if self._current_mission is _mission:
                        self.complete_mission()
                    logger.info("Background agent_task completed: %s", res[:60])
                except asyncio.CancelledError:
                    logger.info("agent_task cancelled")
                    _mission.state = MissionState.CANCELED
                    self._settle_agent_task_turn(
                        agent_task_turn,
                        TurnOutcome.cancel(
                            reason="agent_task_cancelled",
                            metadata={
                                "interaction": "agent_task",
                                "interaction_phase": "final",
                                "skill_name": "agent_task",
                            },
                        ),
                    )
                    if self._current_mission is _mission:
                        self._current_mission = None
                    raise
                except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                    logger.warning("agent_task timed out after %.0fs", step_timeout)
                    _mission.state = MissionState.FAILED
                    self._settle_agent_task_turn(
                        agent_task_turn,
                        TurnOutcome.fail(
                            reason="execution_timeout",
                            metadata={
                                "interaction": "agent_task",
                                "interaction_phase": "final",
                                "skill_name": "agent_task",
                            },
                        ),
                    )
                    if self._current_mission is _mission:
                        self._current_mission = None
                    self._audio.speak("任务超时，已中止。")
                    self._audio.start_playback()
                except Exception as exc:
                    logger.error("Background agent_task failed: %s", exc)
                    _mission.state = MissionState.FAILED
                    self._settle_agent_task_turn(
                        agent_task_turn,
                        TurnOutcome.fail(
                            reason=type(exc).__name__,
                            metadata={
                                "interaction": "agent_task",
                                "interaction_phase": "final",
                                "skill_name": "agent_task",
                            },
                        ),
                    )
                    if self._current_mission is _mission:
                        self._current_mission = None

            self._active_agent_task = asyncio.create_task(_run_agent())
            logger.info("agent_task started as background task (mission=%s)", _mission.mission_id)
            return acknowledgement

        try:
            result = await asyncio.wait_for(
                self._execute_pipeline_skill(
                    skill_name,
                    user_text,
                    combined_context,
                    source=source,
                    conversation_session_id=conversation_session_id,
                    voice_turn_id=voice_turn_id,
                    turn_cancel_token=turn_cancel_token,
                ),
                timeout=step_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            logger.error("Skill '%s' timed out after %.0fs", skill_name, step_timeout)
            result = f"[超时] 技能 {skill_name} 执行超过 {int(step_timeout)} 秒，已中止。"
            if self._current_mission:
                self._current_mission.state = MissionState.FAILED
                self._current_mission.add_step(skill_name, user_text, result)
                self._current_mission.shared_context[skill_name] = result[:500]
                self.complete_mission()
            await self._speak_voice(GENERIC_SKILL_TIMEOUT_MESSAGE, source)
            return result
        except Exception as exc:
            logger.error("Skill '%s' failed: %s", skill_name, exc)
            result = f"[错误] {skill_name}: {exc}"
            if self._current_mission:
                self._current_mission.state = MissionState.FAILED
                self._current_mission.add_step(skill_name, user_text, result)
                self.complete_mission()
            await self._speak_voice(GENERIC_SKILL_FAILURE_MESSAGE, source)
            return result

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
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        """Handle a general (non-skill) turn.

        Tries PlannerAgent first: if the intent decomposes into a multi-step
        skill sequence, executes all steps and returns a combined result.
        Falls back to the LLM pipeline for single-step / conversational input.

        Completes any active mission first (this turn breaks the skill chain).
        """
        self.complete_mission()

        def _turn_cancelled() -> bool:
            return bool(turn_cancel_token is not None and turn_cancel_token.is_set())

        def _cancel_current_mission() -> None:
            if self._current_mission is not None:
                self._current_mission.state = MissionState.CANCELED
                self.complete_mission()

        if _turn_cancelled():
            return ""

        # Attempt multi-step planning before handing off to the LLM.
        # 10 s timeout: if the planner LLM hangs, fall back to normal pipeline.
        _PLANNER_TIMEOUT = 10.0
        if self._planner is not None:
            try:
                planner_context = LLMCallContext(
                    session_id=conversation_session_id,
                    turn_id=voice_turn_id,
                    purpose="general",
                    channel=source if source in {"voice", "text"} else "text",
                    request_class="robot_action",
                    privacy_class="conversation",
                    allow_cache=False,
                )
                steps = await asyncio.wait_for(
                    self._planner.plan(
                        user_text,
                        llm_call_context=planner_context,
                    ),
                    timeout=_PLANNER_TIMEOUT,
                )
            except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                logger.warning("PlannerAgent timed out after %.0fs, falling back", _PLANNER_TIMEOUT)
                steps = None
            except Exception as exc:
                logger.warning("PlannerAgent raised unexpectedly: %s", exc)
                steps = None

            if _turn_cancelled():
                return ""

            if steps:
                # Announce the plan so the user knows what's coming
                _step_labels = [
                    (
                        self._skill_manager.get(s.skill_name).description
                        if self._skill_manager.get(s.skill_name)
                        and self._skill_manager.get(s.skill_name).description
                        else s.skill_name
                    )
                    for s in steps
                ]
                _plan_summary = "、".join(_step_labels)
                await self._speak_voice(f"好的，分{len(steps)}步：{_plan_summary}", source)

                if _turn_cancelled():
                    return ""

                results: list[str] = []
                for step in steps:
                    if _turn_cancelled():
                        _cancel_current_mission()
                        break
                    # Pass the original user request as extra_context so each skill has
                    # full intent beyond the planner's terse per-step instruction.
                    _orig_ctx = f"原始用户请求: {user_text}" if user_text else ""

                    def _start_step_dispatch() -> asyncio.Task[str]:
                        return asyncio.create_task(
                            self.dispatch(
                                step.skill_name,
                                step.intent,
                                source=source,
                                extra_context=_orig_ctx,
                                conversation_session_id=conversation_session_id,
                                voice_turn_id=voice_turn_id,
                                turn_cancel_token=turn_cancel_token,
                            )
                        )

                    atomic_runner = getattr(turn_cancel_token, "try_run", None)
                    if callable(atomic_runner):
                        started, dispatch_task = atomic_runner(_start_step_dispatch)
                        if not started or dispatch_task is None:
                            _cancel_current_mission()
                            break
                        result = await dispatch_task
                    else:
                        if _turn_cancelled():
                            _cancel_current_mission()
                            break
                        result = await self.dispatch(
                            step.skill_name,
                            step.intent,
                            source=source,
                            extra_context=_orig_ctx,
                            conversation_session_id=conversation_session_id,
                            voice_turn_id=voice_turn_id,
                            turn_cancel_token=turn_cancel_token,
                        )
                    results.append(result)
                    if _turn_cancelled():
                        _cancel_current_mission()
                        break
                    # Stop plan execution if a step failed.
                    # The timeout result is stored but execute_skill() was cancelled,
                    # so the user has not heard it yet — speak it explicitly.
                    if (
                        self._current_mission is not None
                        and self._current_mission.state == MissionState.FAILED
                    ):
                        logger.warning(
                            "Plan aborted at step '%s' due to FAILED state", step.skill_name
                        )
                        _step_n = len(results)
                        await self._speak_voice(f"第{_step_n}步执行失败，任务中止。", source)
                        break
                _done_mission = self.complete_mission()
                # Announce plan success and await TTS so it isn't eaten by the
                # next turn's drain_buffers(). Only announce if all steps ran.
                if (
                    not _turn_cancelled()
                    and _done_mission
                    and _done_mission.state != MissionState.CANCELED
                    and len(_done_mission.steps) > 1
                ):
                    _names = "、".join(
                        (
                            self._skill_manager.get(s.skill_name).description
                            if self._skill_manager.get(s.skill_name)
                            and self._skill_manager.get(s.skill_name).description
                            else s.skill_name
                        )
                        for s in _done_mission.steps
                    )
                    await self._speak_voice(f"多步任务完成：{_names}", source)
                return "\n".join(results)

        # Single-step or conversational: delegate to LLM pipeline
        if _turn_cancelled():
            return ""
        pipeline_kwargs: dict[str, Any] = {
            "memory_task": memory_task,
            "source": source,
        }
        if conversation_session_id is not None:
            pipeline_kwargs["conversation_session_id"] = conversation_session_id
        if voice_turn_id is not None:
            pipeline_kwargs["voice_turn_id"] = voice_turn_id
        if turn_cancel_token is not None:
            pipeline_kwargs["turn_cancel_token"] = turn_cancel_token
        return await self._pipeline.process(user_text, **pipeline_kwargs)

    def get_skill(self, skill_name: str):
        """Return the SkillDefinition for a skill name, or None if not found."""
        return self._skill_manager.get(skill_name)

    def complete_mission(self) -> MissionContext | None:
        """End the current mission and return it for logging.

        TTS announcement is intentionally NOT done here because this method is
        synchronous and cannot await TTS completion.  Callers that need to
        announce success (e.g. the plan loop in handle_general) must do so
        after this returns, using asyncio.to_thread(wait_speaking_done).
        This prevents the announcement from being eaten by the next turn's
        drain_buffers() call.
        """
        mission = self._current_mission
        if mission and mission.steps:
            logger.info("Mission completed: %s", mission.summary())
            if len(mission.steps) > 1 and mission.state == MissionState.RUNNING:
                # Record in episodic memory so future turns can recall this mission
                _episodic = getattr(self._pipeline, "_episodic", None)
                if _episodic is not None:
                    _episodic.log("mission_complete", mission.summary())
        if mission:
            if mission.state == MissionState.RUNNING:
                mission.state = MissionState.SUCCEEDED
        self._persist_mission()
        self._last_mission = mission  # preserve for post-completion inspection
        self._current_mission = None
        return mission

    @property
    def last_mission(self) -> MissionContext | None:
        """The most recently completed or failed mission. None if no mission has run."""
        return self._last_mission

    @property
    def has_active_mission(self) -> bool:
        return self._current_mission is not None

    @property
    def has_active_agent_task(self) -> bool:
        """True while a background agent_task is still executing."""
        return self._active_agent_task is not None and not self._active_agent_task.done()

    def cancel_active_agent_task(self) -> bool:
        """Cancel the running background agent task.

        Returns True if a task was cancelled, False if nothing was running.
        Safe to call when no task is active.
        """
        task = self._active_agent_task
        if task and not task.done():
            task.cancel()
            self._active_agent_task = None
            logger.info("Background agent task cancelled by request")
            return True
        self._active_agent_task = None
        return False

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
        # Limit nesting depth — override via ASKME_DISPATCH_MAX_DEPTH env var.
        _MAX_DEPTH = int(os.environ.get("ASKME_DISPATCH_MAX_DEPTH", "3"))
        depth = getattr(self._dispatch_depth, "value", 0)
        if depth >= _MAX_DEPTH:
            logger.warning(
                "dispatch_skill depth=%d >= max=%d, rejecting '%s'", depth, _MAX_DEPTH, skill_name
            )
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
