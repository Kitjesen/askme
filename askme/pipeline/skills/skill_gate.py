"""Skill execution gate -safety checks, context assembly, and compat routing."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time as _time
from typing import TYPE_CHECKING, Any

from askme.llm.core.contracts import LLMCallContext
from askme.pipeline.core.hooks import PipelineHooks, ToolCallRecord
from askme.pipeline.core.protocols import SkillExecutionDisposition
from askme.pipeline.core.trace import get_tracer
from askme.pipeline.core.utils import classify_skill_error, strip_think_blocks
from askme.pipeline.skills.outcome import (
    GENERIC_SKILL_UNAVAILABLE_MESSAGE,
    NAV_LOCATION_UNAVAILABLE_MESSAGE,
    SkillOutcome,
    SkillOutcomeStatus,
)
from askme.skills.governance.audit import SkillAuditLog

if TYPE_CHECKING:
    from askme.agent_shell import AgentShell
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.system import MemorySystem
    from askme.ports import ArmControlPort, AudioFrontendPort, RobotControlPort, SafetyPort
    from askme.skills.core.skill_executor import SkillExecutor
    from askme.skills.core.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_NAV_PREFLIGHT_CACHE_S = 1.5


class SkillGate:
    """Skill execution gate -safety checks, context assembly, and compat routing."""

    def __init__(
        self,
        *,
        skill_manager: SkillManager,
        skill_executor: SkillExecutor,
        audio: AudioFrontendPort,
        conversation: ConversationManager,
        dog_safety: SafetyPort | None = None,
        dog_control: RobotControlPort | None = None,
        arm_controller: ArmControlPort | None = None,
        episodic: EpisodicMemory | None = None,
        memory_system: MemorySystem | None = None,
        agent_shell: AgentShell | None = None,
        prompt_seed: list[dict[str, str]] | None = None,
        max_response_chars: int = 500,
        cancel_token: asyncio.Event | None = None,
        hooks: PipelineHooks | None = None,
    ) -> None:
        self._skill_manager = skill_manager
        self._skill_executor = skill_executor
        self._audio = audio
        self._conversation = conversation
        self._dog_safety = dog_safety
        self._dog_control = dog_control
        self._arm = arm_controller
        self._episodic = episodic
        self._mem = memory_system
        self._agent_shell = agent_shell
        self._prompt_seed = prompt_seed
        self._max_response_chars = max_response_chars
        self._last_spoken_text = ""
        self._cancel_token = cancel_token
        self._hooks = hooks
        self._audit = SkillAuditLog()
        self._preflight_cache: dict[str, tuple[float, SkillOutcome]] = {}

    # Helpers

    def _log_episode(self, kind: str, text: str) -> None:
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    def classify_execution_result(
        self,
        result: str,
        *,
        skill_name: str = "",
    ) -> SkillExecutionDisposition:
        """Map the implementation outcome onto the core settlement contract."""

        if str(result or "").lstrip().startswith("[安全锁定]"):
            return SkillExecutionDisposition(
                status="failed",
                code="estop_active",
            )
        outcome = SkillOutcome.from_legacy_result(result, skill_name=skill_name)
        if outcome.status is SkillOutcomeStatus.SUCCEEDED:
            return SkillExecutionDisposition(
                status="succeeded",
                code=outcome.code,
            )
        if outcome.status is SkillOutcomeStatus.CANCELLED:
            return SkillExecutionDisposition(
                status="cancelled",
                code=outcome.code,
            )
        return SkillExecutionDisposition(
            status="failed",
            code=outcome.code,
        )

    def _record_conversation(
        self,
        user_text: str,
        assistant_text: str,
        *,
        conversation_session_id: str | None,
    ) -> None:
        """Project one skill exchange once, preserving the legacy call shape."""

        if conversation_session_id is None:
            self._conversation.add_user_message(user_text)
            self._conversation.add_assistant_message(assistant_text)
            return
        self._conversation.add_user_message(
            user_text,
            conversation_session_id=conversation_session_id,
        )
        self._conversation.add_assistant_message(
            assistant_text,
            conversation_session_id=conversation_session_id,
        )

    @staticmethod
    def _should_project_conversation(skill_name: str) -> bool:
        """Whether this skill should write the legacy prompt-history projection."""

        return skill_name != "agent_task"

    @staticmethod
    def _llm_call_context(
        *,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        source: str,
        turn_cancel_token: Any | None,
    ) -> LLMCallContext:
        channel = source if source in {"voice", "text"} else "text"
        context = LLMCallContext(
            session_id=conversation_session_id,
            turn_id=voice_turn_id,
            purpose="tool_followup",
            channel=channel,
            request_class="robot_action",
            privacy_class="conversation",
            allow_cache=False,
        )
        if turn_cancel_token is not None:
            object.__setattr__(context, "cancel_token", turn_cancel_token)
        return context

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        """Prepare agent result for TTS + conversation storage."""
        _AGENT_TTS_LIMIT = self._max_response_chars or 200
        if len(result) <= _AGENT_TTS_LIMIT:
            return result, result

        boundary = 0
        for ch in "。！？?":
            idx = result.rfind(ch, 0, _AGENT_TTS_LIMIT)
            if idx > boundary:
                boundary = idx + 1
        if boundary == 0:
            boundary = _AGENT_TTS_LIMIT

        spoken = result[:boundary].rstrip() + " 完整结果已保存到工作区。"

        try:
            workspace = self._agent_shell._workspace if self._agent_shell else None
            if workspace:
                workspace.mkdir(parents=True, exist_ok=True)
                # Resolve and contain: prevent path traversal via crafted result text
                target = (workspace / "last_result.txt").resolve()
                if target.parent.resolve() == workspace.resolve():
                    target.write_text(result, encoding="utf-8")
                else:
                    logger.warning("[SkillGate] Path traversal blocked for last_result.txt")
        except Exception:
            logger.exception("[SkillGate] Workspace result save failed")

        return spoken, result

    def extract_semantic_target(self, user_text: str) -> str:
        """Extract navigation target from natural language commands."""
        text = str(user_text or "").strip()
        if not text:
            return text

        prefixes = ("导航到", "带我去", "前往", "走到", "去")
        filler_suffixes = ("好的", "可以", "那里", "这边", "一下", "嗯", "好", "吧", "啊")

        for prefix in prefixes:
            if not text.startswith(prefix):
                continue
            target = text[len(prefix) :].strip()
            target = re.split(r"[。！？?!，,；;]", target, maxsplit=1)[0].strip()
            changed = True
            while changed:
                changed = False
                for suffix in filler_suffixes:
                    if target.endswith(suffix) and len(target) > len(suffix):
                        target = target[: -len(suffix)].strip()
                        changed = True
                        break
            if target:
                return target
        return user_text

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        """Return a user-facing voice message for a skill execution error."""
        return classify_skill_error(exc, skill_name)

    def _create_thinking_task(self) -> tuple[asyncio.Task[None], None]:
        async def _thinking_indicator() -> None:
            await asyncio.sleep(1.2)
            self._audio.play_thinking()

        return asyncio.create_task(_thinking_indicator()), None

    async def can_execute(
        self,
        skill_name: str,
        user_text: str = "",
        *,
        source: str = "voice",
    ) -> SkillOutcome:
        """Return a side-effect-free readiness outcome for preface gating."""

        del user_text, source
        if self._cancel_token is not None and self._cancel_token.is_set():
            return SkillOutcome(
                SkillOutcomeStatus.CANCELLED,
                "cancel_token",
                result="",
            )

        skill = self._skill_manager.get(skill_name)
        if skill is None:
            return SkillOutcome.blocked(
                code="not_found",
                result=f"[Skill] Not found: {skill_name}",
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

        if self._dog_safety and self._dog_safety.is_configured():
            estop_state = await asyncio.to_thread(self._dog_safety.query_estop_state)
            if estop_state is not None and estop_state.get("enabled"):
                message = f"急停已激活，无法执行 {skill_name}。请先解除急停。"
                return SkillOutcome.blocked(
                    code="estop_active",
                    result=f"[安全锁定] {message}",
                    user_message=message,
                )

        if skill_name == "nav_query":
            cached = self._preflight_cache.get(skill_name)
            if cached is not None and cached[0] > _time.monotonic():
                return cached[1]
            preflight_impl = getattr(type(self._skill_executor), "preflight_skill", None)
            if callable(preflight_impl):
                try:
                    ready, reason = await preflight_impl(self._skill_executor, skill)
                except Exception as exc:
                    logger.warning("[SkillGate] nav preflight failed closed: %s", exc)
                    ready, reason = False, "nav_preflight_error"
                if not ready:
                    outcome = SkillOutcome.blocked(
                        code=str(reason or "nav_not_ready"),
                        result=f"[Skill] Unavailable: {skill_name} ({reason})",
                        user_message=NAV_LOCATION_UNAVAILABLE_MESSAGE,
                    )
                    self._preflight_cache[skill_name] = (
                        _time.monotonic() + _NAV_PREFLIGHT_CACHE_S,
                        outcome,
                    )
                    return outcome

        outcome = SkillOutcome.ready()
        if skill_name == "nav_query":
            self._preflight_cache[skill_name] = (
                _time.monotonic() + _NAV_PREFLIGHT_CACHE_S,
                outcome,
            )
        return outcome

    async def _speak_outcome(self, outcome: SkillOutcome, *, source: str) -> None:
        """Speak only the customer-safe field, exactly once, for voice calls."""

        if source != "voice" or not outcome.should_speak or not outcome.user_message:
            return
        message = outcome.user_message
        speak_and_wait = getattr(self._audio, "speak_and_wait", None)
        declared_speak_and_wait = getattr(type(self._audio), "speak_and_wait", None)
        if callable(speak_and_wait) and callable(declared_speak_and_wait):
            await speak_and_wait(message)
        else:
            self._audio.speak(message)
            self._audio.start_playback()
            await asyncio.to_thread(self._audio.wait_speaking_done)
            self._audio.stop_playback()
        self._last_spoken_text = message

    # Core

    async def execute_skill(
        self,
        skill_name: str,
        user_text: str,
        extra_context: str = "",
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: Any | None = None,
    ) -> str:
        """Execute a named skill and speak the result."""
        audit_start = _time.perf_counter()
        preflight = await self.can_execute(skill_name, user_text, source=source)
        skill = self._skill_manager.get(skill_name)
        if not preflight.can_execute:
            logger.warning(
                "[SkillGate] skill '%s' blocked by preflight: %s",
                skill_name,
                preflight.code,
            )
            audit_fields = {
                "skill_name": skill_name,
                "status": "blocked",
                "user_text": user_text,
                "source": source,
                "reason": preflight.code,
            }
            if skill is not None:
                audit_fields["safety_level"] = skill.safety_level
                audit_fields["execution"] = skill.execution
            self._audit.append(**audit_fields)
            await self._speak_outcome(preflight, source=source)
            return preflight.legacy_result

        # A ready outcome guarantees the definition was present and enabled.
        if skill is None:  # pragma: no cover - defensive against a racy manager swap
            return ""

        if skill.depends:
            for dep in skill.depends:
                dep_skill = self._skill_manager.get(dep)
                if dep_skill is None:
                    logger.warning(
                        "Skill '%s' depends on '%s' which is not available",
                        skill_name,
                        dep,
                    )

        logger.info("Executing skill: %s", skill_name)

        _skill_def = self._skill_manager.get(skill_name)
        _is_agent_shell = _skill_def is not None and _skill_def.execution == "agent_shell"
        if _is_agent_shell and self._agent_shell is not None:
            logger.info("[AgentShell] Routing agent_task to deprecated AgentShell compat stub")
            self._audio.drain_buffers()
            self._audio.start_playback()
            try:
                _now = datetime.datetime.now()
                _agent_timeout = getattr(self._agent_shell, "_default_timeout", 120.0)
                result = await self._agent_shell.run_task(
                    user_text,
                    context={
                        "current_time": _now.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    timeout=_agent_timeout,
                )
                result = strip_think_blocks(result)
                spoken, stored = self._prepare_agent_result(result)
                self._last_spoken_text = spoken
                if self._should_project_conversation(skill_name):
                    self._record_conversation(
                        user_text,
                        stored,
                        conversation_session_id=conversation_session_id,
                    )
                self._log_episode("outcome", f"{skill_name}完成: {result[:100]}")
                if source == "voice":
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                self._audit.append(
                    skill_name=skill_name,
                    status="succeeded",
                    user_text=user_text,
                    source=source,
                    safety_level=skill.safety_level,
                    execution=skill.execution,
                    elapsed_ms=(_time.perf_counter() - audit_start) * 1000,
                    result_preview=result,
                )
                return result
            except Exception as exc:
                logger.error("[AgentShell] %s failed: %s", skill_name, exc)
                self._audio.speak(f"任务执行出错：{exc}")
                self._audit.append(
                    skill_name=skill_name,
                    status="failed",
                    user_text=user_text,
                    source=source,
                    safety_level=skill.safety_level,
                    execution=skill.execution,
                    elapsed_ms=(_time.perf_counter() - audit_start) * 1000,
                    reason=str(exc),
                )
                return f"[AgentShell Error] {exc}"
            finally:
                self._audio.stop_playback()

        self._audio.drain_buffers()
        self._log_episode("action", f"执行技能: {skill_name}")

        _now = datetime.datetime.now()
        context: dict[str, str] = {
            "user_input": user_text,
            "current_time": _now.strftime("%Y-%m-%d %H:%M:%S"),
            "current_date": _now.strftime("%Y-%m-%d"),
            "semantic_target": self.extract_semantic_target(user_text),
        }
        if extra_context:
            context["mission_context"] = extra_context
        if self._arm:
            # arm.get_state() may call hardware registers; run in thread pool.
            context["robot_state"] = json.dumps(
                await asyncio.to_thread(self._arm.get_state), ensure_ascii=False
            )

        if skill_name == "dog_control" and self._dog_control and self._dog_control.is_configured():
            _capability_map = {
                "站起来": "stand",
                "站立": "stand",
                "坐下": "sit",
                "趴下": "sit",
            }
            for phrase, capability in _capability_map.items():
                if phrase in user_text:
                    logger.info(
                        "[DogControl] Dispatching capability '%s' for phrase '%s'",
                        capability,
                        phrase,
                    )
                    dispatch_result = await asyncio.to_thread(
                        self._dog_control.dispatch_capability, capability, {}
                    )
                    if "error" in dispatch_result:
                        logger.warning(
                            "[DogControl] Capability dispatch failed: %s",
                            dispatch_result["error"],
                        )
                    break

        _ep = self._mem.episodic if self._mem is not None else self._episodic
        if skill_name == "patrol_report" and _ep:
            parts = [
                _ep.get_recent_digest(),
                _ep.get_knowledge_context(),
            ]
            patrol_data = "\n".join(p for p in parts if p)
            context["patrol_data"] = patrol_data or ""

        self._audio.start_playback()
        thinking_task: asyncio.Task[None] | None = None
        try:
            thinking_task, _ = self._create_thinking_task()

            _hooks = self._hooks

            def _on_tool_call(tool_name: str) -> None:
                """Called synchronously when a sub-tool fires within the skill."""
                logger.debug("[SkillGate] sub-tool invoked: %s", tool_name)
                if _hooks and _hooks.pre_tool:
                    # Fire pre_tool hooks as a fire-and-forget task so the
                    # synchronous callback doesn't block the skill executor.
                    probe = ToolCallRecord(
                        call_id="",
                        tool_name=tool_name,
                        arguments="",
                        result="",
                        elapsed_ms=0.0,
                    )
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(_hooks.fire_pre_tool(probe))
                    except RuntimeError:
                        pass  # No running loop; hooks can't fire here.

            t0 = _time.perf_counter()
            with get_tracer().span(f"skill.{skill_name}", skill=skill_name):
                raw_result = await self._skill_executor.execute(
                    skill,
                    context,
                    prompt_seed=self._prompt_seed or None,
                    on_tool_call=_on_tool_call,
                    llm_call_context=self._llm_call_context(
                        conversation_session_id=conversation_session_id,
                        voice_turn_id=voice_turn_id,
                        source=source,
                        turn_cancel_token=turn_cancel_token,
                    ),
                )
            elapsed_ms = (_time.perf_counter() - t0) * 1000

            if thinking_task is not None:
                thinking_task.cancel()
                thinking_task = None
            result = strip_think_blocks(raw_result)
            logger.info("Skill result [%.0fms]: %s", elapsed_ms, result[:100])

            # post_tool hook may transform the skill result.
            if self._hooks and self._hooks.post_tool:
                record = ToolCallRecord(
                    call_id="",
                    tool_name=skill_name,
                    arguments=user_text,
                    result=result,
                    elapsed_ms=elapsed_ms,
                )
                result = await self._hooks.fire_post_tool(record)

            execution_outcome = SkillOutcome.from_legacy_result(
                result,
                skill_name=skill_name,
            )
            if execution_outcome.status in {
                SkillOutcomeStatus.FAILED,
                SkillOutcomeStatus.TIMED_OUT,
            }:
                await self._speak_outcome(execution_outcome, source=source)
                if self._should_project_conversation(skill_name):
                    self._record_conversation(
                        user_text,
                        execution_outcome.user_message,
                        conversation_session_id=conversation_session_id,
                    )
                self._log_episode(
                    "error",
                    f"技能返回内部错误 {skill_name}: {result[:100]}",
                )
                self._audit.append(
                    skill_name=skill_name,
                    status="failed",
                    user_text=user_text,
                    source=source,
                    safety_level=skill.safety_level,
                    execution=skill.execution,
                    elapsed_ms=(_time.perf_counter() - audit_start) * 1000,
                    reason=execution_outcome.code,
                    result_preview=result,
                )
                return result

            self._audio.speak(result)
            self._last_spoken_text = result
            if self._should_project_conversation(skill_name):
                self._record_conversation(
                    user_text,
                    result,
                    conversation_session_id=conversation_session_id,
                )
            self._log_episode("outcome", f"直接回复: {result[:100]}")
            if source == "voice":
                await asyncio.to_thread(self._audio.wait_speaking_done)
            self._audit.append(
                skill_name=skill_name,
                status="succeeded",
                user_text=user_text,
                source=source,
                safety_level=skill.safety_level,
                execution=skill.execution,
                elapsed_ms=(_time.perf_counter() - audit_start) * 1000,
                result_preview=result,
            )
            return result
        except Exception as exc:
            logger.error("Skill error (%s): %s", skill_name, exc)
            self._log_episode("error", f"技能错误 {skill_name}: {exc}")
            self._audio.speak(self._classify_skill_error_message(exc, skill_name))
            self._audit.append(
                skill_name=skill_name,
                status="failed",
                user_text=user_text,
                source=source,
                safety_level=skill.safety_level,
                execution=skill.execution,
                elapsed_ms=(_time.perf_counter() - audit_start) * 1000,
                reason=str(exc),
            )
            return f"[Skill Error] {exc}"
        finally:
            if thinking_task is not None:
                thinking_task.cancel()
            self._audio.stop_playback()

    # Late-binding setters

    def set_audio(self, audio: AudioFrontendPort) -> None:
        self._audio = audio

    def set_skill_manager(self, manager: SkillManager) -> None:
        self._skill_manager = manager

    def set_skill_executor(self, executor: SkillExecutor) -> None:
        self._skill_executor = executor

    def set_agent_shell(self, shell: AgentShell) -> None:
        self._agent_shell = shell

    # Properties

    @property
    def last_spoken_text(self) -> str:
        return self._last_spoken_text
