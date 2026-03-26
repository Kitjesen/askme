"""Skill execution gate — safety checks, context assembly, AgentShell routing."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from typing import Any, TYPE_CHECKING

_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks from a complete string."""
    return _RE_THINK.sub("", text).strip()

if TYPE_CHECKING:
    from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
    from askme.llm.conversation import ConversationManager
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.system import MemorySystem
    from askme.robot.arm_controller import ArmController
    from askme.robot.control_client import DogControlClient
    from askme.robot.safety_client import DogSafetyClient
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


class SkillGate:
    """Skill execution gate — safety checks, context assembly, AgentShell routing."""

    def __init__(
        self,
        *,
        skill_manager: SkillManager,
        skill_executor: SkillExecutor,
        audio: AudioAgent,
        conversation: ConversationManager,
        dog_safety: DogSafetyClient | None = None,
        dog_control: DogControlClient | None = None,
        arm_controller: ArmController | None = None,
        episodic: EpisodicMemory | None = None,
        memory_system: MemorySystem | None = None,
        agent_shell: ThunderAgentShell | None = None,
        prompt_seed: str | None = None,
        max_response_chars: int = 500,
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

    # ── Helpers ───────────────────────────────────────────────

    def _log_episode(self, kind: str, text: str) -> None:
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        """Prepare agent result for TTS + conversation storage."""
        _AGENT_TTS_LIMIT = self._max_response_chars or 200
        if len(result) <= _AGENT_TTS_LIMIT:
            return result, result

        boundary = 0
        for ch in "。！？!?":
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
                (workspace / "last_result.txt").write_text(result, encoding="utf-8")
        except Exception:
            pass

        return spoken, result

    def extract_semantic_target(self, user_text: str) -> str:
        """Extract navigation target from natural language commands."""
        patterns = [
            r"导航到(.{1,20}?)(?:吧|啊|嗯|[。！？]|$)",
            r"带我去(.{1,20}?)(?:吧|啊|嗯|[。！？]|$)",
            r"前往(.{1,20}?)(?:吧|啊|嗯|[。！？]|$)",
            r"走到(.{1,20}?)(?:吧|啊|嗯|[。！？]|$)",
            r"去(.{1,20}?)(?:吧|啊|嗯|[。！？]|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, user_text)
            if m:
                target = m.group(1).strip()
                if target:
                    return target
        return user_text

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        """Return a user-facing voice message for a skill execution error."""
        if isinstance(exc, asyncio.TimeoutError):
            return f"{skill_name}执行超时，跳过了。要不要换个方式？"
        if "connect" in str(exc).lower() or "network" in str(exc).lower():
            return f"网络有问题，{skill_name}暂时做不了。"
        return f"{skill_name}执行失败，要不要重试？"

    def _create_thinking_task(self) -> tuple[asyncio.Task[None], None]:
        async def _thinking_indicator() -> None:
            await asyncio.sleep(1.2)
            self._audio.play_thinking()
        return asyncio.create_task(_thinking_indicator()), None

    # ── Core ──────────────────────────────────────────────────

    async def execute_skill(
        self, skill_name: str, user_text: str, extra_context: str = "",
        source: str = "voice",
    ) -> str:
        """Execute a named skill and speak the result."""
        skill = self._skill_manager.get(skill_name)
        if not skill:
            return f"[Skill] Not found: {skill_name}"

        if self._dog_safety and self._dog_safety.is_configured():
            estop_state = await asyncio.to_thread(self._dog_safety.query_estop_state)
            if estop_state is not None and estop_state.get("enabled"):
                msg = f"[安全锁定] 急停已激活，无法执行 {skill_name}。请先解除急停。"
                logger.warning("Safety gate blocked skill '%s': estop is active", skill_name)
                return msg

        if skill.depends:
            for dep in skill.depends:
                dep_skill = self._skill_manager.get(dep)
                if dep_skill is None:
                    logger.warning(
                        "Skill '%s' depends on '%s' which is not available",
                        skill_name, dep,
                    )

        logger.info("Executing skill: %s", skill_name)

        _skill_def = self._skill_manager.get(skill_name)
        _is_agent_shell = (
            _skill_def is not None and _skill_def.execution == "agent_shell"
        )
        if _is_agent_shell and self._agent_shell is not None:
            logger.info("[AgentShell] Routing agent_task to ThunderAgentShell")
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
                self._conversation.add_user_message(user_text)
                self._conversation.add_assistant_message(stored)
                self._log_episode("outcome", f"{skill_name}完成: {result[:100]}")
                if source == "voice":
                    await asyncio.to_thread(self._audio.wait_speaking_done)
                return result
            except Exception as exc:
                logger.error("[AgentShell] %s failed: %s", skill_name, exc)
                self._audio.speak(f"任务执行出错：{exc}")
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
            context["robot_state"] = json.dumps(
                self._arm.get_state(), ensure_ascii=False
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
                        capability, phrase,
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

        _ep = (self._mem.episodic if self._mem is not None else self._episodic)
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

            def _on_tool_call(tool_name: str) -> None:
                pass

            raw_result = await self._skill_executor.execute(
                skill, context, prompt_seed=self._prompt_seed or None,
                on_tool_call=_on_tool_call,
            )
            if thinking_task is not None:
                thinking_task.cancel()
                thinking_task = None
            result = strip_think_blocks(raw_result)
            logger.info("Skill result: %s", result[:100])
            self._audio.speak(result)
            self._last_spoken_text = result
            self._conversation.add_user_message(user_text)
            self._conversation.add_assistant_message(result)
            self._log_episode("outcome", f"直接回复: {result[:100]}")
            if source == "voice":
                await asyncio.to_thread(self._audio.wait_speaking_done)
            return result
        except Exception as exc:
            logger.error("Skill error (%s): %s", skill_name, exc)
            self._log_episode("error", f"技能错误 {skill_name}: {exc}")
            self._audio.speak(self._classify_skill_error_message(exc, skill_name))
            return f"[Skill Error] {exc}"
        finally:
            if thinking_task is not None:
                thinking_task.cancel()
            self._audio.stop_playback()

    # ── Late-binding setters ──────────────────────────────────

    def set_audio(self, audio: AudioAgent) -> None:
        self._audio = audio

    def set_skill_manager(self, manager: SkillManager) -> None:
        self._skill_manager = manager

    def set_skill_executor(self, executor: SkillExecutor) -> None:
        self._skill_executor = executor

    def set_agent_shell(self, shell: ThunderAgentShell) -> None:
        self._agent_shell = shell

    # ── Properties ────────────────────────────────────────────

    @property
    def last_spoken_text(self) -> str:
        return self._last_spoken_text
