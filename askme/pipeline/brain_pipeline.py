"""Brain pipeline — memory retrieval → LLM streaming → tool calls → TTS.

TODO(GAP-CORE-1): Decompose into PromptAssembler + LLMTurnExecutor + SkillGate + StreamingTTSBridge.
See docs/LAYER_GAPS.md for the decomposition plan.
Current: ~1093 lines, target: <400 lines after decomposition.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from typing import Any, TYPE_CHECKING

from askme.pipeline.prompt_builder import PromptBuilder
from askme.pipeline.tool_executor import ToolExecutor
from askme.pipeline.trace import get_tracer

if TYPE_CHECKING:
    from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
    from askme.llm.conversation import ConversationManager
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.llm.client import LLMClient
    from askme.memory.bridge import MemoryBridge
    from askme.memory.system import MemorySystem
    from askme.memory.session import SessionMemory
    from askme.perception.vision_bridge import VisionBridge
    from askme.robot.control_client import DogControlClient
    from askme.robot.safety_client import DogSafetyClient
    from askme.robot.arm_controller import ArmController
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


class _ThinkFilter:
    """Strip ``<think>...</think>`` blocks from incremental streaming text.

    MiniMax-M2.5 (and other reasoning models) emit a ``<think>`` block before
    the actual answer.  This filter removes it in O(n) without buffering the
    entire response — safe for real-time TTS piping.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buf.find("</think>")
                if idx < 0:
                    if len(self._buf) > 8:
                        self._buf = self._buf[-8:]
                    return "".join(out)
                self._buf = self._buf[idx + 8:]
                self._in_think = False
            else:
                idx = self._buf.find("<think>")
                if idx < 0:
                    safe = max(0, len(self._buf) - 7)
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    return "".join(out)
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + 7:]
                self._in_think = True

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        r = self._buf
        self._buf = ""
        return r

    def reset(self) -> None:
        self._in_think = False
        self._buf = ""


_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks from a complete string."""
    return _RE_THINK.sub("", text).strip()


class BrainPipeline:
    """Orchestrates one turn of conversation:
    memory → system prompt → streaming LLM → tool calls → TTS.

    All dependencies are injected via the constructor.
    """

    _DEFAULT_MAX_RESPONSE_CHARS = 500
    _THINKING_DELAY = 1.2
    _SILENT_MARKER = "[SILENT]"
    _TOOL_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        llm: LLMClient,
        conversation: ConversationManager,
        memory: MemoryBridge,
        tools: ToolRegistry,
        skill_manager: SkillManager,
        skill_executor: SkillExecutor,
        audio: AudioAgent,
        splitter: StreamSplitter,
        arm_controller: ArmController | None = None,
        dog_safety_client: DogSafetyClient | None = None,
        dog_control_client: DogControlClient | None = None,
        vision: VisionBridge | None = None,
        session_memory: SessionMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        system_prompt: str = (
            "你是穹沛科技的机器人语音助手，搭载在巡检机器人上。"
            "操作员是工程师和现场人员。"
            "说话简洁口语化，像对讲机里的值班员。"
            "短句为主，不超过80字。"
            "不用markdown、emoji、英文。"
            "不确定时说'不确定，需要确认'，不编造信息。"
            "绝不说自己是AI助手或语言模型。"
        ),
        prompt_seed: list[dict[str, str]] | None = None,
        user_prefix: str = "",
        voice_model: str | None = None,
        general_tool_max_safety_level: str = "normal",
        max_response_chars: int = 0,
        agent_shell: ThunderAgentShell | None = None,
        memory_system: MemorySystem | None = None,
        qp_memory: Any = None,
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._memory = memory
        self._mem = memory_system
        self._tools = tools
        self._skill_manager = skill_manager
        self._skill_executor = skill_executor
        self._audio = audio
        self._splitter = splitter
        self._arm = arm_controller
        self._dog_safety = dog_safety_client
        self._dog_control = dog_control_client
        self._vision = vision
        self._session_memory = session_memory
        self._episodic = episodic_memory
        self._base_prompt = system_prompt
        self._prompt_seed = prompt_seed or []
        self._user_prefix = user_prefix
        self._voice_model = voice_model
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._max_response_chars = (
            max_response_chars if max_response_chars > 0
            else self._DEFAULT_MAX_RESPONSE_CHARS
        )
        self._agent_shell = agent_shell
        self._qp_memory = qp_memory
        self._qp_turn_count = 0
        self._last_spoken_text: str = ""
        self._think_filter = _ThinkFilter()
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._llm_semaphore: asyncio.Semaphore | None = None

        self._prompt_builder = PromptBuilder(
            base_prompt=system_prompt,
            prompt_seed=self._prompt_seed,
            user_prefix=user_prefix,
            tools=tools,
            skill_manager=skill_manager,
            general_tool_max_safety_level=general_tool_max_safety_level,
            dog_safety=dog_safety_client,
            episodic=episodic_memory,
            session_memory=session_memory,
            vision=vision,
            qp_memory=qp_memory,
            memory_system=memory_system,
        )
        self._tool_executor = ToolExecutor(
            tools=tools,
            conversation=conversation,
            episodic=episodic_memory,
            general_tool_max_safety_level=general_tool_max_safety_level,
            prompt_builder=self._prompt_builder,
            stream_and_speak=self._stream_and_speak,
        )

    def _log_episode(self, kind: str, text: str) -> None:
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    # ── Public API ───────────────────────────────────────────

    @property
    def last_spoken_text(self) -> str:
        """The most recent text spoken via TTS. Used by repeat_last skill."""
        return self._last_spoken_text

    # ── Late-binding setters (called by modules built after Pipeline) ──

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (set by VoiceModule/TextModule after build)."""
        self._audio = audio

    def set_skill_manager(self, manager: Any) -> None:
        """Late-bind SkillManager (set by SkillModule after build)."""
        self._skill_manager = manager

    def set_skill_executor(self, executor: Any) -> None:
        """Late-bind SkillExecutor (set by SkillModule after build)."""
        self._skill_executor = executor

    def set_agent_shell(self, shell: Any) -> None:
        """Late-bind ThunderAgentShell (set by ExecutorModule after build)."""
        self._agent_shell = shell

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        """Start an idle-time reflection background task (dream consolidation)."""
        _ep = (self._mem.episodic if self._mem is not None else self._episodic)
        if not _ep:
            return None

        async def _idle_reflect() -> None:
            await asyncio.sleep(idle_seconds)
            _should = (
                self._mem.should_reflect() if self._mem is not None
                else (_ep.should_reflect() if _ep else False)
            )
            if not _should:
                return
            sem = self._llm_semaphore
            if sem is not None and sem.locked():
                logger.info("[Dream] Skipping reflection — LLM busy with user turn")
                return
            logger.info("[Dream] Idle-time reflection triggered")
            try:
                if self._mem is not None:
                    summary = await self._mem.reflect()
                else:
                    summary = await _ep.reflect()
                    _ep.cleanup_old_episodes()
                if summary:
                    logger.info("[Dream] Reflection result: %s", summary[:80])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[Dream] Reflection failed: %s", exc)

        return asyncio.create_task(_idle_reflect())

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[str]:
        """Start memory retrieval as a background task. Call ASAP after ASR returns."""
        return asyncio.create_task(self._memory.retrieve(user_text))

    async def process(
        self, user_text: str, *, memory_task: asyncio.Task[str] | None = None,
        source: str = "voice",
    ) -> str:
        """Run the full brain pipeline for *user_text*. Returns assistant reply."""
        logger.info("Processing: %s", user_text[:60])

        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(1)

        self._audio.drain_buffers()

        if self._dog_safety and self._dog_safety.is_configured():
            _estop_t = asyncio.create_task(
                asyncio.to_thread(self._dog_safety.query_estop_state),
                name="estop_refresh",
            )
            self._pending_tasks.add(_estop_t)
            _estop_t.add_done_callback(self._pending_tasks.discard)

        async def _compress_bg() -> None:
            try:
                await self._conversation.maybe_compress(self._llm)
            except Exception as _e:
                logger.warning("Conversation compression failed (non-critical): %s", _e)

        _ct = asyncio.create_task(_compress_bg(), name="conv_compress")
        self._pending_tasks.add(_ct)
        _ct.add_done_callback(self._pending_tasks.discard)

        if not memory_task:
            memory_task = asyncio.create_task(self._memory.retrieve(user_text))
        vision_task = self._start_vision_capture()

        _tracer = get_tracer()
        try:
            with _tracer.span("memory_retrieve"):
                context_str = await memory_task
        except Exception:
            context_str = ""

        scene_desc = ""
        if vision_task:
            try:
                scene_desc = await vision_task
            except Exception:
                scene_desc = ""

        if scene_desc:
            self._log_episode("perception", scene_desc)

        system_prompt = self._prompt_builder.build_system_prompt(
            context_str,
            scene_desc=scene_desc,
            user_text=user_text,
        )

        self._conversation.add_user_message(user_text)
        messages = self._prepare_messages(
            self._conversation.get_messages(system_prompt)
        )

        self._log_episode("command", f"用户说: {user_text}")

        self._audio.start_playback()
        try:
            async with self._llm_semaphore:  # type: ignore[union-attr]
                full_response = await self._stream_with_tools(
                    messages, system_prompt, model=self._voice_model,
                    source=source,
                )
            if self._SILENT_MARKER in full_response:
                logger.info("[SILENT] Not addressed to robot, suppressing output")
                self._audio.drain_buffers()
                if (
                    self._conversation.history
                    and self._conversation.history[-1].get("role") == "user"
                ):
                    self._conversation.history.pop()
                return ""

            self._conversation.add_assistant_message(full_response)
            self._last_spoken_text = full_response

            if self._mem is not None:
                _mt = asyncio.create_task(self._mem.save_to_vector(user_text, full_response))
                self._pending_tasks.add(_mt)
                _mt.add_done_callback(self._pending_tasks.discard)
            elif self._memory is not None:
                _mt = asyncio.create_task(self._memory.save(user_text, full_response))
                self._pending_tasks.add(_mt)
                _mt.add_done_callback(self._pending_tasks.discard)

            if source == "voice":
                await asyncio.to_thread(self._audio.wait_speaking_done)

            self._log_episode("action", f"回复: {full_response[:100]}")

            if self._qp_memory is not None:
                _qp = self._qp_memory
                _resp = full_response

                async def _qp_voice_bg():
                    try:
                        await asyncio.to_thread(_qp.record_observation, "voice", user_text)
                        await asyncio.to_thread(_qp.process_turn, user_text, _resp)
                        if self._qp_turn_count % 10 == 0:
                            await asyncio.to_thread(_qp.save)
                    except Exception:
                        pass

                self._qp_turn_count += 1
                _t = asyncio.create_task(_qp_voice_bg())
                self._pending_tasks.add(_t)
                _t.add_done_callback(self._pending_tasks.discard)

            _should = (
                self._mem.should_reflect() if self._mem is not None
                else (self._episodic.should_reflect() if self._episodic else False)
            )
            if _should:

                async def _delayed_reflect() -> None:
                    await asyncio.sleep(5)
                    if self._mem is not None:
                        await self._mem.reflect()
                    elif self._episodic and self._episodic.should_reflect():
                        try:
                            await self._episodic.reflect()
                        except Exception as e:
                            logger.error("[Episodic] Reflection failed: %s", e)

                t = asyncio.create_task(_delayed_reflect())
                self._pending_tasks.add(t)
                t.add_done_callback(self._pending_tasks.discard)

            return full_response
        except Exception as exc:
            logger.error("LLM pipeline error: %s", exc)
            self._log_episode("error", f"LLM错误: {exc}")
            self._audio.speak(self._classify_error_message(exc))
            error_msg = f"[系统错误] {type(exc).__name__}"
            last_role = (
                self._conversation.history[-1].get("role")
                if self._conversation.history
                else None
            )
            if last_role == "assistant":
                self._conversation.add_user_message("[系统恢复]")
            self._conversation.add_assistant_message(error_msg)
            return error_msg
        finally:
            self._audio.stop_playback()

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

    async def shutdown(self) -> None:
        """Cancel all in-flight background tasks (delayed reflections, etc.)."""
        tasks = list(self._pending_tasks)
        if tasks:
            logger.info("BrainPipeline shutdown: cancelling %d pending tasks", len(tasks))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()

    def handle_estop(self) -> None:
        """Emergency stop — halt local arm motion and notify dog-safety-service."""
        logger.warning("E-STOP triggered!")
        if self._arm:
            self._arm.emergency_stop()
        if self._dog_safety and self._dog_safety.is_configured():
            self._dog_safety.notify_estop()
            logger.warning("E-STOP: notified dog-safety-service")
        logger.warning("E-STOP: local motion halted.")

    def has_pending_tool_approval(self) -> bool:
        """Whether a dangerous tool invocation is waiting for operator approval."""
        return self._tools.has_pending_approval()

    async def handle_pending_tool_response(self, user_text: str) -> str | None:
        """Resolve or restate the pending dangerous tool based on user input."""
        return await self._tool_executor.handle_pending_tool_response(
            user_text, audio=self._audio,
        )

    async def _respond_without_llm(
        self, user_text: str, assistant_text: str, *, source: str = "voice"
    ) -> str:
        """Speak and record a direct response that doesn't need another LLM turn."""
        return await self._tool_executor.respond_without_llm(
            user_text, assistant_text, audio=self._audio, source=source,
        )

    # ── Internal ─────────────────────────────────────────────

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Delegate to PromptBuilder."""
        return self._prompt_builder.prepare_messages(messages)

    def _build_l0_runtime_block(self) -> str:
        """Return a compact L0 runtime truth block from authoritative services.

        Non-blocking: reads only the in-memory cache on DogSafetyClient.
        Returns an empty string when no services are configured.
        """
        if not (self._dog_safety and self._dog_safety.is_configured()):
            return ""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        estop = self._dog_safety.is_estop_active()
        estop_str = "⚠️ 已激活 — 禁止运动指令" if estop else "正常"
        return f"[运行时状态 {ts}]\n急停: {estop_str}"

    def _build_system_prompt(
        self,
        context_str: str | None,
        *,
        scene_desc: str = "",
        user_text: str = "",
    ) -> str:
        """Delegate to PromptBuilder (kept for backward compat with tests)."""
        return self._prompt_builder.build_system_prompt(
            context_str, scene_desc=scene_desc, user_text=user_text,
        )

    def _start_vision_capture(self) -> asyncio.Task[str] | None:
        if not self._vision or not self._vision.available:
            return None
        if not self._vision._vision_cfg.get("auto_capture", False):
            return None
        return asyncio.create_task(self._vision.describe_scene())

    def _create_thinking_task(
        self, include_slow_network: bool = False,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None] | None]:
        async def _thinking_indicator() -> None:
            await asyncio.sleep(self._THINKING_DELAY)
            self._audio.play_thinking()

        thinking_task = asyncio.create_task(_thinking_indicator())

        slow_network_task: asyncio.Task[None] | None = None
        if include_slow_network:
            async def _slow_network_indicator() -> None:
                await asyncio.sleep(5.0)
                self._audio.play_thinking()

            slow_network_task = asyncio.create_task(_slow_network_indicator())

        return thinking_task, slow_network_task

    async def _consume_llm_stream(
        self,
        stream,
        source: str = "voice",
    ) -> tuple[str, dict[int, dict[str, str]]]:
        """Consume LLM stream: apply think filter, feed splitter -> TTS, enforce truncation.

        Returns (full_text, tool_calls_acc).
        """
        full_response = ""
        tool_calls_acc: dict[int, dict[str, str]] = {}
        spoke_any = False

        is_voice = source == "voice"
        chars_spoken = 0
        truncated = False
        char_limit = self._max_response_chars if is_voice else 0

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

                if spoke_any:
                    self._audio.drain_buffers()
                    spoke_any = False

            if delta.content:
                clean = self._think_filter.feed(delta.content)
                if clean:
                    full_response += clean
                    if not truncated:
                        for sentence in self._splitter.feed(clean):
                            if char_limit and chars_spoken + len(sentence) > char_limit:
                                self._audio.speak(sentence)
                                self._audio.speak("还有更多内容，说继续我就接着说。")
                                spoke_any = True
                                truncated = True
                                logger.info(
                                    "Voice truncation at %d chars (limit %d)",
                                    chars_spoken + len(sentence), char_limit,
                                )
                                break
                            self._audio.speak(sentence)
                            chars_spoken += len(sentence)
                            spoke_any = True

        think_tail = self._think_filter.flush()
        if think_tail:
            full_response += think_tail
            if not truncated:
                for sentence in self._splitter.feed(think_tail):
                    if char_limit and chars_spoken + len(sentence) > char_limit:
                        self._audio.speak(sentence)
                        spoke_any = True
                        truncated = True
                        break
                    self._audio.speak(sentence)
                    chars_spoken += len(sentence)
                    spoke_any = True
        if not truncated:
            remainder = self._splitter.flush()
            if remainder:
                self._audio.speak(remainder)
                spoke_any = True

        return full_response, tool_calls_acc

    async def _stream_with_tools(
        self, messages: list[dict[str, Any]], system_prompt: str,
        model: str | None = None, source: str = "voice",
    ) -> str:
        """Stream LLM response, speak sentences immediately, handle tool calls."""
        import time as _time

        tool_definitions = self._tools.get_definitions(
            max_safety_level=self._general_tool_max_safety_level
        )
        tool_names = [td.get("function", {}).get("name") for td in tool_definitions]
        logger.info("LLM tools available (%d): %s", len(tool_definitions), tool_names)
        ttft_logged = False
        t_start = _time.perf_counter()
        self._splitter.reset()
        self._think_filter.reset()

        is_voice = source == "voice"

        thinking_task: asyncio.Task[None] | None = None
        slow_network_task: asyncio.Task[None] | None = None
        if is_voice:
            thinking_task, slow_network_task = self._create_thinking_task(
                include_slow_network=True,
            )

        try:
            async def _ttft_stream():
                nonlocal ttft_logged, thinking_task, slow_network_task
                async for chunk in self._llm.chat_stream(
                    messages, tools=tool_definitions, tool_choice="auto", model=model,
                ):
                    if not ttft_logged:
                        ttft_logged = True
                        elapsed = _time.perf_counter() - t_start
                        logger.info("TTFT: %.2fs", elapsed)
                        get_tracer().record_span("ttft", elapsed * 1000, model=model or "default")
                        if thinking_task is not None:
                            thinking_task.cancel()
                            thinking_task = None
                        if slow_network_task is not None:
                            slow_network_task.cancel()
                            slow_network_task = None
                    yield chunk

            full_response, tool_calls_acc = await self._consume_llm_stream(
                _ttft_stream(), source=source,
            )
        finally:
            if thinking_task is not None:
                thinking_task.cancel()
            if slow_network_task is not None:
                slow_network_task.cancel()

        if tool_calls_acc:
            self._audio.drain_buffers()
            full_response = await self._tool_executor.execute_tools(
                tool_calls_acc, system_prompt, model=model, source=source,
            )

        return full_response

    async def _stream_and_speak(
        self, messages: list[dict[str, Any]], model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        self._splitter.reset()
        self._think_filter.reset()
        full_response, _ = await self._consume_llm_stream(
            self._llm.chat_stream(messages, model=model), source=source,
        )
        return full_response

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

    def _classify_error_message(self, exc: Exception) -> str:
        """Return a user-facing voice message for an LLM pipeline error."""
        try:
            from openai import APIConnectionError, APITimeoutError
            if isinstance(exc, (asyncio.TimeoutError, APITimeoutError)):
                return "想了一下没想出来，你再说一遍？"
            if isinstance(exc, APIConnectionError):
                return "网络有点问题，基本功能还在。"
        except ImportError:
            pass
        if "timeout" in str(exc).lower():
            return "响应超时，请再说一遍。"
        if "connect" in str(exc).lower() or "network" in str(exc).lower():
            return "网络连接异常，请稍候重试。"
        return "处理出错，请重试。"

    def _classify_skill_error_message(self, exc: Exception, skill_name: str) -> str:
        """Return a user-facing voice message for a skill execution error."""
        if isinstance(exc, asyncio.TimeoutError):
            return f"{skill_name}执行超时，跳过了。要不要换个方式？"
        if "connect" in str(exc).lower() or "network" in str(exc).lower():
            return f"网络有问题，{skill_name}暂时做不了。"
        return f"{skill_name}执行失败，要不要重试？"

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
