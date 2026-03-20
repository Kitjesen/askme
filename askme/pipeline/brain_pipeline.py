"""Brain pipeline — memory retrieval → LLM streaming → tool calls → TTS."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from typing import Any, TYPE_CHECKING

from askme.pipeline.trace import get_tracer

if TYPE_CHECKING:
    from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
    from askme.brain.conversation import ConversationManager
    from askme.brain.episodic_memory import EpisodicMemory
    from askme.brain.llm_client import LLMClient
    from askme.brain.memory_bridge import MemoryBridge
    from askme.brain.memory_system import MemorySystem
    from askme.brain.session_memory import SessionMemory
    from askme.brain.vision_bridge import VisionBridge
    from askme.dog_control_client import DogControlClient
    from askme.dog_safety_client import DogSafetyClient
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

    # Default max response chars for voice mode (0 = unlimited)
    _DEFAULT_MAX_RESPONSE_CHARS = 200
    # Seconds to wait before playing thinking indicator
    _THINKING_DELAY = 1.2
    # Marker LLM returns when it determines the user is not talking to the robot
    _SILENT_MARKER = "[SILENT]"
    # Maximum time (seconds) to wait for a single tool execution.
    # dispatch_skill has its own inner timeout; this is a safety net for
    # other tools (get_time, http_request, etc.) that may hang.
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
        system_prompt: str = "你是一个有用的AI语音助手。用中文简洁口语化回答。",
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
        self._mem = memory_system  # unified facade (preferred over individual components)
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
        self._voice_model = voice_model  # fast model for real-time voice turns
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._max_response_chars = (
            max_response_chars if max_response_chars > 0
            else self._DEFAULT_MAX_RESPONSE_CHARS
        )
        self._agent_shell = agent_shell
        self._qp_memory = qp_memory  # qp_memory.Memory instance (optional)
        self._qp_turn_count = 0  # counter for periodic qp_memory saves
        # Last spoken text — for "repeat last" voice command
        self._last_spoken_text: str = ""
        self._think_filter = _ThinkFilter()
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        # Semaphore(1) ensures reflection and user LLM calls never run
        # concurrently — they share the same API quota.  Reflection always
        # runs with try_acquire (non-blocking) so it never blocks the user.
        self._llm_semaphore: asyncio.Semaphore | None = None  # lazy-init in async context

    def _log_episode(self, kind: str, text: str) -> None:
        """Log to episodic memory via unified facade or direct fallback."""
        if self._mem is not None:
            self._mem.log_event(kind, text)
        elif self._episodic:
            self._episodic.log(kind, text)

    # ── Public API ───────────────────────────────────────────

    @property
    def last_spoken_text(self) -> str:
        """The most recent text spoken via TTS. Used by repeat_last skill."""
        return self._last_spoken_text

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        """Start an idle-time reflection background task (dream consolidation).

        Waits *idle_seconds*, then triggers episodic memory reflection if
        conditions are met. Cancel the returned task when user input arrives.
        Inspired by Letta sleep-time compute and sleep consolidation research.
        """
        if not self._episodic:
            return None

        async def _idle_reflect() -> None:
            await asyncio.sleep(idle_seconds)
            if not (self._episodic and self._episodic.should_reflect()):
                return
            # Only reflect when the LLM is idle — skip if a user turn is using it.
            sem = self._llm_semaphore
            if sem is not None and sem.locked():
                logger.info("[Dream] Skipping reflection — LLM busy with user turn")
                return
            logger.info("[Dream] Idle-time reflection triggered")
            try:
                summary = await self._episodic.reflect()
                if summary:
                    logger.info("[Dream] Reflection result: %s", summary[:80])
                self._episodic.cleanup_old_episodes()
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

        # Lazy-init semaphore (requires running event loop)
        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(1)

        # 0. Clear leftover audio from any previous turn
        self._audio.drain_buffers()

        # 0b. Background estop state refresh — keeps L0 block in system prompt current.
        # Fires a daemon thread via to_thread; result goes into DogSafetyClient cache.
        # Never blocks this turn — the cache may be 0–30 s stale but is always available.
        if self._dog_safety and self._dog_safety.is_configured():
            asyncio.create_task(
                asyncio.to_thread(self._dog_safety.query_estop_state),
                name="estop_refresh",
            )

        # 0c. Sliding window compression — fire-and-forget so it never blocks
        # the user's turn. The compressed history is only needed for future turns.
        async def _compress_bg() -> None:
            try:
                await self._conversation.maybe_compress(self._llm)
            except Exception as _e:
                logger.warning("Conversation compression failed (non-critical): %s", _e)

        _ct = asyncio.create_task(_compress_bg(), name="conv_compress")
        self._pending_tasks.add(_ct)
        _ct.add_done_callback(self._pending_tasks.discard)

        # 1. Retrieve memory + vision scene concurrently
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

        # Auto-log vision perceptions to episodic memory
        if scene_desc and self._episodic:
            self._episodic.log("perception", scene_desc)

        # 2. Build system prompt (with vision context)
        system_prompt = self._build_system_prompt(
            context_str,
            scene_desc=scene_desc,
            user_text=user_text,
        )

        # 3. Record user message and prepare LLM messages
        self._conversation.add_user_message(user_text)
        messages = self._prepare_messages(
            self._conversation.get_messages(system_prompt)
        )

        # 4. Log episode (episodic memory)
        self._log_episode("command", f"用户说: {user_text}")

        # 5. Stream LLM → TTS: start playback, wait for TTS to finish
        self._audio.start_playback()
        try:
            async with self._llm_semaphore:  # type: ignore[union-attr]
                full_response = await self._stream_with_tools(
                    messages, system_prompt, model=self._voice_model,
                    source=source,
                )
            # [SILENT] detection: LLM judged this was not addressed to the robot
            if self._SILENT_MARKER in full_response:
                logger.info("[SILENT] Not addressed to robot, suppressing output")
                self._audio.drain_buffers()
                # Remove the user message we just added to keep history clean
                if (
                    self._conversation.history
                    and self._conversation.history[-1].get("role") == "user"
                ):
                    self._conversation.history.pop()
                return ""

            self._conversation.add_assistant_message(full_response)
            self._last_spoken_text = full_response

            # Save to L4 vector memory (fire-and-forget, non-blocking)
            if self._memory is not None:
                _mt = asyncio.create_task(self._memory.save(user_text, full_response))
                self._pending_tasks.add(_mt)
                _mt.add_done_callback(self._pending_tasks.discard)

            # Wait for TTS to finish speaking before returning
            await asyncio.to_thread(self._audio.wait_speaking_done)

            # Log response as episode
            self._log_episode("action", f"回复: {full_response[:100]}")

            # qp_memory: record voice command as observation + periodic save
            if self._qp_memory is not None:
                try:
                    self._qp_memory.record_observation("voice", user_text)
                    self._qp_turn_count += 1
                    if self._qp_turn_count % 10 == 0:
                        self._qp_memory.save()
                except Exception as _e:
                    logger.debug("qp_memory record failed: %s", _e)

            # Defer reflection to avoid 429 rate-limit collision with
            # the next user query.  A 5-second delay lets the relay
            # quota window reset.
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
            # Store only the exception type — not the full message — to prevent
            # raw stack text from being compressed into long-term memory context.
            error_msg = f"[系统错误] {type(exc).__name__}"
            # Write an error placeholder so conversation history stays
            # user/assistant alternating. But if the history already ends
            # with an assistant message (e.g. partial tool exchange wrote
            # one), inserting another would create assistant->assistant,
            # which causes 400 Bad Request on the next LLM call.
            last_role = (
                self._conversation.history[-1].get("role")
                if self._conversation.history
                else None
            )
            if last_role == "assistant":
                # Insert a synthetic user recovery message to maintain alternation
                self._conversation.add_user_message("[系统恢复]")
            self._conversation.add_assistant_message(error_msg)
            return error_msg
        finally:
            self._audio.stop_playback()

    async def execute_skill(
        self, skill_name: str, user_text: str, extra_context: str = ""
    ) -> str:
        """Execute a named skill and speak the result.

        *extra_context* is injected as ``mission_context`` into the skill's
        execution context dict — used by SkillDispatcher to pass prior mission
        step history so multi-step skills share state.
        """
        skill = self._skill_manager.get(skill_name)
        if not skill:
            return f"[Skill] Not found: {skill_name}"

        # L0 Safety gate: block motion skills when estop is active.
        # query_estop_state() may do a short network call on cache miss — run in
        # a thread so we don't block the event loop.
        if self._dog_safety and self._dog_safety.is_configured():
            estop_state = await asyncio.to_thread(self._dog_safety.query_estop_state)
            if estop_state is not None and estop_state.get("enabled"):
                msg = f"[安全锁定] 急停已激活，无法执行 {skill_name}。请先解除急停。"
                logger.warning("Safety gate blocked skill '%s': estop is active", skill_name)
                return msg

        # Task 2: Dependency check — warn only, never block execution
        if skill.depends:
            for dep in skill.depends:
                dep_skill = self._skill_manager.get(dep)
                if dep_skill is None:
                    logger.warning(
                        "Skill '%s' depends on '%s' which is not available",
                        skill_name, dep,
                    )

        logger.info("Executing skill: %s", skill_name)

        # Route agent-shell skills to ThunderAgentShell for autonomous execution
        _AGENT_SHELL_SKILLS = {
            "agent_task", "find_object", "safety_check",
            "find_person", "check_location", "patrol_scan",
            "solve_problem",
        }
        if skill_name in _AGENT_SHELL_SKILLS and self._agent_shell is not None:
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
                # Agent shell handles all TTS internally (progress + streaming).
                # Never re-speak here — it causes double playback.
                self._last_spoken_text = spoken
                self._conversation.add_user_message(user_text)
                self._conversation.add_assistant_message(stored)
                self._log_episode("outcome", f"{skill_name}完成: {result[:100]}")
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
        # Task 1: dog_control skill — dispatch capability to dog-control-service
        # Maps voice姿态 intents to runtime capability names, then falls through
        # to the LLM prompt for user-facing confirmation.
        if skill_name == "dog_control" and self._dog_control and self._dog_control.is_configured():
            _capability_map = {
                "站起来": "stand",
                "站立": "stand",
                "坐下": "sit",
                "趴下": "sit",  # dog-control-service does not yet distinguish sit vs prone
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

        # Inject episodic data for patrol_report so LLM doesn't fabricate
        if skill_name == "patrol_report" and self._episodic:
            parts = [
                self._episodic.get_recent_digest(),
                self._episodic.get_knowledge_context(),
            ]
            patrol_data = "\n".join(p for p in parts if p)
            context["patrol_data"] = patrol_data or ""

        self._audio.start_playback()
        thinking_task: asyncio.Task[None] | None = None
        try:
            # Thinking indicator: if skill execution takes > _THINKING_DELAY,
            # play a tone so the user knows we're working on it.
            thinking_task, _ = self._create_thinking_task()

            def _on_tool_call(tool_name: str) -> None:
                pass  # no filler — let the result speak for itself

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
            if self._episodic:
                self._episodic.log("outcome", f"直接回复: {result[:100]}")
            await asyncio.to_thread(self._audio.wait_speaking_done)
            return result
        except Exception as exc:
            logger.error("Skill error (%s): %s", skill_name, exc)
            if self._episodic:
                self._episodic.log("error", f"技能错误 {skill_name}: {exc}")
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
        """Emergency stop — halt local arm motion and notify dog-safety-service.

        Local arm stop is synchronous (immediate).
        Runtime safety notification is fire-and-forget (non-blocking).
        """
        logger.warning("E-STOP triggered!")
        if self._arm:
            self._arm.emergency_stop()
        # Notify Thunder runtime safety layer — non-blocking, best-effort
        if self._dog_safety and self._dog_safety.is_configured():
            self._dog_safety.notify_estop()
            logger.warning("E-STOP: notified dog-safety-service")
        logger.warning("E-STOP: local motion halted.")

    def has_pending_tool_approval(self) -> bool:
        """Whether a dangerous tool invocation is waiting for operator approval."""
        return self._tools.has_pending_approval()

    async def handle_pending_tool_response(self, user_text: str) -> str | None:
        """Resolve or restate the pending dangerous tool based on user input."""
        result = self._tools.handle_pending_input(user_text)
        if result is None:
            return None
        return await self._respond_without_llm(user_text, result)

    async def _respond_without_llm(self, user_text: str, assistant_text: str) -> str:
        """Speak and record a direct response that doesn't need another LLM turn."""
        self._audio.drain_buffers()
        self._audio.start_playback()
        self._audio.speak(assistant_text)
        self._conversation.add_user_message(user_text)
        self._conversation.add_assistant_message(assistant_text)
        if self._episodic:
            self._episodic.log("command", f"用户说: {user_text}")
            self._episodic.log("outcome", f"直接回复: {assistant_text[:100]}")
        await asyncio.to_thread(self._audio.wait_speaking_done)
        self._audio.stop_playback()
        return assistant_text

    # ── Internal ─────────────────────────────────────────────

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject prompt seed and user format prefix for relay compatibility.

        The relay service overrides our system prompt with its own developer
        assistant identity. To counter this:
          1. DROP the system message when prompt_seed is present — the seed
             establishes identity via fake user/assistant turns, which the
             relay cannot override.
          2. Prepend a TTS format tag to the latest user message to enforce
             output constraints (no markdown, short, Chinese).

        Original conversation history is NOT modified — transformations are
        applied only to the copy sent to the LLM.
        """
        if not self._prompt_seed and not self._user_prefix:
            return messages

        # When seed is present, skip system message to avoid relay conflict.
        # The relay injects its own system prompt regardless; including ours
        # creates competing identities. Seed messages work better alone.
        #
        # BUT: the dropped system prompt contained tool instructions.
        # Inject tool awareness as a seed exchange so the LLM knows it
        # CAN and SHOULD call tools for factual queries.
        if self._prompt_seed:
            result = list(self._prompt_seed)

            # Inject tool context that would otherwise be lost
            tool_defs = self._tools.get_definitions(
                max_safety_level=self._general_tool_max_safety_level
            )
            if tool_defs:
                tool_names = [
                    td.get("function", {}).get("name", "")
                    for td in tool_defs
                ]
                result.append({
                    "role": "user",
                    "content": (
                        f"你有以下工具可用: {', '.join(tool_names)}。"
                        "涉及时间、文件等真实数据时必须调用工具，不要编造。"
                    ),
                })
                result.append({
                    "role": "assistant",
                    "content": "明白，需要真实数据时我会调用工具获取。",
                })

            rest = [m for m in messages if m.get("role") != "system"]
        else:
            result = [messages[0]]  # keep system prompt (no seed to compete)
            rest = list(messages[1:])

        # Prepend format tag to the last user message
        if self._user_prefix:
            for i in range(len(rest) - 1, -1, -1):
                if rest[i].get("role") == "user":
                    rest[i] = {
                        **rest[i],
                        "content": f"{self._user_prefix}\n{rest[i]['content']}",
                    }
                    break

        result.extend(rest)
        return result

    def _start_vision_capture(self) -> asyncio.Task[str] | None:
        """Start a background vision scene description, or None if vision is unavailable.

        Skipped when ``vision.auto_capture`` is False (default) — vision is
        only used when explicitly called via tools (look_around, find_target).
        This avoids ~3-5s VLM overhead on every conversational turn.
        """
        if not self._vision or not self._vision.available:
            return None
        if not self._vision._vision_cfg.get("auto_capture", False):
            return None
        return asyncio.create_task(self._vision.describe_scene())

    def _create_thinking_task(
        self, include_slow_network: bool = False,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None] | None]:
        """Create background tasks for thinking and optional slow-network indicators.

        Returns (thinking_task, slow_network_task).  slow_network_task is None
        when *include_slow_network* is False.
        """
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
        stream,  # async iterator of chunks
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

            # Accumulate tool call fragments
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

                # Tool calls detected -- drain any sentences already sent to TTS
                if spoke_any:
                    self._audio.drain_buffers()
                    spoke_any = False

            # Text content -- strip <think> blocks, then speak
            if delta.content:
                clean = self._think_filter.feed(delta.content)
                if clean:
                    full_response += clean
                    if not truncated:
                        for sentence in self._splitter.feed(clean):
                            if char_limit and chars_spoken + len(sentence) > char_limit:
                                self._audio.speak(sentence)
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

        # Flush think filter and splitter tail
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

        # Thinking indicator: play a synthesized tone if TTFT > 1.2s.
        # Uses play_thinking() (direct PCM -> aplay) instead of speak("嗯...")
        # to bypass the TTS network, cutting indicator latency from ~2.2s to ~1.25s.
        thinking_task: asyncio.Task[None] | None = None
        slow_network_task: asyncio.Task[None] | None = None
        if is_voice:
            thinking_task, slow_network_task = self._create_thinking_task(
                include_slow_network=True,
            )

        try:
            async def _ttft_stream():
                """Wrap chat_stream to record TTFT and cancel thinking indicators."""
                nonlocal ttft_logged, thinking_task, slow_network_task
                async for chunk in self._llm.chat_stream(
                    messages, tools=tool_definitions, tool_choice="auto", model=model
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
            # Ensure thinking/network indicators are cancelled even on error
            if thinking_task is not None:
                thinking_task.cancel()
            if slow_network_task is not None:
                slow_network_task.cancel()

        # If tool calls detected, drain leftover TTS and do follow-up
        if tool_calls_acc:
            self._audio.drain_buffers()
            full_response = await self._execute_tools(
                tool_calls_acc, system_prompt, model=model, source=source,
            )

        return full_response

    async def _execute_tools(
        self,
        tool_calls_acc: dict[int, dict[str, str]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Execute accumulated tool calls and get follow-up LLM response."""
        logger.info("Tool calls: %d detected", len(tool_calls_acc))

        # Build all tool call objects first
        tool_call_objs = []
        tool_results = []
        approval_response: str | None = None
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            logger.info("  -> %s(%s)", tc["name"], tc["arguments"])
            if self._episodic:
                self._episodic.log("action", f"调用工具: {tc['name']}")
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._tools.execute,
                        tc["name"],
                        tc["arguments"],
                        max_safety_level=self._general_tool_max_safety_level,
                    ),
                    timeout=self._TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Tool '%s' timed out after %.0fs", tc["name"], self._TOOL_TIMEOUT
                )
                result = f"[Error] 工具 {tc['name']} 执行超时（超过 {int(self._TOOL_TIMEOUT)} 秒）"
            logger.info("  <- %s", result)
            if self._episodic:
                self._episodic.log("outcome", f"工具结果 {tc['name']}: {str(result)[:100]}")

            tool_call_objs.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })
            tool_results.append({"tool_call_id": tc["id"], "content": str(result)})
            if self._tools.has_pending_approval():
                approval_response = str(result)
                break

        if approval_response is not None:
            # Do NOT record to history when approval is pending —
            # the tool will be re-executed after operator confirmation.
            return approval_response

        self._conversation.add_tool_exchange(tool_call_objs, tool_results)

        # Follow-up LLM call with tool results
        follow_msgs = self._prepare_messages(
            self._conversation.get_messages(system_prompt)
        )
        return await self._stream_and_speak(follow_msgs, model=model, source=source)

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
                return "响应超时，请再说一遍。"
            if isinstance(exc, APIConnectionError):
                return "网络连接异常，请稍候重试。"
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
            return f"{skill_name}执行超时，已跳过。"
        if "connect" in str(exc).lower() or "network" in str(exc).lower():
            return f"网络异常，{skill_name}执行失败。"
        return f"{skill_name}执行失败，请重试。"

    def _prepare_agent_result(self, result: str) -> tuple[str, str]:
        """Prepare agent result for TTS + conversation storage.

        If result exceeds max_response_chars (voice mode), truncate at a sentence
        boundary for TTS and save the full result to the workspace for later reference.

        Returns (spoken_text, stored_text).
        """
        _AGENT_TTS_LIMIT = self._max_response_chars or 200
        if len(result) <= _AGENT_TTS_LIMIT:
            return result, result

        # Find last sentence end (。！？) within limit
        boundary = _AGENT_TTS_LIMIT
        for ch in "。！？!?":
            idx = result.rfind(ch, 0, _AGENT_TTS_LIMIT)
            if idx > 0 and idx < boundary:
                boundary = idx + 1

        spoken = result[:boundary].rstrip() + " 完整结果已保存到工作区。"

        # Persist full result so user can retrieve later
        try:
            workspace = self._agent_shell._workspace if self._agent_shell else None
            if workspace:
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / "last_result.txt").write_text(result, encoding="utf-8")
        except Exception:
            pass  # best-effort — don't let file I/O block voice response

        return spoken, result

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
        """Assemble system prompt with episodic knowledge, session summaries, and memory context."""
        l0 = self._build_l0_runtime_block()
        prompt = (l0 + "\n") if l0 else ""
        prompt += self._base_prompt

        if self._episodic:
            world_ctx = self._episodic.get_knowledge_context()
            if world_ctx:
                prompt += f"\n{world_ctx}"
            digest_ctx = self._episodic.get_recent_digest()
            if digest_ctx:
                prompt += f"\n{digest_ctx}"
            relevant_ctx = self._episodic.get_relevant_context(user_text)
            if relevant_ctx:
                prompt += f"\n{relevant_ctx}"

        if self._session_memory:
            session_ctx = self._session_memory.get_recent_summaries()
            if session_ctx:
                prompt += f"\n{session_ctx}"

        # qp_memory: spatial/procedural/markdown context (optional, additive)
        # NOTE: WebChat path injects into user message directly (app.py).
        # Voice path injects here into system prompt. Both use get_context_smart().
        # Only inject if not already injected by the caller (avoid double-injection).
        if self._qp_memory is not None and "[站点记忆]" not in (user_text or ""):
            try:
                qp_ctx = self._qp_memory.get_context_smart(query=user_text, max_chars=800)
                if qp_ctx:
                    prompt += f"\n[站点记忆]\n{qp_ctx}"
            except Exception as _e:
                logger.debug("qp_memory context retrieval failed: %s", _e)

        if context_str:
            prompt += f"\nRelevant memory:\n{context_str}"

        if self._vision and self._vision.available:
            prompt += "\n视觉能力: 已启用"
            if scene_desc:
                prompt += f"\n当前视野: {scene_desc}"

        # Tool use guidance: list available tools and instruct LLM to use them
        tool_defs = self._tools.get_definitions(
            max_safety_level=self._general_tool_max_safety_level
        )
        if tool_defs:
            tool_names = [
                td.get("function", {}).get("name", "") for td in tool_defs
            ]
            prompt += (
                f"\n你可以主动调用以下工具: {', '.join(tool_names)}。"
                "当用户提问涉及时间、文件、目录等信息时，主动调用对应工具获取真实数据再回答。"
            )

        skill_catalog = self._skill_manager.get_skill_catalog()
        if skill_catalog != "none":
            prompt += f"\n可用技能: {skill_catalog}"

        return prompt
