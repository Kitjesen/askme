"""Brain pipeline — memory retrieval → LLM streaming → tool calls → TTS."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.brain.conversation import ConversationManager
    from askme.brain.episodic_memory import EpisodicMemory
    from askme.brain.llm_client import LLMClient
    from askme.brain.memory_bridge import MemoryBridge
    from askme.brain.session_memory import SessionMemory
    from askme.brain.vision_bridge import VisionBridge
    from askme.robot.arm_controller import ArmController
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


class BrainPipeline:
    """Orchestrates one turn of conversation:
    memory → system prompt → streaming LLM → tool calls → TTS.

    All dependencies are injected via the constructor.
    """

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
        vision: VisionBridge | None = None,
        session_memory: SessionMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        system_prompt: str = "你是一个有用的AI语音助手。用中文简洁口语化回答。",
        prompt_seed: list[dict[str, str]] | None = None,
        user_prefix: str = "",
        voice_model: str | None = None,
        general_tool_max_safety_level: str = "normal",
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._memory = memory
        self._tools = tools
        self._skill_manager = skill_manager
        self._skill_executor = skill_executor
        self._audio = audio
        self._splitter = splitter
        self._arm = arm_controller
        self._vision = vision
        self._session_memory = session_memory
        self._episodic = episodic_memory
        self._base_prompt = system_prompt
        self._prompt_seed = prompt_seed or []
        self._user_prefix = user_prefix
        self._voice_model = voice_model  # fast model for real-time voice turns
        self._general_tool_max_safety_level = general_tool_max_safety_level

    # ── Public API ───────────────────────────────────────────

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
            if self._episodic and self._episodic.should_reflect():
                logger.info("[Dream] Idle-time reflection triggered")
                summary = await self._episodic.reflect()
                if summary:
                    logger.info("[Dream] Reflection result: %s", summary[:80])
                # Clean up old episode files while we're at it
                self._episodic.cleanup_old_episodes()

        return asyncio.create_task(_idle_reflect())

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[str]:
        """Start memory retrieval as a background task. Call ASAP after ASR returns."""
        return asyncio.create_task(self._memory.retrieve(user_text))

    async def process(
        self, user_text: str, *, memory_task: asyncio.Task[str] | None = None,
    ) -> str:
        """Run the full brain pipeline for *user_text*. Returns assistant reply."""
        logger.info("Processing: %s", user_text[:60])

        # 0. Clear leftover audio from any previous turn
        self._audio.drain_buffers()

        # 0b. Sliding window compression (non-blocking, best-effort)
        try:
            await self._conversation.maybe_compress(self._llm)
        except Exception:
            pass  # compression failure is non-critical

        # 1. Retrieve memory + vision scene concurrently
        if not memory_task:
            memory_task = asyncio.create_task(self._memory.retrieve(user_text))
        vision_task = self._start_vision_capture()

        try:
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
        if self._episodic:
            self._episodic.log("command", f"用户说: {user_text}")

        # 5. Stream LLM + TTS (non-blocking: don't wait for playback to finish)
        self._audio.start_playback()
        try:
            full_response = await self._stream_with_tools(
                messages, system_prompt, model=self._voice_model
            )
            self._conversation.add_assistant_message(full_response)

            # Log response as episode
            if self._episodic:
                self._episodic.log("action", f"回复: {full_response[:100]}")
                # Trigger reflection if conditions met (fire-and-forget with logging)
                if self._episodic.should_reflect():
                    task = asyncio.create_task(self._episodic.reflect())
                    task.add_done_callback(
                        lambda t: logger.error("[Episodic] Reflection failed: %s", t.exception())
                        if not t.cancelled() and t.exception() else None
                    )

            return full_response
        except Exception as exc:
            logger.error("LLM pipeline error: %s", exc)
            if self._episodic:
                self._episodic.log("error", f"LLM错误: {exc}")
            # Audio feedback so user knows something went wrong
            self._audio.speak_error()
            return f"[Error] {exc}"

    async def execute_skill(self, skill_name: str, user_text: str) -> str:
        """Execute a named skill and speak the result."""
        skill = self._skill_manager.get(skill_name)
        if not skill:
            return f"[Skill] Not found: {skill_name}"

        logger.info("Executing skill: %s", skill_name)
        self._audio.drain_buffers()
        if self._episodic:
            self._episodic.log("action", f"执行技能: {skill_name}")

        context: dict[str, str] = {
            "user_input": user_text,
            "current_time": __import__("datetime").datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        if self._arm:
            context["robot_state"] = json.dumps(
                self._arm.get_state(), ensure_ascii=False
            )

        self._audio.start_playback()
        try:
            result = await self._skill_executor.execute(skill, context)
            logger.info("Skill result: %s", result[:100])
            self._audio.speak(result)
            self._conversation.add_user_message(user_text)
            self._conversation.add_assistant_message(result)
            if self._episodic:
                self._episodic.log("outcome", f"技能结果 {skill_name}: {result[:100]}")
            await asyncio.to_thread(self._audio.wait_speaking_done)
            return result
        except Exception as exc:
            logger.error("Skill error (%s): %s", skill_name, exc)
            if self._episodic:
                self._episodic.log("error", f"技能错误 {skill_name}: {exc}")
            return f"[Skill Error] {exc}"
        finally:
            self._audio.stop_playback()

    def handle_estop(self) -> None:
        """Emergency stop — immediately halt robot motion."""
        logger.warning("E-STOP triggered!")
        if self._arm:
            self._arm.emergency_stop()
        logger.warning("E-STOP: All motion halted.")

    def has_pending_tool_approval(self) -> bool:
        """Whether a dangerous tool invocation is waiting for operator approval."""
        return self._tools.has_pending_approval()

    async def handle_pending_tool_response(self, user_text: str) -> str | None:
        """Approve or reject the pending dangerous tool based on user input."""
        if self._tools.matches_confirmation(user_text):
            result = await asyncio.to_thread(self._tools.approve_pending)
            return await self._respond_without_llm(user_text, result)
        if self._tools.matches_rejection(user_text):
            result = self._tools.reject_pending()
            return await self._respond_without_llm(user_text, result)
        return None

    async def _respond_without_llm(self, user_text: str, assistant_text: str) -> str:
        """Speak and record a direct response that doesn't need another LLM turn."""
        self._audio.drain_buffers()
        self._audio.start_playback()
        self._audio.speak(assistant_text)
        self._conversation.add_user_message(user_text)
        self._conversation.add_assistant_message(assistant_text)
        if self._episodic:
            self._episodic.log("command", f"鐢ㄦ埛璇? {user_text}")
            self._episodic.log("outcome", f"鐩存帴鍥炲: {assistant_text[:100]}")
        await asyncio.to_thread(self._audio.wait_speaking_done)
        self._audio.stop_playback()
        return assistant_text

    # ── Internal ─────────────────────────────────────────────

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject prompt seed and user format prefix for relay compatibility.

        The relay service may override our system prompt with its own.
        To counter this:
          1. Insert seed messages (fake user/assistant turn) after system prompt
             to establish our assistant's identity.
          2. Prepend a TTS format tag to the latest user message to enforce
             output constraints (no markdown, short, Chinese).

        Original conversation history is NOT modified — transformations are
        applied only to the copy sent to the LLM.
        """
        if not self._prompt_seed and not self._user_prefix:
            return messages

        result = [messages[0]]  # system prompt
        result.extend(self._prompt_seed)
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
        """Start a background vision scene description, or None if vision is unavailable."""
        if not self._vision or not self._vision.available:
            return None
        return asyncio.create_task(self._vision.describe_scene())

    async def _stream_with_tools(
        self, messages: list[dict[str, Any]], system_prompt: str,
        model: str | None = None,
    ) -> str:
        """Stream LLM response, speak sentences immediately, handle tool calls."""
        tool_definitions = self._tools.get_definitions(
            max_safety_level=self._general_tool_max_safety_level
        )
        full_response = ""
        tool_calls_acc: dict[int, dict[str, str]] = {}
        spoke_any = False
        self._splitter.reset()

        async for chunk in self._llm.chat_stream(
            messages, tools=tool_definitions, tool_choice="auto", model=model
        ):
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

                # Tool calls detected — drain any sentences already sent to TTS
                if spoke_any:
                    self._audio.drain_buffers()
                    spoke_any = False

            # Text content — speak immediately (don't buffer)
            if delta.content:
                full_response += delta.content
                for sentence in self._splitter.feed(delta.content):
                    self._audio.speak(sentence)
                    spoke_any = True

        remainder = self._splitter.flush()
        if remainder:
            self._audio.speak(remainder)
            spoke_any = True

        # If tool calls detected, drain leftover TTS and do follow-up
        if tool_calls_acc:
            if spoke_any:
                self._audio.drain_buffers()
            full_response = await self._execute_tools(
                tool_calls_acc, system_prompt, model=model
            )

        return full_response

    async def _execute_tools(
        self,
        tool_calls_acc: dict[int, dict[str, str]],
        system_prompt: str,
        model: str | None = None,
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
            result = await asyncio.to_thread(
                self._tools.execute,
                tc["name"],
                tc["arguments"],
                max_safety_level=self._general_tool_max_safety_level,
            )
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

        # Append ONE assistant message with all tool calls
        self._conversation.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_call_objs,
        })
        # Append each tool result
        for tr in tool_results:
            self._conversation.history.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            })

        if approval_response is not None:
            return approval_response

        # Follow-up LLM call with tool results
        follow_msgs = self._prepare_messages(
            self._conversation.get_messages(system_prompt)
        )
        return await self._stream_and_speak(follow_msgs, model=model)

    async def _stream_and_speak(
        self, messages: list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        full_response = ""
        self._splitter.reset()

        async for chunk in self._llm.chat_stream(messages, model=model):
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                for sentence in self._splitter.feed(delta.content):
                    self._audio.speak(sentence)

        remainder = self._splitter.flush()
        if remainder:
            self._audio.speak(remainder)

        return full_response

    def _build_system_prompt(
        self,
        context_str: str | None,
        *,
        scene_desc: str = "",
        user_text: str = "",
    ) -> str:
        """Assemble system prompt with episodic knowledge, session summaries, and memory context."""
        prompt = self._base_prompt

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

        if context_str:
            prompt += f"\nRelevant memory:\n{context_str}"

        if self._vision and self._vision.available:
            prompt += "\n视觉能力: 已启用"
            if scene_desc:
                prompt += f"\n当前视野: {scene_desc}"

        skill_catalog = self._skill_manager.get_skill_catalog()
        if skill_catalog != "none":
            prompt += f"\n可用技能: {skill_catalog}"

        return prompt

        # Episodic: World knowledge from reflection (robot experiences)
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

        # Layer 2: Session summaries (medium-term memory)
        if self._session_memory:
            session_ctx = self._session_memory.get_recent_summaries()
            if session_ctx:
                prompt += f"\n{session_ctx}"

        # Layer 3: Long-term memory (MemU vector retrieval)
        if context_str:
            prompt += f"\n记忆上下文:\n{context_str}"
        else:
            prompt += "\n记忆上下文: 无"

        if self._vision and self._vision.available:
            prompt += "\n视觉能力: 已启用（可以看到周围环境）"
            if scene_desc:
                prompt += f"\n当前视野: {scene_desc}"

        skill_catalog = self._skill_manager.get_skill_catalog()
        if skill_catalog != "none":
            prompt += f"\n可用技能: {skill_catalog}"

        return prompt
