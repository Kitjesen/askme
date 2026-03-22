"""Tool execution helpers for BrainPipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.llm.conversation import ConversationManager
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.pipeline.prompt_builder import PromptBuilder
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)

# stream_and_speak signature: (messages, model, source) -> str
_StreamAndSpeakFn = Callable[[list[dict[str, Any]], str | None, str], Awaitable[str]]


class ToolExecutor:
    """Executes tool calls returned by the LLM and handles approval flows."""

    _TOOL_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        tools: ToolRegistry,
        conversation: ConversationManager,
        episodic: EpisodicMemory | None,
        general_tool_max_safety_level: str,
        prompt_builder: PromptBuilder,
        stream_and_speak: _StreamAndSpeakFn,
    ) -> None:
        self._tools = tools
        self._conversation = conversation
        self._episodic = episodic
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._prompt_builder = prompt_builder
        self._stream_and_speak = stream_and_speak

    async def execute_tools(
        self,
        tool_calls_acc: dict[int, dict[str, str]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Execute accumulated tool calls and get follow-up LLM response."""
        logger.info("Tool calls: %d detected", len(tool_calls_acc))

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

        follow_msgs = self._prompt_builder.prepare_messages(
            self._conversation.get_messages(system_prompt)
        )
        return await self._stream_and_speak(follow_msgs, model, source)

    async def respond_without_llm(
        self,
        user_text: str,
        assistant_text: str,
        *,
        audio: AudioAgent,
        source: str = "voice",
    ) -> str:
        """Speak and record a direct response that doesn't need another LLM turn."""
        audio.drain_buffers()
        audio.start_playback()
        audio.speak(assistant_text)
        self._conversation.add_user_message(user_text)
        self._conversation.add_assistant_message(assistant_text)
        if self._episodic:
            self._episodic.log("command", f"用户说: {user_text}")
            self._episodic.log("outcome", f"直接回复: {assistant_text[:100]}")
        if source == "voice":
            await asyncio.to_thread(audio.wait_speaking_done)
            audio.stop_playback()
        return assistant_text

    async def handle_pending_tool_response(
        self,
        user_text: str,
        *,
        audio: AudioAgent,
        source: str = "voice",
    ) -> str | None:
        """Resolve or restate the pending dangerous tool based on user input."""
        result = self._tools.handle_pending_input(user_text)
        if result is None:
            return None
        return await self.respond_without_llm(
            user_text, result, audio=audio, source=source
        )
