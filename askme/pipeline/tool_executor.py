"""Tool execution helpers for BrainPipeline."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import TYPE_CHECKING, Any, Protocol

from askme.pipeline.hooks import _PROCEED, PipelineHooks, ToolCallRecord

if TYPE_CHECKING:
    from askme.llm.conversation import ConversationManager
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.pipeline.prompt_builder import PromptBuilder
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


class _StreamAndSpeakFn(Protocol):
    """stream_and_speak callback — keyword args must be passed by keyword."""

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        model: str | None = ...,
        source: str = ...,
    ) -> str: ...


class ToolExecutor:
    """Executes tool calls returned by the LLM and handles approval flows.

    Supports PipelineHooks for pre/post-tool interception (inspired by Claude
    Code's PreToolUse/PostToolUse hook types):
      - ``pre_tool``  : may short-circuit a call and return an override result
      - ``post_tool`` : may transform the result before it enters conversation
    """

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
        hooks: PipelineHooks | None = None,
    ) -> None:
        self._tools = tools
        self._conversation = conversation
        self._episodic = episodic
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._prompt_builder = prompt_builder
        self._stream_and_speak = stream_and_speak
        self._hooks = hooks

    async def execute_tools(
        self,
        tool_calls_acc: dict[int, dict[str, str]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Execute accumulated tool calls and get follow-up LLM response.

        For each tool call:
          1. Fire ``pre_tool`` hooks — any hook can short-circuit by returning a
             string result, skipping the actual tool execution (like Claude Code's
             PreToolUse hook blocking a dangerous command).
          2. Execute the tool (with timeout).
          3. Fire ``post_tool`` hooks — hooks may transform the result before it
             enters the conversation (like Claude Code's PostToolUse hook).
          4. Produce an immutable ``ToolCallRecord`` for hook context.
        """
        logger.info("Tool calls: %d detected", len(tool_calls_acc))

        tool_call_objs = []
        tool_results = []
        approval_response: str | None = None

        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            tool_name = tc["name"]
            tool_args = tc["arguments"]
            call_id = tc["id"]
            logger.info("  -> %s(%s)", tool_name, tool_args)
            if self._episodic:
                self._episodic.log("action", f"调用工具: {tool_name}")

            timed_out = False

            # ── pre_tool hook (Claude Code: PreToolUse) ────────────────────
            hook_override: str | None = None
            if self._hooks and self._hooks.pre_tool:
                probe = ToolCallRecord(
                    call_id=call_id, tool_name=tool_name,
                    arguments=tool_args, result="",
                    elapsed_ms=0.0,
                )
                override = await self._hooks.fire_pre_tool(probe)
                if override is not _PROCEED:
                    hook_override = override or ""
                    logger.info(
                        "  [pre_tool hook] %s intercepted by hook, result overridden", tool_name
                    )

            if hook_override is not None:
                result = hook_override
                elapsed_ms = 0.0
            else:
                t0 = _time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._tools.execute,
                            tool_name,
                            tool_args,
                            max_safety_level=self._general_tool_max_safety_level,
                        ),
                        timeout=self._TOOL_TIMEOUT,
                    )
                except TimeoutError:
                    logger.error(
                        "Tool '%s' timed out after %.0fs", tool_name, self._TOOL_TIMEOUT
                    )
                    result = (
                        f"[Error] 工具 {tool_name} 执行超时"
                        f"（超过 {int(self._TOOL_TIMEOUT)} 秒）"
                    )
                    timed_out = True
                elapsed_ms = (_time.perf_counter() - t0) * 1000

            logger.info("  <- %s", result)
            if self._episodic:
                self._episodic.log("outcome", f"工具结果 {tool_name}: {str(result)[:100]}")

            # ── post_tool hook (Claude Code: PostToolUse) ──────────────────
            if self._hooks and self._hooks.post_tool:
                record = ToolCallRecord(
                    call_id=call_id, tool_name=tool_name,
                    arguments=tool_args, result=str(result),
                    elapsed_ms=elapsed_ms, timed_out=timed_out,
                )
                result = await self._hooks.fire_post_tool(record)

            tool_call_objs.append({
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            })
            tool_results.append({"tool_call_id": call_id, "content": str(result)})
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
        return await self._stream_and_speak(follow_msgs, model=model, source=source)

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
