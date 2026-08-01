"""Tool execution helpers for BrainPipeline."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Protocol

from askme.conversation import InteractionTurnContext
from askme.pipeline.core.hooks import _PROCEED, PipelineHooks, ToolCallRecord

if TYPE_CHECKING:
    from askme.llm.core.contracts import LLMCallContext
    from askme.memory.core.conversation import ConversationManager
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.pipeline.core.prompt_builder import PromptBuilder
    from askme.pipeline.core.protocols import CancellationToken
    from askme.ports import AudioFrontendPort
    from askme.tools.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class _StreamAndSpeakFn(Protocol):
    """stream_and_speak callback; keyword args must be passed by keyword."""

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
        stream_and_speak: _StreamAndSpeakFn | None,
        hooks: PipelineHooks | None = None,
    ) -> None:
        self._tools = tools
        self._conversation = conversation
        self._episodic = episodic
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._prompt_builder = prompt_builder
        self._stream_and_speak = stream_and_speak
        self._hooks = hooks

    @staticmethod
    def _accepts_keyword(callback: Any, name: str) -> bool:
        """Return whether a callable can accept one named compatibility argument."""

        try:
            parameters = signature(callback).parameters
        except (TypeError, ValueError):
            return False
        named_parameter = parameters.get(name)
        return (
            named_parameter is not None
            and named_parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        ) or any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())

    @staticmethod
    def _interaction_context(
        *,
        source: str,
        conversation_session_id: str | None,
        turn_cancel_token: CancellationToken | None,
        llm_call_context: LLMCallContext | None,
    ) -> InteractionTurnContext | None:
        """Project internal LLM correlation into canonical tool policy context."""

        explicit_thread = str(conversation_session_id or "").strip()
        llm_thread = str(getattr(llm_call_context, "session_id", None) or "").strip()
        if explicit_thread and llm_thread and explicit_thread != llm_thread:
            raise ValueError("LLM and conversation session identities do not match")
        thread_id = llm_thread or explicit_thread
        turn_id = str(getattr(llm_call_context, "turn_id", None) or "").strip()
        if not thread_id or not turn_id:
            return None
        channel = str(getattr(llm_call_context, "channel", None) or source or "text").strip()
        operator_id = str(getattr(llm_call_context, "operator_id", None) or "").strip()
        return InteractionTurnContext(
            thread_id=thread_id,
            turn_id=turn_id,
            channel=channel,
            source=str(source or channel),
            user_text="",
            operator_id=operator_id or None,
            cancel_token=turn_cancel_token,
        )

    def _has_pending_approval(
        self,
        interaction_context: InteractionTurnContext | None,
    ) -> bool:
        callback = self._tools.has_pending_approval
        if interaction_context is None:
            return bool(callback())
        if not self._accepts_keyword(callback, "interaction_context"):
            logger.warning(
                "Scoped tool approval probe rejected: registry does not accept interaction_context"
            )
            return False
        return bool(callback(interaction_context=interaction_context))

    async def execute_tools(
        self,
        tool_calls_acc: dict[int, dict[str, str]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        llm_call_context: LLMCallContext | None = None,
    ) -> str:
        """Execute accumulated tool calls and get follow-up LLM response.

        For each tool call:
          1. Fire ``pre_tool`` hooks; any hook can short-circuit by returning a
             string result, skipping the actual tool execution (like Claude Code's
             PreToolUse hook blocking a dangerous command).
          2. Execute the tool; ToolRegistry owns per-tool timeout/cooldown.
          3. Fire ``post_tool`` hooks; hooks may transform the result before it
             enters the conversation (like Claude Code's PostToolUse hook).
          4. Produce an immutable ``ToolCallRecord`` for hook context.
        """
        logger.info("Tool calls: %d detected", len(tool_calls_acc))
        interaction_context = self._interaction_context(
            source=source,
            conversation_session_id=conversation_session_id,
            turn_cancel_token=turn_cancel_token,
            llm_call_context=llm_call_context,
        )

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

            # pre_tool hook (Claude Code: PreToolUse).
            hook_override: str | None = None
            if self._hooks and self._hooks.pre_tool:
                probe = ToolCallRecord(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=tool_args,
                    result="",
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
                    execute_callback = self._tools.execute
                    execute_kwargs: dict[str, Any] = {
                        "max_safety_level": self._general_tool_max_safety_level,
                    }
                    if interaction_context is not None and not self._accepts_keyword(
                        execute_callback,
                        "interaction_context",
                    ):
                        logger.error(
                            "Scoped tool execution rejected for %s: registry does not "
                            "accept interaction_context",
                            tool_name,
                        )
                        result = (
                            "[Error] Tool registry does not support scoped interaction context."
                        )
                    else:
                        if interaction_context is not None:
                            execute_kwargs["interaction_context"] = interaction_context
                        result = await asyncio.to_thread(
                            execute_callback,
                            tool_name,
                            tool_args,
                            **execute_kwargs,
                        )
                except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                    logger.error("Tool '%s' timed out after %.0fs", tool_name, self._TOOL_TIMEOUT)
                    result = (
                        f"[Error] 工具 {tool_name} 执行超时（超过 {int(self._TOOL_TIMEOUT)} 秒）"
                    )
                    timed_out = True
                elapsed_ms = (_time.perf_counter() - t0) * 1000

            logger.info("  <- %s", result)
            if self._episodic:
                self._episodic.log("outcome", f"工具结果 {tool_name}: {str(result)[:100]}")

            # post_tool hook (Claude Code: PostToolUse).
            if self._hooks and self._hooks.post_tool:
                record = ToolCallRecord(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=str(result),
                    elapsed_ms=elapsed_ms,
                    timed_out=timed_out,
                )
                result = await self._hooks.fire_post_tool(record)

            tool_call_objs.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args},
                }
            )
            tool_results.append({"tool_call_id": call_id, "content": str(result)})
            if self._has_pending_approval(interaction_context):
                approval_response = str(result)
                break

        if approval_response is not None:
            # Do NOT record to history when approval is pending;
            # the tool will be re-executed after operator confirmation.
            return approval_response

        if conversation_session_id is None:
            self._conversation.add_tool_exchange(tool_call_objs, tool_results)
            conversation_messages = self._conversation.get_messages(system_prompt)
        else:
            self._conversation.add_tool_exchange(
                tool_call_objs,
                tool_results,
                conversation_session_id=conversation_session_id,
            )
            conversation_messages = self._conversation.get_messages(
                system_prompt,
                conversation_session_id=conversation_session_id,
            )

        follow_msgs = self._prompt_builder.prepare_messages(
            conversation_messages,
            source=source,
        )
        if self._stream_and_speak is None:
            raise RuntimeError("stream_and_speak callback is not configured")
        return await self._stream_and_speak(follow_msgs, model=model, source=source)

    async def respond_without_llm(
        self,
        user_text: str,
        assistant_text: str,
        *,
        audio: AudioFrontendPort,
        source: str = "voice",
        interaction_context: InteractionTurnContext | None = None,
    ) -> str:
        """Deliver a direct response, then project only completed delivery to history."""

        cancel_token = interaction_context.cancel_token if interaction_context is not None else None
        if cancel_token is not None and cancel_token.is_set():
            return ""

        audio.drain_buffers()
        audio.start_playback()
        audio.speak(assistant_text)
        if source == "voice":
            try:
                await asyncio.to_thread(audio.wait_speaking_done)
            finally:
                audio.stop_playback()
            if cancel_token is not None and cancel_token.is_set():
                audio.drain_buffers()
                return ""

        conversation_session_id = (
            str(interaction_context.thread_id or "").strip()
            if interaction_context is not None
            else ""
        )
        if conversation_session_id:
            self._conversation.add_user_message(
                user_text,
                conversation_session_id=conversation_session_id,
            )
            self._conversation.add_assistant_message(
                assistant_text,
                conversation_session_id=conversation_session_id,
            )
        else:
            self._conversation.add_user_message(user_text)
            self._conversation.add_assistant_message(assistant_text)
        if self._episodic:
            self._episodic.log("command", f"用户说: {user_text}")
            self._episodic.log("outcome", f"直接回复: {assistant_text[:100]}")
        return assistant_text

    async def handle_pending_tool_response(
        self,
        user_text: str,
        *,
        audio: AudioFrontendPort,
        source: str = "voice",
        interaction_context: InteractionTurnContext | None = None,
    ) -> str | None:
        """Resolve one exact pending approval challenge for this later Turn."""

        if interaction_context is None:
            result = self._tools.handle_pending_input(user_text)
        else:
            cancel_token = interaction_context.cancel_token
            if cancel_token is not None and cancel_token.is_set():
                return None
            scope_getter = getattr(self._tools, "pending_approval_scope", None)
            if not callable(scope_getter) or not self._accepts_keyword(
                scope_getter,
                "interaction_context",
            ):
                logger.warning(
                    "Scoped tool approval response rejected: registry does not expose "
                    "pending_approval_scope(interaction_context)"
                )
                return None
            scope = scope_getter(interaction_context=interaction_context)
            approval_id = str(getattr(scope, "approval_id", "") or "").strip()
            if not approval_id:
                return None
            pending_handler = getattr(self._tools, "handle_pending_input", None)
            if not callable(pending_handler) or not self._accepts_keyword(
                pending_handler,
                "interaction_context",
            ):
                logger.warning(
                    "Scoped tool approval response rejected: registry handler does "
                    "not accept interaction_context"
                )
                return None
            if not self._accepts_keyword(pending_handler, "approval_id"):
                logger.warning(
                    "Scoped tool approval response rejected: registry handler does "
                    "not accept approval_id"
                )
                return None
            result = pending_handler(
                user_text,
                interaction_context=interaction_context,
                approval_id=approval_id,
            )
        if result is None:
            return None
        return await self.respond_without_llm(
            user_text,
            result,
            audio=audio,
            source=source,
            interaction_context=interaction_context,
        )
