"""Tests for ToolExecutor — pre/post hook wiring, timeout, approval flow."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from askme.pipeline.hooks import PipelineHooks, ToolCallRecord
from askme.pipeline.tool_executor import ToolExecutor
from askme.tools.tool_registry import BaseTool, ToolRegistry

from askme.conversation import (
    ApprovalScope,
    ConfirmationKind,
    InteractionTurnContext,
)
from askme.llm.core.contracts import LLMCallContext


def _make_executor(
    *,
    tools=None,
    conversation=None,
    episodic=None,
    prompt_builder=None,
    stream_and_speak=None,
    hooks=None,
) -> ToolExecutor:
    """Build a ToolExecutor with sensible mocks for all required deps."""
    if tools is None:
        tools = MagicMock()
        tools.execute = MagicMock(return_value="tool_result")
        tools.has_pending_approval.return_value = False
        tools.handle_pending_input.return_value = None
    if conversation is None:
        conversation = MagicMock()
        conversation.get_messages.return_value = []
    if prompt_builder is None:
        prompt_builder = MagicMock()
        prompt_builder.prepare_messages.return_value = []
    if stream_and_speak is None:
        stream_and_speak = AsyncMock(return_value="follow_up")
    return ToolExecutor(
        tools=tools,
        conversation=conversation,
        episodic=episodic,
        general_tool_max_safety_level="normal",
        prompt_builder=prompt_builder,
        stream_and_speak=stream_and_speak,
        hooks=hooks,
    )


def _make_tool_calls(
    names: list[str], args: str = "{}", ids: list[str] | None = None
) -> dict[int, dict[str, str]]:
    """Build a tool_calls_acc dict as BrainPipeline would."""
    if ids is None:
        ids = [f"call_{i}" for i in range(len(names))]
    return {i: {"name": names[i], "arguments": args, "id": ids[i]} for i in range(len(names))}


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    async def test_single_tool_called(self):
        executor = _make_executor()
        result = await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        executor._tools.execute.assert_called_once_with("nav_to", "{}", max_safety_level="normal")
        assert result == "follow_up"

    async def test_result_added_to_conversation(self):
        executor = _make_executor()
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        executor._conversation.add_tool_exchange.assert_called_once()

    async def test_session_scope_is_used_for_tool_exchange_and_follow_up(self):
        executor = _make_executor()

        await executor.execute_tools(
            _make_tool_calls(["nav_to"]),
            system_prompt="system",
            conversation_session_id="conv-a",
        )

        executor._conversation.add_tool_exchange.assert_called_once()
        assert executor._conversation.add_tool_exchange.call_args.kwargs == {
            "conversation_session_id": "conv-a"
        }
        executor._conversation.get_messages.assert_called_once_with(
            "system",
            conversation_session_id="conv-a",
        )

    async def test_multiple_tools_in_order(self):
        call_log: list[str] = []
        tools = MagicMock()
        tools.execute = MagicMock(
            side_effect=lambda name, args, **kw: call_log.append(name) or "ok"
        )
        tools.has_pending_approval.return_value = False
        executor = _make_executor(tools=tools)
        await executor.execute_tools(_make_tool_calls(["a", "b", "c"]), system_prompt="")
        assert call_log == ["a", "b", "c"]

    async def test_episodic_log_called(self):
        episodic = MagicMock()
        executor = _make_executor(episodic=episodic)
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        assert episodic.log.call_count >= 2  # action + outcome

    async def test_no_episodic_is_safe(self):
        executor = _make_executor(episodic=None)
        result = await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        assert result == "follow_up"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_records_error_in_conversation(self):
        """Timeout → tool result in conversation contains error; execute_tools still returns follow-up."""
        tools = MagicMock()
        tools.execute = MagicMock(return_value="[Error][Timeout] Tool 'slow_tool' exceeded 1.0s.")
        tools.has_pending_approval.return_value = False
        tools.handle_pending_input.return_value = None
        executor = _make_executor(tools=tools)
        await executor.execute_tools(_make_tool_calls(["slow_tool"]), system_prompt="")

        # The tool result recorded in conversation should mention timeout
        call_args = executor._conversation.add_tool_exchange.call_args
        tool_results = call_args[0][1]
        assert "超时" in tool_results[0]["content"] or "Error" in tool_results[0]["content"]


# ---------------------------------------------------------------------------
# pre_tool hooks
# ---------------------------------------------------------------------------


class TestPreToolHook:
    async def test_pre_tool_override_skips_execution(self):
        hooks = PipelineHooks()
        hooks.on_pre_tool(AsyncMock(return_value="intercepted"))
        executor = _make_executor(hooks=hooks)
        result = await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        # Tool execute should NOT have been called
        executor._tools.execute.assert_not_called()
        # The overridden result should appear in tool_results
        call_args = executor._conversation.add_tool_exchange.call_args
        tool_results = call_args[0][1]
        assert tool_results[0]["content"] == "intercepted"

    async def test_pre_tool_none_proceeds_normally(self):
        hooks = PipelineHooks()
        hooks.on_pre_tool(AsyncMock(return_value=None))
        executor = _make_executor(hooks=hooks)
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        executor._tools.execute.assert_called_once()

    async def test_no_hooks_proceeds_normally(self):
        executor = _make_executor(hooks=None)
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        executor._tools.execute.assert_called_once()


# ---------------------------------------------------------------------------
# post_tool hooks
# ---------------------------------------------------------------------------


class TestPostToolHook:
    async def test_post_tool_transforms_result(self):
        hooks = PipelineHooks()

        async def transform(rec: ToolCallRecord) -> str:
            return rec.result.upper()

        hooks.on_post_tool(transform)
        executor = _make_executor(hooks=hooks)
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        call_args = executor._conversation.add_tool_exchange.call_args
        tool_results = call_args[0][1]
        assert tool_results[0]["content"] == "TOOL_RESULT"

    async def test_post_tool_receives_correct_metadata(self):
        received: list[ToolCallRecord] = []

        async def capture(rec: ToolCallRecord) -> str:
            received.append(rec)
            return rec.result

        hooks = PipelineHooks()
        hooks.on_post_tool(capture)
        executor = _make_executor(hooks=hooks)
        await executor.execute_tools(
            _make_tool_calls(["nav_to"], ids=["call_abc"]), system_prompt=""
        )
        assert received[0].tool_name == "nav_to"
        assert received[0].call_id == "call_abc"
        assert received[0].result == "tool_result"
        assert received[0].elapsed_ms >= 0.0
        assert received[0].timed_out is False

    async def test_no_post_hooks_uses_raw_result(self):
        executor = _make_executor(hooks=None)
        await executor.execute_tools(_make_tool_calls(["nav_to"]), system_prompt="")
        call_args = executor._conversation.add_tool_exchange.call_args
        tool_results = call_args[0][1]
        assert tool_results[0]["content"] == "tool_result"


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


class TestApprovalFlow:
    async def test_approval_pending_stops_chain_early(self):
        tools = MagicMock()
        call_log: list[str] = []
        tools.execute = MagicMock(side_effect=lambda name, *a, **kw: call_log.append(name) or "ok")
        # After first tool: approval pending
        tools.has_pending_approval = MagicMock(side_effect=[True, False])
        executor = _make_executor(tools=tools)
        result = await executor.execute_tools(
            _make_tool_calls(["dangerous", "safe"]), system_prompt=""
        )
        # Only first tool was executed
        assert call_log == ["dangerous"]
        # Result is the first tool's output
        assert result == "ok"
        # No conversation record added (approval pending)
        executor._conversation.add_tool_exchange.assert_not_called()

    async def test_scoped_context_reaches_execute_and_pending_probe(self):
        token = asyncio.Event()
        tools = MagicMock()
        tools.execute.return_value = "approval required"
        tools.has_pending_approval = MagicMock(
            side_effect=lambda *, interaction_context: bool(interaction_context)
        )
        executor = _make_executor(tools=tools)
        llm_context = LLMCallContext(
            session_id="thread-a",
            turn_id="prompt-turn",
            channel="text",
            operator_id="operator-7",
        )

        result = await executor.execute_tools(
            _make_tool_calls(["dangerous"]),
            system_prompt="",
            source="text",
            conversation_session_id="thread-a",
            turn_cancel_token=token,
            llm_call_context=llm_context,
        )

        assert result == "approval required"
        interaction = tools.execute.call_args.kwargs["interaction_context"]
        assert interaction.thread_id == "thread-a"
        assert interaction.turn_id == "prompt-turn"
        assert interaction.channel == "text"
        assert interaction.operator_id == "operator-7"
        assert interaction.cancel_token is token
        tools.has_pending_approval.assert_called_once_with(interaction_context=interaction)

    async def test_scoped_probe_does_not_read_legacy_global_pending(self):
        class _LegacyProbeRegistry:
            def __init__(self) -> None:
                self.global_probe_calls = 0

            def execute(
                self,
                name: str,
                args: str,
                *,
                max_safety_level: str,
                interaction_context: InteractionTurnContext,
            ) -> str:
                del name, args, max_safety_level, interaction_context
                return "tool_result"

            def has_pending_approval(self) -> bool:
                self.global_probe_calls += 1
                return True

        tools = _LegacyProbeRegistry()
        executor = _make_executor(tools=tools)
        llm_context = LLMCallContext(
            session_id="thread-b",
            turn_id="thread-b-turn",
            channel="text",
            operator_id="operator-b",
        )

        result = await executor.execute_tools(
            _make_tool_calls(["safe"]),
            system_prompt="",
            source="text",
            conversation_session_id="thread-b",
            llm_call_context=llm_context,
        )

        assert result == "follow_up"
        assert tools.global_probe_calls == 0

    async def test_scoped_execution_does_not_call_legacy_unscoped_registry(self):
        class _LegacyRegistry:
            def __init__(self) -> None:
                self.execute_calls = 0
                self.global_probe_calls = 0

            def execute(
                self,
                name: str,
                args: str,
                *,
                max_safety_level: str,
            ) -> str:
                del name, args, max_safety_level
                self.execute_calls += 1
                return "legacy-tool-result"

            def has_pending_approval(self) -> bool:
                self.global_probe_calls += 1
                return True

        tools = _LegacyRegistry()
        executor = _make_executor(tools=tools)
        llm_context = LLMCallContext(
            session_id="thread-b",
            turn_id="thread-b-turn",
            channel="text",
            operator_id="operator-b",
        )

        result = await executor.execute_tools(
            _make_tool_calls(["dangerous"]),
            system_prompt="",
            source="text",
            conversation_session_id="thread-b",
            llm_call_context=llm_context,
        )

        assert result == "follow_up"
        assert tools.execute_calls == 0
        assert tools.global_probe_calls == 0
        tool_results = executor._conversation.add_tool_exchange.call_args.args[1]
        assert "scoped interaction context" in tool_results[0]["content"]

    async def test_later_turn_uses_exact_scoped_approval_id(self):
        tools = MagicMock()
        executor = _make_executor(tools=tools)
        prompt = InteractionTurnContext(
            thread_id="thread-a",
            turn_id="prompt-turn",
            channel="text",
            source="text",
            user_text="delete",
            operator_id="operator-7",
        )
        later = InteractionTurnContext(
            thread_id="thread-a",
            turn_id="confirm-turn",
            channel="text",
            source="text",
            user_text="确认执行",
            operator_id="operator-7",
        )
        scope = ApprovalScope(
            kind=ConfirmationKind.TOOL_APPROVAL,
            thread_id=prompt.thread_id,
            prompt_turn_id=prompt.turn_id,
            person_id=None,
            operator_id=prompt.operator_id,
            expires_at_monotonic=float("inf"),
            allows_short_reply=True,
            approval_id="approval-exact",
            subject="dangerous",
            risk_level="dangerous",
            payload_digest="sha256:payload",
        )
        tools.pending_approval_scope = MagicMock(
            side_effect=lambda *, interaction_context: (
                scope if interaction_context is later else None
            )
        )
        tools.handle_pending_input = MagicMock(
            side_effect=lambda user_text, *, interaction_context, approval_id: (
                "approved"
                if user_text == "确认执行"
                and interaction_context is later
                and approval_id == "approval-exact"
                else None
            )
        )
        audio = MagicMock()

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=later,
        )

        assert result == "approved"
        tools.handle_pending_input.assert_called_once_with(
            "确认执行",
            interaction_context=later,
            approval_id="approval-exact",
        )
        executor._conversation.add_user_message.assert_called_once_with(
            "确认执行",
            conversation_session_id="thread-a",
        )
        executor._conversation.add_assistant_message.assert_called_once_with(
            "approved",
            conversation_session_id="thread-a",
        )

    async def test_scoped_confirmation_does_not_use_legacy_global_handler(self):
        class _LegacyRegistry:
            def __init__(self) -> None:
                self.global_pending_thread = "thread-a"
                self.global_handler_calls = 0

            def handle_pending_input(self, user_text: str) -> str:
                del user_text
                self.global_handler_calls += 1
                return f"executed:{self.global_pending_thread}"

        tools = _LegacyRegistry()
        executor = _make_executor(tools=tools)
        audio = MagicMock()
        thread_b = InteractionTurnContext(
            thread_id="thread-b",
            turn_id="thread-b-confirm",
            channel="text",
            source="text",
            user_text="确认执行",
            operator_id="operator-b",
        )

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=thread_b,
        )

        assert result is None
        assert tools.global_handler_calls == 0
        audio.speak.assert_not_called()
        executor._conversation.add_user_message.assert_not_called()

    async def test_scoped_confirmation_rejects_unscoped_pending_handler(self):
        class _MixedRegistry:
            def __init__(self) -> None:
                self.global_handler_calls = 0

            def pending_approval_scope(
                self,
                interaction_context: InteractionTurnContext,
            ) -> ApprovalScope:
                return ApprovalScope(
                    kind=ConfirmationKind.TOOL_APPROVAL,
                    thread_id=interaction_context.thread_id,
                    prompt_turn_id="prompt-turn",
                    person_id=None,
                    operator_id=interaction_context.operator_id,
                    expires_at_monotonic=float("inf"),
                    allows_short_reply=True,
                    approval_id="approval-thread-b",
                    subject="dangerous",
                    risk_level="dangerous",
                    payload_digest="sha256:payload",
                )

            def handle_pending_input(self, user_text: str) -> str:
                del user_text
                self.global_handler_calls += 1
                return "executed:thread-a"

        tools = _MixedRegistry()
        executor = _make_executor(tools=tools)
        audio = MagicMock()
        thread_b = InteractionTurnContext(
            thread_id="thread-b",
            turn_id="thread-b-confirm",
            channel="text",
            source="text",
            user_text="确认执行",
            operator_id="operator-b",
        )

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=thread_b,
        )

        assert result is None
        assert tools.global_handler_calls == 0
        audio.speak.assert_not_called()

    async def test_scoped_confirmation_requires_exact_approval_id_support(self):
        class _ContextOnlyRegistry:
            def __init__(self) -> None:
                self.handler_calls = 0

            def pending_approval_scope(
                self,
                interaction_context: InteractionTurnContext,
            ) -> ApprovalScope:
                return ApprovalScope(
                    kind=ConfirmationKind.TOOL_APPROVAL,
                    thread_id=interaction_context.thread_id,
                    prompt_turn_id="prompt-turn",
                    person_id=None,
                    operator_id=interaction_context.operator_id,
                    expires_at_monotonic=float("inf"),
                    allows_short_reply=True,
                    approval_id="approval-thread-b",
                    subject="dangerous",
                    risk_level="dangerous",
                    payload_digest="sha256:payload",
                )

            def handle_pending_input(
                self,
                user_text: str,
                interaction_context: InteractionTurnContext,
            ) -> str:
                del user_text, interaction_context
                self.handler_calls += 1
                return "executed"

        tools = _ContextOnlyRegistry()
        executor = _make_executor(tools=tools)
        audio = MagicMock()
        thread_b = InteractionTurnContext(
            thread_id="thread-b",
            turn_id="thread-b-confirm",
            channel="text",
            source="text",
            user_text="确认执行",
            operator_id="operator-b",
        )

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=thread_b,
        )

        assert result is None
        assert tools.handler_calls == 0
        audio.speak.assert_not_called()

    async def test_unscoped_confirmation_preserves_legacy_global_handler(self):
        class _LegacyRegistry:
            def __init__(self) -> None:
                self.global_handler_calls = 0

            def handle_pending_input(self, user_text: str) -> str:
                self.global_handler_calls += 1
                return f"legacy-approved:{user_text}"

        tools = _LegacyRegistry()
        executor = _make_executor(tools=tools)
        audio = MagicMock()

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
        )

        assert result == "legacy-approved:确认执行"
        assert tools.global_handler_calls == 1
        audio.speak.assert_called_once_with("legacy-approved:确认执行")

    async def test_disappeared_scoped_approval_does_not_deliver_or_record(self):
        tools = MagicMock()
        tools.pending_approval_scope.return_value = None
        executor = _make_executor(tools=tools)
        audio = MagicMock()
        later = InteractionTurnContext(
            thread_id="thread-a",
            turn_id="confirm-turn",
            channel="text",
            source="text",
            user_text="确认执行",
        )

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=later,
        )

        assert result is None
        tools.handle_pending_input.assert_not_called()
        audio.speak.assert_not_called()
        executor._conversation.add_user_message.assert_not_called()

    async def test_cancel_during_voice_delivery_skips_history(self):
        token = asyncio.Event()
        tools = MagicMock()
        tools.pending_approval_scope.return_value = MagicMock(approval_id="approval-exact")
        tools.handle_pending_input.return_value = "approved"
        executor = _make_executor(tools=tools)
        audio = MagicMock()
        audio.wait_speaking_done.side_effect = token.set
        later = InteractionTurnContext(
            thread_id="thread-a",
            turn_id="confirm-turn",
            channel="voice",
            source="voice",
            user_text="确认执行",
            cancel_token=token,
        )

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="voice",
            interaction_context=later,
        )

        assert result == ""
        audio.stop_playback.assert_called_once_with()
        executor._conversation.add_user_message.assert_not_called()
        executor._conversation.add_assistant_message.assert_not_called()

    async def test_real_registry_round_trip_is_scoped_to_exact_later_turn(self):
        class _DangerousTool(BaseTool):
            name = "dangerous"
            description = "dangerous"
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
            }
            safety_level = "dangerous"

            def __init__(self) -> None:
                self.calls = 0

            def execute(self, **kwargs: Any) -> str:
                del kwargs
                self.calls += 1
                return "executed"

        registry = ToolRegistry(
            config={
                "require_confirmation_levels": ["dangerous"],
                "confirmation_phrases": ["确认执行"],
                "rejection_phrases": ["取消"],
                "approval_timeout_seconds": 30.0,
            }
        )
        tool = _DangerousTool()
        registry.register(tool)
        executor = _make_executor(tools=registry)
        executor._general_tool_max_safety_level = "dangerous"
        anonymous_context = LLMCallContext(
            session_id="public-voice",
            turn_id="anonymous-turn",
            channel="voice",
        )
        anonymous_result = await executor.execute_tools(
            _make_tool_calls(["dangerous"]),
            system_prompt="",
            source="voice",
            conversation_session_id="public-voice",
            llm_call_context=anonymous_context,
        )

        assert anonymous_result == "follow_up"
        assert tool.calls == 0
        anonymous_policy_context = InteractionTurnContext(
            thread_id="public-voice",
            turn_id="anonymous-turn",
            channel="voice",
            source="voice",
            user_text="",
        )
        assert registry.pending_approval_scope(anonymous_policy_context) is None
        tool_results = executor._conversation.add_tool_exchange.call_args.args[1]
        assert "已认证操作员" in tool_results[0]["content"]
        prompt_context = LLMCallContext(
            session_id="thread-a",
            turn_id="prompt-turn",
            channel="text",
            operator_id="operator-a",
        )

        prompt = await executor.execute_tools(
            _make_tool_calls(["dangerous"]),
            system_prompt="",
            source="text",
            conversation_session_id="thread-a",
            llm_call_context=prompt_context,
        )

        assert prompt.startswith("[Approval Required]")
        assert tool.calls == 0
        later = InteractionTurnContext(
            thread_id="thread-a",
            turn_id="confirm-turn",
            channel="text",
            source="text",
            user_text="确认执行",
            operator_id="operator-a",
        )
        audio = MagicMock()

        result = await executor.handle_pending_tool_response(
            "确认执行",
            audio=audio,
            source="text",
            interaction_context=later,
        )

        assert result == "executed"
        assert tool.calls == 1
        assert registry.pending_approval_scope(later) is None


# ---------------------------------------------------------------------------
# respond_without_llm
# ---------------------------------------------------------------------------


class TestRespondWithoutLlm:
    async def test_speaks_and_records(self):
        audio = MagicMock()
        audio.wait_speaking_done = MagicMock()
        executor = _make_executor()
        result = await executor.respond_without_llm("用户说了什么", "机器人回答了什么", audio=audio)
        audio.drain_buffers.assert_called_once()
        audio.speak.assert_called_once_with("机器人回答了什么")
        executor._conversation.add_user_message.assert_called_once_with("用户说了什么")
        executor._conversation.add_assistant_message.assert_called_once_with("机器人回答了什么")
        assert result == "机器人回答了什么"

    async def test_waits_for_speaking_done_in_voice_mode(self):
        audio = MagicMock()
        audio.wait_speaking_done = MagicMock()
        executor = _make_executor()
        await executor.respond_without_llm("q", "a", audio=audio, source="voice")
        audio.wait_speaking_done.assert_called_once()
        audio.stop_playback.assert_called_once()

    async def test_no_wait_in_text_mode(self):
        audio = MagicMock()
        executor = _make_executor()
        await executor.respond_without_llm("q", "a", audio=audio, source="text")
        audio.wait_speaking_done.assert_not_called()
        audio.stop_playback.assert_not_called()
