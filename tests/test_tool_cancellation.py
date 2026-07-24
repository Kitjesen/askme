"""Cancellation fences around synchronous tool execution."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.tool_executor import ToolExecutor
from askme.pipeline.hooks import PipelineHooks


def _make_executor(
    *,
    cancel_token: object,
    tools: MagicMock | None = None,
    hooks: PipelineHooks | None = None,
):
    tools = tools or MagicMock()
    tools.execute.return_value = "tool result"
    tools.has_pending_approval.return_value = False
    conversation = MagicMock()
    conversation.get_messages.return_value = []
    prompt_builder = MagicMock()
    prompt_builder.prepare_messages.return_value = []
    follow_up = AsyncMock(return_value="follow-up")
    executor = ToolExecutor(
        tools=tools,
        conversation=conversation,
        episodic=None,
        general_tool_max_safety_level="normal",
        prompt_builder=prompt_builder,
        stream_and_speak=follow_up,
        hooks=hooks,
        cancel_token=cancel_token,
    )
    return executor, conversation, follow_up


def _tool_call():
    return {0: {"id": "call-1", "name": "move", "arguments": "{}"}}


class TestToolCancellation:
    @pytest.mark.asyncio
    async def test_registry_receives_same_cancel_token_as_turn(self):
        token = threading.Event()
        tools = MagicMock()
        executor, _conversation, _follow_up = _make_executor(
            cancel_token=token, tools=tools
        )

        await executor.execute_tools(_tool_call(), "system")

        assert tools.execute.call_args.kwargs["cancel_token"] is token

    @pytest.mark.asyncio
    async def test_pending_approval_execution_receives_same_cancel_token(self):
        token = threading.Event()
        tools = MagicMock()
        tools.handle_pending_input.return_value = None
        executor, _conversation, _follow_up = _make_executor(
            cancel_token=token, tools=tools
        )

        assert await executor.handle_pending_tool_response(
            "确认执行",
            audio=MagicMock(),
        ) is None

        assert tools.handle_pending_input.call_args.kwargs["cancel_token"] is token

    @pytest.mark.asyncio
    async def test_cancelled_before_execution_skips_tool_and_follow_up(self):
        token = asyncio.Event()
        token.set()
        tools = MagicMock()
        executor, conversation, follow_up = _make_executor(
            cancel_token=token, tools=tools
        )

        result = await executor.execute_tools(_tool_call(), "system")

        assert result == ""
        tools.execute.assert_not_called()
        conversation.add_tool_exchange.assert_not_called()
        follow_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_from_post_tool_drops_result_and_remaining_tools(self):
        token = asyncio.Event()
        tools = MagicMock()
        tools.execute.return_value = "tool result"
        hooks = PipelineHooks()

        async def cancel_after_tool(record):
            token.set()
            return record.result

        hooks.on_post_tool(cancel_after_tool)
        executor, conversation, follow_up = _make_executor(
            cancel_token=token, tools=tools, hooks=hooks
        )
        calls = {
            0: {"id": "call-1", "name": "first", "arguments": "{}"},
            1: {"id": "call-2", "name": "second", "arguments": "{}"},
        }

        result = await executor.execute_tools(calls, "system")

        assert result == ""
        assert tools.execute.call_count == 1
        conversation.add_tool_exchange.assert_not_called()
        follow_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_from_pre_tool_prevents_tool_start(self):
        token = asyncio.Event()
        tools = MagicMock()
        hooks = PipelineHooks()

        async def cancel_before_tool(record):
            token.set()
            return None

        hooks.on_pre_tool(cancel_before_tool)
        executor, conversation, follow_up = _make_executor(
            cancel_token=token, tools=tools, hooks=hooks
        )

        result = await executor.execute_tools(_tool_call(), "system")

        assert result == ""
        tools.execute.assert_not_called()
        conversation.add_tool_exchange.assert_not_called()
        follow_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_before_exchange_prevents_history_and_follow_up(self):
        token = asyncio.Event()
        tools = MagicMock()
        tools.execute.return_value = "tool result"

        def cancel_at_exchange_boundary():
            token.set()
            return False

        tools.has_pending_approval.side_effect = cancel_at_exchange_boundary
        executor, conversation, follow_up = _make_executor(
            cancel_token=token, tools=tools
        )

        result = await executor.execute_tools(_tool_call(), "system")

        assert result == ""
        conversation.add_tool_exchange.assert_not_called()
        follow_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_while_building_follow_up_prevents_second_llm(self):
        token = asyncio.Event()
        executor, _conversation, follow_up = _make_executor(cancel_token=token)

        def cancel_during_prompt(*args, **kwargs):
            token.set()
            return []

        executor._prompt_builder.prepare_messages.side_effect = cancel_during_prompt

        result = await executor.execute_tools(_tool_call(), "system")

        assert result == ""
        follow_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_during_sync_tool_discards_completed_result(self):
        token = threading.Event()
        tools = MagicMock()

        def finish_after_cancel(*args, **kwargs):
            token.set()
            return "late tool result"

        tools.execute.side_effect = finish_after_cancel
        executor, conversation, follow_up = _make_executor(
            cancel_token=token, tools=tools
        )

        result = await executor.execute_tools(_tool_call(), "system")

        assert result == ""
        tools.execute.assert_called_once()
        conversation.add_tool_exchange.assert_not_called()
        follow_up.assert_not_awaited()
