"""Tests for ToolExecutor — pre/post hook wiring, timeout, approval flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from askme.pipeline.hooks import PipelineHooks, ToolCallRecord
from askme.pipeline.tool_executor import ToolExecutor


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
    return {
        i: {"name": names[i], "arguments": args, "id": ids[i]}
        for i in range(len(names))
    }


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    async def test_single_tool_called(self):
        executor = _make_executor()
        result = await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
        executor._tools.execute.assert_called_once_with(
            "nav_to", "{}", max_safety_level="normal"
        )
        assert result == "follow_up"

    async def test_result_added_to_conversation(self):
        executor = _make_executor()
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
        executor._conversation.add_tool_exchange.assert_called_once()

    async def test_multiple_tools_in_order(self):
        call_log: list[str] = []
        tools = MagicMock()
        tools.execute = MagicMock(side_effect=lambda name, args, **kw: call_log.append(name) or "ok")
        tools.has_pending_approval.return_value = False
        executor = _make_executor(tools=tools)
        await executor.execute_tools(
            _make_tool_calls(["a", "b", "c"]), system_prompt=""
        )
        assert call_log == ["a", "b", "c"]

    async def test_episodic_log_called(self):
        episodic = MagicMock()
        executor = _make_executor(episodic=episodic)
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
        assert episodic.log.call_count >= 2  # action + outcome

    async def test_no_episodic_is_safe(self):
        executor = _make_executor(episodic=None)
        result = await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
        assert result == "follow_up"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_records_error_in_conversation(self):
        """Timeout → tool result in conversation contains error; execute_tools still returns follow-up."""
        async def fake_wait_for(coro, timeout):
            raise TimeoutError()

        with patch("asyncio.wait_for", new=fake_wait_for):
            executor = _make_executor()
            await executor.execute_tools(
                _make_tool_calls(["slow_tool"]), system_prompt=""
            )

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
        result = await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
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
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
        executor._tools.execute.assert_called_once()

    async def test_no_hooks_proceeds_normally(self):
        executor = _make_executor(hooks=None)
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
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
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
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
        await executor.execute_tools(
            _make_tool_calls(["nav_to"]), system_prompt=""
        )
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


# ---------------------------------------------------------------------------
# respond_without_llm
# ---------------------------------------------------------------------------


class TestRespondWithoutLlm:
    async def test_speaks_and_records(self):
        audio = MagicMock()
        audio.wait_speaking_done = MagicMock()
        executor = _make_executor()
        result = await executor.respond_without_llm(
            "用户说了什么", "机器人回答了什么", audio=audio
        )
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
