"""Tests for PipelineHooks — decorator API, fire_* methods, ToolCallRecord."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from askme.pipeline.hooks import (
    _PROCEED,
    PipelineHooks,
    ToolCallRecord,
    _ProceedType,
    dataclasses_replace,
)

# ── ToolCallRecord ────────────────────────────────────────────────────────────

class TestToolCallRecord:
    def test_frozen(self):
        record = ToolCallRecord(
            call_id="c1", tool_name="get_time", arguments="{}", result="12:00",
            elapsed_ms=5.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            record.result = "changed"  # type: ignore

    def test_defaults(self):
        record = ToolCallRecord(
            call_id="c1", tool_name="t", arguments="{}", result="ok", elapsed_ms=1.0,
        )
        assert record.timed_out is False
        assert record.cancelled is False

    def test_timed_out_flag(self):
        record = ToolCallRecord(
            call_id="c2", tool_name="slow", arguments="{}", result="", elapsed_ms=30000.0,
            timed_out=True,
        )
        assert record.timed_out is True

    def test_cancelled_flag(self):
        record = ToolCallRecord(
            call_id="c3", tool_name="nav", arguments="{}", result="", elapsed_ms=0.0,
            cancelled=True,
        )
        assert record.cancelled is True


# ── _ProceedType singleton ────────────────────────────────────────────────────

class TestProceedType:
    def test_singleton(self):
        a = _ProceedType()
        b = _ProceedType()
        assert a is b

    def test_is_proceed_sentinel(self):
        assert _PROCEED is _ProceedType()

    def test_repr(self):
        assert repr(_PROCEED) == "<_PROCEED>"


# ── dataclasses_replace ───────────────────────────────────────────────────────

class TestDataclassesReplace:
    def test_replaces_field(self):
        record = ToolCallRecord(
            call_id="c1", tool_name="t", arguments="{}", result="original", elapsed_ms=1.0
        )
        updated = dataclasses_replace(record, result="updated")
        assert updated.result == "updated"
        assert record.result == "original"  # original unchanged

    def test_other_fields_preserved(self):
        record = ToolCallRecord(
            call_id="c1", tool_name="t", arguments="{}", result="r", elapsed_ms=10.0,
            timed_out=True,
        )
        updated = dataclasses_replace(record, result="new")
        assert updated.call_id == "c1"
        assert updated.timed_out is True


# ── PipelineHooks — decorator API ─────────────────────────────────────────────

class TestDecoratorApi:
    def test_on_pre_turn_registers(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def my_hook(ctx):
            pass

        assert my_hook in hooks.pre_turn

    def test_on_post_turn_registers(self):
        hooks = PipelineHooks()

        @hooks.on_post_turn
        async def my_hook(ctx, resp):
            pass

        assert my_hook in hooks.post_turn

    def test_on_pre_tool_registers(self):
        hooks = PipelineHooks()

        @hooks.on_pre_tool
        async def my_hook(record):
            return None

        assert my_hook in hooks.pre_tool

    def test_on_post_tool_registers(self):
        hooks = PipelineHooks()

        @hooks.on_post_tool
        async def my_hook(record):
            return record.result

        assert my_hook in hooks.post_tool

    def test_on_estop_registers(self):
        hooks = PipelineHooks()

        @hooks.on_estop
        def my_hook():
            pass

        assert my_hook in hooks.estop

    def test_decorator_returns_function(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def fn(ctx):
            pass

        assert fn is fn  # decorator returns the same fn

    def test_multiple_hooks_registered(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def h1(ctx):
            pass

        @hooks.on_pre_turn
        async def h2(ctx):
            pass

        assert len(hooks.pre_turn) == 2


# ── fire_pre_turn ─────────────────────────────────────────────────────────────

class TestFirePreTurn:
    async def test_no_hooks_returns_false(self):
        hooks = PipelineHooks()
        ctx = MagicMock()
        result = await hooks.fire_pre_turn(ctx)
        assert result is False

    async def test_hook_returns_true_skips(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def skipper(ctx):
            return True

        result = await hooks.fire_pre_turn(MagicMock())
        assert result is True

    async def test_hook_returns_none_proceeds(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def pass_through(ctx):
            return None

        result = await hooks.fire_pre_turn(MagicMock())
        assert result is False

    async def test_hook_exception_swallowed(self):
        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def crasher(ctx):
            raise RuntimeError("hook exploded")

        # Should not raise
        result = await hooks.fire_pre_turn(MagicMock())
        assert result is False

    async def test_second_hook_runs_after_exception(self):
        hooks = PipelineHooks()
        called = []

        @hooks.on_pre_turn
        async def crasher(ctx):
            raise RuntimeError("crash")

        @hooks.on_pre_turn
        async def second(ctx):
            called.append(True)

        await hooks.fire_pre_turn(MagicMock())
        assert called == [True]

    async def test_first_true_short_circuits(self):
        hooks = PipelineHooks()
        called = []

        @hooks.on_pre_turn
        async def skip(ctx):
            return True

        @hooks.on_pre_turn
        async def should_not_run(ctx):
            called.append(True)

        result = await hooks.fire_pre_turn(MagicMock())
        assert result is True
        assert called == []  # second hook not called


# ── fire_post_turn ────────────────────────────────────────────────────────────

class TestFirePostTurn:
    async def test_no_hooks_no_crash(self):
        hooks = PipelineHooks()
        await hooks.fire_post_turn(MagicMock(), "response text")

    async def test_hook_called_with_ctx_and_response(self):
        hooks = PipelineHooks()
        received = []

        @hooks.on_post_turn
        async def capture(ctx, resp):
            received.append(resp)

        await hooks.fire_post_turn(MagicMock(), "hello world")
        assert received == ["hello world"]

    async def test_exception_swallowed(self):
        hooks = PipelineHooks()

        @hooks.on_post_turn
        async def crasher(ctx, resp):
            raise ValueError("oops")

        await hooks.fire_post_turn(MagicMock(), "text")  # Should not raise


# ── fire_pre_tool ─────────────────────────────────────────────────────────────

class TestFirePreTool:
    def _record(self) -> ToolCallRecord:
        return ToolCallRecord(
            call_id="c1", tool_name="test_tool", arguments="{}", result="", elapsed_ms=0.0
        )

    async def test_no_hooks_returns_proceed(self):
        hooks = PipelineHooks()
        result = await hooks.fire_pre_tool(self._record())
        assert result is _PROCEED

    async def test_hook_returning_none_gives_proceed(self):
        hooks = PipelineHooks()

        @hooks.on_pre_tool
        async def no_override(record):
            return None

        result = await hooks.fire_pre_tool(self._record())
        assert result is _PROCEED

    async def test_hook_returning_string_overrides(self):
        hooks = PipelineHooks()

        @hooks.on_pre_tool
        async def override(record):
            return "blocked result"

        result = await hooks.fire_pre_tool(self._record())
        assert result == "blocked result"

    async def test_first_override_wins(self):
        hooks = PipelineHooks()

        @hooks.on_pre_tool
        async def first(record):
            return "first override"

        @hooks.on_pre_tool
        async def second(record):
            return "second override"

        result = await hooks.fire_pre_tool(self._record())
        assert result == "first override"

    async def test_exception_swallowed_returns_proceed(self):
        hooks = PipelineHooks()

        @hooks.on_pre_tool
        async def crasher(record):
            raise RuntimeError("crash")

        result = await hooks.fire_pre_tool(self._record())
        assert result is _PROCEED


# ── fire_post_tool ────────────────────────────────────────────────────────────

class TestFirePostTool:
    def _record(self, result="original") -> ToolCallRecord:
        return ToolCallRecord(
            call_id="c1", tool_name="tool", arguments="{}", result=result, elapsed_ms=5.0
        )

    async def test_no_hooks_returns_original_result(self):
        hooks = PipelineHooks()
        result = await hooks.fire_post_tool(self._record("hello"))
        assert result == "hello"

    async def test_hook_can_modify_result(self):
        hooks = PipelineHooks()

        @hooks.on_post_tool
        async def augment(record):
            return record.result + " [audited]"

        result = await hooks.fire_post_tool(self._record("data"))
        assert result == "data [audited]"

    async def test_chained_hooks(self):
        hooks = PipelineHooks()

        @hooks.on_post_tool
        async def h1(record):
            return record.result + "+h1"

        @hooks.on_post_tool
        async def h2(record):
            return record.result + "+h2"

        result = await hooks.fire_post_tool(self._record("base"))
        assert result == "base+h1+h2"

    async def test_exception_in_hook_uses_last_good_result(self):
        hooks = PipelineHooks()

        @hooks.on_post_tool
        async def crasher(record):
            raise RuntimeError("crash")

        result = await hooks.fire_post_tool(self._record("original"))
        assert result == "original"


# ── fire_estop ────────────────────────────────────────────────────────────────

class TestFireEstop:
    def test_no_hooks_no_crash(self):
        hooks = PipelineHooks()
        hooks.fire_estop()  # Should not raise

    def test_hook_called(self):
        hooks = PipelineHooks()
        called = []

        @hooks.on_estop
        def my_estop():
            called.append(True)

        hooks.fire_estop()
        assert called == [True]

    def test_multiple_hooks_all_called(self):
        hooks = PipelineHooks()
        called = []

        @hooks.on_estop
        def h1():
            called.append("h1")

        @hooks.on_estop
        def h2():
            called.append("h2")

        hooks.fire_estop()
        assert "h1" in called
        assert "h2" in called

    def test_exception_swallowed(self):
        hooks = PipelineHooks()

        @hooks.on_estop
        def crasher():
            raise RuntimeError("crash")

        hooks.fire_estop()  # Should not raise

    def test_second_hook_runs_after_exception(self):
        hooks = PipelineHooks()
        called = []

        @hooks.on_estop
        def crasher():
            raise RuntimeError("crash")

        @hooks.on_estop
        def second():
            called.append(True)

        hooks.fire_estop()
        assert called == [True]
