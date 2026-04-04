"""Tests for PipelineHooks lifecycle system and ToolCallRecord."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from askme.pipeline.hooks import (
    PipelineHooks,
    ToolCallRecord,
    _PROCEED,
    _ProceedType,
    dataclasses_replace,
)


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_frozen(self):
        rec = ToolCallRecord(
            call_id="c1", tool_name="nav", arguments="{}", result="ok",
            elapsed_ms=12.5,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            rec.result = "new"  # type: ignore[misc]

    def test_defaults(self):
        rec = ToolCallRecord(
            call_id="c1", tool_name="t", arguments="{}", result="r", elapsed_ms=1.0,
        )
        assert rec.timed_out is False
        assert rec.cancelled is False

    def test_replace(self):
        rec = ToolCallRecord(
            call_id="c1", tool_name="t", arguments="{}", result="original", elapsed_ms=5.0,
        )
        new_rec = dataclasses_replace(rec, result="replaced")
        assert new_rec.result == "replaced"
        assert new_rec.call_id == "c1"  # unchanged


# ---------------------------------------------------------------------------
# _PROCEED sentinel
# ---------------------------------------------------------------------------


class TestProceedSentinel:
    def test_singleton(self):
        assert _ProceedType() is _PROCEED

    def test_repr(self):
        assert repr(_PROCEED) == "<_PROCEED>"


# ---------------------------------------------------------------------------
# PipelineHooks decorator registration
# ---------------------------------------------------------------------------


class TestHookRegistration:
    def test_on_pre_turn_appends(self):
        hooks = PipelineHooks()
        async def fn(ctx): pass
        hooks.on_pre_turn(fn)
        assert fn in hooks.pre_turn

    def test_on_post_turn_appends(self):
        hooks = PipelineHooks()
        async def fn(ctx, resp): pass
        hooks.on_post_turn(fn)
        assert fn in hooks.post_turn

    def test_on_pre_tool_appends(self):
        hooks = PipelineHooks()
        async def fn(rec): return None
        hooks.on_pre_tool(fn)
        assert fn in hooks.pre_tool

    def test_on_post_tool_appends(self):
        hooks = PipelineHooks()
        async def fn(rec): return rec.result
        hooks.on_post_tool(fn)
        assert fn in hooks.post_tool

    def test_on_estop_appends(self):
        hooks = PipelineHooks()
        def fn(): pass
        hooks.on_estop(fn)
        assert fn in hooks.estop

    def test_decorator_returns_fn(self):
        hooks = PipelineHooks()
        async def fn(ctx): pass
        result = hooks.on_pre_turn(fn)
        assert result is fn

    def test_multiple_hooks_registered(self):
        hooks = PipelineHooks()
        async def a(ctx): pass
        async def b(ctx): pass
        hooks.on_pre_turn(a)
        hooks.on_pre_turn(b)
        assert hooks.pre_turn == [a, b]


# ---------------------------------------------------------------------------
# fire_pre_turn
# ---------------------------------------------------------------------------


class TestFirePreTurn:
    async def test_no_hooks_returns_false(self):
        hooks = PipelineHooks()
        assert await hooks.fire_pre_turn(MagicMock()) is False

    async def test_hook_returning_none_does_not_skip(self):
        hooks = PipelineHooks()
        hooks.on_pre_turn(AsyncMock(return_value=None))
        assert await hooks.fire_pre_turn(MagicMock()) is False

    async def test_hook_returning_true_skips(self):
        hooks = PipelineHooks()
        hooks.on_pre_turn(AsyncMock(return_value=True))
        assert await hooks.fire_pre_turn(MagicMock()) is True

    async def test_first_true_stops_chain(self):
        hooks = PipelineHooks()
        second = AsyncMock(return_value=None)
        hooks.on_pre_turn(AsyncMock(return_value=True))
        hooks.on_pre_turn(second)
        result = await hooks.fire_pre_turn(MagicMock())
        assert result is True
        second.assert_not_called()

    async def test_hook_exception_is_swallowed(self):
        hooks = PipelineHooks()
        hooks.on_pre_turn(AsyncMock(side_effect=RuntimeError("boom")))
        # Should not raise, should not skip
        assert await hooks.fire_pre_turn(MagicMock()) is False

    async def test_ctx_passed_to_hook(self):
        hooks = PipelineHooks()
        spy = AsyncMock(return_value=None)
        hooks.on_pre_turn(spy)
        ctx = MagicMock()
        await hooks.fire_pre_turn(ctx)
        spy.assert_called_once_with(ctx)


# ---------------------------------------------------------------------------
# fire_post_turn
# ---------------------------------------------------------------------------


class TestFirePostTurn:
    async def test_no_hooks_is_noop(self):
        hooks = PipelineHooks()
        await hooks.fire_post_turn(MagicMock(), "reply")  # should not raise

    async def test_ctx_and_response_passed(self):
        hooks = PipelineHooks()
        spy = AsyncMock()
        hooks.on_post_turn(spy)
        ctx = MagicMock()
        await hooks.fire_post_turn(ctx, "hello")
        spy.assert_called_once_with(ctx, "hello")

    async def test_exception_swallowed(self):
        hooks = PipelineHooks()
        hooks.on_post_turn(AsyncMock(side_effect=ValueError("bad")))
        await hooks.fire_post_turn(MagicMock(), "r")  # no raise


# ---------------------------------------------------------------------------
# fire_pre_tool
# ---------------------------------------------------------------------------


def _make_record(**kwargs):
    defaults = dict(call_id="c1", tool_name="t", arguments="{}", result="", elapsed_ms=0.0)
    defaults.update(kwargs)
    return ToolCallRecord(**defaults)


class TestFirePreTool:
    async def test_no_hooks_returns_proceed(self):
        hooks = PipelineHooks()
        result = await hooks.fire_pre_tool(_make_record())
        assert result is _PROCEED

    async def test_hook_returning_none_continues_to_proceed(self):
        hooks = PipelineHooks()
        hooks.on_pre_tool(AsyncMock(return_value=None))
        result = await hooks.fire_pre_tool(_make_record())
        assert result is _PROCEED

    async def test_hook_returning_string_overrides(self):
        hooks = PipelineHooks()
        hooks.on_pre_tool(AsyncMock(return_value="intercepted"))
        result = await hooks.fire_pre_tool(_make_record())
        assert result == "intercepted"

    async def test_first_override_stops_chain(self):
        hooks = PipelineHooks()
        second = AsyncMock(return_value="second")
        hooks.on_pre_tool(AsyncMock(return_value="first"))
        hooks.on_pre_tool(second)
        result = await hooks.fire_pre_tool(_make_record())
        assert result == "first"
        second.assert_not_called()

    async def test_exception_swallowed_continues(self):
        hooks = PipelineHooks()
        hooks.on_pre_tool(AsyncMock(side_effect=RuntimeError("x")))
        result = await hooks.fire_pre_tool(_make_record())
        assert result is _PROCEED


# ---------------------------------------------------------------------------
# fire_post_tool
# ---------------------------------------------------------------------------


class TestFirePostTool:
    async def test_no_hooks_returns_original_result(self):
        hooks = PipelineHooks()
        rec = _make_record(result="orig")
        assert await hooks.fire_post_tool(rec) == "orig"

    async def test_hook_can_transform_result(self):
        hooks = PipelineHooks()
        async def transform(rec):
            return rec.result.upper()
        hooks.on_post_tool(transform)
        rec = _make_record(result="hello")
        assert await hooks.fire_post_tool(rec) == "HELLO"

    async def test_chained_transforms(self):
        hooks = PipelineHooks()
        hooks.on_post_tool(AsyncMock(side_effect=lambda r: r.result + "_a"))
        hooks.on_post_tool(AsyncMock(side_effect=lambda r: r.result + "_b"))
        rec = _make_record(result="x")
        assert await hooks.fire_post_tool(rec) == "x_a_b"

    async def test_exception_swallowed_uses_last_good_result(self):
        hooks = PipelineHooks()
        hooks.on_post_tool(AsyncMock(return_value="good"))
        hooks.on_post_tool(AsyncMock(side_effect=RuntimeError("bad")))
        rec = _make_record(result="orig")
        assert await hooks.fire_post_tool(rec) == "good"


# ---------------------------------------------------------------------------
# fire_estop
# ---------------------------------------------------------------------------


class TestFireEStop:
    def test_no_hooks_is_noop(self):
        hooks = PipelineHooks()
        hooks.fire_estop()  # should not raise

    def test_sync_hook_called(self):
        hooks = PipelineHooks()
        spy = MagicMock()
        hooks.on_estop(spy)
        hooks.fire_estop()
        spy.assert_called_once_with()

    def test_multiple_hooks_all_called(self):
        hooks = PipelineHooks()
        a, b = MagicMock(), MagicMock()
        hooks.on_estop(a)
        hooks.on_estop(b)
        hooks.fire_estop()
        a.assert_called_once()
        b.assert_called_once()

    def test_exception_swallowed(self):
        hooks = PipelineHooks()
        hooks.on_estop(MagicMock(side_effect=RuntimeError("panic")))
        hooks.fire_estop()  # no raise
