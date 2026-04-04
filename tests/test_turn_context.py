"""Tests for TurnContext dataclass and protocol re-exports."""

from __future__ import annotations

import asyncio

import pytest

from askme.pipeline.protocols import TurnContext
from askme.pipeline.hooks import PipelineHooks, ToolCallRecord


# ── TurnContext ───────────────────────────────────────────────────────────────

class TestTurnContext:
    def test_basic_creation(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="你好", source="voice", cancel_token=token)
        assert ctx.user_text == "你好"
        assert ctx.source == "voice"
        assert ctx.cancel_token is token

    def test_frozen(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="hello", source="text", cancel_token=token)
        with pytest.raises((AttributeError, TypeError)):
            ctx.user_text = "changed"  # type: ignore

    def test_voice_model_default_none(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="test", source="voice", cancel_token=token)
        assert ctx.voice_model is None

    def test_voice_model_custom(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="test", source="voice", cancel_token=token,
                         voice_model="MiniMax-M2.7")
        assert ctx.voice_model == "MiniMax-M2.7"

    def test_cancel_token_is_asyncio_event(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="test", source="voice", cancel_token=token)
        assert isinstance(ctx.cancel_token, asyncio.Event)

    def test_cancel_token_can_be_set(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="test", source="voice", cancel_token=token)
        ctx.cancel_token.set()
        assert ctx.cancel_token.is_set()

    def test_source_text(self):
        token = asyncio.Event()
        ctx = TurnContext(user_text="hi", source="text", cancel_token=token)
        assert ctx.source == "text"

    def test_equality(self):
        token = asyncio.Event()
        ctx1 = TurnContext(user_text="hello", source="voice", cancel_token=token)
        ctx2 = TurnContext(user_text="hello", source="voice", cancel_token=token)
        assert ctx1 == ctx2

    def test_inequality_different_text(self):
        token = asyncio.Event()
        ctx1 = TurnContext(user_text="hello", source="voice", cancel_token=token)
        ctx2 = TurnContext(user_text="world", source="voice", cancel_token=token)
        assert ctx1 != ctx2


# ── Protocol re-exports ───────────────────────────────────────────────────────

class TestReexports:
    def test_pipeline_hooks_importable_from_protocols(self):
        from askme.pipeline.protocols import PipelineHooks as PH
        assert PH is PipelineHooks

    def test_tool_call_record_importable_from_protocols(self):
        from askme.pipeline.protocols import ToolCallRecord as TCR
        assert TCR is ToolCallRecord
