"""Tests for new BrainPipeline methods: _extract_semantic_target, _classify_error_message,
_classify_skill_error_message."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_pipeline():
    """Build a minimal BrainPipeline with all dependencies mocked."""
    from askme.pipeline.brain_pipeline import BrainPipeline

    llm = AsyncMock()
    conversation = MagicMock()
    conversation.get_messages.return_value = [{"role": "system", "content": "test"}]
    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value="")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.has_pending_approval.return_value = False
    skill_manager = MagicMock()
    skill_manager.get_skill_catalog.return_value = "none"
    skill_executor = MagicMock()
    audio = MagicMock()
    splitter = MagicMock()
    splitter.reset.return_value = None
    splitter.feed.return_value = []
    splitter.flush.return_value = None

    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
    )
    return pipeline


class TestExtractSemanticTarget:
    @pytest.mark.parametrize("text,expected", [
        ("导航到仓库", "仓库"),
        ("带我去门口", "门口"),
        ("前往会议室吧", "会议室"),
        ("去厨房", "厨房"),
    ])
    def test_navigation_patterns(self, text, expected):
        pipeline = _make_pipeline()
        result = pipeline._extract_semantic_target(text)
        assert result == expected, f"Input: {text!r}, expected {expected!r}, got {result!r}"

    def test_fallback_returns_original(self):
        """No navigation pattern → return the full input unchanged."""
        pipeline = _make_pipeline()
        text = "随便说点什么"
        result = pipeline._extract_semantic_target(text)
        assert result == text


class TestClassifyErrorMessage:
    def test_classify_error_network(self):
        """APIConnectionError → 返り値に '网络' を含む。"""
        from openai import APIConnectionError

        pipeline = _make_pipeline()
        # APIConnectionError requires a 'request' arg in some openai versions;
        # construct via subclass trick or pass a mock request
        exc = APIConnectionError(request=MagicMock())
        result = pipeline._classify_error_message(exc)
        assert "网络" in result

    def test_classify_error_timeout(self):
        """asyncio.TimeoutError → 返り値に '超时' を含む。"""
        pipeline = _make_pipeline()
        exc = asyncio.TimeoutError()
        result = pipeline._classify_error_message(exc)
        assert "超时" in result

    def test_classify_error_general(self):
        """RuntimeError → 返り値に '出错' を含む。"""
        pipeline = _make_pipeline()
        exc = RuntimeError("something went wrong")
        result = pipeline._classify_error_message(exc)
        assert "出错" in result


class TestClassifySkillErrorMessage:
    def test_classify_skill_error_timeout(self):
        """asyncio.TimeoutError → 返り値に '超时' を含む。"""
        pipeline = _make_pipeline()
        exc = asyncio.TimeoutError()
        result = pipeline._classify_skill_error_message(exc, "navigate")
        assert "超时" in result

    def test_classify_skill_error_general(self):
        """RuntimeError → 返り値に '失败' を含む。"""
        pipeline = _make_pipeline()
        exc = RuntimeError("unknown error")
        result = pipeline._classify_skill_error_message(exc, "navigate")
        assert "失败" in result
