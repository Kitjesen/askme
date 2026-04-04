"""Tests for pipeline utility functions: strip_think_blocks, classify_*_error."""

from __future__ import annotations

import asyncio

from askme.pipeline.utils import (
    strip_think_blocks,
    classify_llm_error,
    classify_skill_error,
)


class TestStripThinkBlocks:
    def test_removes_single_block(self):
        assert strip_think_blocks("<think>thinking</think>answer") == "answer"

    def test_removes_multiple_blocks(self):
        assert strip_think_blocks("a<think>x</think>b<think>y</think>c") == "abc"

    def test_plain_text_unchanged(self):
        assert strip_think_blocks("hello world") == "hello world"

    def test_strips_surrounding_whitespace(self):
        assert strip_think_blocks("  <think>x</think>  answer  ") == "answer"

    def test_multiline_think_block(self):
        text = "<think>\nstep1\nstep2\n</think>\nfinal"
        assert strip_think_blocks(text) == "final"

    def test_empty_string(self):
        assert strip_think_blocks("") == ""

    def test_only_think_block(self):
        assert strip_think_blocks("<think>nothing</think>") == ""

    def test_nested_think_like_content(self):
        # Our regex is non-greedy — first </think> closes the block
        result = strip_think_blocks("<think>a</think>middle<think>b</think>end")
        assert result == "middleend"


class TestClassifyLlmError:
    def test_asyncio_timeout(self):
        msg = classify_llm_error(asyncio.TimeoutError())
        assert "超时" in msg or "想了" in msg

    def test_timeout_in_message(self):
        msg = classify_llm_error(RuntimeError("connection timeout"))
        assert "超时" in msg or "想了" in msg

    def test_connection_error(self):
        msg = classify_llm_error(RuntimeError("network error occurred"))
        assert "网络" in msg

    def test_connect_in_message(self):
        msg = classify_llm_error(RuntimeError("failed to connect to server"))
        assert "网络" in msg

    def test_generic_error_fallback(self):
        msg = classify_llm_error(ValueError("something went wrong"))
        assert len(msg) > 0  # returns some message

    def test_returns_string(self):
        assert isinstance(classify_llm_error(Exception("x")), str)


class TestClassifySkillError:
    def test_timeout_includes_skill_name(self):
        msg = classify_skill_error(asyncio.TimeoutError(), "navigate")
        assert "navigate" in msg
        assert "超时" in msg

    def test_connection_error_includes_skill_name(self):
        msg = classify_skill_error(RuntimeError("network failure"), "find_object")
        assert "find_object" in msg
        assert "网络" in msg

    def test_generic_error_includes_skill_name(self):
        msg = classify_skill_error(ValueError("bad state"), "patrol")
        assert "patrol" in msg

    def test_returns_string(self):
        assert isinstance(classify_skill_error(Exception("x"), "skill"), str)

    def test_connect_in_message(self):
        msg = classify_skill_error(RuntimeError("failed to connect"), "arm_move")
        assert "网络" in msg
        assert "arm_move" in msg
