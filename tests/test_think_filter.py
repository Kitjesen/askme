"""Tests for _ThinkFilter — incremental <think>...</think> block stripper."""

from __future__ import annotations

from askme.pipeline.stream_processor import _ThinkFilter


def _feed_all(chunks: list[str]) -> str:
    """Feed all chunks through a fresh filter and flush; return concatenated output."""
    tf = _ThinkFilter()
    out = "".join(tf.feed(c) for c in chunks)
    return out + tf.flush()


class TestBasicFiltering:
    def test_plain_text_passes_through(self):
        assert _feed_all(["hello world"]) == "hello world"

    def test_think_block_removed_single_chunk(self):
        assert _feed_all(["<think>some thought</think>answer"]) == "answer"

    def test_think_block_at_start(self):
        result = _feed_all(["<think>reasoning</think>final"])
        assert result == "final"

    def test_think_block_in_middle(self):
        result = _feed_all(["prefix<think>thought</think>suffix"])
        assert result == "prefixsuffix"

    def test_think_block_at_end(self):
        result = _feed_all(["text<think>thinking</think>"])
        assert result == "text"

    def test_empty_think_block(self):
        result = _feed_all(["a<think></think>b"])
        assert result == "ab"

    def test_no_think_block(self):
        result = _feed_all(["just plain text"])
        assert result == "just plain text"


class TestMultipleThinkBlocks:
    def test_two_think_blocks(self):
        result = _feed_all(["a<think>x</think>b<think>y</think>c"])
        assert result == "abc"

    def test_consecutive_think_blocks(self):
        result = _feed_all(["<think>a</think><think>b</think>end"])
        assert result == "end"


class TestStreamedChunks:
    def test_think_tag_split_across_chunks(self):
        """<think> tag split between chunks."""
        result = _feed_all(["<thi", "nk>thought</think>result"])
        assert "result" in result
        assert "thought" not in result

    def test_close_tag_split_across_chunks(self):
        """</think> closing tag split between chunks."""
        result = _feed_all(["<think>xyz</thi", "nk>answer"])
        assert "answer" in result
        assert "xyz" not in result

    def test_content_before_and_after_split_think(self):
        chunks = ["before <", "think>", "thought", "</think>", " after"]
        result = _feed_all(chunks)
        assert "before" in result
        assert "after" in result
        assert "thought" not in result

    def test_answer_split_across_chunks(self):
        """Normal text (not think) split across many chunks."""
        chunks = list("hello world")  # one char per chunk
        result = _feed_all(chunks)
        assert result == "hello world"

    def test_single_char_chunks_with_think(self):
        text = "<think>x</think>answer"
        chunks = list(text)
        result = _feed_all(chunks)
        assert "answer" in result
        assert "x" not in result


class TestRealWorldPatterns:
    def test_minimax_reasoning_pattern(self):
        """Simulates typical MiniMax-M2.5 output: <think> block then answer."""
        result = _feed_all([
            "<think>\n",
            "Let me think step by step...\n",
            "The answer is 42.\n",
            "</think>\n",
            "42",
        ])
        assert "42" in result
        assert "step by step" not in result

    def test_chinese_content_after_think(self):
        result = _feed_all(["<think>thinking</think>你好世界"])
        assert result == "你好世界"

    def test_multiline_think_block(self):
        result = _feed_all(["<think>\nline1\nline2\n</think>actual answer"])
        assert "actual answer" in result
        assert "line1" not in result


class TestFlush:
    def test_flush_returns_buffered_text(self):
        tf = _ThinkFilter()
        fed = tf.feed("hello wor")  # last 7 chars are held back in buffer
        flushed = tf.flush()
        # Total output = feed result + flush result
        assert fed + flushed == "hello wor"

    def test_flush_inside_think_returns_empty(self):
        tf = _ThinkFilter()
        tf.feed("<think>unfinished")
        assert tf.flush() == ""

    def test_flush_after_complete_text(self):
        tf = _ThinkFilter()
        result = tf.feed("short")  # < 7 chars, all buffered
        flushed = tf.flush()
        assert result + flushed == "short"

    def test_double_flush_is_empty(self):
        tf = _ThinkFilter()
        tf.feed("text")
        tf.flush()
        assert tf.flush() == ""


class TestReset:
    def test_reset_clears_state(self):
        tf = _ThinkFilter()
        tf.feed("<think>partial")  # in think mode
        tf.reset()
        assert tf._in_think is False
        assert tf._buf == ""

    def test_reset_allows_fresh_start(self):
        tf = _ThinkFilter()
        tf.feed("<think>partial")
        tf.reset()
        out = tf.feed("clean text") + tf.flush()
        assert "clean text" in out
        assert "partial" not in out

    def test_reset_mid_think_then_new_think(self):
        tf = _ThinkFilter()
        tf.feed("<think>thought</think>first")
        tf.reset()
        out = tf.feed("<think>second</think>answer") + tf.flush()
        assert "answer" in out
        assert "second" not in out
