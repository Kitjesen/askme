"""Tests for StreamProcessor — LLM stream handling, think filtering, TTS piping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askme.pipeline.stream_processor import StreamProcessor, _ThinkFilter

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(content: str | None = None, tool_calls=None) -> SimpleNamespace:
    """Build a minimal chunk that mimics openai.types.chat.ChatCompletionChunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _make_tool_call_delta(idx: int, id: str = "", name: str = "", arguments: str = ""):
    """Build a tool_call delta fragment."""
    func = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(index=idx, id=id, function=func)
    return tc


async def _stream_chunks(chunks):
    for c in chunks:
        yield c


def _make_processor(**kwargs) -> StreamProcessor:
    defaults = dict(
        llm=MagicMock(),
        audio=MagicMock(),
        tools=MagicMock(),
        tool_executor=MagicMock(),
        splitter=MagicMock(),
        general_tool_max_safety_level=3,
        max_response_chars=0,  # no truncation by default
        voice_model=None,
        cancel_token=None,
    )
    defaults.update(kwargs)
    # splitter.feed returns list of sentences; splitter.flush returns remainder
    defaults["splitter"].feed.side_effect = lambda text: [text] if text else []
    defaults["splitter"].flush.return_value = ""
    return StreamProcessor(**defaults)


# ── _ThinkFilter is already covered in test_think_filter.py.
# We include a minimal smoke test here for integration context. ──────────────

class TestThinkFilterSmoke:
    def test_passthrough_no_think(self):
        tf = _ThinkFilter()
        assert tf.feed("hello") == ""  # 5 chars → 0 emitted (7-char lookahead)
        assert tf.flush() == "hello"

    def test_strips_think_block(self):
        tf = _ThinkFilter()
        text = "<think>reasoning</think>answer"
        out = tf.feed(text)
        remainder = tf.flush()
        assert (out + remainder).strip() == "answer"


# ── consume_llm_stream ────────────────────────────────────────────────────────

class TestConsumeLlmStream:
    @pytest.mark.asyncio
    async def test_plain_text_accumulates(self):
        proc = _make_processor()
        # Provide > 7 chars so think filter emits something
        chunks = [_make_chunk("Hello, world! How are you?")]
        full, tool_calls = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert "Hello, world! How are you?" in full
        assert tool_calls == {}

    @pytest.mark.asyncio
    async def test_multiple_chunks_concatenated(self):
        proc = _make_processor()
        chunks = [
            _make_chunk("First chunk. "),
            _make_chunk("Second chunk."),
        ]
        full, _ = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert "First chunk." in full
        assert "Second chunk." in full

    @pytest.mark.asyncio
    async def test_audio_speak_called_for_text(self):
        proc = _make_processor()
        chunks = [_make_chunk("Hello, world! How are you doing?")]
        await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")
        proc._audio.speak.assert_called()

    @pytest.mark.asyncio
    async def test_none_content_ignored(self):
        proc = _make_processor()
        chunks = [_make_chunk(None), _make_chunk("real content here long enough")]
        full, _ = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert "real content" in full

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty(self):
        proc = _make_processor()
        full, tool_calls = await proc.consume_llm_stream(_stream_chunks([]))
        assert full == ""
        assert tool_calls == {}

    @pytest.mark.asyncio
    async def test_tool_calls_accumulated(self):
        proc = _make_processor()
        tc0 = _make_tool_call_delta(0, id="tc-1", name="navigate", arguments='{"dest":')
        tc0b = _make_tool_call_delta(0, arguments='"warehouse"}')
        chunks = [
            _make_chunk(tool_calls=[tc0]),
            _make_chunk(tool_calls=[tc0b]),
        ]
        _, tool_calls = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert 0 in tool_calls
        assert tool_calls[0]["name"] == "navigate"
        assert tool_calls[0]["arguments"] == '{"dest":"warehouse"}'
        assert tool_calls[0]["id"] == "tc-1"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_accumulate_separately(self):
        proc = _make_processor()
        tc0 = _make_tool_call_delta(0, id="tc-1", name="tool_a", arguments='{}')
        tc1 = _make_tool_call_delta(1, id="tc-2", name="tool_b", arguments='{}')
        chunks = [
            _make_chunk(tool_calls=[tc0]),
            _make_chunk(tool_calls=[tc1]),
        ]
        _, tool_calls = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool_a"
        assert tool_calls[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_tool_call_drains_audio(self):
        proc = _make_processor()
        # First speak some content, then a tool call arrives
        tc0 = _make_tool_call_delta(0, id="tc-1", name="nav", arguments='{}')
        chunks = [
            _make_chunk("Hello, world! How are you doing today here?"),
            _make_chunk(tool_calls=[tc0]),
        ]
        await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")
        proc._audio.drain_buffers.assert_called()

    @pytest.mark.asyncio
    async def test_truncation_voice_mode(self):
        """When voice char limit is hit, truncation hint is spoken."""
        audio = MagicMock()
        splitter = MagicMock()
        # Each call to splitter.feed returns the text as a single sentence
        splitter.feed.side_effect = lambda text: [text] if text else []
        splitter.flush.return_value = ""
        proc = _make_processor(audio=audio, splitter=splitter, max_response_chars=10)
        # Single chunk with 20 chars
        chunks = [_make_chunk("Hello world this is a long sentence to test truncation!")]
        await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")
        # The truncation hint should be spoken
        speak_calls = [c[0][0] for c in audio.speak.call_args_list]
        assert any(StreamProcessor.TRUNCATION_HINT in s for s in speak_calls)

    @pytest.mark.asyncio
    async def test_no_truncation_in_text_mode(self):
        """Text (non-voice) mode ignores char_limit even when max_response_chars is set."""
        audio = MagicMock()
        splitter = MagicMock()
        splitter.feed.side_effect = lambda text: [text] if text else []
        splitter.flush.return_value = ""
        proc = _make_processor(audio=audio, splitter=splitter, max_response_chars=1)
        chunks = [_make_chunk("Hello world, long sentence for testing text mode!")]
        await proc.consume_llm_stream(_stream_chunks(chunks), source="text")
        speak_calls = [c[0][0] for c in audio.speak.call_args_list]
        # Truncation hint should NOT appear in text mode
        assert not any(StreamProcessor.TRUNCATION_HINT in s for s in speak_calls)

    @pytest.mark.asyncio
    async def test_flush_at_end_emits_buffered_content(self):
        """Short text stuck in think-filter buffer is flushed at end."""
        proc = _make_processor()
        # 5 chars — entirely held in lookahead buffer, emitted on flush
        chunks = [_make_chunk("Hello")]
        full, _ = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert "Hello" in full


class TestSetAudio:
    def test_set_audio_replaces_audio(self):
        proc = _make_processor()
        new_audio = MagicMock()
        proc.set_audio(new_audio)
        assert proc._audio is new_audio


class TestReset:
    def test_reset_clears_think_filter_state(self):
        proc = _make_processor()
        proc._think_filter.feed("<think>partial")  # leave in think mode
        assert proc._think_filter._in_think is True
        proc.reset()
        assert proc._think_filter._in_think is False

    def test_reset_calls_splitter_reset(self):
        proc = _make_processor()
        proc.reset()
        proc._splitter.reset.assert_called()
