"""Tests for StreamProcessor — LLM stream handling, think filtering, TTS piping."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from askme.pipeline.stream_processor import StreamProcessor, _ThinkFilter

from askme.llm.core.contracts import LLMCallContext
from askme.pipeline.core import stream_processor as stream_processor_module

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
    async def test_turn_cancel_stops_future_chunks_and_tts(self):
        token = asyncio.Event()

        async def _cancelled_stream():
            yield _make_chunk("First sentence is long enough to speak.")
            token.set()
            yield _make_chunk("This late sentence must never be emitted.")

        proc = _make_processor()
        full, _ = await proc.consume_llm_stream(
            _cancelled_stream(),
            source="voice",
            turn_cancel_token=token,
        )

        assert "First sentence" in full
        assert "late sentence" not in full
        assert proc._audio.speak.call_count == 1

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
    async def test_audio_speak_called_for_voice(self):
        proc = _make_processor()
        chunks = [_make_chunk("Hello, world! How are you doing?")]
        await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")
        proc._audio.speak.assert_called()

    @pytest.mark.asyncio
    async def test_text_mode_does_not_speak_or_drain_audio(self):
        proc = _make_processor()
        tc0 = _make_tool_call_delta(0, id="tc-1", name="nav", arguments="{}")
        chunks = [
            _make_chunk("Hello, world! How are you doing today?"),
            _make_chunk(tool_calls=[tc0]),
        ]
        full, tool_calls = await proc.consume_llm_stream(_stream_chunks(chunks), source="text")

        assert "Hello, world!" in full
        assert tool_calls[0]["name"] == "nav"
        proc._audio.speak.assert_not_called()
        proc._audio.drain_buffers.assert_not_called()

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
        tc0 = _make_tool_call_delta(0, id="tc-1", name="tool_a", arguments="{}")
        tc1 = _make_tool_call_delta(1, id="tc-2", name="tool_b", arguments="{}")
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
        tc0 = _make_tool_call_delta(0, id="tc-1", name="nav", arguments="{}")
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
        audio.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_at_end_emits_buffered_content(self):
        """Short text stuck in think-filter buffer is flushed at end."""
        proc = _make_processor()
        # 5 chars — entirely held in lookahead buffer, emitted on flush
        chunks = [_make_chunk("Hello")]
        full, _ = await proc.consume_llm_stream(_stream_chunks(chunks))
        assert "Hello" in full

    @pytest.mark.asyncio
    async def test_voice_tts_coalesce_speaks_once(self):
        audio = MagicMock()
        splitter = MagicMock()
        splitter.feed.side_effect = lambda text: [text] if text else []
        splitter.flush.return_value = ""
        proc = _make_processor(
            audio=audio,
            splitter=splitter,
            voice_tts_coalesce=True,
        )
        chunks = [_make_chunk("First sentence. "), _make_chunk("Second sentence.")]

        full, _ = await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")

        assert "First sentence." in full
        assert "Second sentence." in full
        audio.speak.assert_called_once_with("First sentence. Second sentence.")

    @pytest.mark.asyncio
    async def test_voice_tts_coalesce_discards_pending_text_on_tool_call(self):
        audio = MagicMock()
        splitter = MagicMock()
        splitter.feed.side_effect = lambda text: [text] if text else []
        splitter.flush.return_value = ""
        proc = _make_processor(
            audio=audio,
            splitter=splitter,
            voice_tts_coalesce=True,
        )
        tc0 = _make_tool_call_delta(0, id="tc-1", name="nav", arguments="{}")
        chunks = [
            _make_chunk("Pending spoken text. "),
            _make_chunk(tool_calls=[tc0]),
        ]

        _, tool_calls = await proc.consume_llm_stream(_stream_chunks(chunks), source="voice")

        assert tool_calls[0]["name"] == "nav"
        audio.drain_buffers.assert_called()
        audio.speak.assert_not_called()


class TestStreamWithTools:
    @pytest.mark.asyncio
    async def test_turn_cancel_after_fuse_creation_suppresses_thinking_tone(self, monkeypatch):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.05)
        token = asyncio.Event()
        audio = MagicMock()
        proc = _make_processor(audio=audio)
        created_feedback_tasks: list[asyncio.Task[None]] = []
        create_thinking_task = proc._create_thinking_task

        def _capture_feedback_tasks(*args, **kwargs):
            tasks = create_thinking_task(*args, **kwargs)
            created_feedback_tasks.extend(task for task in tasks if task is not None)
            return tasks

        monkeypatch.setattr(proc, "_create_thinking_task", _capture_feedback_tasks)
        stream_started = asyncio.Event()
        stream_closed = asyncio.Event()

        async def _cancelled_before_first_chunk():
            stream_started.set()
            try:
                await asyncio.Event().wait()
                yield _make_chunk(None)
            finally:
                stream_closed.set()

        proc._llm.chat_stream.return_value = _cancelled_before_first_chunk()

        stream_task = asyncio.create_task(
            proc.stream_with_tools(
                [],
                "system",
                source="voice",
                turn_cancel_token=token,
            )
        )
        await asyncio.wait_for(stream_started.wait(), timeout=1.0)
        token.set()
        result = await asyncio.wait_for(stream_task, timeout=1.0)

        assert result == ""
        audio.play_thinking.assert_not_called()
        audio.cancel_processing_feedback.assert_not_called()
        assert stream_closed.is_set()
        assert all(task.done() for task in created_feedback_tasks)

    @pytest.mark.asyncio
    async def test_turn_cancel_at_audio_handoff_suppresses_thinking_tone(self, monkeypatch):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.0)
        audio = MagicMock()
        stream_may_finish = asyncio.Event()

        class _CancelAtHandoffToken:
            def __init__(self) -> None:
                self.probes = 0
                self.cancelled = False

            def is_set(self) -> bool:
                self.probes += 1
                if self.probes == 3:
                    self.cancelled = True
                    stream_may_finish.set()
                    return False
                return self.cancelled

            def set(self) -> None:
                self.cancelled = True

            def try_run(self, callback):
                if self.cancelled:
                    return False, None
                return True, callback()

        token = _CancelAtHandoffToken()
        proc = _make_processor(audio=audio)

        async def _cancelled_at_handoff():
            await stream_may_finish.wait()
            yield _make_chunk(None)

        proc._llm.chat_stream.return_value = _cancelled_at_handoff()

        result = await asyncio.wait_for(
            proc.stream_with_tools(
                [],
                "system",
                source="voice",
                turn_cancel_token=token,
            ),
            timeout=1.0,
        )

        assert result == ""
        audio.play_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_content_reaps_cancelled_feedback_tasks(self, monkeypatch):
        proc = _make_processor()
        created_tasks: list[asyncio.Task[None]] = []
        create_thinking_task = proc._create_thinking_task

        def _capture_tasks(*args, **kwargs):
            tasks = create_thinking_task(*args, **kwargs)
            created_tasks.extend(task for task in tasks if task is not None)
            return tasks

        monkeypatch.setattr(proc, "_create_thinking_task", _capture_tasks)
        proc._llm.chat_stream.return_value = _stream_chunks(
            [_make_chunk("这是一条立即返回的语义回复。")]
        )

        result = await proc.stream_with_tools([], "system", source="voice")

        assert result
        assert len(created_tasks) == 2
        assert all(task.done() for task in created_tasks)

    @pytest.mark.asyncio
    async def test_semantic_payload_gate_suppresses_tone_if_task_cancel_loses_race(
        self, monkeypatch
    ):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.0)
        monkeypatch.setattr(StreamProcessor, "SLOW_NETWORK_DELAY", 0.0)
        audio = MagicMock()
        proc = _make_processor(audio=audio)
        created_tasks: list[asyncio.Task[None]] = []
        create_thinking_task = proc._create_thinking_task

        def _capture_tasks(*args, **kwargs):
            tasks = create_thinking_task(*args, **kwargs)
            created_tasks.extend(task for task in tasks if task is not None)
            return tasks

        async def _leave_tasks_running(*tasks):
            del tasks

        monkeypatch.setattr(proc, "_create_thinking_task", _capture_tasks)
        monkeypatch.setattr(proc, "_cancel_and_wait", _leave_tasks_running)
        proc._llm.chat_stream.return_value = _stream_chunks(
            [_make_chunk("边界时刻已有有效语义内容。")]
        )

        result = await proc.stream_with_tools([], "system", source="voice")
        await asyncio.wait_for(asyncio.gather(*created_tasks), timeout=1.0)

        assert result
        audio.play_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_cancel_token_is_available_to_tool_follow_up(self):
        token = asyncio.Event()
        tool_executor = MagicMock()
        proc = _make_processor(tool_executor=tool_executor)

        async def _execute_tools(*args, **kwargs):
            del args, kwargs
            assert proc._tool_turn_cancel_token.get() is token
            return "follow-up"

        tool_executor.execute_tools = AsyncMock(side_effect=_execute_tools)
        proc.consume_llm_stream = AsyncMock(
            return_value=(
                "",
                {0: {"id": "tc-1", "name": "nav", "arguments": "{}"}},
            )
        )
        proc._llm.chat_stream.return_value = _stream_chunks([])

        result = await proc.stream_with_tools(
            [],
            "system",
            source="voice",
            turn_cancel_token=token,
        )

        assert result == "follow-up"

    @pytest.mark.asyncio
    async def test_tool_follow_up_gets_distinct_model_call_context(self):
        proc = _make_processor()
        parent = LLMCallContext(
            trace_id="0123456789abcdef0123456789abcdef",
            turn_id="turn-tool-1",
            call_id="root-call",
            purpose="assistant_response",
            channel="voice",
            request_class="voice_fast",
            latency_budget_ms=900,
        )
        proc._llm.chat_stream.return_value = _stream_chunks([])
        context_token = proc._tool_llm_call_context.set(parent)
        try:
            await proc.stream_and_speak([], source="voice")
        finally:
            proc._tool_llm_call_context.reset(context_token)

        child = proc._llm.chat_stream.call_args.kwargs["context"]
        assert child.trace_id == parent.trace_id
        assert child.turn_id == parent.turn_id
        assert child.purpose == "tool_followup"
        assert child.call_id != parent.call_id
        assert child.call_id

    @pytest.mark.asyncio
    async def test_forwards_conversation_session_to_tool_executor(self):
        tool_executor = MagicMock()
        tool_executor.execute_tools = AsyncMock(return_value="follow-up")
        proc = _make_processor(tool_executor=tool_executor)
        proc.consume_llm_stream = AsyncMock(
            return_value=(
                "",
                {0: {"id": "tc-1", "name": "nav", "arguments": "{}"}},
            )
        )
        proc._llm.chat_stream.return_value = _stream_chunks([])

        result = await proc.stream_with_tools(
            [],
            "system",
            source="text",
            conversation_session_id="conv-a",
        )

        assert result == "follow-up"
        tool_executor.execute_tools.assert_awaited_once_with(
            {0: {"id": "tc-1", "name": "nav", "arguments": "{}"}},
            "system",
            model=None,
            source="text",
            conversation_session_id="conv-a",
        )

    @pytest.mark.asyncio
    async def test_externally_armed_processing_feedback_suppresses_second_fuse(self, monkeypatch):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.01)
        audio = MagicMock()
        audio.processing_feedback_armed = True
        proc = _make_processor(audio=audio)

        async def _slow_stream():
            await asyncio.sleep(0.04)
            yield _make_chunk("好的。")

        proc._llm.chat_stream.return_value = _slow_stream()
        await proc.stream_with_tools([], "system", source="voice")

        audio.play_thinking.assert_not_called()
        audio.cancel_processing_feedback.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_processing_feedback_delay_comes_from_audio_duck_type(self, monkeypatch):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 5.0)
        audio = MagicMock()
        feedback_played = asyncio.Event()
        audio.play_thinking.side_effect = feedback_played.set
        audio.processing_feedback_delay_s = 0.01
        proc = _make_processor(audio=audio)

        async def _slow_stream():
            await asyncio.wait_for(feedback_played.wait(), timeout=0.5)
            yield _make_chunk("好的。")

        proc._llm.chat_stream.return_value = _slow_stream()
        await proc.stream_with_tools([], "system", source="voice")

        audio.play_thinking.assert_called()

    @pytest.mark.asyncio
    async def test_empty_delta_does_not_cancel_thinking_fuse(self, monkeypatch):
        """Empty keep-alive chunks must not suppress the long-tail thinking tone."""
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.05)
        audio = MagicMock()
        proc = _make_processor(audio=audio)

        async def _slow_stream():
            yield _make_chunk(None)  # keep-alive empty delta
            await asyncio.sleep(0.15)  # slower than the 0.05s fuse
            audio.play_thinking.assert_called_once()
            # Feedback uses the dedicated PCM/chime path, never semantic TTS.
            audio.speak.assert_not_called()
            yield _make_chunk("好的，请跟我来。")

        proc._llm.chat_stream.return_value = _slow_stream()
        result = await proc.stream_with_tools([], "system", source="voice")

        assert result
        audio.play_thinking.assert_called()
        audio.speak.assert_called()

    @pytest.mark.asyncio
    async def test_payload_arriving_at_audio_handoff_suppresses_thinking_tone(self, monkeypatch):
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.0)
        audio = MagicMock()
        proc = _make_processor(audio=audio)

        class _PayloadAtHandoff(asyncio.Event):
            def __init__(self) -> None:
                super().__init__()
                self.probes = 0

            def is_set(self) -> bool:
                self.probes += 1
                return self.probes >= 2

        payload_seen = _PayloadAtHandoff()
        thinking_task, slow_network_task = proc._create_thinking_task(
            semantic_payload_seen=payload_seen,
        )
        await asyncio.wait_for(thinking_task, timeout=1.0)

        assert payload_seen.probes == 2
        assert slow_network_task is None
        audio.play_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_ttft_ignores_empty_delta_until_first_payload(self, monkeypatch):
        tracer = MagicMock()
        monkeypatch.setattr(stream_processor_module, "get_tracer", lambda: tracer)
        proc = _make_processor()

        async def _stream_with_keep_alive():
            yield _make_chunk(None)
            tracer.record_span.assert_not_called()
            yield _make_chunk("这是第一个有效内容。")

        proc._llm.chat_stream.return_value = _stream_with_keep_alive()

        result = await proc.stream_with_tools([], "system", source="text")

        assert result
        tracer.record_span.assert_called_once()
        assert tracer.record_span.call_args.args[0] == "ttft"

    @pytest.mark.asyncio
    async def test_fast_content_cancels_thinking_fuse(self, monkeypatch):
        """A fast first content chunk prevents the thinking tone entirely."""
        monkeypatch.setattr(StreamProcessor, "THINKING_DELAY", 0.3)
        audio = MagicMock()
        proc = _make_processor(audio=audio)

        async def _fast_stream():
            yield _make_chunk(None)  # empty delta must not cancel the fuse
            yield _make_chunk("我叫小算。")

        proc._llm.chat_stream.return_value = _fast_stream()
        result = await proc.stream_with_tools([], "system", source="voice")

        assert result
        audio.play_thinking.assert_not_called()


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
