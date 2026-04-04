"""Tests for TurnExecutor — single-turn orchestration (memory → LLM → TTS → save)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.turn_executor import TurnExecutor


def _make_executor(**kwargs) -> TurnExecutor:
    """Build a TurnExecutor with all heavy deps mocked."""
    conversation = MagicMock()
    conversation.history = []
    conversation.add_user_message = MagicMock()
    conversation.add_assistant_message = MagicMock()
    conversation.get_messages = MagicMock(return_value=[])
    conversation.maybe_compress = AsyncMock()

    memory = MagicMock()
    memory.retrieve = AsyncMock(return_value="[memory context]")
    memory.save = AsyncMock()

    audio = MagicMock()
    audio.start_playback = MagicMock()
    audio.stop_playback = MagicMock()
    audio.drain_buffers = MagicMock()
    audio.speak = MagicMock()
    audio.wait_speaking_done = MagicMock()

    prompt_builder = MagicMock()
    prompt_builder.build_system_prompt = MagicMock(return_value="You are a robot.")
    prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs: msgs)

    stream_processor = MagicMock()
    stream_processor.stream_with_tools = AsyncMock(return_value="robot answer")
    stream_processor.stream_and_speak = AsyncMock(return_value="robot answer")

    defaults = dict(
        llm=MagicMock(),
        conversation=conversation,
        memory=memory,
        audio=audio,
        prompt_builder=prompt_builder,
        stream_processor=stream_processor,
    )
    defaults.update(kwargs)
    return TurnExecutor(**defaults)


class TestProcessHappyPath:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        te = _make_executor()
        result = await te.process("hello robot")
        assert result == "robot answer"

    @pytest.mark.asyncio
    async def test_adds_user_message_to_conversation(self):
        te = _make_executor()
        await te.process("hello")
        te._conversation.add_user_message.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_adds_assistant_message_to_conversation(self):
        te = _make_executor()
        await te.process("hello")
        te._conversation.add_assistant_message.assert_called_once_with("robot answer")

    @pytest.mark.asyncio
    async def test_audio_playback_started_and_stopped(self):
        te = _make_executor()
        await te.process("hello")
        te._audio.start_playback.assert_called_once()
        te._audio.stop_playback.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_retrieve_called(self):
        te = _make_executor()
        await te.process("where is the warehouse?")
        te._memory.retrieve.assert_called_once_with("where is the warehouse?")

    @pytest.mark.asyncio
    async def test_memory_save_called_after_response(self):
        te = _make_executor()
        await te.process("hello")
        # save is a background task — wait for it
        await asyncio.gather(*te._pending_tasks, return_exceptions=True)
        te._memory.save.assert_called_once_with("hello", "robot answer")

    @pytest.mark.asyncio
    async def test_prompt_builder_called(self):
        te = _make_executor()
        await te.process("hello")
        te._prompt_builder.build_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_last_spoken_text_updated(self):
        te = _make_executor()
        await te.process("hello")
        assert te.last_spoken_text == "robot answer"

    @pytest.mark.asyncio
    async def test_voice_source_waits_for_speaking(self):
        te = _make_executor()
        await te.process("hello", source="voice")
        te._audio.wait_speaking_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_source_does_not_wait_for_speaking(self):
        te = _make_executor()
        await te.process("hello", source="text")
        te._audio.wait_speaking_done.assert_not_called()


class TestCancelToken:
    @pytest.mark.asyncio
    async def test_skips_turn_when_token_set(self):
        token = asyncio.Event()
        token.set()
        te = _make_executor(cancel_token=token)
        result = await te.process("hello")
        assert result == ""
        te._stream_processor.stream_with_tools.assert_not_called()


class TestSilentMarker:
    @pytest.mark.asyncio
    async def test_silent_response_returns_empty(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(return_value="[SILENT] ignored")
        te = _make_executor(stream_processor=stream_processor)
        te._conversation.history = [{"role": "user", "content": "hello"}]
        result = await te.process("hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_silent_response_does_not_save_assistant_message(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(return_value="[SILENT] ignored")
        te = _make_executor(stream_processor=stream_processor)
        te._conversation.history = [{"role": "user", "content": "hello"}]
        await te.process("hello")
        te._conversation.add_assistant_message.assert_not_called()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_error_returns_error_message(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(
            side_effect=RuntimeError("connection failed")
        )
        te = _make_executor(stream_processor=stream_processor)
        result = await te.process("hello")
        assert result.startswith("[系统错误]")

    @pytest.mark.asyncio
    async def test_llm_error_speaks_error_via_audio(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        te = _make_executor(stream_processor=stream_processor)
        await te.process("hello")
        te._audio.speak.assert_called()

    @pytest.mark.asyncio
    async def test_llm_error_stops_playback(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        te = _make_executor(stream_processor=stream_processor)
        await te.process("hello")
        te._audio.stop_playback.assert_called()

    @pytest.mark.asyncio
    async def test_memory_error_does_not_crash_turn(self):
        memory = MagicMock()
        memory.retrieve = AsyncMock(side_effect=RuntimeError("DB down"))
        memory.save = AsyncMock()
        te = _make_executor(memory=memory)
        result = await te.process("hello")
        # Should succeed despite memory failure
        assert result == "robot answer"


class TestPrebuiltMemoryTask:
    @pytest.mark.asyncio
    async def test_accepts_prebuilt_memory_task(self):
        te = _make_executor()

        async def coro():
            return "[cached context]"

        task = asyncio.create_task(coro())
        result = await te.process("hello", memory_task=task)
        # memory.retrieve should not have been called (task provided)
        te._memory.retrieve.assert_not_called()
        assert result == "robot answer"


class TestHooks:
    @pytest.mark.asyncio
    async def test_pre_turn_hook_skip_returns_empty(self):
        from askme.pipeline.hooks import PipelineHooks
        hooks = PipelineHooks()

        async def skip_hook(ctx):
            return True  # request skip

        hooks.on_pre_turn(skip_hook)
        te = _make_executor(hooks=hooks)
        result = await te.process("hello")
        assert result == ""
        te._stream_processor.stream_with_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_turn_hook_fires_after_response(self):
        from askme.pipeline.hooks import PipelineHooks
        fired: list[str] = []
        hooks = PipelineHooks()

        async def post_hook(ctx, reply):
            fired.append(reply)

        hooks.on_post_turn(post_hook)
        te = _make_executor(hooks=hooks)
        await te.process("hello")
        assert fired == ["robot answer"]


class TestSetAudio:
    def test_set_audio_replaces_audio(self):
        te = _make_executor()
        new_audio = MagicMock()
        te.set_audio(new_audio)
        assert te._audio is new_audio


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_tasks(self):
        te = _make_executor()

        async def long_task():
            await asyncio.sleep(100)

        t = te._track_task(long_task(), name="long")
        assert not t.done()
        await te.shutdown()
        assert t.cancelled()
        assert len(te._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_tasks(self):
        te = _make_executor()
        # Should not raise
        await te.shutdown()
