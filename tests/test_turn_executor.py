"""Tests for TurnExecutor — single-turn orchestration (memory → LLM → TTS → save)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from askme.pipeline.turn_executor import TurnExecutor


def _make_executor(**kwargs) -> TurnExecutor:
    """Build a TurnExecutor with all heavy deps mocked."""
    conversation = MagicMock()
    conversation.history = []
    conversation.max_history = 40
    session_histories: dict[str, list[dict[str, object]]] = {}

    def _history_for(conversation_session_id: str | None = None) -> list[dict[str, object]]:
        session = str(conversation_session_id or "").strip()
        if not session:
            return conversation.history
        return session_histories.setdefault(session, [])

    def _add_user_message(
        content: str,
        *,
        conversation_session_id: str | None = None,
    ) -> None:
        _history_for(conversation_session_id).append({"role": "user", "content": content})

    def _add_assistant_message(
        content: str,
        *,
        conversation_session_id: str | None = None,
    ) -> None:
        _history_for(conversation_session_id).append({"role": "assistant", "content": content})

    def _get_messages(
        system_prompt: str,
        *,
        conversation_session_id: str | None = None,
    ) -> list[dict[str, object]]:
        return [{"role": "system", "content": system_prompt}] + list(
            _history_for(conversation_session_id)
        )

    def _remove_latest_user_message(
        content: str,
        *,
        conversation_session_id: str | None = None,
    ) -> bool:
        history = _history_for(conversation_session_id)
        for i in range(len(history) - 1, -1, -1):
            item = history[i]
            if item.get("role") == "user" and item.get("content") == content:
                history.pop(i)
                return True
        return False

    conversation.add_user_message = MagicMock(side_effect=_add_user_message)
    conversation.add_assistant_message = MagicMock(side_effect=_add_assistant_message)
    conversation.get_messages = MagicMock(side_effect=_get_messages)
    conversation.remove_latest_user_message = MagicMock(side_effect=_remove_latest_user_message)
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
    prompt_builder.build_forced_rag_reply = MagicMock(return_value="")
    prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)

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
    async def test_passes_memory_answer_policy_to_prompt_builder(self):
        memory = MagicMock()
        memory.retrieve = AsyncMock(return_value="")
        memory.save = AsyncMock()
        memory.health.return_value = {
            "last_answer_policy": {
                "state": "stale",
                "action": "refuse_and_request_update",
                "reason": "expired",
            },
        }
        te = _make_executor(memory=memory)
        await te.process("where is A 区?")
        _, kwargs = te._prompt_builder.build_system_prompt.call_args
        assert kwargs["rag_policy"]["state"] == "stale"

    @pytest.mark.asyncio
    async def test_forced_rag_reply_skips_llm_and_saves_reply(self):
        memory = MagicMock()
        memory.retrieve = AsyncMock(return_value="")
        memory.save = AsyncMock()
        memory.health.return_value = {
            "last_answer_policy": {
                "state": "conflict",
                "action": "clarify",
                "reason": "conflict:route",
            },
        }
        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(return_value="这条路线信息有冲突，请管理员确认。")
        prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(return_value="错误的自由回答")
        te = _make_executor(
            memory=memory,
            prompt_builder=prompt_builder,
            stream_processor=stream_processor,
        )

        result = await te.process("A 区怎么走？", source="text")

        assert result == "这条路线信息有冲突，请管理员确认。"
        stream_processor.stream_with_tools.assert_not_called()
        te._conversation.add_assistant_message.assert_called_once_with(result)
        await asyncio.gather(*te._pending_tasks, return_exceptions=True)
        memory.save.assert_called_once_with("A 区怎么走？", result)

    @pytest.mark.asyncio
    async def test_forced_rag_reply_speaks_in_voice_mode(self):
        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(return_value="这条知识已过期，请先刷新。")
        prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(return_value="错误回答")
        te = _make_executor(prompt_builder=prompt_builder, stream_processor=stream_processor)

        result = await te.process("设备在哪里？", source="voice")

        assert result == "这条知识已过期，请先刷新。"
        te._audio.speak.assert_called_once_with(result)
        te._audio.wait_speaking_done.assert_called_once()
        stream_processor.stream_with_tools.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_text_source_does_not_touch_audio_playback(self):
        te = _make_executor()
        await te.process("hello", source="text")
        te._audio.drain_buffers.assert_not_called()
        te._audio.start_playback.assert_not_called()
        te._audio.stop_playback.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_conversation_session_uses_scoped_history(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(
            side_effect=["answer a1", "answer b1", "answer a2"]
        )
        te = _make_executor(stream_processor=stream_processor)

        await te.process("hello a", source="text", conversation_session_id="conv-a")
        await te.process("hello b", source="text", conversation_session_id="conv-b")
        await te.process("again a", source="text", conversation_session_id="conv-a")

        third_messages = stream_processor.stream_with_tools.call_args_list[2].args[0]
        assert {"role": "user", "content": "hello a"} in third_messages
        assert {"role": "assistant", "content": "answer a1"} in third_messages
        assert {"role": "user", "content": "again a"} in third_messages
        assert {"role": "user", "content": "hello b"} not in third_messages
        assert te._conversation.add_user_message.call_args_list == [
            call("hello a", conversation_session_id="conv-a"),
            call("hello b", conversation_session_id="conv-b"),
            call("again a", conversation_session_id="conv-a"),
        ]
        assert te._conversation.add_assistant_message.call_args_list == [
            call("answer a1", conversation_session_id="conv-a"),
            call("answer b1", conversation_session_id="conv-b"),
            call("answer a2", conversation_session_id="conv-a"),
        ]

    @pytest.mark.asyncio
    async def test_default_conversation_path_still_uses_conversation_manager(self):
        te = _make_executor()

        await te.process("hello", source="text")

        te._conversation.add_user_message.assert_called_once_with("hello")
        te._conversation.add_assistant_message.assert_called_once_with("robot answer")


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
            side_effect=TimeoutError()
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
