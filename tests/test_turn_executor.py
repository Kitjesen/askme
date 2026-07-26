"""Tests for TurnExecutor — single-turn orchestration (memory → LLM → TTS → save)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from askme.pipeline.turn_executor import TurnExecutor

from askme.llm.core.contracts import LLMCallContext


def test_internal_tool_protocol_is_detected_before_user_output() -> None:
    assert TurnExecutor._contains_internal_protocol("正常中文回答") is False
    assert TurnExecutor._contains_internal_protocol("<｜｜DSML｜｜tool_calls>") is True
    assert TurnExecutor._contains_internal_protocol("<tool_call>internal</tool_call>") is True


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
        evidence: list[dict[str, object]] | None = None,
        rag: dict[str, object] | None = None,
        conversation_session_id: str | None = None,
    ) -> None:
        message: dict[str, object] = {"role": "assistant", "content": content}
        if evidence is not None:
            message["evidence"] = evidence
        if rag is not None:
            message["rag"] = rag
        _history_for(conversation_session_id).append(message)

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

    def _remove_latest_assistant_message(
        content: str,
        *,
        conversation_session_id: str | None = None,
    ) -> bool:
        history = _history_for(conversation_session_id)
        for i in range(len(history) - 1, -1, -1):
            item = history[i]
            if item.get("role") == "assistant" and item.get("content") == content:
                history.pop(i)
                return True
        return False

    conversation.add_user_message = MagicMock(side_effect=_add_user_message)
    conversation.add_assistant_message = MagicMock(side_effect=_add_assistant_message)
    conversation.get_messages = MagicMock(side_effect=_get_messages)
    conversation.remove_latest_user_message = MagicMock(side_effect=_remove_latest_user_message)
    conversation.remove_latest_assistant_message = MagicMock(
        side_effect=_remove_latest_assistant_message
    )
    conversation.maybe_compress = AsyncMock()

    memory = MagicMock()
    memory.retrieve = AsyncMock(return_value="[memory context]")
    memory.save = AsyncMock()
    memory.admit_turn = AsyncMock()

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
    async def test_voice_turn_propagates_model_call_context(self):
        executor = _make_executor(voice_llm_latency_budget_ms=900)

        await executor.process(
            "你好",
            source="voice",
            conversation_session_id="thread-7",
            voice_turn_id="turn-9",
        )

        context = executor._stream_processor.stream_with_tools.call_args.kwargs["llm_call_context"]
        assert isinstance(context, LLMCallContext)
        assert context.trace_id
        assert context.session_id == "thread-7"
        assert context.turn_id == "turn-9"
        assert context.purpose == "assistant_response"
        assert context.channel == "voice"
        assert context.request_class == "voice_fast"
        assert 1 <= context.latency_budget_ms <= 900
        assert context.privacy_class == "conversation"
        assert context.allow_cache is False

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
    async def test_turn_executor_never_writes_memory_before_committed_event(self):
        te = _make_executor()
        await te.process("hello", voice_turn_id="turn-memory-1")
        await asyncio.gather(*te._pending_tasks, return_exceptions=True)
        te._memory.admit_turn.assert_not_awaited()
        te._memory.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_builder_called(self):
        te = _make_executor()
        await te.process("hello")
        te._prompt_builder.build_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_legacy_qp_writer_is_not_used_for_new_dialogue_turns(self):
        qp_memory = MagicMock()
        te = _make_executor(qp_memory=qp_memory)

        await te.process("remember only what admission accepts")
        await asyncio.gather(*te._pending_tasks, return_exceptions=True)

        qp_memory.record_observation.assert_not_called()
        qp_memory.process_turn.assert_not_called()
        qp_memory.save.assert_not_called()

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
    async def test_forced_rag_reply_waits_for_committed_event_admission(self):
        memory = MagicMock()
        memory.retrieve = AsyncMock(return_value="")
        memory.save = AsyncMock()
        memory.admit_turn = AsyncMock()
        memory.health.return_value = {
            "last_answer_policy": {
                "state": "conflict",
                "action": "clarify",
                "reason": "conflict:route",
            },
        }
        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(
            return_value="这条路线信息有冲突，请管理员确认。"
        )
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
        memory.admit_turn.assert_not_awaited()
        memory.save.assert_not_awaited()

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
        assert (
            stream_processor.stream_with_tools.call_args_list[0].kwargs["conversation_session_id"]
            == "conv-a"
        )
        assert (
            stream_processor.stream_with_tools.call_args_list[1].kwargs["conversation_session_id"]
            == "conv-b"
        )
        assert (
            stream_processor.stream_with_tools.call_args_list[2].kwargs["conversation_session_id"]
            == "conv-a"
        )
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
    async def test_turn_exposes_retrieval_context_for_response_payload(self):
        retrieval = MagicMock()
        retrieval.context = "- scoped fact"
        retrieval.evidence = [{"record_id": "rec-a", "text": "scoped fact"}]
        retrieval.rag = {
            "turn_scoped": True,
            "answer_policy": {"state": "grounded", "action": "answer_with_evidence"},
        }
        memory = MagicMock()
        memory.retrieve_with_context = AsyncMock(return_value=retrieval)
        memory.save = AsyncMock()
        te = _make_executor(memory=memory)

        await te.process("hello", source="text", conversation_session_id="conv-a")

        assert te.current_turn_rag == {
            "evidence": retrieval.evidence,
            "rag": retrieval.rag,
        }
        memory.retrieve_with_context.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_default_conversation_path_still_uses_conversation_manager(self):
        te = _make_executor()

        await te.process("hello", source="text")

        te._conversation.add_user_message.assert_called_once_with("hello")
        te._conversation.add_assistant_message.assert_called_once_with("robot answer")


class TestVoiceMemoryDeadline:
    @pytest.mark.asyncio
    async def test_voice_memory_timeout_does_not_block_llm_or_use_global_policy(self):
        memory = MagicMock()
        memory.retrieve = AsyncMock(return_value="unused")
        memory.save = AsyncMock()
        memory.health.return_value = {"last_answer_policy": {"state": "stale", "action": "refuse"}}
        te = _make_executor(memory=memory, voice_memory_retrieval_deadline_s=0.01)
        memory_task = asyncio.create_task(asyncio.sleep(5, result="late memory"))

        result = await asyncio.wait_for(
            te.process("你好", source="voice", memory_task=memory_task),
            timeout=1.0,
        )

        assert result == "robot answer"
        memory.health.assert_not_called()
        assert memory_task.cancelled()
        _, kwargs = te._prompt_builder.build_system_prompt.call_args
        assert kwargs["rag_policy"] == {
            "state": "latency_budget_exhausted",
            "action": "answer_without_memory",
            "reason": "memory_retrieval_deadline_exceeded",
            "deadline_s": 0.01,
        }

    @pytest.mark.asyncio
    async def test_text_memory_path_still_waits_for_prefetch(self):
        te = _make_executor()
        memory_task = asyncio.create_task(asyncio.sleep(0.02, result="[late context]"))

        result = await te.process("hello", source="text", memory_task=memory_task)

        assert result == "robot answer"
        te._prompt_builder.build_system_prompt.assert_called_once()
        assert te._prompt_builder.build_system_prompt.call_args.args[0] == "[late context]"

    @pytest.mark.asyncio
    async def test_voice_memory_timeout_fails_closed_for_knowledge_dependent_query(self):
        from askme.pipeline.core.rag_policy import forced_rag_reply

        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(side_effect=forced_rag_reply)
        prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        te = _make_executor(
            prompt_builder=prompt_builder,
            voice_memory_retrieval_deadline_s=0.01,
        )
        memory_task = asyncio.create_task(asyncio.sleep(5, result="late memory"))

        result = await asyncio.wait_for(
            te.process("A区卫生间在哪里", source="voice", memory_task=memory_task),
            timeout=1.0,
        )

        assert result == "知识检索当前不可用，我不能在没有依据的情况下回答。请稍后重试。"
        assert memory_task.cancelled()
        te._stream_processor.stream_with_tools.assert_not_called()
        _, kwargs = prompt_builder.build_system_prompt.call_args
        assert kwargs["rag_policy"]["state"] == "unavailable"
        assert kwargs["rag_policy"]["reason"] == "memory_retrieval_deadline_exceeded"


class TestCancelToken:
    @pytest.mark.asyncio
    async def test_skips_turn_when_token_set(self):
        token = asyncio.Event()
        token.set()
        te = _make_executor(cancel_token=token)
        result = await te.process("hello")
        assert result == ""
        te._stream_processor.stream_with_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_cancel_drops_unspoken_assistant_history(self):
        token = asyncio.Event()

        async def _cancel_during_stream(*args, **kwargs):
            del args, kwargs
            token.set()
            return "generated but never played"

        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(side_effect=_cancel_during_stream)
        te = _make_executor(stream_processor=stream_processor)

        result = await te.process(
            "hello",
            source="voice",
            voice_turn_id="voice-turn-1",
            turn_epoch=1,
            turn_cancel_token=token,
        )

        assert result == ""
        te._conversation.add_assistant_message.assert_not_called()
        te._memory.save.assert_not_awaited()
        te._memory.admit_turn.assert_not_awaited()
        te._audio.drain_buffers.assert_called()

    @pytest.mark.asyncio
    async def test_turn_cancel_during_playback_drops_full_generated_history(self):
        token = asyncio.Event()
        te = _make_executor()

        def _interrupt_playback():
            token.set()
            return True

        te._audio.wait_speaking_done.side_effect = _interrupt_playback

        result = await te.process(
            "hello",
            source="voice",
            voice_turn_id="voice-turn-2",
            turn_epoch=2,
            turn_cancel_token=token,
        )

        assert result == ""
        te._conversation.add_assistant_message.assert_not_called()
        te._memory.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_turn_cancel_during_forced_reply_drops_assistant_history(self):
        token = asyncio.Event()
        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(
            return_value="deterministic but interrupted"
        )
        prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        te = _make_executor(prompt_builder=prompt_builder)

        def _interrupt_playback():
            token.set()
            return True

        te._audio.wait_speaking_done.side_effect = _interrupt_playback

        result = await te.process(
            "hello",
            source="voice",
            voice_turn_id="voice-turn-forced",
            turn_epoch=3,
            turn_cancel_token=token,
        )

        assert result == ""
        te._conversation.add_assistant_message.assert_not_called()
        te._stream_processor.stream_with_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_during_normal_history_settlement_rolls_back_assistant(self):
        token = asyncio.Event()
        te = _make_executor()
        add_message = te._conversation.add_assistant_message.side_effect

        def _add_then_cancel(*args, **kwargs):
            add_message(*args, **kwargs)
            token.set()

        te._conversation.add_assistant_message.side_effect = _add_then_cancel

        result = await te.process(
            "hello",
            source="text",
            conversation_session_id="conv-a",
            turn_cancel_token=token,
        )

        assert result == ""
        assert te._conversation.get_messages("sys", conversation_session_id="conv-a") == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        te._conversation.remove_latest_assistant_message.assert_called_once_with(
            "robot answer", conversation_session_id="conv-a"
        )
        te._memory.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_during_forced_history_settlement_rolls_back_assistant(self):
        token = asyncio.Event()
        prompt_builder = MagicMock()
        prompt_builder.build_system_prompt = MagicMock(return_value="sys")
        prompt_builder.build_forced_rag_reply = MagicMock(return_value="forced answer")
        prompt_builder.prepare_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        te = _make_executor(prompt_builder=prompt_builder)
        add_message = te._conversation.add_assistant_message.side_effect

        def _add_then_cancel(*args, **kwargs):
            add_message(*args, **kwargs)
            token.set()

        te._conversation.add_assistant_message.side_effect = _add_then_cancel

        result = await te.process(
            "hello",
            source="text",
            turn_cancel_token=token,
        )

        assert result == ""
        assert not any(message.get("role") == "assistant" for message in te._conversation.history)
        te._memory.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_during_visual_history_settlement_rolls_back_assistant(self):
        token = asyncio.Event()
        te = _make_executor()
        te._is_visual_query = MagicMock(return_value=True)
        te._start_vision_capture = MagicMock(
            return_value=asyncio.create_task(asyncio.sleep(0, result="visual answer"))
        )
        add_message = te._conversation.add_assistant_message.side_effect

        def _add_then_cancel(*args, **kwargs):
            add_message(*args, **kwargs)
            token.set()

        te._conversation.add_assistant_message.side_effect = _add_then_cancel

        result = await te.process(
            "look",
            source="text",
            turn_cancel_token=token,
        )

        assert result == ""
        assert not any(message.get("role") == "assistant" for message in te._conversation.history)


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
        stream_processor.stream_with_tools = AsyncMock(side_effect=TimeoutError())
        te = _make_executor(stream_processor=stream_processor)
        await te.process("hello")
        te._audio.speak.assert_called()

    @pytest.mark.asyncio
    async def test_llm_error_stops_playback(self):
        stream_processor = MagicMock()
        stream_processor.stream_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
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


class TestVisualQueryDetection:
    def test_visual_phrases_trigger_camera_capture(self):
        from askme.pipeline.core.turn_executor import TurnExecutor

        assert TurnExecutor._is_visual_query("小算，你看见了什么")
        assert TurnExecutor._is_visual_query("看一下前面有什么")
        assert TurnExecutor._is_visual_query("摄像头里有人吗")

    def test_non_visual_question_does_not_trigger_camera_capture(self):
        from askme.pipeline.core.turn_executor import TurnExecutor

        assert not TurnExecutor._is_visual_query("你的老板是谁")


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
