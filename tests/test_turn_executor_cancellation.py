"""Cancellation contract for one conversation turn.

An interrupted voice turn may have emitted partial audio, but it must not become
conversation history or durable memory.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.core.turn_executor import TurnExecutor


class _Conversation:
    def __init__(self) -> None:
        self.history: list[dict[str, object]] = []

    def add_user_message(self, content: str, **_: object) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, **_: object) -> None:
        self.history.append({"role": "assistant", "content": content})

    def get_messages(self, system_prompt: str, **_: object) -> list[dict[str, object]]:
        return [{"role": "system", "content": system_prompt}, *self.history]

    def remove_latest_user_message(self, content: str, **_: object) -> bool:
        for index in range(len(self.history) - 1, -1, -1):
            message = self.history[index]
            if message.get("role") == "user" and message.get("content") == content:
                self.history.pop(index)
                return True
        return False

    async def maybe_compress(self, *_: object, **__: object) -> None:
        return None


def _make_turn(
    cancel_token: asyncio.Event, **turn_kwargs: object
) -> tuple[TurnExecutor, SimpleNamespace]:
    conversation = _Conversation()
    memory = MagicMock()
    memory.retrieve = AsyncMock(return_value="memory")
    memory.save = AsyncMock()

    audio = MagicMock()
    audio.drain_buffers = MagicMock()
    audio.start_playback = MagicMock()
    audio.speak = MagicMock()
    audio.wait_speaking_done = MagicMock()
    audio.stop_playback = MagicMock()

    prompt_builder = MagicMock()
    prompt_builder.build_system_prompt.return_value = "system"
    prompt_builder.build_forced_rag_reply.return_value = ""
    prompt_builder.prepare_messages.side_effect = lambda messages, **_: messages

    stream_processor = MagicMock()
    stream_processor.stream_with_tools = AsyncMock(return_value="complete answer")

    hooks = MagicMock()
    hooks.fire_pre_turn = AsyncMock(return_value=False)
    hooks.fire_post_turn = AsyncMock()

    episodic = MagicMock()
    episodic.should_reflect.return_value = False

    turn = TurnExecutor(
        llm=MagicMock(),
        conversation=conversation,
        memory=memory,
        audio=audio,
        prompt_builder=prompt_builder,
        stream_processor=stream_processor,
        episodic=episodic,
        cancel_token=cancel_token,
        hooks=hooks,
        **turn_kwargs,
    )
    return turn, SimpleNamespace(
        conversation=conversation,
        memory=memory,
        audio=audio,
        prompt_builder=prompt_builder,
        stream_processor=stream_processor,
        hooks=hooks,
        episodic=episodic,
    )


@pytest.mark.asyncio
async def test_voice_turn_cancelled_while_playing_is_not_committed() -> None:
    cancel_token = asyncio.Event()
    turn, deps = _make_turn(cancel_token)
    deps.audio.wait_speaking_done.side_effect = cancel_token.set

    result = await turn.process("tell me", source="voice")
    await asyncio.sleep(0)

    assert result == ""
    assert deps.conversation.history == []
    assert turn.last_spoken_text == ""
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_after_memory_retrieval_never_starts_llm() -> None:
    cancel_token = asyncio.Event()
    turn, deps = _make_turn(cancel_token)

    async def _retrieve_then_cancel(_: str) -> str:
        cancel_token.set()
        return "late memory"

    deps.memory.retrieve.side_effect = _retrieve_then_cancel

    result = await turn.process("where is it?", source="voice")

    assert result == ""
    assert deps.conversation.history == []
    deps.stream_processor.stream_with_tools.assert_not_awaited()
    deps.audio.start_playback.assert_not_called()
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_after_behavior_memory_never_starts_llm() -> None:
    cancel_token = asyncio.Event()
    behavior_memory = MagicMock()

    async def _retrieve_then_cancel(_: str) -> str:
        cancel_token.set()
        return "preference"

    behavior_memory.retrieve_behavior = AsyncMock(side_effect=_retrieve_then_cancel)
    behavior_memory.save_behavior_memory = AsyncMock()
    behavior_memory.should_reflect.return_value = False
    turn, deps = _make_turn(cancel_token, memory_system=behavior_memory)

    result = await turn.process("keep it short", source="voice")

    assert result == ""
    assert deps.conversation.history == []
    deps.stream_processor.stream_with_tools.assert_not_awaited()
    deps.audio.start_playback.assert_not_called()
    deps.hooks.fire_post_turn.assert_not_awaited()
    behavior_memory.save_behavior_memory.assert_not_awaited()
    behavior_memory.log_event.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_after_vision_capture_commits_nothing() -> None:
    cancel_token = asyncio.Event()
    vision = MagicMock()
    vision.available = True
    vision.auto_capture_enabled.return_value = False

    async def _describe_then_cancel(_: str) -> str:
        await asyncio.sleep(0)
        cancel_token.set()
        return "a person ahead"

    vision.describe_scene_with_question = AsyncMock(side_effect=_describe_then_cancel)
    turn, deps = _make_turn(cancel_token, vision=vision)

    result = await turn.process("看一下前面", source="voice")

    assert result == ""
    assert deps.conversation.history == []
    deps.stream_processor.stream_with_tools.assert_not_awaited()
    deps.audio.start_playback.assert_not_called()
    deps.audio.speak.assert_not_called()
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()


@pytest.mark.asyncio
async def test_forced_rag_cancelled_while_playing_is_not_committed() -> None:
    cancel_token = asyncio.Event()
    turn, deps = _make_turn(cancel_token)
    deps.prompt_builder.build_forced_rag_reply.return_value = "knowledge conflict"
    deps.audio.wait_speaking_done.side_effect = cancel_token.set

    result = await turn.process("which route?", source="voice")
    await asyncio.sleep(0)

    assert result == ""
    assert deps.conversation.history == []
    assert turn.last_spoken_text == ""
    deps.stream_processor.stream_with_tools.assert_not_awaited()
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_llm_error_does_not_create_error_history() -> None:
    cancel_token = asyncio.Event()
    turn, deps = _make_turn(cancel_token)

    async def _cancel_then_fail(*_: object, **__: object) -> str:
        cancel_token.set()
        raise RuntimeError("socket closed during cancellation")

    deps.stream_processor.stream_with_tools.side_effect = _cancel_then_fail

    result = await turn.process("hello", source="voice")

    assert result == ""
    assert deps.conversation.history == []
    deps.audio.speak.assert_not_called()
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()


@pytest.mark.asyncio
async def test_visual_reply_cancelled_while_playing_is_not_committed() -> None:
    cancel_token = asyncio.Event()
    vision = MagicMock()
    vision.available = True
    vision.auto_capture_enabled.return_value = False
    vision.describe_scene_with_question = AsyncMock(return_value="a person ahead")
    turn, deps = _make_turn(cancel_token, vision=vision)
    deps.audio.wait_speaking_done.side_effect = cancel_token.set

    result = await turn.process("看一下前面", source="voice")

    assert result == ""
    assert deps.conversation.history == []
    assert turn.last_spoken_text == ""
    deps.stream_processor.stream_with_tools.assert_not_awaited()
    deps.hooks.fire_post_turn.assert_not_awaited()
    deps.memory.save.assert_not_awaited()
    deps.episodic.log.assert_not_called()
