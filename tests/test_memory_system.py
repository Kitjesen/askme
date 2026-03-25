"""Tests for MemorySystem unified facade."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.memory.system import MemorySystem


def _make_system(
    has_episodic=True,
    has_vector=True,
    has_session=True,
):
    llm = MagicMock()
    conversation = MagicMock()
    conversation.add_user_message = MagicMock()
    conversation.add_assistant_message = MagicMock()
    conversation.maybe_compress = AsyncMock()

    session = MagicMock() if has_session else None
    if session:
        session.get_recent_summaries = MagicMock(return_value="session context")

    episodic = MagicMock() if has_episodic else None
    if episodic:
        episodic.log = MagicMock()
        episodic.should_reflect = MagicMock(return_value=False)
        episodic.reflect = AsyncMock(return_value="reflection result")
        episodic.cleanup_old_episodes = MagicMock()
        episodic.get_knowledge_context = MagicMock(return_value="world knowledge")
        episodic.get_recent_digest = MagicMock(return_value="recent digest")
        episodic.get_relevant_context = MagicMock(return_value="relevant episodes")

    vector = MagicMock() if has_vector else None
    if vector:
        vector.retrieve = AsyncMock(return_value="vector context")
        vector.save = AsyncMock()

    ms = MemorySystem(
        llm=llm,
        conversation=conversation,
        session_memory=session,
        episodic=episodic,
        vector_memory=vector,
    )
    return ms, llm, conversation, session, episodic, vector


class TestLogEvent:
    def test_logs_to_episodic(self):
        ms, _, _, _, episodic, _ = _make_system()
        ms.log_event("command", "用户说: 导航")
        episodic.log.assert_called_once_with("command", "用户说: 导航", None)

    def test_no_crash_without_episodic(self):
        ms, _, _, _, _, _ = _make_system(has_episodic=False)
        ms.log_event("command", "test")  # should not raise


class TestAddTurn:
    def test_adds_to_conversation(self):
        ms, _, conv, _, _, _ = _make_system()
        ms.add_turn("你好", "你好，有什么任务？")
        conv.add_user_message.assert_called_once_with("你好")
        conv.add_assistant_message.assert_called_once_with("你好，有什么任务？")


class TestGetMemoryContext:
    def test_assembles_all_layers(self):
        ms, _, _, _, _, _ = _make_system()
        ctx = ms.get_memory_context("查温度")
        assert "world knowledge" in ctx
        assert "recent digest" in ctx
        assert "relevant episodes" in ctx
        assert "session context" in ctx

    def test_without_episodic(self):
        ms, _, _, _, _, _ = _make_system(has_episodic=False)
        ctx = ms.get_memory_context("test")
        assert "session context" in ctx
        assert "world knowledge" not in ctx

    def test_without_session(self):
        ms, _, _, _, _, _ = _make_system(has_session=False)
        ctx = ms.get_memory_context("test")
        assert "world knowledge" in ctx
        assert "session context" not in ctx

    def test_no_episodic_session_context(self):
        ms, _, _, _, _, _ = _make_system(has_episodic=False, has_session=False)
        ctx = ms.get_memory_context("test")
        # No episodic/session content, but L6 policy rules may be present
        assert "world knowledge" not in ctx
        assert "session context" not in ctx


class TestReflection:
    def test_should_reflect_delegates(self):
        ms, _, _, _, episodic, _ = _make_system()
        episodic.should_reflect.return_value = True
        assert ms.should_reflect() is True

    def test_should_reflect_false_without_episodic(self):
        ms, _, _, _, _, _ = _make_system(has_episodic=False)
        assert ms.should_reflect() is False

    @pytest.mark.asyncio
    async def test_reflect_runs(self):
        ms, _, _, _, episodic, _ = _make_system()
        episodic.should_reflect.return_value = True
        result = await ms.reflect()
        assert result == "reflection result"
        episodic.reflect.assert_awaited_once()
        episodic.cleanup_old_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_skips_when_not_due(self):
        ms, _, _, _, episodic, _ = _make_system()
        episodic.should_reflect.return_value = False
        result = await ms.reflect()
        assert result is None
        episodic.reflect.assert_not_awaited()


class TestCompress:
    @pytest.mark.asyncio
    async def test_compress_delegates(self):
        ms, llm, conv, _, _, _ = _make_system()
        await ms.compress()
        conv.maybe_compress.assert_awaited_once_with(llm)

    @pytest.mark.asyncio
    async def test_compress_error_handled(self):
        ms, _, conv, _, _, _ = _make_system()
        conv.maybe_compress = AsyncMock(side_effect=RuntimeError("test"))
        await ms.compress()  # should not raise


class TestVectorMemory:
    @pytest.mark.asyncio
    async def test_save_to_vector(self):
        ms, _, _, _, _, vector = _make_system()
        await ms.save_to_vector("user", "assistant")
        vector.save.assert_awaited_once_with("user", "assistant")

    @pytest.mark.asyncio
    async def test_save_noop_without_vector(self):
        ms, _, _, _, _, _ = _make_system(has_vector=False)
        await ms.save_to_vector("user", "assistant")  # should not raise


class TestProperties:
    def test_conversation_property(self):
        ms, _, conv, _, _, _ = _make_system()
        assert ms.conversation is conv

    def test_episodic_property(self):
        ms, _, _, _, episodic, _ = _make_system()
        assert ms.episodic is episodic

    def test_has_episodic(self):
        ms1, _, _, _, _, _ = _make_system(has_episodic=True)
        assert ms1.has_episodic is True
        ms2, _, _, _, _, _ = _make_system(has_episodic=False)
        assert ms2.has_episodic is False
