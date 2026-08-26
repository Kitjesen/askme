"""Cancellation behavior at the LLM/tool streaming seam."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from askme.pipeline.stream_processor import StreamProcessor


def _make_chunk(content: str):
    delta = MagicMock(content=content, tool_calls=None)
    return MagicMock(choices=[MagicMock(delta=delta)])


def _make_processor(*, cancel_token: object) -> StreamProcessor:
    splitter = MagicMock()
    splitter.feed.side_effect = lambda text: [text] if text else []
    splitter.flush.return_value = ""
    return StreamProcessor(
        llm=MagicMock(),
        audio=MagicMock(),
        tools=MagicMock(),
        tool_executor=MagicMock(),
        splitter=splitter,
        general_tool_max_safety_level=3,
        max_response_chars=0,
        cancel_token=cancel_token,
    )


class TestStreamCancellation:
    @pytest.mark.asyncio
    async def test_cancel_while_waiting_for_first_chunk_closes_stream(self):
        token = asyncio.Event()
        started = asyncio.Event()
        closed = asyncio.Event()

        async def stalled_stream():
            try:
                started.set()
                await asyncio.Future()
                yield None
            finally:
                closed.set()

        processor = _make_processor(cancel_token=token)
        consume_task = asyncio.create_task(
            processor.consume_llm_stream(stalled_stream(), source="voice")
        )
        await asyncio.wait_for(started.wait(), timeout=0.2)

        token.set()

        assert await asyncio.wait_for(consume_task, timeout=0.2) == ("", {})
        assert closed.is_set()
        processor._audio.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_during_chunk_processing_drops_late_tts_and_tail(self):
        token = asyncio.Event()
        processor = _make_processor(cancel_token=token)

        def cancel_while_splitting(text: str) -> list[str]:
            token.set()
            return [text]

        processor._splitter.feed.side_effect = cancel_while_splitting
        processor._splitter.flush.return_value = "buffered tail"

        async def one_chunk():
            yield _make_chunk("This sentence is long enough to leave a tail.")

        full, tool_calls = await processor.consume_llm_stream(
            one_chunk(), source="voice"
        )

        assert full
        assert tool_calls == {}
        processor._audio.speak.assert_not_called()
        processor._splitter.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_turn_does_not_start_accumulated_tool_call(self):
        token = asyncio.Event()
        processor = _make_processor(cancel_token=token)
        processor._tools.get_definitions.return_value = []
        processor._tool_executor.execute_tools = AsyncMock(return_value="follow-up")

        async def cancelled_result(*args, **kwargs):
            token.set()
            return "partial", {
                0: {"id": "call-1", "name": "move", "arguments": "{}"}
            }

        processor.consume_llm_stream = AsyncMock(side_effect=cancelled_result)
        processor._llm.chat_stream.return_value = MagicMock()

        result = await processor.stream_with_tools([], "system", source="text")

        assert result == "partial"
        processor._tool_executor.execute_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_follow_up_reuses_token_and_closes_cancelled_stream(self):
        token = asyncio.Event()
        processor = _make_processor(cancel_token=token)
        started = asyncio.Event()
        closed = asyncio.Event()

        async def stalled_follow_up():
            try:
                started.set()
                await asyncio.Future()
                yield None
            finally:
                closed.set()

        processor._llm.chat_stream.return_value = stalled_follow_up()
        follow_task = asyncio.create_task(
            processor.stream_and_speak([], source="voice")
        )
        await asyncio.wait_for(started.wait(), timeout=0.2)

        token.set()

        assert await asyncio.wait_for(follow_task, timeout=0.2) == ""
        assert processor._llm.chat_stream.call_args.kwargs["cancel_token"] is token
        assert closed.is_set()

    @pytest.mark.asyncio
    async def test_primary_llm_wrapper_closes_upstream_after_mid_chunk_cancel(self):
        token = asyncio.Event()
        processor = _make_processor(cancel_token=token)
        processor._tools.get_definitions.return_value = []
        closed = asyncio.Event()

        def cancel_while_splitting(text: str) -> list[str]:
            token.set()
            return []

        processor._splitter.feed.side_effect = cancel_while_splitting

        async def upstream():
            try:
                yield _make_chunk("This content is long enough to reach splitting.")
                await asyncio.Future()
            finally:
                closed.set()

        processor._llm.chat_stream.return_value = upstream()

        await processor.stream_with_tools([], "system", source="voice")

        assert closed.is_set()
