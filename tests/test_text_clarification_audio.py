"""Tests for _TextClarificationAudio and TextLoop process_turn."""

from __future__ import annotations

import queue
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── _TextClarificationAudio ───────────────────────────────────────────────────

class TestTextClarificationAudio:
    def _make(self):
        from askme.pipeline.text_loop import _TextClarificationAudio
        return _TextClarificationAudio()

    def test_speak_appends_to_spoken(self):
        audio = self._make()
        audio.speak("你好")
        assert "你好" in audio.spoken

    def test_speak_multiple_times(self):
        audio = self._make()
        audio.speak("第一句")
        audio.speak("第二句")
        assert len(audio.spoken) == 2

    def test_start_playback_no_op(self):
        audio = self._make()
        audio.start_playback()  # should not raise

    def test_stop_playback_no_op(self):
        audio = self._make()
        audio.stop_playback()

    def test_wait_speaking_done_no_op(self):
        audio = self._make()
        audio.wait_speaking_done()

    def test_drain_buffers_no_op(self):
        audio = self._make()
        audio.drain_buffers()

    def test_listen_loop_returns_input(self):
        audio = self._make()
        with patch("builtins.input", return_value="仓库A"):
            result = audio.listen_loop()
        assert result == "仓库A"

    def test_listen_loop_returns_none_on_empty_input(self):
        audio = self._make()
        with patch("builtins.input", return_value=""):
            result = audio.listen_loop()
        assert result is None

    def test_listen_loop_returns_none_on_eof(self):
        audio = self._make()
        with patch("builtins.input", side_effect=EOFError):
            result = audio.listen_loop()
        assert result is None

    def test_listen_loop_strips_whitespace(self):
        audio = self._make()
        with patch("builtins.input", return_value="  hello  "):
            result = audio.listen_loop()
        assert result == "hello"


# ── TextLoop process_turn integration ────────────────────────────────────────

class TestTextLoopProcessTurn:
    def _make_text_loop(self):
        from askme.pipeline.text_loop import TextLoop

        router = MagicMock()
        pipeline = MagicMock()
        pipeline.process.return_value = asyncio.coroutine(lambda *a, **kw: "response")
        commands = MagicMock()
        conversation = MagicMock()
        skill_manager = MagicMock()
        audio = MagicMock()

        router.route.return_value = MagicMock(
            intent="skill",
            skill_name=None,
            requires_voice_confirm=False,
        )

        with patch("askme.pipeline.text_loop.ProactiveOrchestrator"):
            loop = TextLoop(
                router=router,
                pipeline=pipeline,
                commands=commands,
                conversation=conversation,
                skill_manager=skill_manager,
                audio=audio,
            )
        return loop

    @pytest.mark.asyncio
    async def test_process_turn_returns_string(self):
        import asyncio
        from askme.pipeline.text_loop import TextLoop

        router = MagicMock()
        router.route.return_value = MagicMock(
            intent="chat",
            skill_name=None,
            requires_voice_confirm=False,
        )
        pipeline = MagicMock()
        pipeline.process = AsyncMock(return_value="助手的回答")

        with patch("askme.pipeline.proactive.ProactiveOrchestrator"), \
             patch("askme.pipeline.proactive.orchestrator.ProactiveOrchestrator"), \
             patch("askme.pipeline.text_loop.record_external_turn"):
            loop = TextLoop(
                router=router,
                pipeline=pipeline,
                commands=MagicMock(),
                conversation=MagicMock(),
                skill_manager=MagicMock(),
                audio=MagicMock(),
            )
        result = await loop.process_turn("你好")
        assert isinstance(result, str)


import asyncio
