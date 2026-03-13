"""Tests for TextLoop proactive slot filling and reroute logic."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.brain.intent_router import Intent, IntentType
from askme.pipeline.proactive.base import ProactiveResult
from askme.pipeline.text_loop import TextLoop, _TextClarificationAudio


# ---------------------------------------------------------------------------
# Helper: build a TextLoop with controlled mocks
# ---------------------------------------------------------------------------

def _make_text_loop(input_answers, router_side_effect, proactive_side_effect=None):
    """Build TextLoop with mocked dependencies."""
    mock_router = MagicMock()
    mock_router.route.side_effect = router_side_effect

    mock_pipeline = MagicMock()
    mock_pipeline.start_idle_reflection.return_value = None
    mock_pipeline.handle_pending_tool_response = AsyncMock(return_value=None)
    mock_pipeline.start_memory_prefetch = MagicMock(
        return_value=asyncio.create_task(asyncio.sleep(0))
    )
    mock_pipeline.process = AsyncMock(return_value="LLM回复")

    call_idx = [0]

    def input_fn(prompt=""):
        i = call_idx[0]
        call_idx[0] += 1
        if i < len(input_answers):
            return input_answers[i]
        raise EOFError  # stop the loop

    mock_audio = MagicMock()
    mock_audio.wait_speaking_done = MagicMock()

    mock_dispatcher = MagicMock()
    mock_dispatcher.dispatch = AsyncMock()
    mock_dispatcher.handle_general = AsyncMock(return_value="handle_general回复")

    mock_commands = MagicMock()
    mock_commands.handle.return_value = False  # don't quit

    mock_conversation = MagicMock()
    mock_conversation.history = []

    mock_skill_manager = MagicMock()
    mock_skill_manager.get_skill_catalog.return_value = "navigate, web_search"

    loop = TextLoop(
        router=mock_router,
        pipeline=mock_pipeline,
        commands=mock_commands,
        conversation=mock_conversation,
        skill_manager=mock_skill_manager,
        audio=mock_audio,
        dispatcher=mock_dispatcher,
    )

    if proactive_side_effect is not None:
        mock_proactive = MagicMock()
        mock_proactive.run = proactive_side_effect
        loop._proactive = mock_proactive

    return loop, mock_dispatcher, input_fn


# ---------------------------------------------------------------------------
# Class 1: slot filling via proactive
# ---------------------------------------------------------------------------

class TestTextLoopProactiveSlotFilling:
    """Proactive.run() result controls whether dispatcher.dispatch is called."""

    async def test_slot_filled_by_proactive_dispatches(self):
        """proactive.run() returns proceed=True with enriched_text → dispatcher.dispatch called."""
        enriched = "导航去仓库A"

        def router_side_effect(text):
            return Intent(type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text=text)

        proactive_run = AsyncMock(
            return_value=ProactiveResult(enriched_text=enriched, proceed=True)
        )

        loop, mock_dispatcher, input_fn = _make_text_loop(
            input_answers=["去仓库A"],
            router_side_effect=router_side_effect,
            proactive_side_effect=proactive_run,
        )

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        mock_dispatcher.dispatch.assert_called_once_with(
            "navigate", enriched, source="text"
        )

    async def test_slot_not_filled_does_not_dispatch(self):
        """proactive.run() returns proceed=False, empty interrupt_payload → dispatch NOT called."""

        def router_side_effect(text):
            return Intent(type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text=text)

        proactive_run = AsyncMock(
            return_value=ProactiveResult(
                enriched_text="去仓库A", proceed=False, interrupt_payload=""
            )
        )

        loop, mock_dispatcher, input_fn = _make_text_loop(
            input_answers=["去仓库A"],
            router_side_effect=router_side_effect,
            proactive_side_effect=proactive_run,
        )

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        mock_dispatcher.dispatch.assert_not_called()

    async def test_direct_dispatch_without_proactive_still_works(self):
        """loop._dispatcher = None → pipeline.execute_skill is called (old code path)."""

        def router_side_effect(text):
            return Intent(type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text=text)

        loop, mock_dispatcher, input_fn = _make_text_loop(
            input_answers=["去仓库A"],
            router_side_effect=router_side_effect,
        )
        # Remove the dispatcher so the legacy branch executes
        loop._dispatcher = None

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        loop._pipeline.execute_skill.assert_called_once_with("navigate", "去仓库A")
        mock_dispatcher.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# Class 2: reroute logic
# ---------------------------------------------------------------------------

class TestTextLoopReroute:
    """interrupt_payload triggers reroute; original skill is not dispatched."""

    def _make_reroute_loop(
        self,
        interrupt_payload: str,
        reroute_intent: Intent,
        reroute_proactive_result: ProactiveResult,
    ):
        """Build a loop where the first proactive call bails with interrupt_payload."""
        call_count = [0]

        async def proactive_run(skill_name, user_text, audio, *, source="voice"):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call — user bailed out
                return ProactiveResult(
                    enriched_text=user_text,
                    proceed=False,
                    interrupt_payload=interrupt_payload,
                )
            # Second call (rerouted skill)
            return reroute_proactive_result

        router_calls = [0]

        def router_side_effect(text):
            router_calls[0] += 1
            if router_calls[0] == 1:
                # First router call — original navigate trigger
                return Intent(
                    type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text=text
                )
            # Subsequent calls — reroute
            return reroute_intent

        loop, mock_dispatcher, input_fn = _make_text_loop(
            input_answers=["去仓库A"],
            router_side_effect=router_side_effect,
        )
        mock_proactive = MagicMock()
        mock_proactive.run = proactive_run
        loop._proactive = mock_proactive

        return loop, mock_dispatcher, input_fn

    async def test_reroute_to_voice_trigger(self):
        """interrupt_payload routes to VOICE_TRIGGER navigate → dispatcher.dispatch called."""
        reroute_intent = Intent(
            type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text="去仓库B"
        )
        reroute_result = ProactiveResult(enriched_text="仓库B", proceed=True)

        loop, mock_dispatcher, input_fn = self._make_reroute_loop(
            interrupt_payload="去仓库B",
            reroute_intent=reroute_intent,
            reroute_proactive_result=reroute_result,
        )

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        # dispatch should have been called for the rerouted skill
        mock_dispatcher.dispatch.assert_called_once()
        args = mock_dispatcher.dispatch.call_args
        assert "仓库B" in args[0][1]  # enriched_text contains "仓库B"

    async def test_reroute_to_general_calls_handle_general(self):
        """interrupt_payload routes to GENERAL → dispatcher.handle_general called."""
        reroute_intent = Intent(
            type=IntentType.GENERAL, raw_text="查个时间"
        )
        # Second proactive.run won't be called for GENERAL branch
        reroute_result = ProactiveResult(enriched_text="查个时间", proceed=True)

        loop, mock_dispatcher, input_fn = self._make_reroute_loop(
            interrupt_payload="查个时间",
            reroute_intent=reroute_intent,
            reroute_proactive_result=reroute_result,
        )

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        mock_dispatcher.handle_general.assert_called_once_with("查个时间", source="text")

    async def test_original_not_dispatched_on_reroute(self):
        """On reroute, dispatch is called exactly once (for the rerouted skill, not original)."""
        reroute_intent = Intent(
            type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text="去仓库B"
        )
        reroute_result = ProactiveResult(enriched_text="仓库B", proceed=True)

        loop, mock_dispatcher, input_fn = self._make_reroute_loop(
            interrupt_payload="去仓库B",
            reroute_intent=reroute_intent,
            reroute_proactive_result=reroute_result,
        )

        with patch("builtins.input", side_effect=input_fn):
            await loop.run()

        # Only one dispatch call — the rerouted one
        assert mock_dispatcher.dispatch.call_count == 1


# ---------------------------------------------------------------------------
# Class 3: _TextClarificationAudio unit tests
# ---------------------------------------------------------------------------

class TestTextClarificationAudio:
    """Unit tests for the _TextClarificationAudio adapter."""

    def test_speak_appends_to_spoken(self):
        audio = _TextClarificationAudio()
        audio.speak("问你")
        assert audio.spoken == ["问你"]

    def test_listen_loop_returns_input(self):
        audio = _TextClarificationAudio()
        with patch("builtins.input", return_value="答案"):
            result = audio.listen_loop()
        assert result == "答案"

    def test_listen_loop_eof_returns_none(self):
        audio = _TextClarificationAudio()
        with patch("builtins.input", side_effect=EOFError):
            result = audio.listen_loop()
        assert result is None


# ---------------------------------------------------------------------------
# Class 4: reroute logging
# ---------------------------------------------------------------------------

class TestTextLoopRerouteLogging:
    """Verify that reroute events are logged."""

    async def test_reroute_is_logged(self):
        """logger.info should be called with a message containing 'rerouting'."""
        reroute_intent = Intent(
            type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text="去仓库B"
        )
        reroute_result = ProactiveResult(enriched_text="仓库B", proceed=True)

        call_count = [0]

        async def proactive_run(skill_name, user_text, audio, *, source="voice"):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProactiveResult(
                    enriched_text=user_text,
                    proceed=False,
                    interrupt_payload="去仓库B",
                )
            return reroute_result

        router_calls = [0]

        def router_side_effect(text):
            router_calls[0] += 1
            if router_calls[0] == 1:
                return Intent(
                    type=IntentType.VOICE_TRIGGER, skill_name="navigate", raw_text=text
                )
            return reroute_intent

        loop, mock_dispatcher, input_fn = _make_text_loop(
            input_answers=["去仓库A"],
            router_side_effect=router_side_effect,
        )
        mock_proactive = MagicMock()
        mock_proactive.run = proactive_run
        loop._proactive = mock_proactive

        with patch("askme.pipeline.text_loop.logger") as mock_logger:
            with patch("builtins.input", side_effect=input_fn):
                await loop.run()

        # Collect all logger.info calls and check for "rerouting"
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        assert any("rerouting" in msg for msg in info_messages), (
            f"Expected 'rerouting' in logger.info calls, got: {info_messages}"
        )
