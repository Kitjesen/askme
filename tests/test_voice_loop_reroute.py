"""Tests for the VoiceLoop reroute feature.

When proactive.run() returns proceed=False with a non-empty interrupt_payload
(e.g. user said "算了，去仓库B"), VoiceLoop should:
  - NOT dispatch the original skill
  - Re-route the interrupt_payload through the intent router
  - If the new intent is a VOICE_TRIGGER with a skill_name: run proactive again
    and conditionally dispatch
  - Otherwise: call handle_general with the interrupt_payload
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.brain.intent_router import IntentType
from askme.pipeline.proactive.base import ProactiveResult
from askme.pipeline.voice_loop import VoiceLoop


# ── helpers ───────────────────────────────────────────────────────────────────


def _nav_intent() -> MagicMock:
    intent = MagicMock()
    intent.type = IntentType.VOICE_TRIGGER
    intent.skill_name = "navigate"
    return intent


def _general_intent() -> MagicMock:
    intent = MagicMock()
    intent.type = IntentType.GENERAL
    intent.skill_name = None
    return intent


def _make_loop(
    listen_answers: list[str],
    router_side_effect,
    proactive_side_effect,
):
    """Assemble a VoiceLoop with controlled listen_loop, router and proactive.

    listen_answers: sequence of texts returned by listen_loop (sync).
    After the list is exhausted, the next call raises KeyboardInterrupt to
    terminate the run() loop cleanly.

    router_side_effect: passed as side_effect to mock_router.route.

    proactive_side_effect: passed as side_effect (AsyncMock) to
    mock_proactive.run — the mock's side_effect list is consumed in call order.
    """
    mock_router = MagicMock()
    mock_router.route.side_effect = router_side_effect

    mock_pipeline = MagicMock()
    mock_pipeline.start_idle_reflection.return_value = None
    mock_pipeline.handle_pending_tool_response = AsyncMock(return_value=None)
    mock_pipeline.start_memory_prefetch = MagicMock(
        return_value=asyncio.create_task(asyncio.sleep(0))
    )

    call_idx = [0]

    def listen_loop_fn():
        i = call_idx[0]
        call_idx[0] += 1
        if i < len(listen_answers):
            return listen_answers[i]
        raise KeyboardInterrupt  # stop the loop

    mock_audio = MagicMock()
    mock_audio.listen_loop.side_effect = listen_loop_fn
    mock_audio.is_muted = False

    mock_dispatcher = MagicMock()
    mock_dispatcher.dispatch = AsyncMock()
    mock_dispatcher.handle_general = AsyncMock()

    loop = VoiceLoop(
        router=mock_router,
        pipeline=mock_pipeline,
        audio=mock_audio,
        dispatcher=mock_dispatcher,
    )

    # Override the internally-created ProactiveOrchestrator.
    mock_proactive = MagicMock()
    mock_proactive.run = AsyncMock(side_effect=proactive_side_effect)
    loop._proactive = mock_proactive

    return loop, mock_dispatcher, mock_proactive, mock_router


# ── Class 1: TestRerouteToVoiceTrigger ────────────────────────────────────────


class TestRerouteToVoiceTrigger:
    """Interrupt_payload routes to a VOICE_TRIGGER → runs proactive again."""

    async def test_reroute_dispatches_rerouted_skill(self):
        """Rerouted proactive returns proceed=True → dispatch called with rerouted skill."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"  # original skill being interrupted

        rerouted_intent = _nav_intent()  # navigate

        proactive_results = [
            # First call: original intent bails with interrupt_payload
            ProactiveResult(
                enriched_text="去仓库B",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库B",
            ),
            # Second call: rerouted intent proceeds
            ProactiveResult(enriched_text="仓库B", proceed=True),
        ]

        # Router: first call is original text → original_intent; second call is
        # interrupt_payload → rerouted_intent.
        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, mock_proactive, _ = _make_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_called_once()
        call_kwargs = mock_dispatcher.dispatch.call_args
        assert call_kwargs.args[0] == "navigate" or call_kwargs.kwargs.get("skill_name") == "navigate"
        # enriched_text from the second proactive run must be passed
        assert "仓库B" in (call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")

    async def test_original_intent_not_dispatched_on_reroute(self):
        """When reroute fires, the original skill is never dispatched."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        rerouted_intent = _nav_intent()

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库B",
            ),
            ProactiveResult(enriched_text="仓库B", proceed=True),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # dispatch should be called exactly once — for the rerouted skill only
        assert mock_dispatcher.dispatch.call_count == 1
        call_args = mock_dispatcher.dispatch.call_args
        dispatched_skill = call_args.args[0]
        assert dispatched_skill == "navigate", (
            f"Expected 'navigate' (rerouted), got '{dispatched_skill}'"
        )

    async def test_reroute_proactive_also_runs_slot_filling(self):
        """Rerouted intent also goes through proactive.run() — called twice total."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        rerouted_intent = _nav_intent()

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="去仓库B",
            ),
            ProactiveResult(enriched_text="仓库B", proceed=True),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, _, mock_proactive, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        assert mock_proactive.run.call_count == 2

    async def test_reroute_if_second_proactive_not_proceed_no_dispatch(self):
        """If rerouted proactive also returns proceed=False, dispatch is never called."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        rerouted_intent = _nav_intent()

        proactive_results = [
            # First: bail with interrupt_payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="去仓库B",
            ),
            # Second: also bail (user gave up again)
            ProactiveResult(enriched_text="", proceed=False, cancelled_by="user"),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_not_called()


# ── Class 2: TestRerouteToGeneral ─────────────────────────────────────────────


class TestRerouteToGeneral:
    """Interrupt_payload routes to GENERAL intent → handle_general is called."""

    async def test_reroute_to_general_calls_handle_general(self):
        """GENERAL reroute → handle_general called with the interrupt_payload text."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        router_responses = [original_intent, _general_intent()]

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="查个时间",
            ),
        ]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.handle_general.assert_called_once()
        call_args = mock_dispatcher.handle_general.call_args
        assert call_args.args[0] == "查个时间"

    async def test_reroute_to_general_not_dispatch(self):
        """When reroute goes to handle_general, dispatch is never called."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        router_responses = [original_intent, _general_intent()]

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="查个时间",
            ),
        ]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_not_called()


# ── Class 3: TestNoRerouteWhenPayloadEmpty ────────────────────────────────────


class TestNoRerouteWhenPayloadEmpty:
    """No reroute logic runs when interrupt_payload is empty or proceed=True."""

    async def test_pure_interrupt_no_reroute(self):
        """proceed=False with empty interrupt_payload → neither dispatch nor handle_general."""
        nav_intent = _nav_intent()

        # Router returns VOICE_TRIGGER for the single user input only;
        # reroute branch should NOT be entered (interrupt_payload is "")
        router_responses = [nav_intent]

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="",  # empty — no reroute
            ),
        ]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_not_called()
        mock_dispatcher.handle_general.assert_not_called()

    async def test_proceed_true_no_reroute_logic(self):
        """proceed=True → dispatch called normally; handle_general never called."""
        nav_intent = _nav_intent()
        nav_intent.skill_name = "navigate"

        router_responses = [nav_intent]

        proactive_results = [
            ProactiveResult(enriched_text="去仓库A", proceed=True),
        ]

        loop, mock_dispatcher, _, _ = _make_loop(
            listen_answers=["去仓库A"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_called_once()
        dispatched_skill = mock_dispatcher.dispatch.call_args.args[0]
        assert dispatched_skill == "navigate"
        mock_dispatcher.handle_general.assert_not_called()


# ── Class 4: TestRerouteLogging ───────────────────────────────────────────────


class TestRerouteLogging:
    """VoiceLoop logs when a reroute happens."""

    async def test_reroute_is_logged(self):
        """When reroute fires, logger.info is called with the interrupt_payload."""
        original_intent = _nav_intent()
        original_intent.skill_name = "inspect"

        rerouted_intent = _nav_intent()

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                interrupt_payload="去仓库B",
            ),
            ProactiveResult(enriched_text="仓库B", proceed=True),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, _, _, _ = _make_loop(
            listen_answers=["检查"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        with patch("askme.pipeline.voice_loop.logger") as mock_logger:
            await loop.run()

        # logger.info should have been called with a message containing the payload
        info_calls = mock_logger.info.call_args_list
        reroute_logged = any(
            "reroute" in str(call).lower() or "去仓库B" in str(call)
            for call in info_calls
        )
        assert reroute_logged, (
            f"Expected a reroute log entry; got info calls: {info_calls}"
        )
