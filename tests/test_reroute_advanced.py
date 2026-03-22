"""Advanced reroute scenarios for VoiceLoop.

Covers edge cases and interactions NOT tested in:
  - tests/test_voice_loop_reroute.py  (basic reroute cases)
  - tests/test_reroute.py             (parse + agent unit tests)

Scenarios:
  1. Multi-level reroute: a rerouted payload itself triggers another bail-out
  2. Runtime bridge interaction: bridge fail / handled / bypassed for reroute payload
  3. Memory prefetch and idle-task reset on rerouted GENERAL intent
  4. Edge cases: empty skill_name, second proactive proceed=False
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from askme.llm.intent_router import IntentType
from askme.pipeline.proactive.base import ProactiveResult
from askme.pipeline.voice_loop import VoiceLoop


# ── Helpers ────────────────────────────────────────────────────────────────────


def _nav_intent(skill_name: str = "navigate") -> MagicMock:
    intent = MagicMock()
    intent.type = IntentType.VOICE_TRIGGER
    intent.skill_name = skill_name
    return intent


def _general_intent() -> MagicMock:
    intent = MagicMock()
    intent.type = IntentType.GENERAL
    intent.skill_name = None
    return intent


def _make_voice_loop(
    listen_answers: list[str],
    router_side_effect,
    proactive_side_effect,
    voice_runtime_bridge=None,
):
    """Assemble a VoiceLoop with controlled listen_loop, router and proactive.

    listen_answers: sequence of texts returned by listen_loop (sync).
    After the list is exhausted, the next call raises KeyboardInterrupt to
    terminate the run() loop cleanly.

    router_side_effect: passed as side_effect to mock_router.route.

    proactive_side_effect: passed as side_effect (AsyncMock) to
    mock_proactive.run — the mock's side_effect list is consumed in call order.

    voice_runtime_bridge: optional bridge mock (or None).
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
    mock_dispatcher.has_active_agent_task = False

    loop = VoiceLoop(
        router=mock_router,
        pipeline=mock_pipeline,
        audio=mock_audio,
        dispatcher=mock_dispatcher,
        voice_runtime_bridge=voice_runtime_bridge,
    )

    # Override the internally-created ProactiveOrchestrator.
    mock_proactive = MagicMock()
    mock_proactive.run = AsyncMock(side_effect=proactive_side_effect)
    loop._proactive = mock_proactive

    return loop, mock_dispatcher, mock_proactive, mock_router, mock_pipeline


# ── Class 1: TestMultiLevelReroute ────────────────────────────────────────────


class TestMultiLevelReroute:
    """Rerouted payload itself produces another interrupt_payload (user keeps bailing)."""

    async def test_reroute_then_second_reroute_proceeds(self):
        """First reroute → interrupt_payload="去仓库A"; second proactive for "去仓库A"
        also returns interrupt_payload="" (bail-out with no new payload).
        → dispatcher.dispatch must NOT be called at all.
        """
        # First call: original user text "去那里"
        original_intent = _nav_intent("inspect")
        # Second call: rerouted payload "去仓库A"
        rerouted_intent = _nav_intent("navigate")

        proactive_results = [
            # First proactive: bail with payload "去仓库A"
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库A",
            ),
            # Second proactive (for rerouted "去仓库A"): bail with no payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="",
            ),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, mock_proactive, _, _ = _make_voice_loop(
            listen_answers=["去那里"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # Both levels bailed — dispatch must never be called
        mock_dispatcher.dispatch.assert_not_called()
        # Proactive was called twice (once per reroute level)
        assert mock_proactive.run.call_count == 2

    async def test_reroute_to_skill_payload_is_itself_an_interrupt(self):
        """First reroute → interrupt_payload="算了" (an interrupt word itself).
        Router routes "算了" → GENERAL.
        → dispatcher.handle_general called with "算了".
        """
        original_intent = _nav_intent("inspect")

        proactive_results = [
            # Bail out, payload itself is an interrupt word
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="算了",
            ),
        ]

        # Router: first for original text → VOICE_TRIGGER; second for "算了" → GENERAL
        router_responses = [original_intent, _general_intent()]

        loop, mock_dispatcher, _, _, _ = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # Rerouted payload "算了" → GENERAL → handle_general called
        mock_dispatcher.handle_general.assert_called_once()
        call_args = mock_dispatcher.handle_general.call_args
        assert call_args.args[0] == "算了"
        # dispatch should NOT be called
        mock_dispatcher.dispatch.assert_not_called()

    async def test_reroute_chain_final_dispatch_has_correct_content(self):
        """First reroute → interrupt_payload="去仓库A" → VOICE_TRIGGER navigate.
        Second proactive for "去仓库A": proceed=True, enriched_text="导航去仓库A（已确认）".
        → dispatcher.dispatch called with enriched_text containing "仓库A".
        """
        original_intent = _nav_intent("inspect")
        rerouted_intent = _nav_intent("navigate")

        proactive_results = [
            # First: bail with payload "去仓库A"
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库A",
            ),
            # Second (for rerouted "去仓库A"): proceed with enriched text
            ProactiveResult(
                enriched_text="导航去仓库A（已确认）",
                proceed=True,
            ),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, _, _, _ = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        mock_dispatcher.dispatch.assert_called_once()
        call_args = mock_dispatcher.dispatch.call_args
        # Dispatched skill must be "navigate"
        assert call_args.args[0] == "navigate" or call_args.kwargs.get("skill_name") == "navigate"
        # Enriched text must contain "仓库A"
        enriched = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("user_text", "")
        assert "仓库A" in enriched, (
            f"Expected enriched_text to contain '仓库A', got: {enriched!r}"
        )


# ── Class 2: TestRerouteWithRuntimeBridge ─────────────────────────────────────


class TestRerouteWithRuntimeBridge:
    """Verify reroute logic interaction with the voice_runtime_bridge."""

    async def test_bridge_fails_local_reroute_still_works(self):
        """Bridge raises Exception for original user_text "去那里".
        proactive returns interrupt_payload="去仓库B".
        Router routes "去仓库B" → VOICE_TRIGGER navigate.
        Second proactive returns proceed=True.
        → dispatcher.dispatch called correctly (bridge failure doesn't block reroute).
        """
        # Bridge that raises for any call
        mock_bridge = MagicMock()
        mock_bridge.handle_voice_text.side_effect = Exception("bridge connection error")

        original_intent = _nav_intent("navigate")
        rerouted_intent = _nav_intent("navigate")

        proactive_results = [
            # First proactive: bail with reroute payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库B",
            ),
            # Second proactive: proceed
            ProactiveResult(
                enriched_text="去仓库B",
                proceed=True,
            ),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, _, _, _ = _make_voice_loop(
            listen_answers=["去那里"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
            voice_runtime_bridge=mock_bridge,
        )

        await loop.run()

        # Bridge failure doesn't block local reroute — dispatch must be called
        mock_dispatcher.dispatch.assert_called_once()
        call_args = mock_dispatcher.dispatch.call_args
        assert call_args.args[0] == "navigate"

    async def test_bridge_handles_original_no_reroute(self):
        """Bridge returns handled=True for original user_text "去那里".
        → reroute logic never runs (proactive.run never called).
        """
        mock_bridge = MagicMock()
        mock_bridge.handle_voice_text.return_value = {
            "handled": True,
            "turn": {
                "action_type": "skill",
                "skill_name": "navigate",
            },
        }

        original_intent = _nav_intent("navigate")

        loop, mock_dispatcher, mock_proactive, _, _ = _make_voice_loop(
            listen_answers=["去那里"],
            router_side_effect=[original_intent],
            proactive_side_effect=[],  # never called
            voice_runtime_bridge=mock_bridge,
        )

        await loop.run()

        # Bridge handled it — proactive.run must NOT be called
        mock_proactive.run.assert_not_called()

    async def test_reroute_payload_not_sent_to_bridge(self):
        """Bridge is configured.
        Original user_text "去那里" → bridge returns handled=False (not handled).
        proactive returns interrupt_payload="去仓库B".
        → bridge is NOT called again for "去仓库B" (reroute bypasses bridge).
        → dispatcher.dispatch called directly.
        """
        mock_bridge = MagicMock()
        # Returns not-handled for the original text
        mock_bridge.handle_voice_text.return_value = {"handled": False}

        original_intent = _nav_intent("navigate")
        rerouted_intent = _nav_intent("navigate")

        proactive_results = [
            # First proactive: bail with reroute payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库B",
            ),
            # Second proactive: proceed
            ProactiveResult(
                enriched_text="去仓库B",
                proceed=True,
            ),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, _, _, _ = _make_voice_loop(
            listen_answers=["去那里"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
            voice_runtime_bridge=mock_bridge,
        )

        await loop.run()

        # Bridge must have been called exactly once — for "去那里" only
        assert mock_bridge.handle_voice_text.call_count == 1, (
            f"Bridge should be called exactly once (for original text), "
            f"got {mock_bridge.handle_voice_text.call_count} calls: "
            f"{mock_bridge.handle_voice_text.call_args_list}"
        )
        # Verify it was called with the original text
        assert mock_bridge.handle_voice_text.call_args.args[0] == "去那里"
        # Verify "去仓库B" was NEVER passed to bridge
        all_call_args = [c.args[0] for c in mock_bridge.handle_voice_text.call_args_list]
        assert "去仓库B" not in all_call_args, (
            "Reroute payload '去仓库B' must NOT be sent to the runtime bridge."
        )
        # dispatch should be called with the rerouted skill
        mock_dispatcher.dispatch.assert_called_once()


# ── Class 3: TestRerouteMemoryAndIdle ─────────────────────────────────────────


class TestRerouteMemoryAndIdle:
    """Memory prefetch and idle-task reset on rerouted GENERAL intent."""

    async def test_reroute_to_general_starts_memory_prefetch(self):
        """proactive returns interrupt_payload="查个时间", router returns GENERAL.
        Verify pipeline.start_memory_prefetch is called with "查个时间".
        (Not called for the original user_text since VOICE_TRIGGER cancels memory.)
        """
        original_intent = _nav_intent("inspect")

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="查个时间",
            ),
        ]

        router_responses = [original_intent, _general_intent()]

        loop, mock_dispatcher, _, _, mock_pipeline = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # start_memory_prefetch must be called at least once with the rerouted payload
        prefetch_calls = mock_pipeline.start_memory_prefetch.call_args_list
        rerouted_prefetch = [c for c in prefetch_calls if c.args[0] == "查个时间"]
        assert rerouted_prefetch, (
            f"Expected start_memory_prefetch to be called with '查个时间'. "
            f"Actual calls: {prefetch_calls}"
        )

    async def test_reroute_to_general_resets_idle_task(self):
        """proactive returns interrupt_payload="查个时间", router returns GENERAL.
        Verify pipeline.start_idle_reflection is called after handle_general.
        (This is the P2 fix: idle_task reset after rerouted general handling.)
        """
        original_intent = _nav_intent("inspect")

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="查个时间",
            ),
        ]

        router_responses = [original_intent, _general_intent()]

        loop, mock_dispatcher, _, _, mock_pipeline = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # handle_general should be called for the rerouted payload
        mock_dispatcher.handle_general.assert_called_once()
        call_args = mock_dispatcher.handle_general.call_args
        assert call_args.args[0] == "查个时间"

        # start_idle_reflection must be called after handle_general (idle reset)
        assert mock_pipeline.start_idle_reflection.call_count >= 1, (
            "start_idle_reflection must be called at least once to reset the idle timer "
            "after a rerouted GENERAL intent is handled."
        )


# ── Class 4: TestRerouteEdgeCases ─────────────────────────────────────────────


class TestRerouteEdgeCases:
    """Edge cases: empty skill_name, second proactive proceed=False with no payload."""

    async def test_empty_skill_name_in_reroute_does_not_dispatch(self):
        """Router routes interrupt_payload to VOICE_TRIGGER but skill_name is "" (empty).
        → dispatcher.dispatch must NOT be called (code checks `and _reroute_intent.skill_name`).
        """
        original_intent = _nav_intent("inspect")

        # Rerouted intent: VOICE_TRIGGER but skill_name is empty string
        empty_skill_intent = MagicMock()
        empty_skill_intent.type = IntentType.VOICE_TRIGGER
        empty_skill_intent.skill_name = ""  # empty — should prevent dispatch

        proactive_results = [
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库A",
            ),
        ]

        router_responses = [original_intent, empty_skill_intent]

        loop, mock_dispatcher, mock_proactive, _, _ = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # Empty skill_name → no dispatch, no second proactive run
        mock_dispatcher.dispatch.assert_not_called()
        # Proactive should be called once (for the original text), NOT a second time
        # because the rerouted VOICE_TRIGGER has empty skill_name
        assert mock_proactive.run.call_count == 1, (
            f"Proactive should only run once (empty rerouted skill_name stops re-run). "
            f"Got {mock_proactive.run.call_count} calls."
        )

    async def test_reroute_proceeds_false_in_second_proactive_no_dispatch(self):
        """First proactive: interrupt_payload="去仓库A".
        Router: VOICE_TRIGGER navigate.
        Second proactive: proceed=False, interrupt_payload="" (second bail with no new intent).
        → dispatcher.dispatch NOT called.
        """
        original_intent = _nav_intent("inspect")
        rerouted_intent = _nav_intent("navigate")

        proactive_results = [
            # First: bail with reroute payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="去仓库A",
            ),
            # Second: also bail, no new payload
            ProactiveResult(
                enriched_text="",
                proceed=False,
                cancelled_by="user",
                interrupt_payload="",
            ),
        ]

        router_responses = [original_intent, rerouted_intent]

        loop, mock_dispatcher, mock_proactive, _, _ = _make_voice_loop(
            listen_answers=["检查一下"],
            router_side_effect=router_responses,
            proactive_side_effect=proactive_results,
        )

        await loop.run()

        # Second proactive also bailed → no dispatch
        mock_dispatcher.dispatch.assert_not_called()
        # Both proactive runs happened
        assert mock_proactive.run.call_count == 2, (
            f"Expected 2 proactive runs (original + rerouted), "
            f"got {mock_proactive.run.call_count}."
        )
