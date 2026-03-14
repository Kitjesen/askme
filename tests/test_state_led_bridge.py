"""Tests for StateLedBridge — priority resolution and polling behaviour."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from askme.led_controller import LedStateKind, NullLedController
from askme.pipeline.state_led_bridge import StateLedBridge
from askme.voice.audio_agent import AgentState


def _make_audio(state: AgentState) -> MagicMock:
    audio = MagicMock()
    audio.state = state
    return audio


def _make_dispatcher(*, active: bool) -> MagicMock:
    d = MagicMock()
    d.has_active_agent_task = active
    return d


def _make_safety(*, estop: bool) -> MagicMock:
    s = MagicMock()
    s.is_estop_active.return_value = estop
    return s


# ── StateLedBridge.resolve() priority tests ───────────────────────────────────


class TestResolvePriority:
    def test_idle_state(self):
        bridge = StateLedBridge(audio=_make_audio(AgentState.IDLE))
        assert bridge.resolve() == LedStateKind.IDLE

    def test_listening_state(self):
        bridge = StateLedBridge(audio=_make_audio(AgentState.LISTENING))
        assert bridge.resolve() == LedStateKind.LISTENING

    def test_processing_state(self):
        bridge = StateLedBridge(audio=_make_audio(AgentState.PROCESSING))
        assert bridge.resolve() == LedStateKind.PROCESSING

    def test_speaking_state(self):
        bridge = StateLedBridge(audio=_make_audio(AgentState.SPEAKING))
        assert bridge.resolve() == LedStateKind.SPEAKING

    def test_muted_overrides_idle(self):
        bridge = StateLedBridge(audio=_make_audio(AgentState.MUTED))
        assert bridge.resolve() == LedStateKind.MUTED

    def test_agent_task_overrides_idle(self):
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.IDLE),
            dispatcher=_make_dispatcher(active=True),
        )
        assert bridge.resolve() == LedStateKind.AGENT_TASK

    def test_agent_task_overrides_processing(self):
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.PROCESSING),
            dispatcher=_make_dispatcher(active=True),
        )
        assert bridge.resolve() == LedStateKind.AGENT_TASK

    def test_muted_overrides_agent_task(self):
        """MUTED takes priority over AGENT_TASK (mic explicitly silenced)."""
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.MUTED),
            dispatcher=_make_dispatcher(active=True),
        )
        assert bridge.resolve() == LedStateKind.MUTED

    def test_estop_overrides_everything(self):
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.SPEAKING),
            dispatcher=_make_dispatcher(active=True),
            safety=_make_safety(estop=True),
        )
        assert bridge.resolve() == LedStateKind.ESTOP

    def test_no_estop_when_safety_ok(self):
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.IDLE),
            safety=_make_safety(estop=False),
        )
        assert bridge.resolve() == LedStateKind.IDLE

    def test_safety_error_treated_as_no_estop(self):
        """If safety client raises, bridge must not crash and must not set ESTOP."""
        bad_safety = MagicMock()
        bad_safety.is_estop_active.side_effect = ConnectionError("service down")
        bridge = StateLedBridge(
            audio=_make_audio(AgentState.IDLE),
            safety=bad_safety,
        )
        assert bridge.resolve() == LedStateKind.IDLE


# ── Polling loop tests ────────────────────────────────────────────────────────


class TestPollingLoop:
    async def test_run_calls_set_state_on_change(self):
        """run() calls led.set_state exactly once for each distinct state change."""
        led = MagicMock()
        led.set_state = MagicMock()

        audio = _make_audio(AgentState.IDLE)
        bridge = StateLedBridge(audio=audio, led=led)

        task = asyncio.create_task(bridge.run())
        await asyncio.sleep(0.05)  # let one poll cycle fire

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        led.set_state.assert_called_once_with(LedStateKind.IDLE)

    async def test_run_does_not_call_set_state_when_state_unchanged(self):
        """If state does not change, set_state is called only once (first time)."""
        led = MagicMock()
        led.set_state = MagicMock()

        bridge = StateLedBridge(audio=_make_audio(AgentState.IDLE), led=led)
        task = asyncio.create_task(bridge.run())
        await asyncio.sleep(0.5)  # 2+ poll cycles

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert led.set_state.call_count == 1  # only the initial transition

    async def test_run_detects_state_change(self):
        """run() calls set_state again when AgentState changes mid-poll."""
        led = MagicMock()
        led.set_state = MagicMock()

        audio = _make_audio(AgentState.IDLE)
        bridge = StateLedBridge(audio=audio, led=led)

        task = asyncio.create_task(bridge.run())
        await asyncio.sleep(0.05)

        # Simulate state change
        audio.state = AgentState.LISTENING
        await asyncio.sleep(0.3)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        calls = [c.args[0] for c in led.set_state.call_args_list]
        assert LedStateKind.IDLE in calls
        assert LedStateKind.LISTENING in calls


# ── NullLedController ─────────────────────────────────────────────────────────


class TestNullLedController:
    def test_set_state_does_not_raise(self):
        ctrl = NullLedController()
        for kind in LedStateKind:
            ctrl.set_state(kind)  # must not raise
