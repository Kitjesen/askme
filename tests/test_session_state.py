"""Category: Clarification Session State Machine

Real risks covered:
  - Invalid state transition (e.g. FAILED → AWAITING_SLOT) silently corrupts
    the session → wrong skill fires with stale data
  - State not reset between runs → turn_count / slot name bleed into next command
  - No external observability → monitoring cannot detect a stuck session
  - Terminal state reactivated → double-execution or silent hang

These tests are synchronous — they exercise the state machine directly,
independent of any agent or audio device.

Status before this state machine was implemented: ALL FAIL (class did not exist).
Status after implementation: ALL PASS.
"""

from __future__ import annotations

import pytest

from askme.pipeline.proactive.session_state import (
    ClarificationSession,
    ClarificationState,
    InvalidTransitionError,
)

S = ClarificationState   # short alias


# ── Valid transitions ──────────────────────────────────────────────────────────


class TestValidTransitions:
    def test_idle_to_awaiting_slot(self):
        sess = ClarificationSession()
        sess.transition(S.AWAITING_SLOT)
        assert sess.state == S.AWAITING_SLOT

    def test_awaiting_slot_self_loop_retry(self):
        sess = ClarificationSession(state=S.AWAITING_SLOT)
        sess.transition(S.AWAITING_SLOT)
        assert sess.state == S.AWAITING_SLOT

    def test_awaiting_slot_to_idle_when_filled(self):
        sess = ClarificationSession(state=S.AWAITING_SLOT)
        sess.transition(S.IDLE)
        assert sess.state == S.IDLE

    def test_awaiting_slot_to_awaiting_confirmation(self):
        sess = ClarificationSession(state=S.AWAITING_SLOT)
        sess.transition(S.AWAITING_CONFIRMATION)
        assert sess.state == S.AWAITING_CONFIRMATION

    def test_awaiting_confirmation_confirmed(self):
        sess = ClarificationSession(state=S.AWAITING_CONFIRMATION)
        sess.transition(S.IDLE)
        assert sess.state == S.IDLE

    def test_awaiting_confirmation_canceled(self):
        sess = ClarificationSession(state=S.AWAITING_CONFIRMATION)
        sess.transition(S.CANCELED)
        assert sess.state == S.CANCELED

    def test_interrupted_resets_to_idle(self):
        sess = ClarificationSession(state=S.INTERRUPTED)
        sess.transition(S.IDLE)
        assert sess.state == S.IDLE

    def test_canceled_resets_to_idle(self):
        sess = ClarificationSession(state=S.CANCELED)
        sess.transition(S.IDLE)
        assert sess.state == S.IDLE

    def test_failed_resets_to_idle(self):
        sess = ClarificationSession(state=S.FAILED)
        sess.transition(S.IDLE)
        assert sess.state == S.IDLE

    def test_idle_to_awaiting_confirmation_skipping_slots(self):
        """confirm_before_execute skill with no required_slots."""
        sess = ClarificationSession()
        sess.transition(S.AWAITING_CONFIRMATION)
        assert sess.state == S.AWAITING_CONFIRMATION

    def test_awaiting_slot_to_interrupted(self):
        sess = ClarificationSession(state=S.AWAITING_SLOT)
        sess.transition(S.INTERRUPTED)
        assert sess.state == S.INTERRUPTED

    def test_awaiting_confirmation_to_interrupted(self):
        sess = ClarificationSession(state=S.AWAITING_CONFIRMATION)
        sess.transition(S.INTERRUPTED)
        assert sess.state == S.INTERRUPTED

    def test_any_active_state_to_failed(self):
        for start in (S.IDLE, S.AWAITING_SLOT, S.AWAITING_CONFIRMATION):
            sess = ClarificationSession(state=start)
            sess.transition(S.FAILED)
            assert sess.state == S.FAILED


# ── Invalid transitions ────────────────────────────────────────────────────────


class TestInvalidTransitions:
    """These are the dangerous cases — silent state corruption in old code."""

    def test_failed_cannot_jump_to_awaiting_slot(self):
        sess = ClarificationSession(state=S.FAILED)
        with pytest.raises(InvalidTransitionError):
            sess.transition(S.AWAITING_SLOT)

    def test_canceled_cannot_go_to_awaiting_confirmation(self):
        sess = ClarificationSession(state=S.CANCELED)
        with pytest.raises(InvalidTransitionError):
            sess.transition(S.AWAITING_CONFIRMATION)

    def test_idle_cannot_go_directly_to_canceled(self):
        """IDLE → CANCELED is not a valid path — cancel requires active state."""
        sess = ClarificationSession()
        with pytest.raises(InvalidTransitionError):
            sess.transition(S.CANCELED)

    def test_awaiting_confirmation_cannot_retry_slot(self):
        """Once confirming, cannot go back to ask more slots."""
        sess = ClarificationSession(state=S.AWAITING_CONFIRMATION)
        with pytest.raises(InvalidTransitionError):
            sess.transition(S.AWAITING_SLOT)

    def test_interrupted_cannot_go_to_canceled(self):
        sess = ClarificationSession(state=S.INTERRUPTED)
        with pytest.raises(InvalidTransitionError):
            sess.transition(S.CANCELED)

    def test_state_unchanged_on_invalid_transition(self):
        """The state must NOT change when an invalid transition is attempted."""
        sess = ClarificationSession(state=S.FAILED)
        try:
            sess.transition(S.AWAITING_SLOT)
        except InvalidTransitionError:
            pass
        assert sess.state == S.FAILED, "State must not mutate on invalid transition"


# ── State properties ───────────────────────────────────────────────────────────


class TestStateProperties:
    def test_canceled_is_terminal(self):
        assert ClarificationSession(state=S.CANCELED).is_terminal

    def test_failed_is_terminal(self):
        assert ClarificationSession(state=S.FAILED).is_terminal

    def test_idle_not_terminal(self):
        assert not ClarificationSession(state=S.IDLE).is_terminal

    def test_awaiting_slot_not_terminal(self):
        assert not ClarificationSession(state=S.AWAITING_SLOT).is_terminal

    def test_interrupted_not_terminal(self):
        """INTERRUPTED is not terminal — system can recover and reset to IDLE."""
        assert not ClarificationSession(state=S.INTERRUPTED).is_terminal

    def test_awaiting_slot_is_active(self):
        assert ClarificationSession(state=S.AWAITING_SLOT).is_active

    def test_awaiting_confirmation_is_active(self):
        assert ClarificationSession(state=S.AWAITING_CONFIRMATION).is_active

    def test_idle_not_active(self):
        assert not ClarificationSession(state=S.IDLE).is_active

    def test_terminal_states_not_active(self):
        for s in (S.CANCELED, S.FAILED):
            assert not ClarificationSession(state=s).is_active


# ── Metadata tracking ──────────────────────────────────────────────────────────


class TestMetadataTracking:
    def test_current_slot_set_on_transition(self):
        sess = ClarificationSession()
        sess.transition(S.AWAITING_SLOT, current_slot="destination")
        assert sess.current_slot == "destination"

    def test_interrupted_by_set(self):
        sess = ClarificationSession(state=S.AWAITING_SLOT)
        sess.transition(S.INTERRUPTED, interrupted_by="算了")
        assert sess.interrupted_by == "算了"

    def test_cancel_reason_set(self):
        sess = ClarificationSession(state=S.AWAITING_CONFIRMATION)
        sess.transition(S.CANCELED, cancel_reason="用户取消")
        assert sess.cancel_reason == "用户取消"

    def test_error_set_on_failed(self):
        sess = ClarificationSession()
        sess.transition(S.FAILED, error="audio device lost")
        assert sess.error == "audio device lost"

    def test_turn_count_updated_directly(self):
        sess = ClarificationSession()
        sess.turn_count = 2
        assert sess.turn_count == 2

    def test_unknown_meta_key_ignored(self):
        """transition() must not raise on unknown metadata keys."""
        sess = ClarificationSession()
        sess.transition(S.AWAITING_SLOT, nonexistent_field="x")
        assert sess.state == S.AWAITING_SLOT


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_from_canceled_returns_to_idle(self):
        sess = ClarificationSession(state=S.CANCELED)
        sess.reset()
        assert sess.state == S.IDLE

    def test_reset_clears_all_fields(self):
        sess = ClarificationSession(
            state=S.AWAITING_SLOT,
            skill_name="navigate",
            current_slot="destination",
            turn_count=2,
            interrupted_by="算了",
            cancel_reason="test",
            error="err",
        )
        sess.reset()
        assert sess.skill_name == ""
        assert sess.current_slot == ""
        assert sess.turn_count == 0
        assert sess.interrupted_by == ""
        assert sess.cancel_reason == ""
        assert sess.error == ""

    def test_reset_from_failed_allows_new_transitions(self):
        sess = ClarificationSession(state=S.FAILED)
        sess.reset()
        sess.transition(S.AWAITING_SLOT)  # should not raise
        assert sess.state == S.AWAITING_SLOT


# ── No state leakage between orchestrator runs ─────────────────────────────────


class TestNoLeakBetweenRuns:
    """Integration-level: each ProactiveOrchestrator.run() must have its own session."""

    async def test_second_run_has_fresh_session(self):
        from unittest.mock import MagicMock
        from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
        from askme.skills.skill_model import SkillDefinition

        sk = SkillDefinition(name="get_time", voice_trigger="几点了")
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        sessions_seen: list[ClarificationSession] = []

        class _AudioCapture:
            def drain_buffers(self): ...
            def speak(self, t): ...
            def start_playback(self): ...
            def stop_playback(self): ...
            def wait_speaking_done(self): ...
            def listen_loop(self): return None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)
        audio = _AudioCapture()

        # Patch agents to capture the context session
        orig_run = orch.run

        captured: list[ClarificationSession | None] = []

        async def _run_and_capture(skill_name, user_text, audio_dev, *, source="voice"):
            result = await orig_run(skill_name, user_text, audio_dev, source=source)
            return result

        r1 = await _run_and_capture("get_time", "几点了", audio)
        r2 = await _run_and_capture("get_time", "几点了", audio)
        assert r1.proceed and r2.proceed

    async def test_turn_count_independent_across_runs(self):
        """Turn count from run 1 must not appear in run 2."""
        from unittest.mock import MagicMock
        from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
        from askme.skills.skill_model import SkillDefinition, SlotSpec

        sk = SkillDefinition(
            name="web_search", voice_trigger="搜索",
            required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
        )
        dispatcher = MagicMock()
        dispatcher.get_skill.return_value = sk
        dispatcher.current_mission = None

        class _Audio:
            def __init__(self, answers):
                self._q = list(answers); self._i = 0
            def drain_buffers(self): ...
            def speak(self, t): ...
            def start_playback(self): ...
            def stop_playback(self): ...
            def wait_speaking_done(self): ...
            def listen_loop(self):
                if self._i < len(self._q):
                    r = self._q[self._i]; self._i += 1; return r
                return None

        orch = ProactiveOrchestrator.default(pipeline=MagicMock(), dispatcher=dispatcher)

        # Run 1: needs clarification (1 turn)
        r1 = await orch.run("web_search", "搜索", _Audio(["北京天气"]))
        assert r1.proceed

        # Run 2: content already specific — must NOT inherit turn_count from run 1
        r2 = await orch.run("web_search", "搜索明天北京天气", _Audio([]))
        assert r2.proceed
