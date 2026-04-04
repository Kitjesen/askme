"""Tests for ClarificationSession state machine."""

from __future__ import annotations

import pytest

from askme.pipeline.proactive.session_state import (
    ClarificationSession,
    ClarificationState,
    InvalidTransitionError,
)


# ── ClarificationState ────────────────────────────────────────────────────────

class TestClarificationState:
    def test_all_states_have_string_values(self):
        for s in ClarificationState:
            assert isinstance(s.value, str)

    def test_idle_value(self):
        assert ClarificationState.IDLE.value == "idle"


# ── ClarificationSession defaults ────────────────────────────────────────────

class TestSessionDefaults:
    def test_initial_state_is_idle(self):
        s = ClarificationSession()
        assert s.state == ClarificationState.IDLE

    def test_not_terminal_initially(self):
        s = ClarificationSession()
        assert s.is_terminal is False

    def test_not_active_initially(self):
        s = ClarificationSession()
        assert s.is_active is False

    def test_skill_name_optional(self):
        s = ClarificationSession(skill_name="patrol")
        assert s.skill_name == "patrol"


# ── is_terminal and is_active ─────────────────────────────────────────────────

class TestProperties:
    def test_canceled_is_terminal(self):
        s = ClarificationSession()
        s.state = ClarificationState.CANCELED
        assert s.is_terminal is True

    def test_failed_is_terminal(self):
        s = ClarificationSession()
        s.state = ClarificationState.FAILED
        assert s.is_terminal is True

    def test_awaiting_slot_is_active(self):
        s = ClarificationSession()
        s.state = ClarificationState.AWAITING_SLOT
        assert s.is_active is True

    def test_awaiting_confirmation_is_active(self):
        s = ClarificationSession()
        s.state = ClarificationState.AWAITING_CONFIRMATION
        assert s.is_active is True

    def test_interrupted_not_terminal(self):
        s = ClarificationSession()
        s.state = ClarificationState.INTERRUPTED
        assert s.is_terminal is False

    def test_idle_not_active(self):
        s = ClarificationSession()
        assert s.is_active is False


# ── transition ────────────────────────────────────────────────────────────────

class TestTransition:
    def test_idle_to_awaiting_slot(self):
        s = ClarificationSession()
        s.transition(ClarificationState.AWAITING_SLOT)
        assert s.state == ClarificationState.AWAITING_SLOT

    def test_idle_to_awaiting_confirmation(self):
        s = ClarificationSession()
        s.transition(ClarificationState.AWAITING_CONFIRMATION)
        assert s.state == ClarificationState.AWAITING_CONFIRMATION

    def test_awaiting_slot_self_loop(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_SLOT)
        s.transition(ClarificationState.AWAITING_SLOT)
        assert s.state == ClarificationState.AWAITING_SLOT

    def test_awaiting_slot_to_idle(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_SLOT)
        s.transition(ClarificationState.IDLE)
        assert s.state == ClarificationState.IDLE

    def test_awaiting_confirmation_to_idle_on_confirm(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_CONFIRMATION)
        s.transition(ClarificationState.IDLE)
        assert s.state == ClarificationState.IDLE

    def test_awaiting_confirmation_to_canceled(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_CONFIRMATION)
        s.transition(ClarificationState.CANCELED)
        assert s.state == ClarificationState.CANCELED

    def test_invalid_transition_raises(self):
        s = ClarificationSession()
        with pytest.raises(InvalidTransitionError):
            s.transition(ClarificationState.INTERRUPTED)  # IDLE -> INTERRUPTED not allowed

    def test_canceled_to_idle_allowed(self):
        s = ClarificationSession(state=ClarificationState.CANCELED)
        s.transition(ClarificationState.IDLE)
        assert s.state == ClarificationState.IDLE

    def test_meta_written_on_transition(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_CONFIRMATION)
        s.transition(ClarificationState.CANCELED, cancel_reason="user denied")
        assert s.cancel_reason == "user denied"

    def test_interrupted_transition_sets_interrupted_by(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_SLOT)
        s.transition(ClarificationState.INTERRUPTED, interrupted_by="算了去仓库B")
        assert s.interrupted_by == "算了去仓库B"

    def test_unknown_meta_key_ignored(self):
        s = ClarificationSession(state=ClarificationState.AWAITING_SLOT)
        s.transition(ClarificationState.IDLE, nonexistent_key="value")  # should not raise


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_to_idle(self):
        s = ClarificationSession(state=ClarificationState.CANCELED)
        s.reset()
        assert s.state == ClarificationState.IDLE

    def test_reset_clears_all_fields(self):
        s = ClarificationSession(
            state=ClarificationState.CANCELED,
            skill_name="patrol",
            current_slot="destination",
            turn_count=3,
            interrupted_by="急停",
            cancel_reason="user said no",
            error="some error",
        )
        s.reset()
        assert s.skill_name == ""
        assert s.current_slot == ""
        assert s.turn_count == 0
        assert s.interrupted_by == ""
        assert s.cancel_reason == ""
        assert s.error == ""
