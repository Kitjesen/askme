"""Tests for ClarificationSession — clarification state machine."""

from __future__ import annotations

import pytest

from askme.pipeline.proactive.session_state import (
    ClarificationSession,
    ClarificationState,
    InvalidTransitionError,
)

S = ClarificationState


class TestInit:
    def test_initial_state_is_idle(self):
        s = ClarificationSession()
        assert s.state == S.IDLE

    def test_not_active_in_idle(self):
        s = ClarificationSession()
        assert s.is_active is False

    def test_not_terminal_in_idle(self):
        s = ClarificationSession()
        assert s.is_terminal is False

    def test_turn_count_zero(self):
        s = ClarificationSession()
        assert s.turn_count == 0


class TestValidTransitions:
    def test_idle_to_awaiting_slot(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        assert s.state == S.AWAITING_SLOT

    def test_idle_to_awaiting_confirmation(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_CONFIRMATION)
        assert s.state == S.AWAITING_CONFIRMATION

    def test_idle_to_failed(self):
        s = ClarificationSession()
        s.transition(S.FAILED)
        assert s.state == S.FAILED

    def test_awaiting_slot_self_loop(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.AWAITING_SLOT)  # retry same slot
        assert s.state == S.AWAITING_SLOT

    def test_awaiting_slot_to_confirmation(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.AWAITING_CONFIRMATION)
        assert s.state == S.AWAITING_CONFIRMATION

    def test_awaiting_slot_to_idle(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.IDLE)
        assert s.state == S.IDLE

    def test_awaiting_slot_to_canceled(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        assert s.state == S.CANCELED

    def test_awaiting_slot_to_interrupted(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.INTERRUPTED)
        assert s.state == S.INTERRUPTED

    def test_awaiting_confirmation_to_idle(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_CONFIRMATION)
        s.transition(S.IDLE)
        assert s.state == S.IDLE

    def test_awaiting_confirmation_to_canceled(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_CONFIRMATION)
        s.transition(S.CANCELED)
        assert s.state == S.CANCELED

    def test_interrupted_to_idle(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.INTERRUPTED)
        s.transition(S.IDLE)
        assert s.state == S.IDLE

    def test_canceled_to_idle(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        s.transition(S.IDLE)
        assert s.state == S.IDLE

    def test_failed_to_idle(self):
        s = ClarificationSession()
        s.transition(S.FAILED)
        s.transition(S.IDLE)
        assert s.state == S.IDLE


class TestInvalidTransitions:
    def test_idle_to_interrupted_invalid(self):
        s = ClarificationSession()
        with pytest.raises(InvalidTransitionError):
            s.transition(S.INTERRUPTED)

    def test_awaiting_confirmation_to_awaiting_slot_invalid(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_CONFIRMATION)
        with pytest.raises(InvalidTransitionError):
            s.transition(S.AWAITING_SLOT)

    def test_interrupted_to_canceled_invalid(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.INTERRUPTED)
        with pytest.raises(InvalidTransitionError):
            s.transition(S.CANCELED)

    def test_canceled_to_failed_invalid(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        with pytest.raises(InvalidTransitionError):
            s.transition(S.FAILED)

    def test_state_unchanged_after_invalid_transition(self):
        s = ClarificationSession()
        try:
            s.transition(S.INTERRUPTED)
        except InvalidTransitionError:
            pass
        assert s.state == S.IDLE


class TestTransitionMeta:
    def test_sets_skill_name(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT, skill_name="navigate")
        assert s.skill_name == "navigate"

    def test_sets_current_slot(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT, current_slot="destination")
        assert s.current_slot == "destination"

    def test_sets_interrupted_by(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.INTERRUPTED, interrupted_by="停下")
        assert s.interrupted_by == "停下"

    def test_sets_cancel_reason(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED, cancel_reason="user said no")
        assert s.cancel_reason == "user said no"

    def test_sets_error(self):
        s = ClarificationSession()
        s.transition(S.FAILED, error="TTS crashed")
        assert s.error == "TTS crashed"

    def test_unknown_meta_key_silently_ignored(self):
        s = ClarificationSession()
        # 'nonexistent' is not an attribute — should not raise
        s.transition(S.AWAITING_SLOT, nonexistent="value")
        assert s.state == S.AWAITING_SLOT


class TestProperties:
    def test_is_active_awaiting_slot(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        assert s.is_active is True

    def test_is_active_awaiting_confirmation(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_CONFIRMATION)
        assert s.is_active is True

    def test_is_not_active_in_canceled(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        assert s.is_active is False

    def test_is_terminal_in_canceled(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        assert s.is_terminal is True

    def test_is_terminal_in_failed(self):
        s = ClarificationSession()
        s.transition(S.FAILED)
        assert s.is_terminal is True

    def test_is_not_terminal_in_interrupted(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.INTERRUPTED)
        assert s.is_terminal is False


class TestReset:
    def test_reset_returns_to_idle(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT, skill_name="nav", current_slot="dest")
        s.turn_count = 5
        s.reset()
        assert s.state == S.IDLE

    def test_reset_clears_skill_name(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT, skill_name="navigate")
        s.reset()
        assert s.skill_name == ""

    def test_reset_clears_turn_count(self):
        s = ClarificationSession()
        s.turn_count = 10
        s.reset()
        assert s.turn_count == 0

    def test_reset_clears_error(self):
        s = ClarificationSession()
        s.transition(S.FAILED, error="boom")
        s.reset()
        assert s.error == ""

    def test_reset_allows_new_transitions(self):
        s = ClarificationSession()
        s.transition(S.AWAITING_SLOT)
        s.transition(S.CANCELED)
        s.reset()
        # Should be able to start fresh
        s.transition(S.AWAITING_SLOT)
        assert s.state == S.AWAITING_SLOT
