"""Clarification Session State Machine.

Tracks the lifecycle of a single clarification interaction.

State diagram:
  IDLE
    ↓  (slot needed)
  AWAITING_SLOT  ←─ (retry same/next slot)
    ↓  (slots filled, no confirm needed)
  IDLE
    ↓  (slots filled, confirm_before_execute=True)
  AWAITING_CONFIRMATION
    ↓ confirmed         ↓ denied/ambiguous    ↓ interrupted
  IDLE               CANCELED             INTERRUPTED → IDLE

Any state can transition to FAILED (system/audio error).
INTERRUPTED / CANCELED / FAILED all reset to IDLE when the run ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClarificationState(Enum):
    IDLE                  = "idle"
    AWAITING_SLOT         = "awaiting_slot"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    INTERRUPTED           = "interrupted"
    CANCELED              = "canceled"
    FAILED                = "failed"


# Legal transitions: state → set of allowed next states
_ALLOWED: dict[ClarificationState, frozenset[ClarificationState]] = {
    ClarificationState.IDLE: frozenset({
        ClarificationState.AWAITING_SLOT,
        ClarificationState.AWAITING_CONFIRMATION,
        ClarificationState.FAILED,
    }),
    ClarificationState.AWAITING_SLOT: frozenset({
        ClarificationState.AWAITING_SLOT,           # retry (self-loop)
        ClarificationState.AWAITING_CONFIRMATION,   # slots done, need confirm
        ClarificationState.IDLE,                    # slots done, no confirm
        ClarificationState.INTERRUPTED,
        ClarificationState.CANCELED,
        ClarificationState.FAILED,
    }),
    ClarificationState.AWAITING_CONFIRMATION: frozenset({
        ClarificationState.IDLE,       # confirmed → proceed
        ClarificationState.CANCELED,   # denied
        ClarificationState.INTERRUPTED,
        ClarificationState.FAILED,
    }),
    ClarificationState.INTERRUPTED: frozenset({
        ClarificationState.IDLE,       # reset after handling
    }),
    ClarificationState.CANCELED: frozenset({
        ClarificationState.IDLE,       # reset
    }),
    ClarificationState.FAILED: frozenset({
        ClarificationState.IDLE,       # reset
    }),
}


class InvalidTransitionError(Exception):
    """Raised when a state machine transition is not permitted."""


@dataclass
class ClarificationSession:
    """Tracks state for one clarification interaction.

    One instance is created per ProactiveOrchestrator.run() call.
    It is never reused across separate runs — the orchestrator creates a
    fresh session each time so state cannot leak between commands.
    """

    state: ClarificationState = ClarificationState.IDLE
    skill_name: str = ""
    current_slot: str = ""    # slot currently being collected
    turn_count: int = 0       # total clarification turns taken
    interrupted_by: str = ""  # text that triggered the interrupt
    cancel_reason: str = ""   # reason for cancellation
    error: str = ""           # description if FAILED

    @property
    def is_terminal(self) -> bool:
        """True once the session ended without proceeding to execution."""
        return self.state in {ClarificationState.CANCELED, ClarificationState.FAILED}

    @property
    def is_active(self) -> bool:
        """True while waiting for user input."""
        return self.state in {
            ClarificationState.AWAITING_SLOT,
            ClarificationState.AWAITING_CONFIRMATION,
        }

    def transition(self, new_state: ClarificationState, **meta: Any) -> None:
        """Move to *new_state*, optionally writing metadata fields.

        Raises :class:`InvalidTransitionError` if the transition is not in the
        allowed set for the current state.
        """
        allowed = _ALLOWED.get(self.state, frozenset())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Invalid transition: {self.state.value!r} → {new_state.value!r}"
            )
        self.state = new_state
        for key, value in meta.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self) -> None:
        """Return to IDLE, clearing all transient fields."""
        self.state = ClarificationState.IDLE
        self.skill_name = ""
        self.current_slot = ""
        self.turn_count = 0
        self.interrupted_by = ""
        self.cancel_reason = ""
        self.error = ""
