"""Domain models for durable conversation lifecycle tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ThreadStatus(StrEnum):
    """Lifecycle of a product-level conversation thread."""

    OPEN = "open"
    IDLE = "idle"
    CLOSED = "closed"
    EXPIRED = "expired"
    ERASED = "erased"


class TurnStatus(StrEnum):
    """Lifecycle of one user/assistant business exchange."""

    STARTED = "started"
    LISTENING = "listening"
    TRANSCRIBED = "transcribed"
    ROUTED = "routed"
    GENERATING = "generating"
    SPEAKING = "speaking"
    AWAITING_COMMIT = "awaiting_commit"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


class GenerationStatus(StrEnum):
    """Lifecycle of one replaceable provider response attempt."""

    STARTED = "started"
    PROVIDER_TRANSCRIBING = "provider_transcribing"
    PROVIDER_RESPONDING = "provider_responding"
    HELD_FOR_APPROVAL = "held_for_approval"
    APPROVED = "approved"
    DISCARDED = "discarded"
    TRUNCATED = "truncated"
    ROLLED_BACK = "rolled_back"
    PROVIDER_FAILED = "provider_failed"


@dataclass(frozen=True, slots=True)
class ConversationThread:
    """A logical conversation that can outlive any provider connection."""

    thread_id: str
    channel: str
    person_id: str | None
    operator_id: str | None
    robot_id: str | None
    site_id: str | None
    status: ThreadStatus
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    closed_at: datetime | None = None
    timezone: str = "UTC"
    local_day: str = ""

    @property
    def started_at(self) -> datetime:
        """Product-language alias for the creation timestamp."""

        return self.created_at

    @property
    def session_id(self) -> str:
        """Temporary compatibility alias for legacy gateway callers."""

        return self.thread_id


@dataclass(frozen=True, slots=True)
class CommittedTurnEvent:
    """Provider-neutral projection of one durable ``turn.committed`` event."""

    event_id: str
    sequence: int
    occurred_at: datetime
    thread_id: str
    turn_id: str
    turn_sequence: int
    source: str
    user_text: str
    assistant_text: str
    heard_text: str
    played_ms: int
    playback_disposition: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """The atomic commit/audit unit for one conversational exchange."""

    turn_id: str
    thread_id: str
    sequence: int
    source: str
    status: TurnStatus
    user_text: str
    assistant_text: str
    heard_text: str
    played_ms: int
    playback_disposition: str | None
    cancel_reason: str | None
    failure_reason: str | None
    suppression_reason: str | None
    generation_ids: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None
    settled_at: datetime | None = None

    @property
    def conversation_session_id(self) -> str:
        """Temporary compatibility alias for legacy history callers."""

        return self.thread_id


@dataclass(frozen=True, slots=True)
class TurnGeneration:
    """One provider attempt; reconnects create generations, not threads."""

    generation_id: str
    turn_id: str
    thread_id: str
    epoch: int
    provider: str
    provider_session_id: str | None
    provider_generation_id: str | None
    status: GenerationStatus
    response_text: str
    heard_text: str
    played_ms: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    settled_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LegacyMigrationResult:
    """Summary of a deterministic, read-only legacy import."""

    thread_count: int = 0
    turn_count: int = 0
    message_count: int = 0
    thread_ids: tuple[str, ...] = field(default_factory=tuple)


class ConversationLedgerError(RuntimeError):
    """Base exception for Conversation Core failures."""


class TurnInProgress(ConversationLedgerError):
    """Raised when a Thread already owns a different non-terminal Turn."""

    def __init__(self, thread_id: str, blocking_turn_id: str) -> None:
        self.thread_id = str(thread_id)
        self.blocking_turn_id = str(blocking_turn_id)
        super().__init__(
            f"thread {self.thread_id!r} already has active turn "
            f"{self.blocking_turn_id!r}"
        )


class ConflictingThreadAliases(ConversationLedgerError, ValueError):
    """Raised when compatibility aliases identify different threads."""


class InvalidTransition(ConversationLedgerError, ValueError):
    """Raised when an entity attempts an illegal lifecycle transition."""


class EntityNotFound(ConversationLedgerError, LookupError):
    """Raised when a referenced thread, turn, or generation does not exist."""


class DuplicateEntity(ConversationLedgerError, ValueError):
    """Raised when an explicit ID is already owned by another parent."""


class LedgerCorruptionError(ConversationLedgerError, ValueError):
    """Raised for corruption anywhere except an incomplete trailing record."""
