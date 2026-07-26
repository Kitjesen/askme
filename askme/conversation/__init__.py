"""Conversation Core: durable Thread, Turn, and Generation ownership."""

from askme.conversation.identity import canonical_thread_id
from askme.conversation.ledger import VoiceTurnLedger
from askme.conversation.migration import migrate_legacy_history
from askme.conversation.models import (
    CommittedTurnEvent,
    ConflictingThreadAliases,
    ConversationLedgerError,
    ConversationThread,
    DuplicateEntity,
    EntityNotFound,
    GenerationStatus,
    InvalidTransition,
    LedgerCorruptionError,
    LegacyMigrationResult,
    ThreadStatus,
    TurnGeneration,
    TurnInProgress,
    TurnRecord,
    TurnStatus,
)

__all__ = [
    "ConflictingThreadAliases",
    "CommittedTurnEvent",
    "ConversationLedgerError",
    "ConversationThread",
    "DuplicateEntity",
    "EntityNotFound",
    "GenerationStatus",
    "InvalidTransition",
    "LedgerCorruptionError",
    "LegacyMigrationResult",
    "ThreadStatus",
    "TurnGeneration",
    "TurnInProgress",
    "TurnRecord",
    "TurnStatus",
    "VoiceTurnLedger",
    "canonical_thread_id",
    "migrate_legacy_history",
]
