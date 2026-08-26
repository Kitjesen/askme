"""Conversation Core: durable Thread, Turn, and Generation ownership."""

from askme.conversation.identity import canonical_thread_id
from askme.conversation.interaction import (
    ApprovalScope,
    CancellationToken,
    ConfirmationKind,
    ConfirmationScope,
    GenerationStarted,
    InteractionInput,
    InteractionTurnContext,
    InteractionTurnManager,
    TurnOutcome,
)
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
    "ApprovalScope",
    "CancellationToken",
    "ConfirmationKind",
    "ConfirmationScope",
    "ConflictingThreadAliases",
    "CommittedTurnEvent",
    "ConversationLedgerError",
    "ConversationThread",
    "DuplicateEntity",
    "EntityNotFound",
    "GenerationStarted",
    "GenerationStatus",
    "InteractionInput",
    "InteractionTurnContext",
    "InteractionTurnManager",
    "InvalidTransition",
    "LedgerCorruptionError",
    "LegacyMigrationResult",
    "ThreadStatus",
    "TurnGeneration",
    "TurnInProgress",
    "TurnOutcome",
    "TurnRecord",
    "TurnStatus",
    "VoiceTurnLedger",
    "canonical_thread_id",
    "migrate_legacy_history",
]
