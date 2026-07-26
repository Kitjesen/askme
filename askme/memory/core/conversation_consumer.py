"""Durable consumer for committed Conversation Core turns.

The conversation ledger owns turn truth.  This module owns only memory
admission progress: it reads committed-turn projections, submits user-authored
text to the governed memory seam, and checkpoints each acknowledged event.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from askme.conversation.models import CommittedTurnEvent
from askme.memory.core.turn_admission import MemoryCandidate, TurnAdmissionResult

_CHECKPOINT_SCHEMA = "askme.memory.conversation-consumer-checkpoint"
_CHECKPOINT_VERSION = 1
_CONSUMER_NAME = "conversation-memory"
_IDEMPOTENCY_PREFIX = "conversation-memory:v1:"


class CommittedTurnSource(Protocol):
    """Minimal read seam supplied by Conversation Core."""

    def list_committed_turn_events(
        self,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Sequence[CommittedTurnEvent]: ...


class TurnAdmissionSink(Protocol):
    """Minimal write seam supplied by durable Memory."""

    async def admit_turn(self, user_text: str, **kwargs: Any) -> TurnAdmissionResult: ...


class ConversationMemoryConsumerError(RuntimeError):
    """Base error for committed-turn memory consumption."""


class ConversationMemoryCheckpointError(ConversationMemoryConsumerError):
    """Base error for an unreadable or incompatible memory checkpoint."""


class ConversationMemoryCheckpointCorruptError(ConversationMemoryCheckpointError):
    """Raised when checkpoint JSON does not satisfy the durable schema."""


class ConversationMemoryCheckpointMismatchError(ConversationMemoryCheckpointError):
    """Raised when a valid checkpoint belongs to another contract or source."""

    def __init__(self, field: str, *, expected: Any, actual: Any) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"conversation memory checkpoint {field} mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )


class ConversationMemoryCheckpointWriteError(ConversationMemoryCheckpointError):
    """Raised when an acknowledged event cannot be atomically checkpointed."""


class ErasureDeletionUnsupportedError(ConversationMemoryConsumerError):
    """Raised when processing is attempted without an erasure deletion guarantee."""


class ConversationMemoryProcessingError(ConversationMemoryConsumerError):
    """Raised when a sink does not acknowledge an event completely."""

    def __init__(self, event: CommittedTurnEvent, reason: str) -> None:
        self.event_id = event.event_id
        self.sequence = event.sequence
        self.reason = str(reason)
        super().__init__(
            f"memory did not acknowledge committed event {event.event_id!r} "
            f"at sequence {event.sequence}: {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class ConversationMemoryCheckpoint:
    """Memory-owned progress through one stable conversation event source."""

    schema: str
    version: int
    consumer: str
    source_id: str
    last_sequence: int
    last_event_id: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConversationMemoryRunResult:
    """Observable outcome of one bounded source poll."""

    fetched_count: int
    acknowledged_count: int
    admitted_count: int
    rejected_count: int
    last_sequence: int
    last_event_id: str


@dataclass(frozen=True, slots=True)
class ConversationMemoryConsumerStatus:
    """Rollout gate for a consumer that cannot infer later thread erasure."""

    processing_allowed: bool
    erasure_deletion_supported: bool
    blocked_reason: str


class ConversationMemoryConsumer:
    """Consume committed turns in order and atomically checkpoint acknowledgements."""

    def __init__(
        self,
        *,
        source: CommittedTurnSource,
        sink: TurnAdmissionSink,
        checkpoint_path: str | Path,
        source_id: str,
        batch_size: int = 100,
        erasure_deletion_supported: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        clean_source_id = str(source_id or "").strip()
        if not clean_source_id:
            raise ValueError("source_id must be a stable non-empty identifier")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 1000
        ):
            raise ValueError("batch_size must be between 1 and 1000")
        self._source = source
        self._sink = sink
        self._checkpoint_path = Path(checkpoint_path)
        self._source_id = clean_source_id
        self._batch_size = batch_size
        self._erasure_deletion_supported = bool(erasure_deletion_supported)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_lock = asyncio.Lock()

    def status(self) -> ConversationMemoryConsumerStatus:
        """Return the explicit privacy-safety gate without polling the source."""

        allowed = self._erasure_deletion_supported
        return ConversationMemoryConsumerStatus(
            processing_allowed=allowed,
            erasure_deletion_supported=allowed,
            blocked_reason="" if allowed else "erasure_deletion_unsupported",
        )

    async def run_once(self) -> ConversationMemoryRunResult:
        """Poll and acknowledge at most one configured batch of committed turns."""

        async with self._run_lock:
            return await self._run_once_locked()

    async def _run_once_locked(self) -> ConversationMemoryRunResult:
        """Run one poll while holding the per-consumer linearization lock."""

        if not self._erasure_deletion_supported:
            raise ErasureDeletionUnsupportedError(
                "committed-turn memory processing is disabled because erased-thread "
                "deletion propagation is not supported"
            )

        checkpoint = self._load_checkpoint()
        events = list(
            self._source.list_committed_turn_events(
                after_sequence=checkpoint.last_sequence,
                limit=self._batch_size,
            )
        )
        self._validate_event_page(events, after_sequence=checkpoint.last_sequence)
        acknowledged = 0
        admitted = 0
        rejected = 0

        for event in events:
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            try:
                result = await self._sink.admit_turn(
                    event.user_text,
                    source_turn_id=event.turn_id,
                    source_event_id=event.event_id,
                    source_sequence=event.sequence,
                    source_thread_id=event.thread_id,
                    idempotency_key=f"{_IDEMPOTENCY_PREFIX}{event.event_id}",
                    source=event.source,
                    occurred_at=event.occurred_at,
                    customer_id=_scope_text(metadata.get("customer_id")),
                    project_id=_scope_text(metadata.get("project_id")),
                    user_id=_scope_text(metadata.get("user_id")),
                )
            except ConversationMemoryConsumerError:
                raise
            except Exception as exc:
                raise ConversationMemoryProcessingError(
                    event,
                    f"sink_exception:{type(exc).__name__}",
                ) from exc

            result = self._validated_admission_result(event, result)
            errors = result.persistence_errors
            if result.admitted:
                fully_persisted = (
                    not errors and result.persisted_count == len(result.candidates)
                )
                if not fully_persisted:
                    raise ConversationMemoryProcessingError(
                        event,
                        "partial_or_failed_persistence",
                    )
                admitted += 1
            elif errors:
                raise ConversationMemoryProcessingError(event, "rejection_with_persistence_error")
            else:
                rejected += 1

            checkpoint = self._checkpoint_for(event)
            self._write_checkpoint(checkpoint)
            acknowledged += 1

        return ConversationMemoryRunResult(
            fetched_count=len(events),
            acknowledged_count=acknowledged,
            admitted_count=admitted,
            rejected_count=rejected,
            last_sequence=checkpoint.last_sequence,
            last_event_id=checkpoint.last_event_id,
        )

    @staticmethod
    def _validated_admission_result(
        event: CommittedTurnEvent,
        result: Any,
    ) -> TurnAdmissionResult:
        if not isinstance(result, TurnAdmissionResult):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:type",
            )
        if type(result.admitted) is not bool:
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:admitted",
            )
        if not isinstance(result.candidates, tuple) or any(
            not isinstance(candidate, MemoryCandidate) for candidate in result.candidates
        ):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:candidates",
            )
        if (
            not isinstance(result.persisted_count, int)
            or isinstance(result.persisted_count, bool)
            or result.persisted_count < 0
            or result.persisted_count > len(result.candidates)
        ):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:persisted_count",
            )
        if not isinstance(result.persistence_errors, tuple) or any(
            not isinstance(error, str) for error in result.persistence_errors
        ):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:persistence_errors",
            )
        if not isinstance(result.rejected_reason, str):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:rejected_reason",
            )
        if result.admitted and not result.candidates:
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:admitted_without_candidates",
            )
        if not result.admitted and (result.candidates or result.persisted_count):
            raise ConversationMemoryProcessingError(
                event,
                "malformed_admission_result:rejected_with_persistence",
            )
        return result

    @staticmethod
    def _validate_event_page(
        events: Sequence[CommittedTurnEvent],
        *,
        after_sequence: int,
    ) -> None:
        previous_sequence = after_sequence
        for event in events:
            if event.sequence <= previous_sequence:
                raise ConversationMemoryProcessingError(event, "source_sequence_not_strict")
            previous_sequence = event.sequence

    def _load_checkpoint(self) -> ConversationMemoryCheckpoint:
        if not self._checkpoint_path.exists():
            return ConversationMemoryCheckpoint(
                schema=_CHECKPOINT_SCHEMA,
                version=_CHECKPOINT_VERSION,
                consumer=_CONSUMER_NAME,
                source_id=self._source_id,
                last_sequence=0,
                last_event_id="",
                updated_at="",
            )
        try:
            payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConversationMemoryCheckpointCorruptError(
                f"cannot read conversation memory checkpoint {self._checkpoint_path}: "
                f"{type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise ConversationMemoryCheckpointCorruptError(
                "conversation memory checkpoint must be a JSON object"
            )

        expected_fields = set(ConversationMemoryCheckpoint.__dataclass_fields__)
        if set(payload) != expected_fields:
            raise ConversationMemoryCheckpointCorruptError(
                "conversation memory checkpoint fields do not match its schema"
            )
        expected_identity = {
            "schema": _CHECKPOINT_SCHEMA,
            "version": _CHECKPOINT_VERSION,
            "consumer": _CONSUMER_NAME,
            "source_id": self._source_id,
        }
        for field_name, expected in expected_identity.items():
            actual = payload.get(field_name)
            if actual != expected:
                raise ConversationMemoryCheckpointMismatchError(
                    field_name,
                    expected=expected,
                    actual=actual,
                )

        last_sequence = payload.get("last_sequence")
        last_event_id = payload.get("last_event_id")
        updated_at = payload.get("updated_at")
        if (
            not isinstance(last_sequence, int)
            or isinstance(last_sequence, bool)
            or last_sequence < 0
            or not isinstance(last_event_id, str)
            or not isinstance(updated_at, str)
        ):
            raise ConversationMemoryCheckpointCorruptError(
                "conversation memory checkpoint progress fields are invalid"
            )
        if (last_sequence == 0) != (last_event_id == ""):
            raise ConversationMemoryCheckpointCorruptError(
                "conversation memory checkpoint sequence and event identity disagree"
            )
        if last_sequence > 0:
            try:
                parsed_updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ConversationMemoryCheckpointCorruptError(
                    "conversation memory checkpoint updated_at is invalid"
                ) from exc
            if parsed_updated_at.tzinfo is None or parsed_updated_at.utcoffset() is None:
                raise ConversationMemoryCheckpointCorruptError(
                    "conversation memory checkpoint updated_at must include a timezone"
                )
        elif updated_at:
            raise ConversationMemoryCheckpointCorruptError(
                "an initial conversation memory checkpoint cannot have updated_at"
            )
        return ConversationMemoryCheckpoint(**payload)

    def _checkpoint_for(self, event: CommittedTurnEvent) -> ConversationMemoryCheckpoint:
        return ConversationMemoryCheckpoint(
            schema=_CHECKPOINT_SCHEMA,
            version=_CHECKPOINT_VERSION,
            consumer=_CONSUMER_NAME,
            source_id=self._source_id,
            last_sequence=event.sequence,
            last_event_id=event.event_id,
            updated_at=_iso_utc(self._clock()),
        )

    def _write_checkpoint(self, checkpoint: ConversationMemoryCheckpoint) -> None:
        temp_path = self._checkpoint_path.with_name(
            f".{self._checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(asdict(checkpoint), handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._checkpoint_path)
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ConversationMemoryCheckpointWriteError(
                f"cannot atomically write conversation memory checkpoint "
                f"{self._checkpoint_path}: {type(exc).__name__}"
            ) from exc


def _scope_text(value: Any) -> str:
    return str(value or "").strip()


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


__all__ = [
    "CommittedTurnSource",
    "ConversationMemoryCheckpoint",
    "ConversationMemoryCheckpointCorruptError",
    "ConversationMemoryCheckpointError",
    "ConversationMemoryCheckpointMismatchError",
    "ConversationMemoryCheckpointWriteError",
    "ConversationMemoryConsumer",
    "ConversationMemoryConsumerError",
    "ConversationMemoryConsumerStatus",
    "ConversationMemoryProcessingError",
    "ConversationMemoryRunResult",
    "ErasureDeletionUnsupportedError",
    "TurnAdmissionSink",
]
