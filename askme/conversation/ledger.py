"""Thread-safe, append-only JSONL ledger for conversation lifecycle events."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from datetime import timezone as fixed_timezone
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, overload
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from askme.conversation.identity import canonical_thread_id
from askme.conversation.models import (
    CommittedTurnEvent,
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

_T = TypeVar("_T")
_SCHEMA_VERSION = 1
_TERMINAL_TURNS = {
    TurnStatus.COMMITTED,
    TurnStatus.CANCELLED,
    TurnStatus.FAILED,
    TurnStatus.SUPPRESSED,
}
_TERMINAL_GENERATIONS = {
    GenerationStatus.APPROVED,
    GenerationStatus.DISCARDED,
    GenerationStatus.TRUNCATED,
    GenerationStatus.ROLLED_BACK,
    GenerationStatus.PROVIDER_FAILED,
}
_THREAD_TRANSITIONS: dict[ThreadStatus, set[ThreadStatus]] = {
    ThreadStatus.OPEN: {
        ThreadStatus.IDLE,
        ThreadStatus.CLOSED,
        ThreadStatus.EXPIRED,
        ThreadStatus.ERASED,
    },
    ThreadStatus.IDLE: {
        ThreadStatus.OPEN,
        ThreadStatus.CLOSED,
        ThreadStatus.EXPIRED,
        ThreadStatus.ERASED,
    },
    ThreadStatus.CLOSED: {ThreadStatus.ERASED},
    ThreadStatus.EXPIRED: {ThreadStatus.ERASED},
    ThreadStatus.ERASED: set(),
}

_TURN_TRANSITIONS: dict[TurnStatus, set[TurnStatus]] = {
    TurnStatus.STARTED: {
        TurnStatus.LISTENING,
        TurnStatus.TRANSCRIBED,
        TurnStatus.ROUTED,
        TurnStatus.GENERATING,
        TurnStatus.SPEAKING,
        TurnStatus.AWAITING_COMMIT,
        *_TERMINAL_TURNS,
    },
    TurnStatus.LISTENING: {
        TurnStatus.TRANSCRIBED,
        TurnStatus.ROUTED,
        TurnStatus.GENERATING,
        *_TERMINAL_TURNS,
    },
    TurnStatus.TRANSCRIBED: {
        TurnStatus.ROUTED,
        TurnStatus.GENERATING,
        TurnStatus.SPEAKING,
        TurnStatus.AWAITING_COMMIT,
        *_TERMINAL_TURNS,
    },
    TurnStatus.ROUTED: {
        TurnStatus.GENERATING,
        TurnStatus.SPEAKING,
        TurnStatus.AWAITING_COMMIT,
        *_TERMINAL_TURNS,
    },
    TurnStatus.GENERATING: {
        TurnStatus.SPEAKING,
        TurnStatus.AWAITING_COMMIT,
        *_TERMINAL_TURNS,
    },
    TurnStatus.SPEAKING: {
        TurnStatus.AWAITING_COMMIT,
        *_TERMINAL_TURNS,
    },
    TurnStatus.AWAITING_COMMIT: set(_TERMINAL_TURNS),
    TurnStatus.COMMITTED: set(),
    TurnStatus.CANCELLED: set(),
    TurnStatus.FAILED: set(),
    TurnStatus.SUPPRESSED: set(),
}

_GENERATION_TRANSITIONS: dict[GenerationStatus, set[GenerationStatus]] = {
    GenerationStatus.STARTED: {
        GenerationStatus.PROVIDER_TRANSCRIBING,
        GenerationStatus.PROVIDER_RESPONDING,
        GenerationStatus.HELD_FOR_APPROVAL,
        *_TERMINAL_GENERATIONS,
    },
    GenerationStatus.PROVIDER_TRANSCRIBING: {
        GenerationStatus.PROVIDER_RESPONDING,
        GenerationStatus.HELD_FOR_APPROVAL,
        *_TERMINAL_GENERATIONS,
    },
    GenerationStatus.PROVIDER_RESPONDING: {
        GenerationStatus.HELD_FOR_APPROVAL,
        *_TERMINAL_GENERATIONS,
    },
    GenerationStatus.HELD_FOR_APPROVAL: {
        GenerationStatus.APPROVED,
        GenerationStatus.DISCARDED,
        GenerationStatus.ROLLED_BACK,
        GenerationStatus.PROVIDER_FAILED,
    },
    GenerationStatus.APPROVED: set(),
    GenerationStatus.DISCARDED: set(),
    GenerationStatus.TRUNCATED: set(),
    GenerationStatus.ROLLED_BACK: set(),
    GenerationStatus.PROVIDER_FAILED: set(),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clean_identity(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _event_operation(event_type: str) -> str:
    if event_type in {"generation.started", "generation.replaced"}:
        return "generation.start"
    return event_type


def _local_day(value: datetime, timezone: str) -> str:
    zone: tzinfo
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError):
        # Windows images may not ship the IANA database. Keep the product's
        # primary China deployment correct without making ledger startup
        # depend on an optional tzdata wheel; unknown zones fail safe to UTC.
        fixed_offsets = {
            "Asia/Shanghai": 8,
            "Asia/Chongqing": 8,
            "Asia/Hong_Kong": 8,
            "Asia/Singapore": 8,
        }
        hours = fixed_offsets.get(timezone)
        zone = UTC if hours is None else fixed_timezone(timedelta(hours=hours))
    return _as_utc(value).astimezone(zone).date().isoformat()


class VoiceTurnLedger:
    """Durable single-writer owner for Thread, Turn, and Generation state.

    Each operation is validated under one re-entrant lock, appended and fsynced
    as one JSON line, and only then applied to the in-memory read model. Replay
    tolerates an incomplete final line (a power-loss write window) but treats
    corruption in any completed line as an integrity error.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        fsync: bool = True,
        recover_incomplete: bool = True,
    ) -> None:
        self.path = Path(path)
        self._clock = clock or _utc_now
        self._fsync = bool(fsync)
        self._lock = RLock()
        self._threads: dict[str, ConversationThread] = {}
        self._turns: dict[str, TurnRecord] = {}
        self._generations: dict[str, TurnGeneration] = {}
        self._turn_ids_by_thread: dict[str, list[str]] = {}
        self._generation_ids_by_turn: dict[str, list[str]] = {}
        self._committed_turn_events: list[CommittedTurnEvent] = []
        self._event_targets: dict[str, tuple[str, str]] = {}
        self._event_types: dict[str, str] = {}
        self._event_sequence = 0
        self._replay()
        if recover_incomplete:
            self._recover_incomplete_turns()

    # ------------------------------------------------------------------
    # Read model
    # ------------------------------------------------------------------

    def get_thread(self, thread_id: str) -> ConversationThread | None:
        with self._lock:
            return self._copy(self._threads.get(str(thread_id)))

    def get_turn(self, turn_id: str) -> TurnRecord | None:
        with self._lock:
            return self._copy(self._turns.get(str(turn_id)))

    def get_generation(self, generation_id: str) -> TurnGeneration | None:
        with self._lock:
            return self._copy(self._generations.get(str(generation_id)))

    def list_threads(self) -> list[ConversationThread]:
        with self._lock:
            return [self._copy(item) for item in self._threads.values()]

    def list_turns(self, *, thread_id: str | None = None) -> list[TurnRecord]:
        with self._lock:
            if thread_id is None:
                return [self._copy(item) for item in self._turns.values()]
            return [
                self._copy(self._turns[item_id])
                for item_id in self._turn_ids_by_thread.get(str(thread_id), [])
            ]

    def list_generations(self, *, turn_id: str | None = None) -> list[TurnGeneration]:
        with self._lock:
            if turn_id is None:
                return [self._copy(item) for item in self._generations.values()]
            return [
                self._copy(self._generations[item_id])
                for item_id in self._generation_ids_by_turn.get(str(turn_id), [])
            ]

    def list_committed_turn_events(
        self,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[CommittedTurnEvent]:
        """Return commits in global order, omitting events from erased Threads."""

        if after_sequence < 0:
            raise ValueError("after_sequence must be greater than or equal to 0")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            events: list[CommittedTurnEvent] = []
            for event in self._committed_turn_events:
                if event.sequence <= after_sequence:
                    continue
                if self._threads[event.thread_id].status is ThreadStatus.ERASED:
                    continue
                events.append(self._copy(event))
                if len(events) == limit:
                    break
            return events

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._event_sequence

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def resolve_thread(
        self,
        *,
        thread_id: str | None = None,
        conversation_thread_id: str | None = None,
        conversation_session_id: str | None = None,
        conversation_id: str | None = None,
        chat_session_id: str | None = None,
        session_id: str | None = None,
        channel: str = "voice",
        person_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        timezone: str = "UTC",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> ConversationThread:
        explicit_id = canonical_thread_id(
            thread_id=thread_id,
            conversation_thread_id=conversation_thread_id,
            conversation_session_id=conversation_session_id,
            conversation_id=conversation_id,
            chat_session_id=chat_session_id,
            session_id=session_id,
        )
        normalized_channel = str(channel or "voice").strip() or "voice"
        identity = (
            normalized_channel,
            _clean_identity(person_id),
            _clean_identity(operator_id),
            _clean_identity(robot_id),
            _clean_identity(site_id),
        )
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "thread",
                explicit_id,
                event_type="thread.created",
            )
            if duplicate is not None:
                return duplicate
            if explicit_id is not None and explicit_id in self._threads:
                existing = self._threads[explicit_id]
                conflicts = []
                supplied_identity = {
                    "channel": normalized_channel,
                    "person_id": identity[1],
                    "operator_id": identity[2],
                    "robot_id": identity[3],
                    "site_id": identity[4],
                }
                for field_name, supplied in supplied_identity.items():
                    if supplied is not None and getattr(existing, field_name) != supplied:
                        conflicts.append(
                            f"{field_name}={supplied!r} (stored {getattr(existing, field_name)!r})"
                        )
                if conflicts:
                    raise DuplicateEntity(
                        f"thread {explicit_id!r} identity conflict: {', '.join(conflicts)}"
                    )
                return self._copy(existing)
            # A channel alone is not a safe conversation identity.  Without
            # an explicit thread alias or at least one stable participant /
            # robot / site key, fail new instead of merging unrelated local
            # users into one process-wide anonymous thread.
            if explicit_id is None and any(identity[1:]):
                for existing in reversed(tuple(self._threads.values())):
                    existing_identity = (
                        existing.channel,
                        existing.person_id,
                        existing.operator_id,
                        existing.robot_id,
                        existing.site_id,
                    )
                    if existing.status in {ThreadStatus.OPEN, ThreadStatus.IDLE} and (
                        existing_identity == identity
                    ):
                        return self._copy(existing)

            resolved_id = explicit_id or str(uuid4())
            occurred_at = self._event_time(at)
            payload = {
                "thread_id": resolved_id,
                "channel": normalized_channel,
                "person_id": identity[1],
                "operator_id": identity[2],
                "robot_id": identity[3],
                "site_id": identity[4],
                "timezone": str(timezone or "UTC"),
                "metadata": deepcopy(metadata or {}),
                "created_at": occurred_at.isoformat(),
            }
            self._record(
                "thread.created",
                payload,
                entity_type="thread",
                entity_id=resolved_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._threads[resolved_id])

    def transition_thread(
        self,
        thread_id: str,
        status: ThreadStatus | str,
        *,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> ConversationThread:
        target = ThreadStatus(status)
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "thread",
                str(thread_id).strip(),
                event_type=(
                    "thread.erased"
                    if target is ThreadStatus.ERASED
                    else "thread.transitioned"
                ),
            )
            if duplicate is not None:
                return duplicate
            thread = self._require_thread(thread_id)
            if target == thread.status:
                return self._copy(thread)
            if target not in _THREAD_TRANSITIONS[thread.status]:
                raise InvalidTransition(f"thread {thread_id}: {thread.status} -> {target}")
            occurred_at = self._event_time(at)
            if target is ThreadStatus.ERASED:
                # Settle every in-flight fact before the tombstone becomes
                # visible.  Keeping this under the ledger lock prevents a
                # late provider completion from racing between cancellation
                # and redaction.
                for active_turn_id in self._turn_ids_by_thread.get(thread.thread_id, []):
                    active_turn = self._turns[active_turn_id]
                    if active_turn.status in _TERMINAL_TURNS:
                        continue
                    self.cancel_turn(
                        active_turn.turn_id,
                        reason="thread_erased",
                        played_ms=0,
                        heard_text="",
                        metadata={"erased_with_thread": True},
                        event_id=(
                            f"thread-erasure:{thread.thread_id}:turn:{active_turn.turn_id}"
                        ),
                        at=occurred_at,
                    )
            self._record(
                "thread.erased" if target is ThreadStatus.ERASED else "thread.transitioned",
                {
                    "thread_id": thread.thread_id,
                    "status": target.value,
                    "metadata": deepcopy(metadata or {}),
                },
                entity_type="thread",
                entity_id=thread.thread_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._threads[thread.thread_id])

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def start_turn(
        self,
        thread_id: str,
        *,
        turn_id: str | None = None,
        source: str = "voice",
        user_text: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        normalized_thread_id = str(thread_id).strip()
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "turn",
                str(turn_id).strip() if turn_id else None,
                event_type="turn.started",
            )
            if duplicate is not None:
                return duplicate
            thread = self._require_thread(normalized_thread_id)
            if thread.status not in {ThreadStatus.OPEN, ThreadStatus.IDLE}:
                raise InvalidTransition(
                    f"cannot start turn on {thread.status} thread {normalized_thread_id!r}"
                )
            resolved_turn_id = str(turn_id).strip() if turn_id else str(uuid4())
            existing = self._turns.get(resolved_turn_id)
            if existing is not None:
                if existing.thread_id != normalized_thread_id:
                    raise DuplicateEntity(
                        f"turn {resolved_turn_id!r} belongs to thread {existing.thread_id!r}"
                    )
                if user_text is not None and existing.user_text != str(user_text):
                    raise DuplicateEntity(
                        f"turn {resolved_turn_id!r} user_text conflicts with its "
                        "existing payload"
                    )
                return self._copy(existing)
            active_turn = next(
                (
                    self._turns[active_turn_id]
                    for active_turn_id in self._turn_ids_by_thread.get(
                        normalized_thread_id,
                        [],
                    )
                    if self._turns[active_turn_id].status not in _TERMINAL_TURNS
                ),
                None,
            )
            if active_turn is not None:
                raise TurnInProgress(
                    normalized_thread_id,
                    active_turn.turn_id,
                )
            occurred_at = self._event_time(at)
            if thread.status is ThreadStatus.IDLE:
                self._record(
                    "thread.transitioned",
                    {
                        "thread_id": thread.thread_id,
                        "status": ThreadStatus.OPEN.value,
                        "metadata": {},
                    },
                    entity_type="thread",
                    entity_id=thread.thread_id,
                    event_id=None,
                    occurred_at=occurred_at,
                )
            turn_sequence = len(self._turn_ids_by_thread.get(normalized_thread_id, [])) + 1
            self._record(
                "turn.started",
                {
                    "turn_id": resolved_turn_id,
                    "thread_id": normalized_thread_id,
                    "turn_sequence": turn_sequence,
                    "source": str(source or "voice"),
                    "user_text": str(user_text or ""),
                    "metadata": deepcopy(metadata or {}),
                    "created_at": occurred_at.isoformat(),
                },
                entity_type="turn",
                entity_id=resolved_turn_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._turns[resolved_turn_id])

    def transition_turn(
        self,
        turn_id: str,
        status: TurnStatus | str,
        *,
        user_text: str | None = None,
        assistant_text: str | None = None,
        heard_text: str | None = None,
        played_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        target = TurnStatus(status)
        if target in _TERMINAL_TURNS:
            raise InvalidTransition(
                f"use the dedicated {target.value} settlement method; "
                "generic transition_turn cannot enter a terminal state"
            )
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "turn",
                str(turn_id).strip(),
                event_type="turn.transitioned",
            )
            if duplicate is not None:
                return duplicate
            turn = self._require_turn(turn_id)
            self._ensure_thread_not_erased(turn.thread_id)
            if target == turn.status:
                if target in _TERMINAL_TURNS:
                    return self._copy(turn)
            elif target not in _TURN_TRANSITIONS[turn.status]:
                raise InvalidTransition(f"turn {turn_id}: {turn.status} -> {target}")
            occurred_at = self._event_time(at)
            payload: dict[str, Any] = {
                "turn_id": turn.turn_id,
                "status": target.value,
                "metadata": deepcopy(metadata or {}),
            }
            if user_text is not None:
                payload["user_text"] = str(user_text)
            if assistant_text is not None:
                payload["assistant_text"] = str(assistant_text)
            if heard_text is not None:
                payload["heard_text"] = str(heard_text)
            if played_ms is not None:
                payload["played_ms"] = max(0, int(played_ms))
            self._record(
                "turn.transitioned",
                payload,
                entity_type="turn",
                entity_id=turn.turn_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._turns[turn.turn_id])

    def commit_turn(
        self,
        turn_id: str,
        *,
        user_text: str | None = None,
        assistant_text: str | None = None,
        heard_text: str | None = None,
        played_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "turn",
                str(turn_id).strip(),
                event_type="turn.committed",
            )
            if duplicate is not None:
                return duplicate
            turn = self._require_turn(turn_id)
            self._ensure_thread_not_erased(turn.thread_id)
            if not self._ensure_turn_settleable(turn, TurnStatus.COMMITTED):
                return self._copy(turn)
            approved_generation_ids = [
                generation_id
                for generation_id in self._generation_ids_by_turn.get(turn.turn_id, [])
                if self._generations[generation_id].status is GenerationStatus.APPROVED
            ]
            if len(approved_generation_ids) > 1:
                raise InvalidTransition(
                    f"turn {turn.turn_id} already has multiple approved generations"
                )
            final_assistant = turn.assistant_text if assistant_text is None else str(assistant_text)
            final_heard = final_assistant if heard_text is None else str(heard_text)
            occurred_at = self._event_time(at)
            payload: dict[str, Any] = {
                "turn_id": turn.turn_id,
                "status": TurnStatus.COMMITTED.value,
                "assistant_text": final_assistant,
                "heard_text": final_heard,
                "playback_disposition": "delivered",
                "metadata": deepcopy(metadata or {}),
            }
            if user_text is not None:
                payload["user_text"] = str(user_text)
            if played_ms is not None:
                payload["played_ms"] = max(0, int(played_ms))
            self._record(
                "turn.committed",
                payload,
                entity_type="turn",
                entity_id=turn.turn_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._turns[turn.turn_id])

    def cancel_turn(
        self,
        turn_id: str,
        *,
        reason: str = "cancelled",
        played_ms: int = 0,
        heard_text: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "turn",
                str(turn_id).strip(),
                event_type="turn.cancelled",
            )
            if duplicate is not None:
                return duplicate
            turn = self._require_turn(turn_id)
            self._ensure_thread_not_erased(turn.thread_id)
            if not self._ensure_turn_settleable(turn, TurnStatus.CANCELLED):
                return self._copy(turn)
            normalized_played_ms = max(0, int(played_ms))
            if normalized_played_ms == 0:
                final_heard = ""
                disposition = "delete_unheard"
            else:
                final_heard = (
                    str(heard_text)
                    if heard_text is not None
                    else turn.heard_text
                )
                disposition = "truncate_played"
            occurred_at = self._event_time(at)
            self._record(
                "turn.cancelled",
                {
                    "turn_id": turn.turn_id,
                    "status": TurnStatus.CANCELLED.value,
                    "assistant_text": final_heard,
                    "heard_text": final_heard,
                    "played_ms": normalized_played_ms,
                    "playback_disposition": disposition,
                    "cancel_reason": str(reason or "cancelled"),
                    "metadata": deepcopy(metadata or {}),
                },
                entity_type="turn",
                entity_id=turn.turn_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._turns[turn.turn_id])

    def fail_turn(
        self,
        turn_id: str,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        return self._terminal_turn_event(
            turn_id,
            status=TurnStatus.FAILED,
            event_type="turn.failed",
            reason_field="failure_reason",
            reason=reason,
            metadata=metadata,
            event_id=event_id,
            at=at,
        )

    def suppress_turn(
        self,
        turn_id: str,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnRecord:
        return self._terminal_turn_event(
            turn_id,
            status=TurnStatus.SUPPRESSED,
            event_type="turn.suppressed",
            reason_field="suppression_reason",
            reason=reason,
            metadata=metadata,
            event_id=event_id,
            at=at,
        )

    # ------------------------------------------------------------------
    # Generation lifecycle
    # ------------------------------------------------------------------

    def start_generation(
        self,
        turn_id: str,
        *,
        provider: str,
        provider_session_id: str | None = None,
        provider_generation_id: str | None = None,
        generation_id: str | None = None,
        response_text: str = "",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnGeneration:
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "generation",
                str(generation_id).strip() if generation_id else None,
                event_type="generation.start",
            )
            if duplicate is not None:
                return duplicate
            turn = self._require_turn(turn_id)
            self._ensure_thread_not_erased(turn.thread_id)
            if turn.status in _TERMINAL_TURNS:
                raise InvalidTransition(
                    f"cannot start generation for terminal turn {turn.turn_id}: {turn.status}"
                )
            resolved_id = str(generation_id).strip() if generation_id else str(uuid4())
            existing = self._generations.get(resolved_id)
            if existing is not None:
                if existing.turn_id != turn.turn_id:
                    raise DuplicateEntity(
                        f"generation {resolved_id!r} belongs to turn {existing.turn_id!r}"
                    )
                return self._copy(existing)
            epoch = len(self._generation_ids_by_turn.get(turn.turn_id, [])) + 1
            occurred_at = self._event_time(at)
            replaced_generation_ids = [
                existing_id
                for existing_id in self._generation_ids_by_turn.get(turn.turn_id, [])
                if self._generations[existing_id].status not in _TERMINAL_GENERATIONS
            ]
            self._record(
                "generation.replaced" if replaced_generation_ids else "generation.started",
                {
                    "generation_id": resolved_id,
                    "turn_id": turn.turn_id,
                    "thread_id": turn.thread_id,
                    "epoch": epoch,
                    "provider": str(provider or "unknown"),
                    "provider_session_id": _clean_identity(provider_session_id),
                    "provider_generation_id": _clean_identity(provider_generation_id),
                    "response_text": str(response_text or ""),
                    "replaced_generation_ids": replaced_generation_ids,
                    "metadata": deepcopy(metadata or {}),
                    "created_at": occurred_at.isoformat(),
                },
                entity_type="generation",
                entity_id=resolved_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._generations[resolved_id])

    def transition_generation(
        self,
        generation_id: str,
        status: GenerationStatus | str,
        *,
        response_text: str | None = None,
        heard_text: str | None = None,
        played_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> TurnGeneration:
        target = GenerationStatus(status)
        if target is GenerationStatus.APPROVED:
            raise InvalidTransition(
                "generation approval is owned by commit_turn; "
                "generic transition_generation cannot approve output"
            )
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "generation",
                str(generation_id).strip(),
                event_type="generation.transitioned",
            )
            if duplicate is not None:
                return duplicate
            generation = self._require_generation(generation_id)
            self._ensure_thread_not_erased(generation.thread_id)
            if target == generation.status:
                if target in _TERMINAL_GENERATIONS:
                    return self._copy(generation)
            elif target not in _GENERATION_TRANSITIONS[generation.status]:
                raise InvalidTransition(
                    f"generation {generation_id}: {generation.status} -> {target}"
                )
            occurred_at = self._event_time(at)
            payload: dict[str, Any] = {
                "generation_id": generation.generation_id,
                "status": target.value,
                "metadata": deepcopy(metadata or {}),
            }
            if response_text is not None:
                payload["response_text"] = str(response_text)
            if heard_text is not None:
                payload["heard_text"] = str(heard_text)
            if played_ms is not None:
                payload["played_ms"] = max(0, int(played_ms))
            self._record(
                "generation.transitioned",
                payload,
                entity_type="generation",
                entity_id=generation.generation_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._generations[generation.generation_id])

    # ------------------------------------------------------------------
    # Migration entry point
    # ------------------------------------------------------------------

    def migrate_legacy_history(self, history_path: str | Path) -> LegacyMigrationResult:
        """Import legacy JSON history without writing to the source file."""

        from askme.conversation.migration import migrate_legacy_history

        return migrate_legacy_history(history_path, self)

    # ------------------------------------------------------------------
    # Internal validation and persistence
    # ------------------------------------------------------------------

    def _terminal_turn_event(
        self,
        turn_id: str,
        *,
        status: TurnStatus,
        event_type: str,
        reason_field: str,
        reason: str,
        metadata: dict[str, Any] | None,
        event_id: str | None,
        at: datetime | None,
    ) -> TurnRecord:
        with self._lock:
            duplicate = self._duplicate_result(
                event_id,
                "turn",
                str(turn_id).strip(),
                event_type=event_type,
            )
            if duplicate is not None:
                return duplicate
            turn = self._require_turn(turn_id)
            self._ensure_thread_not_erased(turn.thread_id)
            if not self._ensure_turn_settleable(turn, status):
                return self._copy(turn)
            occurred_at = self._event_time(at)
            self._record(
                event_type,
                {
                    "turn_id": turn.turn_id,
                    "status": status.value,
                    reason_field: str(reason),
                    "metadata": deepcopy(metadata or {}),
                },
                entity_type="turn",
                entity_id=turn.turn_id,
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return self._copy(self._turns[turn.turn_id])

    def _ensure_turn_settleable(self, turn: TurnRecord, target: TurnStatus) -> bool:
        if turn.status == target:
            return False
        if target not in _TURN_TRANSITIONS[turn.status]:
            raise InvalidTransition(f"turn {turn.turn_id}: {turn.status} -> {target}")
        return True

    def _event_time(self, supplied: datetime | None) -> datetime:
        return _as_utc(supplied if supplied is not None else self._clock())

    def _duplicate_result(
        self,
        event_id: str | None,
        entity_type: str,
        entity_id: str | None = None,
        *,
        event_type: str,
    ) -> Any | None:
        if not event_id:
            return None
        target = self._event_targets.get(str(event_id))
        if target is None:
            return None
        existing_event_type = self._event_types.get(str(event_id), "")
        if _event_operation(existing_event_type) != _event_operation(event_type):
            raise DuplicateEntity(
                f"event {event_id!r} already records {existing_event_type!r}; "
                f"cannot reuse it for {event_type!r}"
            )
        existing_type, existing_entity_id = target
        if existing_type != entity_type:
            raise DuplicateEntity(
                f"event {event_id!r} already targets "
                f"{existing_type} {existing_entity_id!r}"
            )
        normalized_entity_id = str(entity_id).strip() if entity_id else None
        if (
            normalized_entity_id is not None
            and normalized_entity_id != existing_entity_id
        ):
            raise DuplicateEntity(
                f"event {event_id!r} already targets "
                f"{existing_type} {existing_entity_id!r}; "
                f"cannot reuse it for {entity_type} {normalized_entity_id!r}"
            )
        if entity_type == "thread":
            return self._copy(self._threads[existing_entity_id])
        if entity_type == "turn":
            return self._copy(self._turns[existing_entity_id])
        return self._copy(self._generations[existing_entity_id])

    def _record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        entity_type: str,
        entity_id: str,
        event_id: str | None,
        occurred_at: datetime,
    ) -> None:
        resolved_event_id = str(event_id).strip() if event_id else str(uuid4())
        event = {
            "schema_version": _SCHEMA_VERSION,
            "sequence": self._event_sequence + 1,
            "event_id": resolved_event_id,
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "occurred_at": occurred_at.isoformat(),
            "payload": payload,
        }
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            if self._fsync:
                os.fsync(stream.fileno())
        self._apply_event(event)

    def _recover_incomplete_turns(self) -> None:
        """Durably fail orphaned turns left nonterminal by a prior process."""

        incomplete_turn_ids = [
            turn.turn_id
            for turn in self._turns.values()
            if turn.status not in _TERMINAL_TURNS
            and not bool(turn.metadata.get("legacy_import"))
        ]
        for turn_id in incomplete_turn_ids:
            self.fail_turn(
                turn_id,
                reason="process_restart",
                metadata={"recovered_on_startup": True},
                event_id=f"turn-recovery:{turn_id}",
            )

    def _replay(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        expected_sequence = 1
        byte_offset = 0
        for index, raw_line in enumerate(lines):
            line_start = byte_offset
            byte_offset += len(raw_line)
            is_last = index == len(lines) - 1
            if is_last and not raw_line.endswith((b"\n", b"\r")):
                with self.path.open("r+b") as stream:
                    stream.truncate(line_start)
                    stream.flush()
                    if self._fsync:
                        os.fsync(stream.fileno())
                break
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerCorruptionError(
                    f"invalid completed JSONL record at line {index + 1}"
                ) from exc
            if not isinstance(event, dict):
                raise LedgerCorruptionError(f"record {index + 1} is not an object")
            if event.get("schema_version") != _SCHEMA_VERSION:
                raise LedgerCorruptionError(
                    f"record {index + 1} has unsupported schema_version "
                    f"{event.get('schema_version')!r}"
                )
            sequence = event.get("sequence")
            if sequence != expected_sequence:
                raise LedgerCorruptionError(
                    f"record {index + 1} sequence {sequence!r}; expected {expected_sequence}"
                )
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in self._event_targets:
                raise LedgerCorruptionError(f"duplicate or empty event_id at line {index + 1}")
            try:
                self._apply_event(event)
            except (KeyError, TypeError, ValueError) as exc:
                raise LedgerCorruptionError(
                    f"invalid event payload at line {index + 1}: {exc}"
                ) from exc
            expected_sequence += 1

    def _apply_event(self, event: dict[str, Any]) -> None:
        event_type = str(event["event_type"])
        payload = event["payload"]
        occurred_at = _as_utc(str(event["occurred_at"]))
        if event_type == "thread.created":
            self._apply_thread_created(payload)
        elif event_type in {"thread.transitioned", "thread.erased"}:
            thread = self._require_thread(payload["thread_id"])
            status = ThreadStatus(payload["status"])
            if status != thread.status and status not in _THREAD_TRANSITIONS[thread.status]:
                raise InvalidTransition(
                    f"thread {thread.thread_id}: {thread.status} -> {status}"
                )
            metadata = deepcopy(thread.metadata)
            metadata.update(deepcopy(payload.get("metadata") or {}))
            self._threads[thread.thread_id] = replace(
                thread,
                status=status,
                metadata=metadata,
                updated_at=occurred_at,
                last_activity_at=occurred_at,
                closed_at=(
                    occurred_at
                    if status in {ThreadStatus.CLOSED, ThreadStatus.EXPIRED, ThreadStatus.ERASED}
                    else None
                ),
            )
            if status is ThreadStatus.ERASED:
                self._redact_thread(thread.thread_id, occurred_at)
        elif event_type == "turn.started":
            self._apply_turn_started(payload)
            self._touch_thread(payload["thread_id"], occurred_at)
        elif event_type.startswith("turn."):
            self._apply_turn_changed(payload, event_type=event_type, occurred_at=occurred_at)
            if event_type == "turn.committed":
                turn = self._require_turn(payload["turn_id"])
                self._committed_turn_events.append(
                    CommittedTurnEvent(
                        event_id=str(event["event_id"]),
                        sequence=int(event["sequence"]),
                        occurred_at=occurred_at,
                        thread_id=turn.thread_id,
                        turn_id=turn.turn_id,
                        turn_sequence=turn.sequence,
                        source=turn.source,
                        user_text=turn.user_text,
                        assistant_text=turn.assistant_text,
                        heard_text=turn.heard_text,
                        played_ms=turn.played_ms,
                        playback_disposition=turn.playback_disposition,
                        metadata=deepcopy(turn.metadata),
                    )
                )
        elif event_type in {"generation.started", "generation.replaced"}:
            self._apply_generation_started(payload, occurred_at)
        elif event_type == "generation.transitioned":
            self._apply_generation_changed(payload, occurred_at)
        else:
            raise ValueError(f"unknown event_type {event_type!r}")
        self._event_sequence = int(event["sequence"])
        self._event_targets[str(event["event_id"])] = (
            str(event["entity_type"]),
            str(event["entity_id"]),
        )
        self._event_types[str(event["event_id"])] = event_type

    def _apply_thread_created(self, payload: dict[str, Any]) -> None:
        created_at = _as_utc(payload["created_at"])
        thread_id = str(payload["thread_id"])
        if thread_id in self._threads:
            raise DuplicateEntity(f"duplicate thread creation for {thread_id!r}")
        timezone = str(payload.get("timezone") or "UTC")
        self._threads[thread_id] = ConversationThread(
            thread_id=thread_id,
            channel=str(payload["channel"]),
            person_id=_clean_identity(payload.get("person_id")),
            operator_id=_clean_identity(payload.get("operator_id")),
            robot_id=_clean_identity(payload.get("robot_id")),
            site_id=_clean_identity(payload.get("site_id")),
            status=ThreadStatus.OPEN,
            metadata=deepcopy(payload.get("metadata") or {}),
            created_at=created_at,
            updated_at=created_at,
            last_activity_at=created_at,
            timezone=timezone,
            local_day=_local_day(created_at, timezone),
        )
        self._turn_ids_by_thread.setdefault(thread_id, [])

    def _apply_turn_started(self, payload: dict[str, Any]) -> None:
        created_at = _as_utc(payload["created_at"])
        turn_id = str(payload["turn_id"])
        thread_id = str(payload["thread_id"])
        if turn_id in self._turns:
            raise DuplicateEntity(f"duplicate turn creation for {turn_id!r}")
        thread = self._require_thread(thread_id)
        if thread.status not in {ThreadStatus.OPEN, ThreadStatus.IDLE}:
            raise InvalidTransition(
                f"cannot start turn on {thread.status} thread {thread_id!r}"
            )
        expected_sequence = len(self._turn_ids_by_thread.get(thread_id, [])) + 1
        if int(payload["turn_sequence"]) != expected_sequence:
            raise ValueError(
                f"turn {turn_id!r} sequence {payload['turn_sequence']!r}; "
                f"expected {expected_sequence}"
            )
        self._turns[turn_id] = TurnRecord(
            turn_id=turn_id,
            thread_id=thread_id,
            sequence=int(payload["turn_sequence"]),
            source=str(payload["source"]),
            status=TurnStatus.STARTED,
            user_text=str(payload.get("user_text") or ""),
            assistant_text="",
            heard_text="",
            played_ms=0,
            playback_disposition=None,
            cancel_reason=None,
            failure_reason=None,
            suppression_reason=None,
            generation_ids=(),
            metadata=deepcopy(payload.get("metadata") or {}),
            created_at=created_at,
            updated_at=created_at,
        )
        self._turn_ids_by_thread.setdefault(thread_id, []).append(turn_id)
        self._generation_ids_by_turn.setdefault(turn_id, [])

    def _apply_turn_changed(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
        occurred_at: datetime,
    ) -> None:
        turn = self._require_turn(payload["turn_id"])
        self._ensure_thread_not_erased(turn.thread_id)
        status = TurnStatus(payload["status"])
        if status != turn.status and status not in _TURN_TRANSITIONS[turn.status]:
            raise InvalidTransition(f"turn {turn.turn_id}: {turn.status} -> {status}")
        if status == turn.status and status in _TERMINAL_TURNS:
            raise InvalidTransition(f"turn {turn.turn_id} is already terminal: {status}")
        metadata = deepcopy(turn.metadata)
        metadata.update(deepcopy(payload.get("metadata") or {}))
        changed = replace(
            turn,
            status=status,
            user_text=str(payload.get("user_text", turn.user_text)),
            assistant_text=str(payload.get("assistant_text", turn.assistant_text)),
            heard_text=str(payload.get("heard_text", turn.heard_text)),
            played_ms=max(0, int(payload.get("played_ms", turn.played_ms))),
            playback_disposition=payload.get(
                "playback_disposition", turn.playback_disposition
            ),
            cancel_reason=payload.get("cancel_reason", turn.cancel_reason),
            failure_reason=payload.get("failure_reason", turn.failure_reason),
            suppression_reason=payload.get("suppression_reason", turn.suppression_reason),
            metadata=metadata,
            updated_at=occurred_at,
            committed_at=(occurred_at if status is TurnStatus.COMMITTED else turn.committed_at),
            settled_at=(occurred_at if status in _TERMINAL_TURNS else turn.settled_at),
        )
        self._turns[turn.turn_id] = changed
        if event_type == "turn.committed":
            generation_ids = self._generation_ids_by_turn.get(turn.turn_id, [])
            approved_ids = [
                generation_id
                for generation_id in generation_ids
                if self._generations[generation_id].status is GenerationStatus.APPROVED
            ]
            if len(approved_ids) > 1:
                raise InvalidTransition(
                    f"turn {turn.turn_id} has multiple approved generations"
                )
            active_ids = [
                generation_id
                for generation_id in generation_ids
                if self._generations[generation_id].status not in _TERMINAL_GENERATIONS
            ]
            winner_id = approved_ids[0] if approved_ids else (
                max(active_ids, key=lambda item: self._generations[item].epoch)
                if active_ids
                else None
            )
            for generation_id in generation_ids:
                generation = self._generations[generation_id]
                if generation_id == winner_id:
                    self._generations[generation_id] = replace(
                        generation,
                        status=GenerationStatus.APPROVED,
                        response_text=changed.assistant_text,
                        heard_text=changed.heard_text,
                        played_ms=changed.played_ms,
                        updated_at=occurred_at,
                        settled_at=occurred_at,
                    )
                elif generation.status not in _TERMINAL_GENERATIONS:
                    self._generations[generation_id] = replace(
                        generation,
                        status=GenerationStatus.ROLLED_BACK,
                        updated_at=occurred_at,
                        settled_at=occurred_at,
                    )
            generation_status = None
        elif event_type == "turn.cancelled":
            generation_status = (
                GenerationStatus.TRUNCATED
                if changed.played_ms > 0
                else GenerationStatus.DISCARDED
            )
        elif event_type == "turn.failed":
            generation_status = GenerationStatus.PROVIDER_FAILED
        elif event_type == "turn.suppressed":
            generation_status = GenerationStatus.DISCARDED
        else:
            generation_status = None
        if generation_status is not None:
            for generation_id in self._generation_ids_by_turn.get(turn.turn_id, []):
                generation = self._generations[generation_id]
                if generation.status in _TERMINAL_GENERATIONS:
                    continue
                self._generations[generation_id] = replace(
                    generation,
                    status=generation_status,
                    response_text=(
                        changed.assistant_text
                        if generation_status is GenerationStatus.APPROVED
                        else generation.response_text
                    ),
                    heard_text=changed.heard_text,
                    played_ms=changed.played_ms,
                    updated_at=occurred_at,
                    settled_at=occurred_at,
                )
        self._touch_thread(turn.thread_id, occurred_at)

    def _apply_generation_started(
        self,
        payload: dict[str, Any],
        occurred_at: datetime,
    ) -> None:
        generation_id = str(payload["generation_id"])
        turn_id = str(payload["turn_id"])
        thread_id = str(payload["thread_id"])
        created_at = _as_utc(payload["created_at"])
        if generation_id in self._generations:
            raise DuplicateEntity(f"duplicate generation creation for {generation_id!r}")
        turn = self._require_turn(turn_id)
        self._ensure_thread_not_erased(turn.thread_id)
        if turn.thread_id != thread_id:
            raise ValueError(
                f"generation {generation_id!r} thread {thread_id!r} does not match "
                f"turn thread {turn.thread_id!r}"
            )
        if turn.status in _TERMINAL_TURNS:
            raise InvalidTransition(
                f"cannot start generation for terminal turn {turn_id!r}: {turn.status}"
            )
        expected_epoch = len(self._generation_ids_by_turn.get(turn_id, [])) + 1
        if int(payload["epoch"]) != expected_epoch:
            raise ValueError(
                f"generation {generation_id!r} epoch {payload['epoch']!r}; "
                f"expected {expected_epoch}"
            )
        replaced_ids = [str(item) for item in payload.get("replaced_generation_ids") or []]
        active_ids = {
            existing_id
            for existing_id in self._generation_ids_by_turn.get(turn_id, [])
            if self._generations[existing_id].status not in _TERMINAL_GENERATIONS
        }
        if set(replaced_ids) != active_ids:
            raise ValueError(
                f"generation replacement mismatch: recorded {replaced_ids!r}; "
                f"active {sorted(active_ids)!r}"
            )
        for previous_id in replaced_ids:
            previous = self._generations[previous_id]
            self._generations[previous_id] = replace(
                previous,
                status=GenerationStatus.ROLLED_BACK,
                updated_at=occurred_at,
                settled_at=occurred_at,
            )
        self._generations[generation_id] = TurnGeneration(
            generation_id=generation_id,
            turn_id=turn_id,
            thread_id=thread_id,
            epoch=int(payload["epoch"]),
            provider=str(payload["provider"]),
            provider_session_id=_clean_identity(payload.get("provider_session_id")),
            provider_generation_id=_clean_identity(payload.get("provider_generation_id")),
            status=GenerationStatus.STARTED,
            response_text=str(payload.get("response_text") or ""),
            heard_text="",
            played_ms=0,
            metadata=deepcopy(payload.get("metadata") or {}),
            created_at=created_at,
            updated_at=created_at,
        )
        self._generation_ids_by_turn.setdefault(turn_id, []).append(generation_id)
        next_turn_status = (
            TurnStatus.GENERATING
            if turn.status
            in {TurnStatus.STARTED, TurnStatus.LISTENING, TurnStatus.TRANSCRIBED, TurnStatus.ROUTED}
            else turn.status
        )
        self._turns[turn_id] = replace(
            turn,
            status=next_turn_status,
            generation_ids=(*turn.generation_ids, generation_id),
            updated_at=occurred_at,
        )
        self._touch_thread(thread_id, occurred_at)

    def _apply_generation_changed(
        self,
        payload: dict[str, Any],
        occurred_at: datetime,
    ) -> None:
        generation = self._require_generation(payload["generation_id"])
        self._ensure_thread_not_erased(generation.thread_id)
        status = GenerationStatus(payload["status"])
        if status != generation.status and status not in _GENERATION_TRANSITIONS[generation.status]:
            raise InvalidTransition(
                f"generation {generation.generation_id}: {generation.status} -> {status}"
            )
        if status == generation.status and status in _TERMINAL_GENERATIONS:
            raise InvalidTransition(
                f"generation {generation.generation_id} is already terminal: {status}"
            )
        metadata = deepcopy(generation.metadata)
        metadata.update(deepcopy(payload.get("metadata") or {}))
        self._generations[generation.generation_id] = replace(
            generation,
            status=status,
            response_text=str(payload.get("response_text", generation.response_text)),
            heard_text=str(payload.get("heard_text", generation.heard_text)),
            played_ms=max(0, int(payload.get("played_ms", generation.played_ms))),
            metadata=metadata,
            updated_at=occurred_at,
            settled_at=(occurred_at if status in _TERMINAL_GENERATIONS else None),
        )
        self._touch_thread(generation.thread_id, occurred_at)

    def _redact_thread(self, thread_id: str, occurred_at: datetime) -> None:
        """Materialize an erasure tombstone without exposing prior content."""

        thread = self._require_thread(thread_id)
        self._threads[thread_id] = replace(
            thread,
            person_id=None,
            operator_id=None,
            metadata={},
            updated_at=occurred_at,
            last_activity_at=occurred_at,
            closed_at=occurred_at,
        )
        for turn_id in self._turn_ids_by_thread.get(thread_id, []):
            turn = self._turns[turn_id]
            self._turns[turn_id] = replace(
                turn,
                user_text="",
                assistant_text="",
                heard_text="",
                cancel_reason=None,
                failure_reason=None,
                suppression_reason=None,
                metadata={},
                updated_at=occurred_at,
            )
            for generation_id in self._generation_ids_by_turn.get(turn_id, []):
                generation = self._generations[generation_id]
                self._generations[generation_id] = replace(
                    generation,
                    provider_session_id=None,
                    provider_generation_id=None,
                    response_text="",
                    heard_text="",
                    metadata={},
                    updated_at=occurred_at,
                )

    def _touch_thread(self, thread_id: str, occurred_at: datetime) -> None:
        thread = self._require_thread(thread_id)
        self._threads[thread.thread_id] = replace(
            thread,
            updated_at=occurred_at,
            last_activity_at=occurred_at,
        )

    def _require_thread(self, thread_id: str) -> ConversationThread:
        try:
            return self._threads[str(thread_id)]
        except KeyError as exc:
            raise EntityNotFound(f"unknown thread {thread_id!r}") from exc

    def _ensure_thread_not_erased(self, thread_id: str) -> ConversationThread:
        thread = self._require_thread(thread_id)
        if thread.status is ThreadStatus.ERASED:
            raise InvalidTransition(f"thread {thread.thread_id!r} is erased")
        return thread

    def _require_turn(self, turn_id: str) -> TurnRecord:
        try:
            return self._turns[str(turn_id)]
        except KeyError as exc:
            raise EntityNotFound(f"unknown turn {turn_id!r}") from exc

    def _require_generation(self, generation_id: str) -> TurnGeneration:
        try:
            return self._generations[str(generation_id)]
        except KeyError as exc:
            raise EntityNotFound(f"unknown generation {generation_id!r}") from exc

    @staticmethod
    @overload
    def _copy(value: None) -> None: ...

    @staticmethod
    @overload
    def _copy(value: _T) -> _T: ...

    @staticmethod
    def _copy(value: _T | None) -> _T | None:
        return deepcopy(value)
