"""Privacy-safe observational evidence for realtime voice turns.

Conversation Core owns Thread, Turn, and Generation truth.  This module only
records timestamped evidence supplied by those systems and their media peers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeGuard

MAX_ATTRIBUTE_COUNT = 32
MAX_ATTRIBUTE_STRING_LENGTH = 256
MAX_IDENTITY_LENGTH = 256
DEFAULT_EXPORT_QUEUE_LIMIT = 256
DEFAULT_EMERGENCY_EVENT_LIMIT = 64
MAX_QUERY_LIMIT = 1_000
_SAFE_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9._~:/@+=-]{1,256}")
_STABLE_SCOPE_FIELDS = (
    "thread_id",
    "turn_id",
    "trace_id",
)
_EVENT_LOCAL_SCOPE_FIELDS = (
    "generation_id",
    "provider_session_id",
)
_OPTIONAL_SCOPE_FIELDS = (*_STABLE_SCOPE_FIELDS, *_EVENT_LOCAL_SCOPE_FIELDS)
VOICE_TIMELINE_ATTRIBUTE_ALLOWLIST = frozenset(
    {
        "attempt",
        "audio_class",
        "audio_segment_id",
        "buffered_ms",
        "byte_count",
        "cache_hit",
        "cancelled",
        "channels",
        "character_count",
        "clause_index",
        "clock_id",
        "codec",
        "confidence",
        "degraded",
        "device_id",
        "duration_ms",
        "endpoint_reason",
        "epoch",
        "error_type",
        "evidence_kind",
        "fallback_reason",
        "frame_samples",
        "instrumented",
        "interrupt_reason",
        "language",
        "latency_ms",
        "locale",
        "media_transport",
        "mode",
        "model",
        "offset_ms",
        "outcome",
        "peak",
        "played_ms",
        "provider",
        "provenance_verified",
        "queue_depth",
        "reason_code",
        "rms",
        "route",
        "sample_rate_hz",
        "sequence",
        "source",
        "status",
        "token_count",
        "transport",
    }
)


class TimelineConflict(ValueError):
    """An event or scope identity was reused with incompatible evidence."""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        super().__init__(f"timeline identity conflict for event_id={event_id!r}")


class TimelineStoreError(RuntimeError):
    """A completed local timeline record could not be safely replayed."""


class TimelineClockError(ValueError):
    """An injected timeline clock returned a non-finite numeric value."""

    def __init__(self, clock_name: str) -> None:
        self.clock_name = clock_name
        super().__init__(f"timeline {clock_name} clock must return a finite number")


class VoiceTimelineStage(StrEnum):
    """Canonical milestones that may be observed during a voice turn."""

    LISTEN_STARTED = "listen_started"
    FIRST_AUDIO_FRAME = "first_audio_frame"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    ENDPOINT_COMMITTED = "endpoint_committed"
    ASR_FINAL = "asr_final"
    TURN_CORRELATED = "turn_correlated"
    TURN_ADMITTED = "turn_admitted"
    LLM_REQUESTED = "llm_requested"
    FIRST_LLM_PAYLOAD = "first_llm_payload"
    FIRST_SEMANTIC = "first_semantic"
    FIRST_CLAUSE = "first_clause"
    TTS_FIRST_PCM = "tts_first_pcm"
    SPEAKER_RENDER_STARTED = "speaker_render_started"
    SPEAKER_PHYSICAL_STARTED = "speaker_physical_started"
    INTERRUPT_DETECTED = "interrupt_detected"
    INTERRUPT_CONFIRMED = "interrupt_confirmed"
    INTERRUPT_DISMISSED = "interrupt_dismissed"
    SPEAKER_RENDER_STOPPED = "speaker_render_stopped"
    SPEAKER_PHYSICAL_STOPPED = "speaker_physical_stopped"
    UPSTREAM_CLOSED = "upstream_closed"
    FALLBACK_SELECTED = "fallback_selected"
    TURN_FINISHED = "turn_finished"
    ERROR = "error"


class TimelineRecordStatus(StrEnum):
    """Caller-visible disposition of one record attempt."""

    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    DROPPED_PRIVACY = "dropped_privacy"
    DEGRADED_PERSISTENCE = "degraded_persistence"


@dataclass(frozen=True, slots=True)
class VoiceTimelineScope:
    """Identifiers known for one observed voice-turn milestone.

    Thread, Turn, and Trace are stable late bindings for a voice turn.
    Generation and provider session identify only the event that observed them.
    """

    voice_turn_id: str
    thread_id: str | None = None
    turn_id: str | None = None
    generation_id: str | None = None
    provider_session_id: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "voice_turn_id": self.voice_turn_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "provider_session_id": self.provider_session_id,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True, slots=True)
class VoiceTimelineEventInput:
    """One caller-supplied observation before sequence and time assignment."""

    event_id: str
    stage: VoiceTimelineStage
    scope: VoiceTimelineScope
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimelineQuery:
    """Bounded filters for timeline snapshots."""

    voice_turn_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    generation_id: str | None = None
    provider_session_id: str | None = None
    trace_id: str | None = None
    stage: VoiceTimelineStage | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("timeline query limit must be an integer")
        if not 1 <= self.limit <= MAX_QUERY_LIMIT:
            raise ValueError(f"timeline query limit must be between 1 and {MAX_QUERY_LIMIT}")
        if self.stage is not None:
            object.__setattr__(self, "stage", VoiceTimelineStage(self.stage))


@dataclass(frozen=True, slots=True)
class VoiceTimelineRecord:
    """A locally ordered and timestamped timeline event."""

    sequence: int
    event_id: str
    stage: VoiceTimelineStage
    scope: VoiceTimelineScope
    attributes: Mapping[str, object]
    recorded_at_epoch_s: float
    recorded_at_monotonic_s: float
    payload_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "stage": self.stage.value,
            "scope": self.scope.to_dict(),
            "attributes": dict(sorted(self.attributes.items())),
            "recorded_at_epoch_s": self.recorded_at_epoch_s,
            "recorded_at_monotonic_s": self.recorded_at_monotonic_s,
            "payload_hash": self.payload_hash,
        }


class VoiceTimelineExporter(Protocol):
    """Non-blocking true-external adapter used by the export worker."""

    def offer(self, record: VoiceTimelineRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class TimelineRecordReceipt:
    """Result of one record attempt."""

    event_id: str
    status: TimelineRecordStatus
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status.value,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class VoiceTimelineSnapshot:
    """Immutable query result plus health counters."""

    events: tuple[VoiceTimelineRecord, ...]
    export_error_count: int = 0
    last_export_error_type: str | None = None
    persistence_degraded: bool = False
    persistence_error_count: int = 0
    last_persistence_error_type: str | None = None
    emergency_event_count: int = 0
    emergency_dropped_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "export_error_count": self.export_error_count,
            "last_export_error_type": self.last_export_error_type,
            "persistence_degraded": self.persistence_degraded,
            "persistence_error_count": self.persistence_error_count,
            "last_persistence_error_type": self.last_persistence_error_type,
            "emergency_event_count": self.emergency_event_count,
            "emergency_dropped_count": self.emergency_dropped_count,
        }


class MemoryVoiceTimelineStore:
    """Process-local store adapter suitable for tests and ephemeral runtimes."""

    def __init__(self) -> None:
        self._records: list[VoiceTimelineRecord] = []

    def load(self) -> tuple[VoiceTimelineRecord, ...]:
        return tuple(self._records)

    def append(self, record: VoiceTimelineRecord) -> None:
        self._records.append(record)


class JsonlVoiceTimelineStore:
    """Durable append-only JSONL adapter with strict deterministic replay."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._last_sequence = 0

    def load(self) -> tuple[VoiceTimelineRecord, ...]:
        if not self.path.exists():
            self._last_sequence = 0
            return ()
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise TimelineStoreError("timeline store could not be read") from exc

        records: list[VoiceTimelineRecord] = []
        bindings: dict[str, VoiceTimelineScope] = {}
        seen_event_ids: set[str] = set()
        cursor = 0
        while cursor < len(raw):
            newline = raw.find(b"\n", cursor)
            if newline < 0:
                tail = raw[cursor:]
                parsed = _decode_json(tail)
                if parsed is None:
                    self._truncate(cursor)
                    break
                record = _record_from_json(parsed)
                self._accept_replayed_record(
                    record,
                    records=records,
                    bindings=bindings,
                    seen_event_ids=seen_event_ids,
                )
                self._append_final_newline()
                cursor = len(raw)
                break

            raw_line = raw[cursor:newline]
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            parsed = _decode_json(raw_line)
            if parsed is None:
                raise TimelineStoreError("timeline store contains completed corruption")
            record = _record_from_json(parsed)
            self._accept_replayed_record(
                record,
                records=records,
                bindings=bindings,
                seen_event_ids=seen_event_ids,
            )
            cursor = newline + 1

        self._last_sequence = records[-1].sequence if records else 0
        return tuple(records)

    def append(self, record: VoiceTimelineRecord) -> None:
        if record.sequence != self._last_sequence + 1:
            raise TimelineStoreError("timeline append sequence is not contiguous")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **record.to_dict(),
        }
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab", buffering=0) as stream:
            written = 0
            while written < len(encoded):
                count = stream.write(encoded[written:])
                if not count:
                    raise OSError("timeline append made no progress")
                written += count
            os.fsync(stream.fileno())
        self._last_sequence = record.sequence

    def _accept_replayed_record(
        self,
        record: VoiceTimelineRecord,
        *,
        records: list[VoiceTimelineRecord],
        bindings: dict[str, VoiceTimelineScope],
        seen_event_ids: set[str],
    ) -> None:
        if record.sequence != len(records) + 1:
            raise TimelineStoreError("timeline store sequence is not contiguous")
        if record.event_id in seen_event_ids:
            raise TimelineStoreError("timeline store repeats an event identity")
        current = bindings.get(
            record.scope.voice_turn_id,
            VoiceTimelineScope(voice_turn_id=record.scope.voice_turn_id),
        )
        try:
            bindings[record.scope.voice_turn_id] = _merge_stable_scope(
                current,
                record.scope,
                event_id=record.event_id,
            )
        except TimelineConflict as exc:
            raise TimelineStoreError("timeline store contains conflicting scope bindings") from exc
        seen_event_ids.add(record.event_id)
        records.append(record)

    def _truncate(self, size: int) -> None:
        try:
            with self.path.open("r+b", buffering=0) as stream:
                stream.truncate(size)
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TimelineStoreError("incomplete timeline tail could not be repaired") from exc

    def _append_final_newline(self) -> None:
        try:
            with self.path.open("ab", buffering=0) as stream:
                stream.write(b"\n")
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TimelineStoreError("timeline final newline could not be repaired") from exc


class VoiceTurnTimeline:
    """Deep timeline module exposed through only ``record`` and ``snapshot``."""

    def __init__(
        self,
        store: MemoryVoiceTimelineStore | JsonlVoiceTimelineStore | None = None,
        *,
        epoch_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        exporter: VoiceTimelineExporter | None = None,
        export_queue_limit: int = DEFAULT_EXPORT_QUEUE_LIMIT,
        emergency_limit: int = DEFAULT_EMERGENCY_EVENT_LIMIT,
    ) -> None:
        self._store = store or MemoryVoiceTimelineStore()
        self._epoch_clock = epoch_clock
        self._monotonic_clock = monotonic_clock
        self._exporter = exporter
        if export_queue_limit < 1:
            raise ValueError("export_queue_limit must be at least 1")
        self._export_queue_limit = export_queue_limit
        self._export_queue: deque[VoiceTimelineRecord] = deque()
        self._export_queue_lock = threading.Lock()
        self._export_worker_active = False
        if emergency_limit < 1:
            raise ValueError("emergency_limit must be at least 1")
        self._emergency_limit = emergency_limit
        self._emergency_records: deque[VoiceTimelineRecord] = deque()
        self._emergency_dropped_count = 0
        self._persistence_degraded = False
        self._persistence_error_count = 0
        self._last_persistence_error_type: str | None = None
        self._lock = threading.RLock()
        self._records = list(self._store.load())
        self._next_sequence = max((record.sequence for record in self._records), default=0) + 1
        self._records_by_event_id = {record.event_id: record for record in self._records}
        self._stable_scope_by_voice_turn: dict[str, VoiceTimelineScope] = {}
        for record in sorted(self._records, key=lambda item: item.sequence):
            current = self._stable_scope_by_voice_turn.get(
                record.scope.voice_turn_id,
                VoiceTimelineScope(voice_turn_id=record.scope.voice_turn_id),
            )
            self._stable_scope_by_voice_turn[
                record.scope.voice_turn_id
            ] = _merge_stable_scope(
                current,
                record.scope,
                event_id=record.event_id,
            )
        self._export_error_count = 0
        self._last_export_error_type: str | None = None

    def record(self, event: VoiceTimelineEventInput) -> TimelineRecordReceipt:
        """Record one observation and return its local disposition."""

        if not _event_identity_is_safe(event) or not _privacy_attributes_are_safe(
            event.attributes
        ):
            return TimelineRecordReceipt(
                event_id=event.event_id,
                status=TimelineRecordStatus.DROPPED_PRIVACY,
            )
        with self._lock:
            payload_hash = _event_payload_hash(event)
            existing = self._records_by_event_id.get(event.event_id)
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    raise TimelineConflict(event.event_id)
                return TimelineRecordReceipt(
                    event_id=event.event_id,
                    status=TimelineRecordStatus.DUPLICATE,
                    sequence=existing.sequence,
                )
            current_scope = self._stable_scope_by_voice_turn.get(
                event.scope.voice_turn_id,
                VoiceTimelineScope(voice_turn_id=event.scope.voice_turn_id),
            )
            stable_scope = _merge_stable_scope(
                current_scope,
                event.scope,
                event_id=event.event_id,
            )
            epoch_value = self._epoch_clock()
            if not _is_finite_number(epoch_value):
                raise TimelineClockError("epoch")
            monotonic_value = self._monotonic_clock()
            if not _is_finite_number(monotonic_value):
                raise TimelineClockError("monotonic")
            record = VoiceTimelineRecord(
                sequence=self._next_sequence,
                event_id=event.event_id,
                stage=VoiceTimelineStage(event.stage),
                scope=event.scope,
                attributes=dict(event.attributes),
                recorded_at_epoch_s=float(epoch_value),
                recorded_at_monotonic_s=float(monotonic_value),
                payload_hash=payload_hash,
            )
            status = TimelineRecordStatus.RECORDED
            if not self._persistence_degraded:
                try:
                    self._store.append(record)
                except Exception as exc:
                    self._persistence_degraded = True
                    self._persistence_error_count += 1
                    self._last_persistence_error_type = type(exc).__name__
            if self._persistence_degraded:
                status = TimelineRecordStatus.DEGRADED_PERSISTENCE
                self._remember_emergency(record)
            else:
                self._records.append(record)
            self._records_by_event_id[record.event_id] = record
            self._stable_scope_by_voice_turn[record.scope.voice_turn_id] = stable_scope
            self._next_sequence += 1
            receipt = TimelineRecordReceipt(
                event_id=event.event_id,
                status=status,
                sequence=record.sequence,
            )
            self._enqueue_export(
                replace(record, scope=_effective_scope(stable_scope, record.scope))
            )
        return receipt

    def snapshot(self, query: TimelineQuery | None = None) -> VoiceTimelineSnapshot:
        """Return a deterministic, sequence-ordered view of matching evidence."""

        selected_query = query or TimelineQuery()
        with self._lock:
            local_records = sorted(
                [*self._records, *self._emergency_records],
                key=lambda record: record.sequence,
            )
            effective_records = [
                replace(
                    record,
                    scope=_effective_scope(
                        self._stable_scope_by_voice_turn[record.scope.voice_turn_id],
                        record.scope,
                    ),
                )
                for record in local_records
            ]
            records = [
                record
                for record in effective_records
                if _record_matches(record, selected_query)
            ][: selected_query.limit]
            export_error_count = self._export_error_count
            last_export_error_type = self._last_export_error_type
            persistence_degraded = self._persistence_degraded
            persistence_error_count = self._persistence_error_count
            last_persistence_error_type = self._last_persistence_error_type
            emergency_event_count = len(self._emergency_records)
            emergency_dropped_count = self._emergency_dropped_count
        return VoiceTimelineSnapshot(
            events=tuple(records),
            export_error_count=export_error_count,
            last_export_error_type=last_export_error_type,
            persistence_degraded=persistence_degraded,
            persistence_error_count=persistence_error_count,
            last_persistence_error_type=last_persistence_error_type,
            emergency_event_count=emergency_event_count,
            emergency_dropped_count=emergency_dropped_count,
        )

    def _remember_emergency(self, record: VoiceTimelineRecord) -> None:
        if len(self._emergency_records) >= self._emergency_limit:
            evicted = self._emergency_records.popleft()
            if self._records_by_event_id.get(evicted.event_id) is evicted:
                self._records_by_event_id.pop(evicted.event_id, None)
            self._emergency_dropped_count += 1
        self._emergency_records.append(record)

    def _enqueue_export(self, record: VoiceTimelineRecord) -> None:
        if self._exporter is None:
            return
        should_start_worker = False
        with self._export_queue_lock:
            if len(self._export_queue) >= self._export_queue_limit:
                self._export_error_count += 1
                self._last_export_error_type = "TimelineExportQueueFull"
                return
            self._export_queue.append(record)
            if not self._export_worker_active:
                self._export_worker_active = True
                should_start_worker = True
        if not should_start_worker:
            return
        try:
            threading.Thread(
                target=self._drain_export_queue,
                name="voice-timeline-export",
                daemon=True,
            ).start()
        except Exception as exc:  # Thread startup cannot fail local recording.
            with self._export_queue_lock:
                self._export_worker_active = False
                self._export_queue.clear()
            self._export_error_count += 1
            self._last_export_error_type = type(exc).__name__

    def _drain_export_queue(self) -> None:
        exporter = self._exporter
        if exporter is None:
            return
        while True:
            with self._export_queue_lock:
                if not self._export_queue:
                    self._export_worker_active = False
                    return
                record = self._export_queue.popleft()
            try:
                exporter.offer(record)
            except Exception as exc:  # A true-external adapter cannot fail local recording.
                with self._lock:
                    self._export_error_count += 1
                    self._last_export_error_type = type(exc).__name__


def _record_matches(record: VoiceTimelineRecord, query: TimelineQuery) -> bool:
    if query.voice_turn_id is not None and record.scope.voice_turn_id != query.voice_turn_id:
        return False
    for field_name in _OPTIONAL_SCOPE_FIELDS:
        expected = getattr(query, field_name)
        if expected is not None and getattr(record.scope, field_name) != expected:
            return False
    if query.stage is not None and record.stage is not VoiceTimelineStage(query.stage):
        return False
    return True


def _merge_stable_scope(
    current: VoiceTimelineScope,
    incoming: VoiceTimelineScope,
    *,
    event_id: str,
) -> VoiceTimelineScope:
    if current.voice_turn_id != incoming.voice_turn_id:
        raise TimelineConflict(event_id)
    merged: dict[str, str | None] = {}
    for field_name in _STABLE_SCOPE_FIELDS:
        current_value = getattr(current, field_name)
        incoming_value = getattr(incoming, field_name)
        if current_value is not None and incoming_value is not None:
            if current_value != incoming_value:
                raise TimelineConflict(event_id)
        merged[field_name] = current_value or incoming_value
    return VoiceTimelineScope(
        voice_turn_id=current.voice_turn_id,
        thread_id=merged["thread_id"],
        turn_id=merged["turn_id"],
        trace_id=merged["trace_id"],
    )


def _effective_scope(
    stable: VoiceTimelineScope,
    observed: VoiceTimelineScope,
) -> VoiceTimelineScope:
    return VoiceTimelineScope(
        voice_turn_id=stable.voice_turn_id,
        thread_id=stable.thread_id,
        turn_id=stable.turn_id,
        generation_id=observed.generation_id,
        provider_session_id=observed.provider_session_id,
        trace_id=stable.trace_id,
    )


def _event_payload_hash(event: VoiceTimelineEventInput) -> str:
    canonical = {
        "attributes": dict(event.attributes),
        "event_id": event.event_id,
        "scope": event.scope.to_dict(),
        "stage": VoiceTimelineStage(event.stage).value,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_json(raw: bytes) -> dict[str, object] | None:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _record_from_json(value: dict[str, object]) -> VoiceTimelineRecord:
    expected_keys = {
        "schema_version",
        "sequence",
        "event_id",
        "stage",
        "scope",
        "attributes",
        "recorded_at_epoch_s",
        "recorded_at_monotonic_s",
        "payload_hash",
    }
    if set(value) != expected_keys or value.get("schema_version") != 1:
        raise TimelineStoreError("timeline record schema is invalid")

    sequence = value["sequence"]
    event_id = value["event_id"]
    raw_stage = value["stage"]
    raw_scope = value["scope"]
    attributes = value["attributes"]
    epoch_s = value["recorded_at_epoch_s"]
    monotonic_s = value["recorded_at_monotonic_s"]
    payload_hash = value["payload_hash"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise TimelineStoreError("timeline record sequence is invalid")
    if not _identity_is_safe(event_id):
        raise TimelineStoreError("timeline record event identity is invalid")
    if not isinstance(raw_stage, str):
        raise TimelineStoreError("timeline record stage is invalid")
    try:
        stage = VoiceTimelineStage(raw_stage)
    except ValueError as exc:
        raise TimelineStoreError("timeline record stage is invalid") from exc
    if not isinstance(raw_scope, dict):
        raise TimelineStoreError("timeline record scope is invalid")
    scope_keys = {"voice_turn_id", *_OPTIONAL_SCOPE_FIELDS}
    if set(raw_scope) != scope_keys:
        raise TimelineStoreError("timeline record scope is invalid")
    if not _identity_is_safe(raw_scope.get("voice_turn_id")):
        raise TimelineStoreError("timeline record voice turn identity is invalid")
    if any(
        value is not None and not _identity_is_safe(value)
        for key, value in raw_scope.items()
        if key != "voice_turn_id"
    ):
        raise TimelineStoreError("timeline record scope is invalid")
    scope = VoiceTimelineScope(**raw_scope)
    if not isinstance(attributes, dict) or not _privacy_attributes_are_safe(attributes):
        raise TimelineStoreError("timeline record attributes are invalid")
    if not _is_finite_number(epoch_s) or not _is_finite_number(monotonic_s):
        raise TimelineStoreError("timeline record timestamp is invalid")
    if (
        not isinstance(payload_hash, str)
        or len(payload_hash) != 64
        or any(character not in "0123456789abcdef" for character in payload_hash)
    ):
        raise TimelineStoreError("timeline record payload hash is invalid")
    event_input = VoiceTimelineEventInput(
        event_id=event_id,
        stage=stage,
        scope=scope,
        attributes=attributes,
    )
    if _event_payload_hash(event_input) != payload_hash:
        raise TimelineStoreError("timeline record payload hash does not match")
    return VoiceTimelineRecord(
        sequence=sequence,
        event_id=event_id,
        stage=stage,
        scope=scope,
        attributes=dict(attributes),
        recorded_at_epoch_s=float(epoch_s),
        recorded_at_monotonic_s=float(monotonic_s),
        payload_hash=payload_hash,
    )


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _privacy_attributes_are_safe(attributes: Mapping[str, object]) -> bool:
    if not isinstance(attributes, Mapping) or len(attributes) > MAX_ATTRIBUTE_COUNT:
        return False
    for key, value in attributes.items():
        if not isinstance(key, str) or key not in VOICE_TIMELINE_ATTRIBUTE_ALLOWLIST:
            return False
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, str):
            if len(value) > MAX_ATTRIBUTE_STRING_LENGTH:
                return False
            continue
        if isinstance(value, int):
            if -(2**63) <= value <= 2**63 - 1:
                continue
            return False
        if isinstance(value, float):
            if math.isfinite(value):
                continue
            return False
        return False
    return True


def _event_identity_is_safe(event: VoiceTimelineEventInput) -> bool:
    if not _identity_is_safe(event.event_id):
        return False
    if not _identity_is_safe(event.scope.voice_turn_id):
        return False
    return all(
        value is None or _identity_is_safe(value)
        for value in (
            event.scope.thread_id,
            event.scope.turn_id,
            event.scope.generation_id,
            event.scope.provider_session_id,
            event.scope.trace_id,
        )
    )


def _identity_is_safe(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) <= MAX_IDENTITY_LENGTH
        and _SAFE_IDENTITY_PATTERN.fullmatch(value) is not None
    )
