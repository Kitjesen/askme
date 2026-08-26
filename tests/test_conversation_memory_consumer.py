"""Committed conversation events are admitted to durable memory exactly once per replay key."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from askme.conversation.models import CommittedTurnEvent
from askme.memory.core.conversation_consumer import (
    ConversationMemoryCheckpointCorruptError,
    ConversationMemoryCheckpointMismatchError,
    ConversationMemoryCheckpointWriteError,
    ConversationMemoryConsumer,
    ConversationMemoryProcessingError,
    ErasureDeletionUnsupportedError,
)
from askme.memory.core.turn_admission import TurnAdmissionResult, TurnMemoryAdmission
from askme.memory.retrieval.bridge import MemoryBridge


@dataclass
class _CommittedEventSource:
    events: list[CommittedTurnEvent]

    def __post_init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def list_committed_turn_events(
        self,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[CommittedTurnEvent]:
        self.requests.append((after_sequence, limit))
        return [event for event in self.events if event.sequence > after_sequence][:limit]


class _LifecycleFilteringSource(_CommittedEventSource):
    def __init__(
        self,
        *,
        committed: list[CommittedTurnEvent],
        lifecycle_noise: list[tuple[str, str]],
    ) -> None:
        super().__init__(committed)
        self.lifecycle_noise = list(lifecycle_noise)


class _RecordingSink:
    def __init__(self, results: list[TurnAdmissionResult | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def admit_turn(self, user_text: str, **kwargs: Any) -> TurnAdmissionResult:
        self.calls.append((user_text, dict(kwargs)))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _RecordingKnowledgeCatalog:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def upsert_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, int]:
        self.payloads.extend(payloads)
        return {"accepted": len(payloads)}


class _IdempotentSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.records: dict[str, str] = {}

    async def admit_turn(self, user_text: str, **kwargs: Any) -> TurnAdmissionResult:
        copied = dict(kwargs)
        self.calls.append((user_text, copied))
        self.records.setdefault(str(copied["idempotency_key"]), user_text)
        return _admitted_result()


class _BlockingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.first_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()

    async def admit_turn(self, user_text: str, **kwargs: Any) -> TurnAdmissionResult:
        self.calls.append((user_text, dict(kwargs)))
        if len(self.calls) == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
        return _admitted_result()


class _MalformedSink:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def admit_turn(self, user_text: str, **kwargs: Any) -> TurnAdmissionResult:
        self.calls.append((user_text, dict(kwargs)))
        return self.result


class _FailingMemPalaceCollection:
    def __init__(self) -> None:
        self.upsert_count = 0

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_count += 1
        raise OSError("disk unavailable")


def _patch_mempalace(collection: Any):
    palace_module = ModuleType("mempalace.palace")
    setattr(palace_module, "get_collection", MagicMock(return_value=collection))
    package = ModuleType("mempalace")
    setattr(package, "palace", palace_module)
    return patch.dict(
        sys.modules,
        {"mempalace": package, "mempalace.palace": palace_module},
    )


def _event(
    sequence: int,
    *,
    event_id: str | None = None,
    user_text: str = "请记住我喜欢简短回答",
    assistant_text: str = "好的，我以后会简短回答。",
    occurred_at: datetime | None = None,
) -> CommittedTurnEvent:
    return CommittedTurnEvent(
        event_id=event_id or f"commit-{sequence}",
        sequence=sequence,
        occurred_at=occurred_at or datetime(2026, 7, 26, 8, sequence, tzinfo=UTC),
        thread_id="thread-7",
        turn_id=f"turn-{sequence}",
        turn_sequence=sequence,
        source="voice",
        user_text=user_text,
        assistant_text=assistant_text,
        heard_text=user_text,
        played_ms=640,
        playback_disposition="completed",
        metadata={
            "customer_id": "customer-a",
            "project_id": "project-b",
            "user_id": "person-c",
        },
    )


def _admitted_result() -> TurnAdmissionResult:
    classified = TurnMemoryAdmission().classify(
        "请记住我喜欢简短回答",
        user_id="person-c",
    )
    assert classified.admitted is True
    return replace(classified, persisted_count=len(classified.candidates))


@pytest.mark.asyncio
async def test_committed_event_admits_only_user_text_with_exact_backlinks(
    tmp_path: Path,
) -> None:
    event = _event(7, event_id="commit-fixed")
    source = _CommittedEventSource([event])
    sink = _RecordingSink(
        [TurnAdmissionResult(False, rejected_reason="not_durable_memory")]
    )
    checkpoint_path = tmp_path / "memory-consumer.json"
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    result = await consumer.run_once()

    assert source.requests == [(0, 100)]
    assert sink.calls == [
        (
            event.user_text,
            {
                "source_turn_id": event.turn_id,
                "source_event_id": event.event_id,
                "source_sequence": event.sequence,
                "source_thread_id": event.thread_id,
                "idempotency_key": "conversation-memory:v1:commit-fixed",
                "source": event.source,
                "occurred_at": event.occurred_at,
                "customer_id": "customer-a",
                "project_id": "project-b",
                "user_id": "person-c",
            },
        )
    ]
    assert event.assistant_text not in repr(sink.calls)
    assert result.acknowledged_count == 1
    assert result.rejected_count == 1
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint == {
        "schema": "askme.memory.conversation-consumer-checkpoint",
        "version": 1,
        "consumer": "conversation-memory",
        "source_id": "ledger:site-a",
        "last_sequence": 7,
        "last_event_id": "commit-fixed",
        "updated_at": checkpoint["updated_at"],
    }
    assert datetime.fromisoformat(checkpoint["updated_at"]).tzinfo is not None


def test_turn_admission_uses_observed_time_and_preserves_source_backlinks() -> None:
    observed_at = datetime(2026, 7, 20, 12, 30, tzinfo=UTC)
    admission = TurnMemoryAdmission(user_memory_ttl_days=10)

    result = admission.classify(
        "请记住我喜欢简短回答",
        source="voice",
        source_turn_id="turn-7",
        source_event_id="commit-7",
        source_sequence=17,
        source_thread_id="thread-7",
        idempotency_key="conversation-memory:v1:commit-7",
        observed_at=observed_at,
        user_id="person-c",
    )

    assert result.admitted is True
    candidate = result.candidates[0]
    assert candidate.created_at == "2026-07-20T12:30:00+00:00"
    assert candidate.last_confirmed_at == candidate.created_at
    metadata = candidate.to_metadata()
    assert metadata["source"] == "voice"
    assert metadata["source_turn_id"] == "turn-7"
    assert metadata["source_event_id"] == "commit-7"
    assert metadata["source_sequence"] == 17
    assert metadata["source_thread_id"] == "thread-7"
    assert metadata["idempotency_key"] == "conversation-memory:v1:commit-7"
    assert metadata["occurred_at"] == "2026-07-20T12:30:00+00:00"
    replayed = admission.classify(
        "请记住我喜欢简短回答",
        source="voice",
        source_turn_id="turn-7",
        source_event_id="commit-7",
        source_sequence=17,
        source_thread_id="thread-7",
        idempotency_key="conversation-memory:v1:commit-7",
        observed_at=observed_at,
        user_id="person-c",
    )
    assert replayed.candidates == result.candidates


@pytest.mark.asyncio
async def test_memory_bridge_preserves_committed_event_identity_in_backend_metadata(
    tmp_path: Path,
) -> None:
    catalog = _RecordingKnowledgeCatalog()
    bridge = MemoryBridge(
        config={"memory": {"enabled": True, "backend": "vector"}, "brain": {}},
        data_dir=tmp_path,
        knowledge_catalog=catalog,
    )
    occurred_at = datetime(2026, 7, 20, 12, 30, tzinfo=UTC)

    result = await bridge.admit_turn(
        "A区卫生间在东侧",
        source="voice",
        source_turn_id="turn-7",
        source_event_id="commit-7",
        source_sequence=17,
        source_thread_id="thread-7",
        idempotency_key="conversation-memory:v1:commit-7",
        occurred_at=occurred_at,
        customer_id="customer-a",
        project_id="project-b",
        user_id="person-c",
    )

    assert result.persisted_count == 1
    assert result.persistence_errors == ()
    assert len(catalog.payloads) == 1
    metadata = catalog.payloads[0]["metadata"]
    assert metadata["source"] == "voice"
    assert metadata["source_turn_id"] == "turn-7"
    assert metadata["source_event_id"] == "commit-7"
    assert metadata["source_sequence"] == 17
    assert metadata["source_thread_id"] == "thread-7"
    assert metadata["idempotency_key"] == "conversation-memory:v1:commit-7"
    assert metadata["occurred_at"] == "2026-07-20T12:30:00+00:00"
    assert metadata["created_at"] == metadata["occurred_at"]


@pytest.mark.asyncio
async def test_mempalace_write_failure_is_not_counted_as_persisted(tmp_path: Path) -> None:
    collection = _FailingMemPalaceCollection()
    bridge = MemoryBridge(
        config={
            "memory": {
                "enabled": True,
                "backend": "vector",
                "user_long_term_memory_backend": "mempalace",
                "mempalace_palace_path": str(tmp_path / "palace"),
                "user_id": "person-c",
            },
            "brain": {},
        },
        data_dir=tmp_path,
    )

    with _patch_mempalace(collection):
        result = await bridge.admit_turn(
            "请记住我喜欢简短回答",
            source_turn_id="turn-7",
            source_event_id="commit-7",
            source_sequence=17,
            source_thread_id="thread-7",
            idempotency_key="conversation-memory:v1:commit-7",
            occurred_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
            user_id="person-c",
        )

    assert collection.upsert_count == 1
    assert result.admitted is True
    assert result.persisted_count == 0
    assert len(result.persistence_errors) == 1


@pytest.mark.asyncio
async def test_default_erasure_gate_reports_blocked_and_refuses_processing(
    tmp_path: Path,
) -> None:
    source = _CommittedEventSource([_event(1)])
    sink = _RecordingSink(
        [TurnAdmissionResult(False, rejected_reason="not_durable_memory")]
    )
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=tmp_path / "memory-consumer.json",
        source_id="ledger:site-a",
    )

    status = consumer.status()

    assert status.processing_allowed is False
    assert status.erasure_deletion_supported is False
    assert status.blocked_reason == "erasure_deletion_unsupported"
    with pytest.raises(ErasureDeletionUnsupportedError):
        await consumer.run_once()
    assert source.requests == []
    assert sink.calls == []


@pytest.mark.asyncio
async def test_restart_resumes_after_last_acknowledged_global_sequence(
    tmp_path: Path,
) -> None:
    source = _CommittedEventSource([_event(3), _event(8)])
    sink = _RecordingSink([_admitted_result(), _admitted_result()])
    checkpoint_path = tmp_path / "memory-consumer.json"

    first = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        batch_size=1,
        erasure_deletion_supported=True,
    )
    first_result = await first.run_once()
    restarted = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        batch_size=1,
        erasure_deletion_supported=True,
    )
    second_result = await restarted.run_once()

    assert source.requests == [(0, 1), (3, 1)]
    assert [call[1]["source_sequence"] for call in sink.calls] == [3, 8]
    assert first_result.last_sequence == 3
    assert second_result.last_sequence == 8
    assert second_result.admitted_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        replace(_admitted_result(), persisted_count=0),
        replace(_admitted_result(), persistence_errors=("backend:error",)),
        OSError("backend offline"),
    ],
    ids=["partial", "reported-error", "exception"],
)
async def test_first_unacknowledged_failure_stops_without_advancing_past_it(
    tmp_path: Path,
    failure: TurnAdmissionResult | Exception,
) -> None:
    source = _CommittedEventSource([_event(1), _event(2), _event(3)])
    sink = _RecordingSink(
        [
            TurnAdmissionResult(False, rejected_reason="not_durable_memory"),
            failure,
            _admitted_result(),
        ]
    )
    checkpoint_path = tmp_path / "memory-consumer.json"
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    with pytest.raises(ConversationMemoryProcessingError) as exc_info:
        await consumer.run_once()

    assert exc_info.value.event_id == "commit-2"
    assert [call[1]["source_sequence"] for call in sink.calls] == [1, 2]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["last_sequence"] == 1
    assert checkpoint["last_event_id"] == "commit-1"


@pytest.mark.asyncio
async def test_out_of_order_source_page_fails_closed_before_any_admission(
    tmp_path: Path,
) -> None:
    source = _CommittedEventSource([_event(8), _event(3)])
    sink = _RecordingSink([_admitted_result(), _admitted_result()])
    checkpoint_path = tmp_path / "memory-consumer.json"
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    with pytest.raises(ConversationMemoryProcessingError) as exc_info:
        await consumer.run_once()

    assert exc_info.value.reason == "source_sequence_not_strict"
    assert sink.calls == []
    assert checkpoint_path.exists() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("checkpoint_text", "expected_error"),
    [
        ("{not-json", ConversationMemoryCheckpointCorruptError),
        (
            json.dumps(
                {
                    "schema": "wrong-schema",
                    "version": 1,
                    "consumer": "conversation-memory",
                    "source_id": "ledger:site-a",
                    "last_sequence": 1,
                    "last_event_id": "commit-1",
                    "updated_at": "2026-07-20T12:30:00+00:00",
                }
            ),
            ConversationMemoryCheckpointMismatchError,
        ),
        (
            json.dumps(
                {
                    "schema": "askme.memory.conversation-consumer-checkpoint",
                    "version": 1,
                    "consumer": "conversation-memory",
                    "source_id": "ledger:different-site",
                    "last_sequence": 1,
                    "last_event_id": "commit-1",
                    "updated_at": "2026-07-20T12:30:00+00:00",
                }
            ),
            ConversationMemoryCheckpointMismatchError,
        ),
    ],
    ids=["corrupt-json", "schema-mismatch", "source-mismatch"],
)
async def test_invalid_checkpoint_fails_closed_before_source_poll(
    tmp_path: Path,
    checkpoint_text: str,
    expected_error: type[Exception],
) -> None:
    checkpoint_path = tmp_path / "memory-consumer.json"
    checkpoint_path.write_text(checkpoint_text, encoding="utf-8")
    source = _CommittedEventSource([_event(2)])
    sink = _RecordingSink([_admitted_result()])
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    with pytest.raises(expected_error):
        await consumer.run_once()

    assert source.requests == []
    assert sink.calls == []


@pytest.mark.asyncio
async def test_checkpoint_failure_replays_same_identity_without_duplicate_backend_record(
    tmp_path: Path,
) -> None:
    source = _CommittedEventSource([_event(7, event_id="commit-fixed")])
    sink = _IdempotentSink()
    checkpoint_path = tmp_path / "memory-consumer.json"

    first = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )
    with (
        patch(
            "askme.memory.core.conversation_consumer.os.replace",
            side_effect=OSError("replace failed"),
        ),
        pytest.raises(ConversationMemoryCheckpointWriteError),
    ):
        await first.run_once()

    restarted = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )
    result = await restarted.run_once()

    assert result.acknowledged_count == 1
    assert len(sink.calls) == 2
    first_kwargs = sink.calls[0][1]
    replay_kwargs = sink.calls[1][1]
    assert replay_kwargs == first_kwargs
    assert replay_kwargs["idempotency_key"] == "conversation-memory:v1:commit-fixed"
    assert replay_kwargs["source_event_id"] == "commit-fixed"
    assert replay_kwargs["source_sequence"] == 7
    assert len(sink.records) == 1


@pytest.mark.asyncio
async def test_started_and_cancelled_lifecycle_noise_never_reaches_memory(
    tmp_path: Path,
) -> None:
    committed = _event(5, user_text="请记住我喜欢简短回答")
    source = _LifecycleFilteringSource(
        committed=[committed],
        lifecycle_noise=[
            ("started", "started text must stay invisible"),
            ("cancelled", "cancelled text must stay invisible"),
        ],
    )
    sink = _RecordingSink([_admitted_result()])
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=tmp_path / "memory-consumer.json",
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    result = await consumer.run_once()

    assert result.acknowledged_count == 1
    assert [call[0] for call in sink.calls] == [committed.user_text]
    assert "started text" not in repr(sink.calls)
    assert "cancelled text" not in repr(sink.calls)


@pytest.mark.asyncio
async def test_concurrent_run_once_calls_are_linearized_after_checkpoint(
    tmp_path: Path,
) -> None:
    source = _CommittedEventSource([_event(7)])
    sink = _BlockingSink()
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=tmp_path / "memory-consumer.json",
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    first_task = asyncio.create_task(consumer.run_once())
    await sink.first_call_started.wait()
    second_task = asyncio.create_task(consumer.run_once())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    calls_before_first_checkpoint = len(sink.calls)
    sink.release_first_call.set()
    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert calls_before_first_checkpoint == 1
    assert len(sink.calls) == 1
    assert source.requests == [(0, 100), (7, 100)]
    assert first_result.acknowledged_count == 1
    assert second_result.acknowledged_count == 0
    assert second_result.last_sequence == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_result",
    [
        TurnAdmissionResult(False, persisted_count=1),
        TurnAdmissionResult(True, candidates=(), persisted_count=0),
        replace(_admitted_result(), persisted_count=-1),
        replace(_admitted_result(), persisted_count=2),
        replace(_admitted_result(), candidates=cast(Any, None)),
        {"admitted": False, "persisted_count": 0},
    ],
    ids=[
        "rejection-persisted",
        "admitted-without-candidates",
        "negative-count",
        "count-exceeds-candidates",
        "invalid-candidates",
        "wrong-result-type",
    ],
)
async def test_malformed_sink_result_fails_closed_without_checkpoint(
    tmp_path: Path,
    malformed_result: Any,
) -> None:
    source = _CommittedEventSource([_event(7)])
    sink = _MalformedSink(malformed_result)
    checkpoint_path = tmp_path / "memory-consumer.json"
    consumer = ConversationMemoryConsumer(
        source=source,
        sink=sink,
        checkpoint_path=checkpoint_path,
        source_id="ledger:site-a",
        erasure_deletion_supported=True,
    )

    with pytest.raises(ConversationMemoryProcessingError) as exc_info:
        await consumer.run_once()

    assert exc_info.value.event_id == "commit-7"
    assert len(sink.calls) == 1
    assert checkpoint_path.exists() is False


def test_batch_size_rejects_bool_even_though_bool_is_an_int(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ConversationMemoryConsumer(
            source=_CommittedEventSource([]),
            sink=_RecordingSink([]),
            checkpoint_path=tmp_path / "memory-consumer.json",
            source_id="ledger:site-a",
            batch_size=True,
            erasure_deletion_supported=True,
        )
