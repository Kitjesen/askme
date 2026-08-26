from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from askme.voice.core.turn_timeline import (
    JsonlVoiceTimelineStore,
    MemoryVoiceTimelineStore,
    TimelineClockError,
    TimelineConflict,
    TimelineQuery,
    TimelineRecordStatus,
    TimelineStoreError,
    VoiceTimelineEventInput,
    VoiceTimelineScope,
    VoiceTimelineStage,
    VoiceTurnTimeline,
)


def test_record_orders_events_with_injected_clocks() -> None:
    epoch_values = iter([1_000.125, 1_000.5])
    monotonic_values = iter([20.0, 20.25])
    timeline = VoiceTurnTimeline(
        store=MemoryVoiceTimelineStore(),
        epoch_clock=lambda: next(epoch_values),
        monotonic_clock=lambda: next(monotonic_values),
    )

    first = timeline.record(
        VoiceTimelineEventInput(
            event_id="event-1",
            stage=VoiceTimelineStage.LISTEN_STARTED,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-1"),
            attributes={"source": "microphone"},
        )
    )
    second = timeline.record(
        VoiceTimelineEventInput(
            event_id="event-2",
            stage=VoiceTimelineStage.FIRST_AUDIO_FRAME,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-1"),
            attributes={"frame_samples": 320},
        )
    )

    snapshot = timeline.snapshot(TimelineQuery(voice_turn_id="voice-turn-1", limit=10))

    assert first.status is TimelineRecordStatus.RECORDED
    assert second.status is TimelineRecordStatus.RECORDED
    assert [event.sequence for event in snapshot.events] == [1, 2]
    assert [event.recorded_at_epoch_s for event in snapshot.events] == [1_000.125, 1_000.5]
    assert [event.recorded_at_monotonic_s for event in snapshot.events] == [20.0, 20.25]
    assert snapshot.to_dict() == snapshot.to_dict()


def test_event_id_retry_is_idempotent_but_payload_reuse_conflicts() -> None:
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore())
    original = VoiceTimelineEventInput(
        event_id="stable-event",
        stage=VoiceTimelineStage.SPEECH_END,
        scope=VoiceTimelineScope(voice_turn_id="voice-turn-1"),
        attributes={"duration_ms": 420.5, "reason_code": "silence"},
    )

    first = timeline.record(original)
    duplicate = timeline.record(
        VoiceTimelineEventInput(
            event_id="stable-event",
            stage=VoiceTimelineStage.SPEECH_END,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-1"),
            attributes={"reason_code": "silence", "duration_ms": 420.5},
        )
    )

    assert duplicate.status is TimelineRecordStatus.DUPLICATE
    assert duplicate.sequence == first.sequence
    assert len(timeline.snapshot(TimelineQuery()).events) == 1

    with pytest.raises(TimelineConflict):
        timeline.record(
            VoiceTimelineEventInput(
                event_id="stable-event",
                stage=VoiceTimelineStage.SPEECH_END,
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-1"),
                attributes={"duration_ms": 999.0, "reason_code": "silence"},
            )
        )


def test_privacy_guard_drops_unsafe_attributes_before_storage_or_export() -> None:
    class CollectingExporter:
        def __init__(self) -> None:
            self.records = []
            self.offered = Event()

        def offer(self, record) -> None:
            self.records.append(record)
            self.offered.set()

    exporter = CollectingExporter()
    timeline = VoiceTurnTimeline(
        store=MemoryVoiceTimelineStore(),
        exporter=exporter,
    )
    accepted = timeline.record(
        VoiceTimelineEventInput(
            event_id="safe-event",
            stage=VoiceTimelineStage.FIRST_AUDIO_FRAME,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-private"),
            attributes={
                "source": "microphone",
                "frame_samples": 320,
                "duration_ms": 20.0,
                "instrumented": True,
            },
        )
    )
    unsafe_attributes = [
        {"transcript": "the raw words must never escape"},
        {"pcm": b"\x00\x01"},
        {"metrics": {"rms": 0.5}},
        {"duration_ms": float("inf")},
        {"source": "x" * 257},
        {"not_explicitly_allowed": "value"},
    ]

    dropped = [
        timeline.record(
            VoiceTimelineEventInput(
                event_id=f"unsafe-{index}",
                stage=VoiceTimelineStage.ERROR,
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-private"),
                attributes=attributes,
            )
        )
        for index, attributes in enumerate(unsafe_attributes)
    ]

    assert accepted.status is TimelineRecordStatus.RECORDED
    assert all(receipt.status is TimelineRecordStatus.DROPPED_PRIVACY for receipt in dropped)
    assert all(receipt.sequence is None for receipt in dropped)
    assert [event.event_id for event in timeline.snapshot(TimelineQuery()).events] == [
        "safe-event"
    ]
    assert exporter.offered.wait(timeout=1.0)
    assert [record.event_id for record in exporter.records] == ["safe-event"]


def test_late_correlation_retroactively_scopes_events_and_rejects_rebinding() -> None:
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore())
    timeline.record(
        VoiceTimelineEventInput(
            event_id="early-listen",
            stage=VoiceTimelineStage.LISTEN_STARTED,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-late"),
        )
    )
    effective_scope = VoiceTimelineScope(
        voice_turn_id="voice-turn-late",
        thread_id="thread-7",
        turn_id="turn-7",
        generation_id="generation-2",
        provider_session_id="provider-session-b",
        trace_id="trace-9",
    )
    timeline.record(
        VoiceTimelineEventInput(
            event_id="correlation",
            stage=VoiceTimelineStage.TURN_CORRELATED,
            scope=effective_scope,
        )
    )

    snapshot = timeline.snapshot(
        TimelineQuery(
            thread_id="thread-7",
            turn_id="turn-7",
            trace_id="trace-9",
        )
    )

    assert [event.event_id for event in snapshot.events] == ["early-listen", "correlation"]
    assert snapshot.events[0].scope == VoiceTimelineScope(
        voice_turn_id="voice-turn-late",
        thread_id="thread-7",
        turn_id="turn-7",
        trace_id="trace-9",
    )
    assert snapshot.events[1].scope == effective_scope
    assert [
        event.event_id
        for event in timeline.snapshot(TimelineQuery(generation_id="generation-2")).events
    ] == ["correlation"]
    assert [
        event.event_id
        for event in timeline.snapshot(
            TimelineQuery(provider_session_id="provider-session-b")
        ).events
    ] == ["correlation"]

    with pytest.raises(TimelineConflict):
        timeline.record(
            VoiceTimelineEventInput(
                event_id="conflicting-correlation",
                stage=VoiceTimelineStage.TURN_CORRELATED,
                scope=VoiceTimelineScope(
                    voice_turn_id="voice-turn-late",
                    thread_id="thread-other",
                ),
            )
        )
    assert len(timeline.snapshot(TimelineQuery()).events) == 2


def test_generation_and_provider_session_remain_event_local_within_voice_turn() -> None:
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore())
    voice_turn_id = "voice-turn-multi-generation"
    timeline.record(
        VoiceTimelineEventInput(
            event_id="before-generation",
            stage=VoiceTimelineStage.ASR_FINAL,
            scope=VoiceTimelineScope(voice_turn_id=voice_turn_id),
        )
    )
    stable_scope = {
        "voice_turn_id": voice_turn_id,
        "thread_id": "thread-multi",
        "turn_id": "turn-multi",
        "trace_id": "trace-multi",
    }
    first = timeline.record(
        VoiceTimelineEventInput(
            event_id="generation-one",
            stage=VoiceTimelineStage.LLM_REQUESTED,
            scope=VoiceTimelineScope(
                **stable_scope,
                generation_id="generation-1",
                provider_session_id="provider-session-a",
            ),
        )
    )
    second = timeline.record(
        VoiceTimelineEventInput(
            event_id="generation-two",
            stage=VoiceTimelineStage.FALLBACK_SELECTED,
            scope=VoiceTimelineScope(
                **stable_scope,
                generation_id="generation-2",
                provider_session_id="provider-session-b",
            ),
        )
    )

    all_events = timeline.snapshot(TimelineQuery(voice_turn_id=voice_turn_id)).events
    generation_one = timeline.snapshot(TimelineQuery(generation_id="generation-1"))
    generation_two = timeline.snapshot(TimelineQuery(generation_id="generation-2"))

    assert first.status is TimelineRecordStatus.RECORDED
    assert second.status is TimelineRecordStatus.RECORDED
    assert all_events[0].scope == VoiceTimelineScope(**stable_scope)
    assert all_events[1].scope.generation_id == "generation-1"
    assert all_events[1].scope.provider_session_id == "provider-session-a"
    assert all_events[2].scope.generation_id == "generation-2"
    assert all_events[2].scope.provider_session_id == "provider-session-b"
    assert [event.event_id for event in generation_one.events] == ["generation-one"]
    assert [event.event_id for event in generation_two.events] == ["generation-two"]
    assert [
        event.event_id
        for event in timeline.snapshot(
            TimelineQuery(provider_session_id="provider-session-b")
        ).events
    ] == ["generation-two"]


def test_blocking_exporter_does_not_delay_local_recording() -> None:
    class BlockingExporter:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        def offer(self, _record) -> None:
            self.started.set()
            self.release.wait(timeout=2.0)

    exporter = BlockingExporter()
    timeline = VoiceTurnTimeline(
        store=MemoryVoiceTimelineStore(),
        exporter=exporter,
    )
    event = VoiceTimelineEventInput(
        event_id="non-blocking-export",
        stage=VoiceTimelineStage.LLM_REQUESTED,
        scope=VoiceTimelineScope(voice_turn_id="voice-turn-export"),
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            receipt = pool.submit(timeline.record, event).result(timeout=0.25)
    finally:
        exporter.release.set()

    assert receipt.status is TimelineRecordStatus.RECORDED
    assert exporter.started.wait(timeout=1.0)
    assert [item.event_id for item in timeline.snapshot(TimelineQuery()).events] == [
        "non-blocking-export"
    ]


def test_export_failure_is_sanitized_and_duplicate_is_not_reexported() -> None:
    class SecretExportError(RuntimeError):
        pass

    class FailingExporter:
        def __init__(self) -> None:
            self.call_count = 0

        def offer(self, _record) -> None:
            self.call_count += 1
            raise SecretExportError("credential=must-not-appear")

    exporter = FailingExporter()
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore(), exporter=exporter)
    event = VoiceTimelineEventInput(
        event_id="export-failure",
        stage=VoiceTimelineStage.UPSTREAM_CLOSED,
        scope=VoiceTimelineScope(voice_turn_id="voice-turn-export"),
        attributes={"reason_code": "remote_close"},
    )

    recorded = timeline.record(event)
    duplicate = timeline.record(event)
    deadline = time.monotonic() + 1.0
    snapshot = timeline.snapshot(TimelineQuery())
    while snapshot.export_error_count == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
        snapshot = timeline.snapshot(TimelineQuery())

    assert recorded.status is TimelineRecordStatus.RECORDED
    assert duplicate.status is TimelineRecordStatus.DUPLICATE
    assert exporter.call_count == 1
    assert snapshot.export_error_count == 1
    assert snapshot.last_export_error_type == "SecretExportError"
    assert "credential" not in repr(snapshot.to_dict())
    assert [item.event_id for item in snapshot.events] == ["export-failure"]


def test_jsonl_store_is_canonical_and_replays_identity_and_correlation(tmp_path) -> None:
    path = tmp_path / "voice-timeline.jsonl"
    first_input = VoiceTimelineEventInput(
        event_id="persisted-early",
        stage=VoiceTimelineStage.SPEECH_START,
        scope=VoiceTimelineScope(voice_turn_id="voice-turn-persisted"),
        attributes={"confidence": 0.875, "source": "vad"},
    )
    correlation_input = VoiceTimelineEventInput(
        event_id="persisted-correlation",
        stage=VoiceTimelineStage.TURN_CORRELATED,
        scope=VoiceTimelineScope(
            voice_turn_id="voice-turn-persisted",
            thread_id="thread-persisted",
            turn_id="turn-persisted",
            generation_id="generation-persisted-1",
            provider_session_id="provider-session-persisted-a",
            trace_id="trace-persisted",
        ),
    )
    retry_input = VoiceTimelineEventInput(
        event_id="persisted-retry",
        stage=VoiceTimelineStage.FALLBACK_SELECTED,
        scope=VoiceTimelineScope(
            voice_turn_id="voice-turn-persisted",
            thread_id="thread-persisted",
            turn_id="turn-persisted",
            generation_id="generation-persisted-2",
            provider_session_id="provider-session-persisted-b",
            trace_id="trace-persisted",
        ),
    )
    timeline = VoiceTurnTimeline(
        store=JsonlVoiceTimelineStore(path),
        epoch_clock=iter([10.0, 11.0, 12.0]).__next__,
        monotonic_clock=iter([20.0, 21.0, 22.0]).__next__,
    )
    timeline.record(first_input)
    timeline.record(correlation_input)
    timeline.record(retry_input)

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    decoded = [json.loads(line) for line in raw_lines]

    assert path.read_bytes().endswith(b"\n")
    assert raw_lines == [
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for item in decoded
    ]
    assert [item["sequence"] for item in decoded] == [1, 2, 3]
    assert all(len(item["payload_hash"]) == 64 for item in decoded)

    restarted = VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))
    replayed = restarted.snapshot(TimelineQuery(turn_id="turn-persisted"))

    assert [event.event_id for event in replayed.events] == [
        "persisted-early",
        "persisted-correlation",
        "persisted-retry",
    ]
    assert all(event.scope.thread_id == "thread-persisted" for event in replayed.events)
    assert replayed.events[0].scope == VoiceTimelineScope(
        voice_turn_id="voice-turn-persisted",
        thread_id="thread-persisted",
        turn_id="turn-persisted",
        trace_id="trace-persisted",
    )
    assert replayed.events[1].scope == correlation_input.scope
    assert replayed.events[2].scope == retry_input.scope
    assert [
        event.event_id
        for event in restarted.snapshot(
            TimelineQuery(generation_id="generation-persisted-2")
        ).events
    ] == ["persisted-retry"]
    assert restarted.record(first_input).status is TimelineRecordStatus.DUPLICATE
    with pytest.raises(TimelineConflict):
        restarted.record(
            VoiceTimelineEventInput(
                event_id="persisted-early",
                stage=VoiceTimelineStage.SPEECH_END,
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-persisted"),
            )
        )


def test_jsonl_replay_repairs_only_an_incomplete_final_line(tmp_path) -> None:
    path = tmp_path / "voice-timeline.jsonl"
    timeline = VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))
    timeline.record(
        VoiceTimelineEventInput(
            event_id="before-crash",
            stage=VoiceTimelineStage.LISTEN_STARTED,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-crash"),
        )
    )
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"sequence":2,"event_id":')

    recovered = VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))
    assert [event.event_id for event in recovered.snapshot(TimelineQuery()).events] == [
        "before-crash"
    ]
    recovered.record(
        VoiceTimelineEventInput(
            event_id="after-recovery",
            stage=VoiceTimelineStage.TURN_FINISHED,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-crash"),
        )
    )

    replayed = VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))
    assert [event.sequence for event in replayed.snapshot(TimelineQuery()).events] == [1, 2]
    assert path.read_bytes().endswith(b"\n")


def test_jsonl_replay_rejects_completed_corruption(tmp_path) -> None:
    path = tmp_path / "voice-timeline.jsonl"
    timeline = VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))
    timeline.record(
        VoiceTimelineEventInput(
            event_id="hash-protected",
            stage=VoiceTimelineStage.ASR_FINAL,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-corrupt"),
            attributes={"provider": "local"},
        )
    )
    completed = json.loads(path.read_text(encoding="utf-8"))
    completed["attributes"]["provider"] = "tampered"
    path.write_text(
        json.dumps(completed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TimelineStoreError):
        VoiceTurnTimeline(store=JsonlVoiceTimelineStore(path))


def test_store_failure_degrades_into_a_bounded_emergency_buffer() -> None:
    class DiskFullError(OSError):
        pass

    class FailingStore:
        def __init__(self) -> None:
            self.append_count = 0

        def load(self):
            return ()

        def append(self, _record) -> None:
            self.append_count += 1
            raise DiskFullError("path-and-secret-must-not-escape")

    store = FailingStore()
    timeline = VoiceTurnTimeline(store=store, emergency_limit=2)

    receipts = [
        timeline.record(
            VoiceTimelineEventInput(
                event_id=f"degraded-{index}",
                stage=VoiceTimelineStage.ERROR,
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-degraded"),
                attributes={"error_type": "storage"},
            )
        )
        for index in range(3)
    ]
    snapshot = timeline.snapshot(TimelineQuery())

    assert all(
        receipt.status is TimelineRecordStatus.DEGRADED_PERSISTENCE
        for receipt in receipts
    )
    assert [event.sequence for event in snapshot.events] == [2, 3]
    assert store.append_count == 1
    assert snapshot.persistence_degraded is True
    assert snapshot.persistence_error_count == 1
    assert snapshot.last_persistence_error_type == "DiskFullError"
    assert snapshot.emergency_event_count == 2
    assert snapshot.emergency_dropped_count == 1
    assert "path-and-secret" not in repr(snapshot.to_dict())


def test_query_is_bounded_and_concurrent_records_keep_a_total_order() -> None:
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore())

    def record_index(index: int):
        return timeline.record(
            VoiceTimelineEventInput(
                event_id=f"concurrent-{index}",
                stage=(
                    VoiceTimelineStage.FIRST_CLAUSE
                    if index % 2 == 0
                    else VoiceTimelineStage.FIRST_SEMANTIC
                ),
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-concurrent"),
                attributes={"clause_index": index},
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(record_index, range(40)))

    snapshot = timeline.snapshot(
        TimelineQuery(stage=VoiceTimelineStage.FIRST_CLAUSE, limit=7)
    )

    assert len(snapshot.events) == 7
    assert all(event.stage is VoiceTimelineStage.FIRST_CLAUSE for event in snapshot.events)
    assert [event.sequence for event in timeline.snapshot(TimelineQuery(limit=100)).events] == list(
        range(1, 41)
    )
    assert {receipt.sequence for receipt in receipts} == set(range(1, 41))
    with pytest.raises(ValueError):
        TimelineQuery(limit=0)
    with pytest.raises(ValueError):
        TimelineQuery(limit=1_001)


def test_identity_privacy_gate_rejects_unsafe_tokens_before_storage_or_export() -> None:
    class CollectingExporter:
        def __init__(self) -> None:
            self.records = []
            self.offered = Event()

        def offer(self, record) -> None:
            self.records.append(record)
            self.offered.set()

    exporter = CollectingExporter()
    timeline = VoiceTurnTimeline(store=MemoryVoiceTimelineStore(), exporter=exporter)
    invalid_events = [
        VoiceTimelineEventInput(
            event_id="raw transcript\ninside identity",
            stage=VoiceTimelineStage.ERROR,
            scope=VoiceTimelineScope(voice_turn_id="voice-turn-safe"),
        ),
        VoiceTimelineEventInput(
            event_id="event-empty-voice-turn",
            stage=VoiceTimelineStage.ERROR,
            scope=VoiceTimelineScope(voice_turn_id=""),
        ),
        VoiceTimelineEventInput(
            event_id="event-spaced-thread",
            stage=VoiceTimelineStage.ERROR,
            scope=VoiceTimelineScope(
                voice_turn_id="voice-turn-safe",
                thread_id="thread id with spaces",
            ),
        ),
        VoiceTimelineEventInput(
            event_id="event-overlong-generation",
            stage=VoiceTimelineStage.ERROR,
            scope=VoiceTimelineScope(
                voice_turn_id="voice-turn-safe",
                generation_id="g" * 257,
            ),
        ),
    ]

    dropped = [timeline.record(event) for event in invalid_events]
    maximum_safe = "i" * 256
    accepted = timeline.record(
        VoiceTimelineEventInput(
            event_id=maximum_safe,
            stage=VoiceTimelineStage.TURN_FINISHED,
            scope=VoiceTimelineScope(voice_turn_id=maximum_safe),
        )
    )

    assert all(receipt.status is TimelineRecordStatus.DROPPED_PRIVACY for receipt in dropped)
    assert accepted.status is TimelineRecordStatus.RECORDED
    assert [event.event_id for event in timeline.snapshot(TimelineQuery()).events] == [
        maximum_safe
    ]
    assert exporter.offered.wait(timeout=1.0)
    assert [event.event_id for event in exporter.records] == [maximum_safe]


@pytest.mark.parametrize(
    ("epoch_value", "monotonic_value"),
    [(float("nan"), 1.0), (1.0, float("inf"))],
)
def test_non_finite_injected_clocks_are_rejected_before_jsonl_or_export(
    tmp_path,
    epoch_value: float,
    monotonic_value: float,
) -> None:
    class CollectingExporter:
        def __init__(self) -> None:
            self.records = []

        def offer(self, record) -> None:
            self.records.append(record)

    path = tmp_path / "invalid-clock.jsonl"
    exporter = CollectingExporter()
    timeline = VoiceTurnTimeline(
        store=JsonlVoiceTimelineStore(path),
        exporter=exporter,
        epoch_clock=lambda: epoch_value,
        monotonic_clock=lambda: monotonic_value,
    )

    with pytest.raises(TimelineClockError):
        timeline.record(
            VoiceTimelineEventInput(
                event_id="invalid-clock",
                stage=VoiceTimelineStage.LISTEN_STARTED,
                scope=VoiceTimelineScope(voice_turn_id="voice-turn-clock"),
            )
        )

    assert timeline.snapshot(TimelineQuery()).events == ()
    assert exporter.records == []
    assert not path.exists()
