from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from askme.voice.core.turn_timeline import (
    TimelineQuery,
    VoiceTimelineStage,
    VoiceTurnTimeline,
)
from askme.voice.core.turn_trace import VoiceTurnTraceRecorder


def _timeline_events(
    timeline: VoiceTurnTimeline,
    voice_turn_id: str,
) -> list[dict[str, Any]]:
    return [
        event.to_dict()
        for event in timeline.snapshot(TimelineQuery(voice_turn_id=voice_turn_id, limit=100)).events
    ]


def test_trace_start_never_retains_sensitive_turn_metadata() -> None:
    recorder = VoiceTurnTraceRecorder()
    secrets = {
        "transcript": "private-start-transcript",
        "prompt": "private-start-prompt",
        "content": "private-start-content",
        "error": "private-start-error",
        "message": "private-start-message",
        "exception": "private-start-exception",
    }

    trace = recorder.start(
        source="microphone",
        media_transport="sounddevice",
        metadata={
            **secrets,
            "thread_id": "thread-safe",
            "turn_id": "turn-safe",
            "trace_id": "trace-safe",
        },
    )

    assert trace.metadata == {
        "thread_id": "thread-safe",
        "trace_id": "trace-safe",
        "turn_id": "turn-safe",
    }
    serialized = repr(trace)
    assert all(secret not in serialized for secret in secrets.values())


def test_trace_rejects_content_shaped_identity_fields_at_ingestion() -> None:
    recorder = VoiceTurnTraceRecorder()
    trace = recorder.start(
        source="private source sentence",
        media_transport="private transport sentence",
    )

    recorder.mark(
        "private stage sentence",
        audio_segment_id="private segment sentence",
        text="private payload sentence",
    )
    recorder.finish("error", error_type="private error sentence")

    serialized = repr(trace)
    assert trace.source == "unknown"
    assert trace.media_transport == "unknown"
    assert set(trace.stages) == {"listen_started"}
    assert "error_type" not in trace.metadata
    assert "private" not in serialized


def test_trace_lifecycle_retains_provenance_but_never_sensitive_text() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(
        source="microphone",
        media_transport="sounddevice",
        metadata={"thread_id": "thread-safe"},
    )
    transcript = "private-mark-transcript"
    mark_secrets = {
        "text": transcript,
        "transcript": "private-mark-transcript-alias",
        "prompt": "private-mark-prompt",
        "content": "private-mark-content",
        "error": "private-mark-error",
        "message": "private-mark-message",
        "exception": "private-mark-exception",
    }
    finish_secrets = {
        "transcript": "private-finish-transcript",
        "prompt": "private-finish-prompt",
        "content": "private-finish-content",
        "error": "private-finish-error",
        "message": "private-finish-message",
        "exception": "private-finish-exception",
    }

    recorder.mark(
        "asr_final",
        **mark_secrets,
        asr_source="local",
        provider="whisper-local",
        latency_ms=87.5,
    )
    recorder.mark_audio(
        "physical_first_semantic_audio",
        audio_class="semantic",
        audio_segment_id="answer-1",
        **mark_secrets,
        evidence_kind="physical_acoustic",
        instrumented=True,
        clock_id="capture-1",
        provenance={
            "validated": True,
            "message": "private-nested-provenance-message",
        },
    )
    recorder.finish(
        "error",
        **finish_secrets,
        asr_source="local",
        error_type="RuntimeError",
    )

    assert trace.metadata == {
        "asr_source": "local",
        "error_type": "RuntimeError",
        "thread_id": "thread-safe",
    }
    assert trace.stages["asr_final"].metadata == {"asr_source": "local"}
    assert trace.stages["physical_first_semantic_audio"].metadata == {
        "clock_id": "capture-1",
        "evidence_kind": "physical_acoustic",
        "instrumented": True,
        "provenance": {"validated": True},
    }
    serialized = repr(trace)
    secret_values = [
        *mark_secrets.values(),
        *finish_secrets.values(),
        "private-nested-provenance-message",
    ]
    assert all(secret not in serialized for secret in secret_values)

    events = _timeline_events(timeline, trace.voice_turn_id)
    asr_event = next(
        event for event in events if event["stage"] == VoiceTimelineStage.ASR_FINAL.value
    )
    physical_event = next(
        event
        for event in events
        if event["stage"] == VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED.value
    )
    assert asr_event["attributes"] == {
        "character_count": len(transcript),
        "latency_ms": 87.5,
        "provider": "whisper-local",
    }
    assert physical_event["attributes"] == {
        "audio_class": "semantic",
        "audio_segment_id": "answer-1",
        "clock_id": "capture-1",
        "evidence_kind": "physical_acoustic",
        "instrumented": True,
    }
    assert all(secret not in repr(events) for secret in secret_values)


def test_trace_adapter_routes_late_playback_to_explicit_voice_turn() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)

    first = recorder.start(source="microphone", media_transport="sounddevice")
    recorder.finish("accepted")
    second = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.mark_for(
        first.voice_turn_id,
        "tts_playback_started",
        generation_id="generation-a",
        provider="local-tts",
    )
    recorder.mark_for(
        first.voice_turn_id,
        "playback_done",
        generation_id="generation-a",
        played_ms=240,
    )

    snapshot = recorder.snapshot()
    first_stages = {stage["name"] for stage in snapshot["latest"]["stages"]}
    second_stages = {stage["name"] for stage in snapshot["current"]["stages"]}
    first_events = _timeline_events(timeline, first.voice_turn_id)
    second_events = _timeline_events(timeline, second.voice_turn_id)

    assert {"tts_playback_started", "playback_done"} <= first_stages
    assert "tts_playback_started" not in second_stages
    assert re.fullmatch(r"[0-9a-f]{32}", first.voice_turn_id)
    assert re.fullmatch(r"[0-9a-f]{32}", second.voice_turn_id)
    assert [event["stage"] for event in first_events][-2:] == [
        VoiceTimelineStage.SPEAKER_RENDER_STARTED.value,
        VoiceTimelineStage.SPEAKER_RENDER_STOPPED.value,
    ]
    assert first_events[-2]["scope"]["generation_id"] == "generation-a"
    assert [event["stage"] for event in second_events] == [VoiceTimelineStage.LISTEN_STARTED.value]


def test_stale_explicit_finish_cannot_close_newer_active_turn() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    first = recorder.start(source="microphone", media_transport="sounddevice")
    recorder.finish("accepted")
    second = recorder.start(source="microphone", media_transport="sounddevice")

    finished = recorder.finish_for(
        first.voice_turn_id,
        "error",
        error="late private failure",
    )

    snapshot = recorder.snapshot()

    assert finished is False
    assert snapshot["current"]["voice_turn_id"] == second.voice_turn_id
    assert snapshot["current"]["status"] == "active"
    assert snapshot["latest"]["status"] == "accepted"
    assert "late private failure" not in repr(_timeline_events(timeline, first.voice_turn_id))


def test_trace_adapter_projects_asr_metadata_without_content() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.mark(
        "asr_final",
        provider="whisper-local",
        latency_ms=87.5,
        text="the private transcript must never leave legacy trace",
        error="private upstream error",
        confidence=0.99,
    )

    event = _timeline_events(timeline, trace.voice_turn_id)[-1]
    serialized = repr(event)

    assert event["stage"] == VoiceTimelineStage.ASR_FINAL.value
    assert event["attributes"] == {
        "character_count": 52,
        "latency_ms": 87.5,
        "provider": "whisper-local",
    }
    assert "private transcript" not in serialized
    assert "private upstream error" not in serialized
    assert "private transcript" not in repr(recorder.snapshot())
    assert "private upstream error" not in repr(recorder.snapshot())


def test_trace_adapter_keeps_generation_and_provider_session_event_local() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(
        source="microphone",
        media_transport="sounddevice",
        metadata={
            "generation_id": "capture-generation",
            "provider_session_id": "capture-session",
            "thread_id": "thread-a",
        },
    )

    recorder.mark("asr_final", provider="local-asr")

    events = _timeline_events(timeline, trace.voice_turn_id)

    assert events[0]["scope"]["generation_id"] == "capture-generation"
    assert events[0]["scope"]["provider_session_id"] == "capture-session"
    assert events[1]["scope"]["generation_id"] is None
    assert events[1]["scope"]["provider_session_id"] is None
    assert all(event["scope"]["thread_id"] == "thread-a" for event in events)


def test_trace_adapter_maps_interrupt_lifecycle() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.mark("barge_in_detected", peak=1200, rms=48.5)
    recorder.mark_barge_in(peak=1400, rms=52.0)
    recorder.mark("barge_in_recovered", reason_code="vad_dismissed")

    events = _timeline_events(timeline, trace.voice_turn_id)

    assert [event["stage"] for event in events] == [
        VoiceTimelineStage.LISTEN_STARTED.value,
        VoiceTimelineStage.INTERRUPT_DETECTED.value,
        VoiceTimelineStage.INTERRUPT_CONFIRMED.value,
        VoiceTimelineStage.INTERRUPT_DISMISSED.value,
    ]
    assert events[-1]["attributes"] == {"reason_code": "vad_dismissed"}


def test_trace_adapter_routes_late_interrupt_to_explicit_output_turn() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    output_turn = recorder.start(source="microphone", media_transport="sounddevice")
    recorder.finish("accepted")
    capture_turn = recorder.start(source="microphone", media_transport="sounddevice")

    assert recorder.mark_barge_in_for(output_turn.voice_turn_id, peak=1400, rms=52.0)

    output_events = _timeline_events(timeline, output_turn.voice_turn_id)
    capture_events = _timeline_events(timeline, capture_turn.voice_turn_id)
    snapshot = recorder.snapshot()

    assert output_events[-1]["stage"] == VoiceTimelineStage.INTERRUPT_CONFIRMED.value
    assert VoiceTimelineStage.INTERRUPT_CONFIRMED.value not in {
        event["stage"] for event in capture_events
    }
    assert snapshot["counters"]["barge_in_count"] == 1


def test_trace_adapter_finish_never_claims_conversation_turn_finished() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.finish("error", error="credential-shaped private error")

    events = _timeline_events(timeline, trace.voice_turn_id)
    stages = [event["stage"] for event in events]
    serialized = repr(events)

    assert stages[-2:] == [
        VoiceTimelineStage.UPSTREAM_CLOSED.value,
        VoiceTimelineStage.ERROR.value,
    ]
    assert VoiceTimelineStage.TURN_FINISHED.value not in stages
    assert events[-2]["attributes"] == {"status": "error"}
    assert events[-1]["attributes"] == {"error_type": "LegacyTraceError"}
    assert "credential-shaped" not in serialized
    assert "credential-shaped" not in repr(recorder.snapshot())


def test_trace_adapter_reports_scope_conflicts_separately() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(
        source="microphone",
        media_transport="sounddevice",
        metadata={"thread_id": "thread-a"},
    )

    recorder.mark(
        "asr_final",
        thread_id="thread-b",
        provider="local-asr",
        text="private transcript",
    )

    snapshot = recorder.snapshot()

    assert {stage["name"] for stage in snapshot["current"]["stages"]} >= {
        "listen_started",
        "asr_final",
    }
    assert snapshot["counters"]["timeline_conflict_count"] == 1
    assert snapshot["counters"]["last_timeline_error_type"] == "TimelineConflict"
    assert [event["stage"] for event in _timeline_events(timeline, trace.voice_turn_id)] == [
        VoiceTimelineStage.LISTEN_STARTED.value
    ]


def test_ambiguous_default_async_mark_is_not_projected_as_a_timeline_fact() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    first = recorder.start(source="microphone", media_transport="sounddevice")
    recorder.finish("accepted")
    second = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.mark("tts_playback_started", generation_id="generation-a")

    snapshot = recorder.snapshot()

    assert "tts_playback_started" in {stage["name"] for stage in snapshot["current"]["stages"]}
    assert snapshot["counters"]["timeline_ambiguous_event_count"] == 1
    assert VoiceTimelineStage.SPEAKER_RENDER_STARTED.value not in {
        event["stage"] for event in _timeline_events(timeline, second.voice_turn_id)
    }
    assert VoiceTimelineStage.SPEAKER_RENDER_STARTED.value not in {
        event["stage"] for event in _timeline_events(timeline, first.voice_turn_id)
    }


def test_trace_adapter_keeps_unknown_stage_legacy_only() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    trace = recorder.start(source="microphone", media_transport="sounddevice")

    recorder.mark("experimental_private_stage", text="private payload")

    stages = {stage["name"] for stage in recorder.snapshot()["current"]["stages"]}
    events = _timeline_events(timeline, trace.voice_turn_id)

    assert "experimental_private_stage" in stages
    assert [event["stage"] for event in events] == [VoiceTimelineStage.LISTEN_STARTED.value]


def test_trace_adapter_isolates_timeline_failures_from_legacy_trace() -> None:
    class FailingTimeline:
        def record(self, event: object) -> None:
            del event
            raise RuntimeError("private exporter failure detail")

    recorder = VoiceTurnTraceRecorder(timeline=FailingTimeline())  # type: ignore[arg-type]

    trace = recorder.start(source="microphone", media_transport="sounddevice")
    recorder.mark("asr_final", text="private transcript")
    recorder.finish("accepted")

    snapshot = recorder.snapshot()

    assert trace.voice_turn_id == snapshot["latest"]["voice_turn_id"]
    assert {stage["name"] for stage in snapshot["latest"]["stages"]} >= {
        "listen_started",
        "asr_final",
    }
    assert snapshot["counters"]["timeline_status"] == "degraded"
    assert snapshot["counters"]["timeline_error_count"] == 3
    assert snapshot["counters"]["last_timeline_error_type"] == "RuntimeError"
    assert "private exporter failure detail" not in repr(snapshot["counters"])


def test_trace_adapter_serializes_concurrent_start_mark_and_snapshot() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)

    def exercise(index: int) -> str:
        trace = recorder.start(source="microphone", media_transport="sounddevice")
        recorder.mark_for(
            trace.voice_turn_id,
            "first_audio_frame",
            frame_samples=160 + index,
        )
        recorder.snapshot()
        return trace.voice_turn_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        voice_turn_ids = list(executor.map(exercise, range(40)))

    assert len(set(voice_turn_ids)) == 40
    assert recorder.snapshot()["counters"]["timeline_error_count"] == 0
    for voice_turn_id in voice_turn_ids:
        assert {event["stage"] for event in _timeline_events(timeline, voice_turn_id)} >= {
            VoiceTimelineStage.LISTEN_STARTED.value,
            VoiceTimelineStage.FIRST_AUDIO_FRAME.value,
        }
