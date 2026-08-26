from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import askme.providers.telemetry.otel_voice_timeline as otel_timeline
from askme.providers.telemetry.otel_voice_timeline import (
    OpenTelemetryApiUnavailableError,
    OpenTelemetryVoiceTimelineExporter,
    VoiceTimelineExportPrivacyError,
)
from askme.voice.core.turn_timeline import (
    VoiceTimelineRecord,
    VoiceTimelineScope,
    VoiceTimelineStage,
)


@dataclass
class FakeStatus:
    status_code: object


class FakeSpan:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.statuses: list[object] = []
        self.end_times: list[int | None] = []
        self.add_event_error: Exception | None = None
        self.end_error: Exception | None = None

    def add_event(
        self,
        name: str,
        *,
        attributes: dict[str, object],
        timestamp: int,
    ) -> None:
        if self.add_event_error is not None:
            raise self.add_event_error
        self.events.append(
            {
                "name": name,
                "attributes": attributes,
                "timestamp": timestamp,
            }
        )

    def set_status(self, status: object) -> None:
        self.statuses.append(status)

    def end(self, *, end_time: int | None = None) -> None:
        self.end_times.append(end_time)
        if self.end_error is not None:
            raise self.end_error


class FakeTracer:
    def __init__(self, span: FakeSpan | None = None) -> None:
        self.span = span or FakeSpan()
        self.starts: list[dict[str, Any]] = []
        self.start_error: Exception | None = None

    def start_span(self, name: str, **kwargs: object) -> FakeSpan:
        if self.start_error is not None:
            raise self.start_error
        self.starts.append({"name": name, **kwargs})
        return self.span


def make_record(
    *,
    stage: VoiceTimelineStage = VoiceTimelineStage.FIRST_AUDIO_FRAME,
    attributes: dict[str, object] | None = None,
    scope: VoiceTimelineScope | None = None,
) -> VoiceTimelineRecord:
    return VoiceTimelineRecord(
        sequence=7,
        event_id="event-7",
        stage=stage,
        scope=scope
        or VoiceTimelineScope(
            voice_turn_id="voice-turn-7",
            thread_id="thread-7",
            turn_id="turn-7",
            generation_id="generation-7",
            provider_session_id="provider-session-7",
            trace_id="upstream-trace-7",
        ),
        attributes=attributes or {"source": "microphone", "duration_ms": 12.5},
        recorded_at_epoch_s=1_234.567_89,
        recorded_at_monotonic_s=99.0,
        payload_hash="a" * 64,
    )


def test_offer_creates_timestamped_observation_with_namespaced_safe_attributes() -> None:
    tracer = FakeTracer()
    exporter = OpenTelemetryVoiceTimelineExporter(tracer)

    exporter.offer(make_record())

    timestamp_ns = 1_234_567_890_000
    assert tracer.starts == [
        {
            "name": "askme.voice.timeline.observation",
            "attributes": {
                "askme.voice.timeline.sequence": 7,
                "askme.voice.timeline.event_id": "event-7",
                "askme.voice.timeline.stage": "first_audio_frame",
                "askme.voice.scope.voice_turn_id": "voice-turn-7",
                "askme.voice.scope.thread_id": "thread-7",
                "askme.voice.scope.turn_id": "turn-7",
                "askme.voice.scope.generation_id": "generation-7",
                "askme.voice.scope.provider_session_id": "provider-session-7",
                "askme.voice.scope.trace_id": "upstream-trace-7",
                "askme.voice.attribute.source": "microphone",
                "askme.voice.attribute.duration_ms": 12.5,
            },
            "start_time": timestamp_ns,
        }
    ]
    assert tracer.span.events == [
        {
            "name": "askme.voice.timeline.stage.first_audio_frame",
            "attributes": {
                "askme.voice.timeline.sequence": 7,
                "askme.voice.timeline.event_id": "event-7",
                "askme.voice.timeline.stage": "first_audio_frame",
                "askme.voice.scope.voice_turn_id": "voice-turn-7",
                "askme.voice.scope.thread_id": "thread-7",
                "askme.voice.scope.turn_id": "turn-7",
                "askme.voice.scope.generation_id": "generation-7",
                "askme.voice.scope.provider_session_id": "provider-session-7",
                "askme.voice.scope.trace_id": "upstream-trace-7",
                "askme.voice.attribute.source": "microphone",
                "askme.voice.attribute.duration_ms": 12.5,
            },
            "timestamp": timestamp_ns,
        }
    ]
    assert tracer.span.end_times == [timestamp_ns]
    assert tracer.span.statuses == []
    # The correlation trace ID is evidence only; it was not forged into an OTel parent/context.
    assert "context" not in tracer.starts[0]
    assert "parent" not in tracer.starts[0]


def test_error_stage_marks_only_that_observation_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error_code = object()
    monkeypatch.setattr(
        otel_timeline,
        "_load_error_status",
        lambda: FakeStatus(status_code=error_code),
    )
    error_tracer = FakeTracer()
    normal_tracer = FakeTracer()

    OpenTelemetryVoiceTimelineExporter(error_tracer).offer(
        make_record(stage=VoiceTimelineStage.ERROR, attributes={"error_type": "TimeoutError"})
    )
    OpenTelemetryVoiceTimelineExporter(normal_tracer).offer(make_record())

    assert error_tracer.span.statuses == [FakeStatus(status_code=error_code)]
    assert normal_tracer.span.statuses == []


@pytest.mark.parametrize(
    "unsafe_attributes",
    [
        {"transcript": "raw user words"},
        {"audio": b"\x00\x01"},
        {"error_message": "secret provider payload"},
        {"arbitrary": "not explicitly allowed"},
        {"source": ["microphone"]},
    ],
)
def test_privacy_policy_rejects_non_allowlisted_or_nonscalar_attributes(
    unsafe_attributes: dict[str, object],
) -> None:
    tracer = FakeTracer()

    with pytest.raises(VoiceTimelineExportPrivacyError, match="privacy policy"):
        OpenTelemetryVoiceTimelineExporter(tracer).offer(
            make_record(attributes=unsafe_attributes)
        )

    assert tracer.starts == []


def test_privacy_policy_rejects_unsafe_scope_before_touching_tracer() -> None:
    tracer = FakeTracer()
    scope = VoiceTimelineScope(voice_turn_id="raw words are not an identity")

    with pytest.raises(VoiceTimelineExportPrivacyError, match="privacy policy"):
        OpenTelemetryVoiceTimelineExporter(tracer).offer(make_record(scope=scope))

    assert tracer.starts == []


@pytest.mark.parametrize("failure_point", ["start", "event", "end"])
def test_tracer_failures_propagate_to_timeline_health_accounting(failure_point: str) -> None:
    tracer = FakeTracer()
    expected = RuntimeError(f"{failure_point} failed")
    if failure_point == "start":
        tracer.start_error = expected
    elif failure_point == "event":
        tracer.span.add_event_error = expected
    else:
        tracer.span.end_error = expected

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        OpenTelemetryVoiceTimelineExporter(tracer).offer(make_record())


def test_missing_opentelemetry_api_is_reported_as_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise OpenTelemetryApiUnavailableError(
            "OpenTelemetry trace API is required to mark ERROR observations"
        )

    monkeypatch.setattr(otel_timeline, "_load_error_status", unavailable)
    tracer = FakeTracer()

    with pytest.raises(OpenTelemetryApiUnavailableError, match="trace API"):
        OpenTelemetryVoiceTimelineExporter(tracer).offer(
            make_record(stage=VoiceTimelineStage.ERROR, attributes={"error_type": "TimeoutError"})
        )

    assert tracer.span.end_times == [1_234_567_890_000]
