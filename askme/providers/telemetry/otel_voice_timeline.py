"""Optional OpenTelemetry adapter for privacy-safe voice-turn observations.

This module deliberately owns neither an OpenTelemetry SDK nor its lifecycle.
Applications provide an already configured tracer; batching, flushing, and
shutdown remain composition-root responsibilities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Final, TypeGuard

from askme.voice.core.turn_timeline import (
    MAX_ATTRIBUTE_COUNT,
    MAX_ATTRIBUTE_STRING_LENGTH,
    MAX_IDENTITY_LENGTH,
    VOICE_TIMELINE_ATTRIBUTE_ALLOWLIST,
    VoiceTimelineRecord,
    VoiceTimelineStage,
)

_SPAN_NAME: Final = "askme.voice.timeline.observation"
_EVENT_NAME_PREFIX: Final = "askme.voice.timeline.stage."
_IDENTITY_PATTERN: Final = re.compile(r"[A-Za-z0-9._~:/@+=-]{1,256}")
_SCOPE_ATTRIBUTE_NAMES: Final = {
    "voice_turn_id": "askme.voice.scope.voice_turn_id",
    "thread_id": "askme.voice.scope.thread_id",
    "turn_id": "askme.voice.scope.turn_id",
    "generation_id": "askme.voice.scope.generation_id",
    "provider_session_id": "askme.voice.scope.provider_session_id",
    "trace_id": "askme.voice.scope.trace_id",
}


class OpenTelemetryApiUnavailableError(RuntimeError):
    """The optional OpenTelemetry trace API needed by an event is unavailable."""


class VoiceTimelineExportPrivacyError(ValueError):
    """A record did not satisfy the export adapter's privacy contract."""

    def __init__(self) -> None:
        # Keep the exception static: rejected field names and values may themselves be private.
        super().__init__("voice timeline record rejected by export privacy policy")


class OpenTelemetryVoiceTimelineExporter:
    """Export one timeline record as a short, non-parenting observation span.

    ``tracer`` is intentionally duck typed so this optional adapter stays importable
    when the OpenTelemetry API is not installed.  A real deployment should pass a
    tracer obtained from its composition-owned ``TracerProvider``.
    """

    def __init__(self, tracer: object) -> None:
        if not callable(getattr(tracer, "start_span", None)):
            raise TypeError("tracer must provide start_span()")
        self._tracer: Any = tracer

    def offer(self, record: VoiceTimelineRecord) -> None:
        """Synchronously offer one observation and propagate adapter failures."""

        stage = VoiceTimelineStage(record.stage)
        attributes = _span_attributes(record, stage=stage)
        timestamp_ns = _epoch_nanoseconds(record.recorded_at_epoch_s)

        # Do not pass a context or parent.  scope.trace_id is correlation evidence,
        # not an OpenTelemetry trace context, and forging it would break causality.
        span = self._tracer.start_span(
            _SPAN_NAME,
            attributes=dict(attributes),
            start_time=timestamp_ns,
        )
        try:
            span.add_event(
                f"{_EVENT_NAME_PREFIX}{stage.value}",
                attributes=dict(attributes),
                timestamp=timestamp_ns,
            )
            if stage is VoiceTimelineStage.ERROR:
                span.set_status(_load_error_status())
        finally:
            # A zero-duration observation keeps queued export latency out of the span.
            span.end(end_time=timestamp_ns)


def _span_attributes(
    record: VoiceTimelineRecord,
    *,
    stage: VoiceTimelineStage,
) -> dict[str, str | bool | int | float]:
    if (
        isinstance(record.sequence, bool)
        or not isinstance(record.sequence, int)
        or record.sequence < 1
        or not _safe_identity(record.event_id)
    ):
        raise VoiceTimelineExportPrivacyError

    attributes: dict[str, str | bool | int | float] = {
        "askme.voice.timeline.sequence": record.sequence,
        "askme.voice.timeline.event_id": record.event_id,
        "askme.voice.timeline.stage": stage.value,
    }
    for field_name, attribute_name in _SCOPE_ATTRIBUTE_NAMES.items():
        value = getattr(record.scope, field_name, None)
        if value is None and field_name != "voice_turn_id":
            continue
        if not _safe_identity(value):
            raise VoiceTimelineExportPrivacyError
        attributes[attribute_name] = value

    source_attributes = record.attributes
    if not isinstance(source_attributes, Mapping) or len(source_attributes) > MAX_ATTRIBUTE_COUNT:
        raise VoiceTimelineExportPrivacyError
    for key, value in source_attributes.items():
        if not isinstance(key, str) or key not in VOICE_TIMELINE_ATTRIBUTE_ALLOWLIST:
            raise VoiceTimelineExportPrivacyError
        if value is None:
            # None is accepted by the local evidence model but is not an OTel attribute value.
            continue
        if not _safe_attribute_value(value):
            raise VoiceTimelineExportPrivacyError
        attributes[f"askme.voice.attribute.{key}"] = value
    return attributes


def _safe_identity(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) <= MAX_IDENTITY_LENGTH
        and _IDENTITY_PATTERN.fullmatch(value) is not None
    )


def _safe_attribute_value(value: object) -> TypeGuard[str | bool | int | float]:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return len(value) <= MAX_ATTRIBUTE_STRING_LENGTH
    if isinstance(value, int):
        return -(2**63) <= value <= 2**63 - 1
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _epoch_nanoseconds(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("voice timeline recorded_at_epoch_s must be finite and non-negative")
    return int(round(float(value) * 1_000_000_000))


def _load_error_status() -> object:
    try:
        from opentelemetry.trace import Status, StatusCode
    except (ImportError, AttributeError) as exc:
        raise OpenTelemetryApiUnavailableError(
            "OpenTelemetry trace API is required to mark ERROR observations"
        ) from exc
    return Status(StatusCode.ERROR)


__all__ = [
    "OpenTelemetryApiUnavailableError",
    "OpenTelemetryVoiceTimelineExporter",
    "VoiceTimelineExportPrivacyError",
]
