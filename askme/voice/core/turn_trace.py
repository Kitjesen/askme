"""Structured voice-turn trace helpers.

The realtime media layer is evolving from a local microphone loop toward
pluggable transports such as WebRTC/LiveKit.  This module records media and
turn-detection milestones without depending on a specific transport.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4

from askme.voice.core.turn_timeline import (
    TimelineConflict,
    TimelineRecordStatus,
    VoiceTimelineEventInput,
    VoiceTimelineScope,
    VoiceTimelineStage,
    VoiceTurnTimeline,
)

# These buckets are retained for API compatibility. They are absolute offsets
# from listen_started and therefore are not speech-end latency or stage duration.
LEGACY_LATENCY_BUCKET_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "mic_first_frame_ms": ("first_audio_frame", "mic_first_frame"),
    "vad_start_ms": ("vad_start", "speech_start"),
    "vad_end_ms": ("vad_end", "speech_end"),
    "asr_first_partial_ms": (
        "asr_first_partial",
        "first_asr_partial",
        "asr_partial",
    ),
    "asr_final_ms": ("asr_final", "asr_done"),
    "intent_route_ms": ("intent_route", "intent_routed", "route_intent"),
    "llm_ttft_ms": ("llm_ttft", "llm_first_token", "llm_first_delta"),
    "llm_done_ms": ("llm_done", "llm_completed"),
    "tts_first_audio_ms": (
        "tts_first_audio",
        "tts_first_audio_chunk",
        "tts_audio_first_chunk",
    ),
    "playback_start_ms": ("playback_start", "tts_playback_started"),
    "playback_done_ms": ("playback_done", "tts_playback_done"),
}
# Public compatibility alias retained for existing importers.
LATENCY_BUCKET_STAGE_ALIASES = LEGACY_LATENCY_BUCKET_STAGE_ALIASES

# Customer latency is derived only from two explicitly named events.
# barge_in_confirmed is a reference event, never proof of speaker stop.
DERIVED_LATENCY_BUCKET_EVENTS: dict[
    str,
    tuple[tuple[str, ...], tuple[str, ...], str | None],
] = {
    "speech_end_to_endpoint_commit_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("endpoint_committed", "fast_endpoint_committed"),
        None,
    ),
    "speech_end_to_asr_final_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("asr_final", "asr_done"),
        None,
    ),
    "speech_end_to_turn_admitted_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("interaction_admitted", "turn_admitted"),
        None,
    ),
    "speech_end_to_ack_render_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("ack_render_started",),
        "ack",
    ),
    "speech_end_to_ack_physical_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("ack_physical_started",),
        "ack",
    ),
    "speech_end_to_feedback_render_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("feedback_render_started", "processing_feedback_render_started"),
        "feedback",
    ),
    "speech_end_to_feedback_physical_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("feedback_physical_started", "processing_feedback_physical_started"),
        "feedback",
    ),
    "speech_end_to_first_llm_payload_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("llm_first_payload",),
        None,
    ),
    "speech_end_to_first_llm_semantic_text_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("llm_first_semantic_text",),
        None,
    ),
    "speech_end_to_tts_first_pcm_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("tts_first_semantic_pcm", "tts_first_pcm"),
        "semantic",
    ),
    "speech_end_to_render_first_semantic_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("render_first_semantic_nonzero",),
        "semantic",
    ),
    "speech_end_to_physical_first_semantic_audio_ms": (
        ("speech_last_active_sample", "speech_end"),
        ("physical_first_semantic_audio",),
        "semantic",
    ),
    "barge_in_to_render_stop_ms": (
        ("barge_in_confirmed",),
        ("render_speaker_stopped", "speaker_render_stopped"),
        None,
    ),
    "barge_in_to_physical_speaker_stop_ms": (
        ("barge_in_confirmed",),
        ("physical_speaker_stopped",),
        None,
    ),
    # Compatibility name: now requires a real render or physical stop event.
    "barge_in_stop_ms": (
        ("barge_in_confirmed",),
        (
            "render_speaker_stopped",
            "speaker_render_stopped",
            "physical_speaker_stopped",
        ),
        None,
    ),
}

AUDIO_CLASSES = frozenset({"ack", "feedback", "semantic", "safety", "error"})
PHYSICAL_PROVENANCE_BUCKETS = frozenset(
    {
        "speech_end_to_ack_physical_ms",
        "speech_end_to_feedback_physical_ms",
        "speech_end_to_physical_first_semantic_audio_ms",
        "barge_in_to_physical_speaker_stop_ms",
    }
)
LATENCY_BUCKET_NAMES: tuple[str, ...] = (
    *LEGACY_LATENCY_BUCKET_STAGE_ALIASES,
    *DERIVED_LATENCY_BUCKET_EVENTS,
)
_SUMMARY_WINDOW_SIZE = 300
_SAFE_TIMELINE_TOKEN = re.compile(r"[A-Za-z0-9._~:/@+=-]{1,256}")
_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_.]{0,127}")
_TIMELINE_EVENT_ID_PREFIX = "legacy-trace:v1"
_EXPLICIT_TURN_TIMELINE_STAGES = frozenset(
    {
        "turn_correlated",
        "interaction_admitted",
        "turn_admitted",
        "llm_requested",
        "llm_first_payload",
        "llm_first_token",
        "llm_first_delta",
        "llm_first_semantic_text",
        "first_clause",
        "tts_first_semantic_pcm",
        "tts_first_pcm",
        "tts_first_audio",
        "tts_first_audio_chunk",
        "tts_playback_started",
        "playback_start",
        "ack_render_started",
        "feedback_render_started",
        "processing_feedback_render_started",
        "render_first_semantic_nonzero",
        "ack_physical_started",
        "feedback_physical_started",
        "processing_feedback_physical_started",
        "physical_first_semantic_audio",
        "barge_in_detected",
        "barge_in_confirmed",
        "barge_in_dismissed",
        "barge_in_recovered",
        "render_speaker_stopped",
        "speaker_render_stopped",
        "playback_done",
        "tts_playback_done",
        "physical_speaker_stopped",
        "fallback_selected",
    }
)
_LEGACY_TURN_SNAPSHOT_FIELDS = frozenset({"asr_source"})
_NO_LEGACY_SNAPSHOT_FIELDS: frozenset[str] = frozenset()
_TRACE_SCOPE_METADATA_FIELDS = ("thread_id", "turn_id", "trace_id")
_PHYSICAL_PROVENANCE_STAGE_NAMES = frozenset(
    {
        "ack_physical_started",
        "feedback_physical_started",
        "processing_feedback_physical_started",
        "physical_first_semantic_audio",
        "physical_speaker_stopped",
    }
)
_LEGACY_STAGE_SNAPSHOT_FIELDS: dict[str, frozenset[str]] = {
    "first_audio_frame": frozenset({"chunk_samples"}),
    "mic_first_frame": frozenset({"chunk_samples"}),
    "vad_start": frozenset({"peak", "rms"}),
    "speech_start": frozenset({"peak", "rms"}),
    "vad_end": frozenset({"peak", "rms"}),
    "speech_end": frozenset({"peak", "rms"}),
    "barge_in_confirmed": frozenset({"peak", "rms"}),
    "asr_final": frozenset({"asr_source"}),
    "asr_done": frozenset({"asr_source"}),
}
_SAFE_ASR_SOURCE_LABELS = frozenset({"cloud", "cloud_partial", "local", "local_partial"})
MIN_P95_SAMPLES = 100
MIN_P99_SAMPLES = 300
DEFAULT_VOICE_TURN_SLO_MS: dict[str, float] = {
    "speech_end_to_endpoint_commit_ms": 400.0,
    "speech_end_to_asr_final_ms": 1200.0,
    "speech_end_to_ack_physical_ms": 600.0,
    "speech_end_to_feedback_physical_ms": 900.0,
    "speech_end_to_first_llm_payload_ms": 1800.0,
    "speech_end_to_tts_first_pcm_ms": 1800.0,
    "speech_end_to_render_first_semantic_ms": 1800.0,
    "speech_end_to_physical_first_semantic_audio_ms": 1800.0,
    "barge_in_to_render_stop_ms": 250.0,
    "barge_in_to_physical_speaker_stop_ms": 250.0,
    "barge_in_stop_ms": 250.0,
}
DEFAULT_REQUIRED_VOICE_TURN_BUCKETS: tuple[str, ...] = (
    "speech_end_to_endpoint_commit_ms",
    "speech_end_to_asr_final_ms",
    "speech_end_to_first_llm_payload_ms",
    "speech_end_to_tts_first_pcm_ms",
    "speech_end_to_render_first_semantic_ms",
    "speech_end_to_physical_first_semantic_audio_ms",
)
DEFAULT_REQUIRED_PHYSICAL_PROVENANCE_BUCKETS: tuple[str, ...] = (
    "speech_end_to_ack_physical_ms",
    "speech_end_to_feedback_physical_ms",
    "speech_end_to_physical_first_semantic_audio_ms",
    "barge_in_to_physical_speaker_stop_ms",
)

_TIMELINE_STAGE_BY_LEGACY_NAME: dict[str, VoiceTimelineStage] = {
    "listen_started": VoiceTimelineStage.LISTEN_STARTED,
    "first_audio_frame": VoiceTimelineStage.FIRST_AUDIO_FRAME,
    "mic_first_frame": VoiceTimelineStage.FIRST_AUDIO_FRAME,
    "vad_start": VoiceTimelineStage.SPEECH_START,
    "speech_start": VoiceTimelineStage.SPEECH_START,
    "vad_end": VoiceTimelineStage.SPEECH_END,
    "speech_end": VoiceTimelineStage.SPEECH_END,
    "speech_last_active_sample": VoiceTimelineStage.SPEECH_END,
    "endpoint_committed": VoiceTimelineStage.ENDPOINT_COMMITTED,
    "fast_endpoint_committed": VoiceTimelineStage.ENDPOINT_COMMITTED,
    "asr_final": VoiceTimelineStage.ASR_FINAL,
    "asr_done": VoiceTimelineStage.ASR_FINAL,
    "turn_correlated": VoiceTimelineStage.TURN_CORRELATED,
    "interaction_admitted": VoiceTimelineStage.TURN_ADMITTED,
    "turn_admitted": VoiceTimelineStage.TURN_ADMITTED,
    "llm_requested": VoiceTimelineStage.LLM_REQUESTED,
    "llm_first_payload": VoiceTimelineStage.FIRST_LLM_PAYLOAD,
    "llm_first_token": VoiceTimelineStage.FIRST_LLM_PAYLOAD,
    "llm_first_delta": VoiceTimelineStage.FIRST_LLM_PAYLOAD,
    "llm_first_semantic_text": VoiceTimelineStage.FIRST_SEMANTIC,
    "first_clause": VoiceTimelineStage.FIRST_CLAUSE,
    "tts_first_semantic_pcm": VoiceTimelineStage.TTS_FIRST_PCM,
    "tts_first_pcm": VoiceTimelineStage.TTS_FIRST_PCM,
    "tts_first_audio": VoiceTimelineStage.TTS_FIRST_PCM,
    "tts_first_audio_chunk": VoiceTimelineStage.TTS_FIRST_PCM,
    "tts_playback_started": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "playback_start": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "ack_render_started": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "feedback_render_started": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "processing_feedback_render_started": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "render_first_semantic_nonzero": VoiceTimelineStage.SPEAKER_RENDER_STARTED,
    "ack_physical_started": VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED,
    "feedback_physical_started": VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED,
    "processing_feedback_physical_started": VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED,
    "physical_first_semantic_audio": VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED,
    "barge_in_detected": VoiceTimelineStage.INTERRUPT_DETECTED,
    "barge_in_confirmed": VoiceTimelineStage.INTERRUPT_CONFIRMED,
    "barge_in_dismissed": VoiceTimelineStage.INTERRUPT_DISMISSED,
    "barge_in_recovered": VoiceTimelineStage.INTERRUPT_DISMISSED,
    "render_speaker_stopped": VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
    "speaker_render_stopped": VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
    "playback_done": VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
    "tts_playback_done": VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
    "physical_speaker_stopped": VoiceTimelineStage.SPEAKER_PHYSICAL_STOPPED,
    "fallback_selected": VoiceTimelineStage.FALLBACK_SELECTED,
}

_SAFE_TRACE_STATUSES = frozenset(
    {
        "accepted",
        "active",
        "cancelled",
        "completed",
        "empty",
        "error",
        "failed",
        "forced_empty",
        "interrupted",
        "noise_filtered",
        "stopped",
        "superseded",
        "timeout",
        "unavailable",
        "wake_word_not_detected",
    }
)


def _round_ms(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low_index = int(rank)
    high_index = min(low_index + 1, len(ordered) - 1)
    fraction = rank - low_index
    return ordered[low_index] + (ordered[high_index] - ordered[low_index]) * fraction


@dataclass
class VoiceTraceStage:
    """One named milestone inside a voice turn."""

    name: str
    offset_ms: float
    audio_class: str | None = None
    audio_segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": _safe_timeline_token(self.name) or "legacy_stage",
            "offset_ms": round(max(self.offset_ms, 0.0), 2),
            "audio_class": self.audio_class,
            "audio_segment_id": _safe_timeline_token(self.audio_segment_id),
            "metadata": _safe_legacy_snapshot_metadata(
                self.metadata,
                allowed_fields=_LEGACY_STAGE_SNAPSHOT_FIELDS.get(
                    self.name,
                    _NO_LEGACY_SNAPSHOT_FIELDS,
                ),
            ),
        }


@dataclass
class VoiceTurnTrace:
    """Trace for a single operator voice turn."""

    voice_turn_id: str
    source: str
    media_transport: str
    started_at_epoch_s: float
    started_at_monotonic_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, VoiceTraceStage] = field(default_factory=dict)
    ended_at_epoch_s: float | None = None
    total_ms: float | None = None
    status: str = "active"

    @classmethod
    def start(
        cls,
        *,
        source: str,
        media_transport: str,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceTurnTrace:
        return cls(
            voice_turn_id=uuid4().hex,
            source=_safe_timeline_token(source) or "unknown",
            media_transport=_safe_timeline_token(media_transport) or "unknown",
            started_at_epoch_s=time.time(),
            started_at_monotonic_s=time.monotonic(),
            metadata=_safe_trace_turn_metadata(
                metadata or {},
                include_scope=True,
            ),
        )

    def mark(self, name: str, **metadata: Any) -> None:
        """Record the first occurrence of a stage."""
        stage_name = _safe_timeline_token(name)
        if stage_name is None or stage_name in self.stages:
            return
        raw_audio_class = metadata.pop("audio_class", None)
        audio_class = (
            str(raw_audio_class).strip().lower()
            if raw_audio_class is not None
            else _default_audio_class(stage_name)
        )
        if audio_class is not None and audio_class not in AUDIO_CLASSES:
            raise ValueError(
                f"audio_class must be one of {sorted(AUDIO_CLASSES)}, got {audio_class!r}"
            )
        raw_segment_id = metadata.pop(
            "audio_segment_id",
            metadata.pop("segment_id", None),
        )
        audio_segment_id = _safe_timeline_token(raw_segment_id)
        offset_ms = (time.monotonic() - self.started_at_monotonic_s) * 1000.0
        self.stages[stage_name] = VoiceTraceStage(
            name=stage_name,
            offset_ms=offset_ms,
            audio_class=audio_class,
            audio_segment_id=audio_segment_id,
            metadata=_safe_trace_stage_metadata(stage_name, metadata),
        )

    def _find_stage(
        self,
        aliases: tuple[str, ...],
        *,
        required_audio_class: str | None = None,
    ) -> VoiceTraceStage | None:
        for stage_name in aliases:
            stage = self.stages.get(stage_name)
            if stage is None:
                continue
            if required_audio_class is not None and stage.audio_class != required_audio_class:
                continue
            return stage
        return None

    def legacy_latency_buckets(self) -> dict[str, float | None]:
        """Return compatibility offsets measured from listen start."""

        buckets: dict[str, float | None] = {}
        for bucket_name, stage_aliases in LEGACY_LATENCY_BUCKET_STAGE_ALIASES.items():
            stage = self._find_stage(stage_aliases)
            buckets[bucket_name] = _round_ms(stage.offset_ms) if stage is not None else None
        return buckets

    def derived_latency_buckets(self) -> dict[str, float | None]:
        """Return event-to-event latency using explicit reference milestones."""

        buckets: dict[str, float | None] = {}
        for bucket_name, (
            reference_aliases,
            event_aliases,
            required_audio_class,
        ) in DERIVED_LATENCY_BUCKET_EVENTS.items():
            reference = self._find_stage(reference_aliases)
            event = self._find_stage(
                event_aliases,
                required_audio_class=required_audio_class,
            )
            latency_ms = None
            if reference is not None and event is not None:
                delta_ms = event.offset_ms - reference.offset_ms
                if delta_ms >= 0.0:
                    latency_ms = _round_ms(delta_ms)
            buckets[bucket_name] = latency_ms
        return buckets

    def latency_bucket_provenance(self) -> dict[str, dict[str, Any]]:
        """Describe the events and evidence behind every derived bucket."""

        provenance: dict[str, dict[str, Any]] = {}
        for bucket_name, (
            reference_aliases,
            event_aliases,
            required_audio_class,
        ) in DERIVED_LATENCY_BUCKET_EVENTS.items():
            reference = self._find_stage(reference_aliases)
            event = self._find_stage(
                event_aliases,
                required_audio_class=required_audio_class,
            )
            requires_physical = bucket_name in PHYSICAL_PROVENANCE_BUCKETS
            physical_valid = _physical_provenance_valid(event) if requires_physical else None
            provenance[bucket_name] = {
                "reference_event": reference.name if reference is not None else None,
                "event": event.name if event is not None else None,
                "audio_class": event.audio_class if event is not None else None,
                "audio_segment_id": (event.audio_segment_id if event is not None else None),
                "requires_physical_provenance": requires_physical,
                "physical_provenance_valid": physical_valid,
                "evidence_kind": (
                    _safe_timeline_token(event.metadata.get("evidence_kind"))
                    if event is not None
                    else None
                ),
                "clock_id": (
                    _safe_timeline_token(event.metadata.get("clock_id"))
                    if event is not None
                    else None
                ),
            }
        return provenance

    def latency_buckets(self) -> dict[str, float | None]:
        """Return compatibility offsets plus explicit event-delta buckets.

        ACK and processing feedback have dedicated buckets and can never
        satisfy a semantic-audio bucket.
        """

        return {
            **self.legacy_latency_buckets(),
            **self.derived_latency_buckets(),
        }

    def missing_latency_buckets(self) -> list[str]:
        return [
            bucket_name
            for bucket_name, latency_ms in self.latency_buckets().items()
            if latency_ms is None
        ]

    def finish(self, status: str = "completed", **metadata: Any) -> None:
        if self.ended_at_epoch_s is not None:
            return
        self.status = _safe_trace_status(status)
        self.ended_at_epoch_s = time.time()
        self.total_ms = (time.monotonic() - self.started_at_monotonic_s) * 1000.0
        self.metadata.update(_safe_trace_turn_metadata(metadata, include_scope=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_turn_id": self.voice_turn_id,
            "source": _safe_timeline_token(self.source) or "unknown",
            "media_transport": _safe_timeline_token(self.media_transport) or "unknown",
            "status": _safe_trace_status(self.status),
            "started_at_epoch_s": round(self.started_at_epoch_s, 3),
            "ended_at_epoch_s": (
                round(self.ended_at_epoch_s, 3) if self.ended_at_epoch_s is not None else None
            ),
            "total_ms": round(self.total_ms, 2) if self.total_ms is not None else None,
            "stages": [stage.to_dict() for stage in self.stages.values()],
            "legacy_latency_buckets": self.legacy_latency_buckets(),
            "derived_latency_buckets": self.derived_latency_buckets(),
            "latency_buckets": self.latency_buckets(),
            "latency_bucket_provenance": self.latency_bucket_provenance(),
            "missing_latency_buckets": self.missing_latency_buckets(),
            "metadata": _safe_legacy_snapshot_metadata(
                self.metadata,
                allowed_fields=_LEGACY_TURN_SNAPSHOT_FIELDS,
            ),
        }


class VoiceTurnTraceRecorder:
    """Tracks the active and latest completed voice-turn traces."""

    def __init__(self, *, timeline: VoiceTurnTimeline | None = None) -> None:
        self._lock = RLock()
        self._current: VoiceTurnTrace | None = None
        self._latest: VoiceTurnTrace | None = None
        self._history: list[VoiceTurnTrace] = []
        self._barge_in_count = 0
        self._timeline = timeline
        self._timeline_status = "disabled" if timeline is None else "healthy"
        self._timeline_error_count = 0
        self._timeline_conflict_count = 0
        self._timeline_ambiguous_event_count = 0
        self._last_timeline_error_type: str | None = None

    @staticmethod
    def bucket_names() -> tuple[str, ...]:
        return LATENCY_BUCKET_NAMES

    @staticmethod
    def default_slo_ms() -> dict[str, float]:
        return dict(DEFAULT_VOICE_TURN_SLO_MS)

    def _remember(self, trace: VoiceTurnTrace) -> None:
        if trace in self._history:
            return
        self._history.append(trace)
        del self._history[:-_SUMMARY_WINDOW_SIZE]

    def _find_trace(self, voice_turn_id: str) -> VoiceTurnTrace | None:
        if self._current is not None and self._current.voice_turn_id == voice_turn_id:
            return self._current
        if self._latest is not None and self._latest.voice_turn_id == voice_turn_id:
            return self._latest
        return next(
            (trace for trace in reversed(self._history) if trace.voice_turn_id == voice_turn_id),
            None,
        )

    def _record_timeline_event(
        self,
        trace: VoiceTurnTrace,
        *,
        event_name: str,
        stage: VoiceTimelineStage,
        metadata: Mapping[str, Any],
        event_suffix: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        safe_attributes = (
            dict(attributes)
            if attributes is not None
            else _timeline_attributes(stage, trace, metadata)
        )
        suffix = event_suffix or event_name
        event = VoiceTimelineEventInput(
            event_id=(f"{_TIMELINE_EVENT_ID_PREFIX}:{trace.voice_turn_id}:{suffix}"),
            stage=stage,
            scope=_timeline_scope(trace, metadata),
            attributes=safe_attributes,
        )
        try:
            receipt = timeline.record(event)
            if receipt.status is TimelineRecordStatus.DROPPED_PRIVACY:
                self._mark_timeline_degraded("TimelinePrivacyDrop")
            elif receipt.status is TimelineRecordStatus.DEGRADED_PERSISTENCE:
                self._mark_timeline_degraded("TimelinePersistenceDegraded")
        except TimelineConflict:
            self._timeline_conflict_count += 1
            self._mark_timeline_degraded("TimelineConflict")
        except Exception as exc:
            self._mark_timeline_degraded(type(exc).__name__)

    def _mark_timeline_degraded(self, error_type: str) -> None:
        self._timeline_status = "degraded"
        self._timeline_error_count += 1
        self._last_timeline_error_type = (
            error_type if _safe_timeline_token(error_type) is not None else "TimelineError"
        )

    def _mark_trace(
        self,
        trace: VoiceTurnTrace,
        name: str,
        metadata: Mapping[str, Any],
        *,
        project_timeline: bool = True,
    ) -> bool:
        stage_name = _safe_timeline_token(name)
        if stage_name is None or stage_name in trace.stages:
            return False
        trace.mark(stage_name, **dict(metadata))
        timeline_stage = _TIMELINE_STAGE_BY_LEGACY_NAME.get(stage_name)
        if timeline_stage is not None and project_timeline:
            self._record_timeline_event(
                trace,
                event_name=stage_name,
                stage=timeline_stage,
                metadata=metadata,
            )
        return True

    def _finish_trace(
        self,
        trace: VoiceTurnTrace,
        status: str,
        metadata: Mapping[str, Any],
    ) -> bool:
        if trace.ended_at_epoch_s is not None:
            return False
        trace.finish(status, **dict(metadata))
        safe_status = _safe_trace_status(status)
        self._record_timeline_event(
            trace,
            event_name="finish",
            event_suffix="finish:upstream-closed",
            stage=VoiceTimelineStage.UPSTREAM_CLOSED,
            metadata=metadata,
            attributes={"status": safe_status},
        )
        if safe_status in {"error", "failed"}:
            self._record_timeline_event(
                trace,
                event_name="finish_error",
                event_suffix="finish:error",
                stage=VoiceTimelineStage.ERROR,
                metadata=metadata,
                attributes={"error_type": "LegacyTraceError"},
            )
        return True

    def start(
        self,
        *,
        source: str,
        media_transport: str,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceTurnTrace:
        with self._lock:
            if self._current is not None and self._current.ended_at_epoch_s is None:
                self._finish_trace(self._current, "superseded", {})
                self._latest = self._current
                self._remember(self._current)
            self._current = VoiceTurnTrace.start(
                source=source,
                media_transport=media_transport,
                metadata=metadata,
            )
            self._mark_trace(self._current, "listen_started", metadata or {})
            return self._current

    def mark(self, name: str, **metadata: Any) -> None:
        with self._lock:
            target = self._current or self._latest
            if target is not None:
                stage_name = str(name or "").strip()
                ambiguous = (
                    self._current is not None
                    and self._latest is not None
                    and stage_name in _EXPLICIT_TURN_TIMELINE_STAGES
                )
                if ambiguous:
                    self._timeline_ambiguous_event_count += 1
                self._mark_trace(
                    target,
                    stage_name,
                    metadata,
                    project_timeline=not ambiguous,
                )

    def mark_for(self, voice_turn_id: str, name: str, **metadata: Any) -> bool:
        """Record a stage against an explicit voice turn.

        Playback and provider callbacks may arrive after the next capture turn
        has started.  Explicit routing prevents those events from being
        attributed to whichever trace happens to be current.
        """

        with self._lock:
            target = self._find_trace(str(voice_turn_id or ""))
            return bool(target is not None and self._mark_trace(target, name, metadata))

    def mark_audio(
        self,
        name: str,
        *,
        audio_class: str,
        audio_segment_id: str | None = None,
        **metadata: Any,
    ) -> None:
        """Record a typed audio milestone."""

        self.mark(
            name,
            audio_class=audio_class,
            audio_segment_id=audio_segment_id,
            **metadata,
        )

    def finish(self, status: str = "completed", **metadata: Any) -> None:
        with self._lock:
            if self._current is None:
                return
            self._finish_trace(self._current, status, metadata)
            self._latest = self._current
            self._remember(self._current)
            self._current = None

    def finish_for(
        self,
        voice_turn_id: str,
        status: str = "completed",
        **metadata: Any,
    ) -> bool:
        """Finish an explicit trace without disturbing a newer active trace."""

        with self._lock:
            target = self._find_trace(str(voice_turn_id or ""))
            if target is None or not self._finish_trace(target, status, metadata):
                return False
            self._remember(target)
            if target is self._current:
                self._latest = target
                self._current = None
            elif self._current is None:
                self._latest = target
            return True

    def mark_barge_in(self, **metadata: Any) -> None:
        with self._lock:
            target = self._current or self._latest
            if target is None:
                return
            ambiguous = self._current is not None and self._latest is not None
            if ambiguous:
                self._timeline_ambiguous_event_count += 1
            if self._mark_trace(
                target,
                "barge_in_confirmed",
                metadata,
                project_timeline=not ambiguous,
            ):
                self._barge_in_count += 1

    def mark_barge_in_for(self, voice_turn_id: str, **metadata: Any) -> bool:
        """Record a confirmed interruption against its explicit output turn."""

        with self._lock:
            target = self._find_trace(str(voice_turn_id or ""))
            if target is None or not self._mark_trace(
                target,
                "barge_in_confirmed",
                metadata,
            ):
                return False
            self._barge_in_count += 1
            return True

    def latency_summary(self) -> dict[str, Any]:
        with self._lock:
            history = self._history[-_SUMMARY_WINDOW_SIZE:]
            latest_buckets = self._latest.latency_buckets() if self._latest is not None else {}
            bucket_summaries: dict[str, dict[str, Any]] = {}

            for bucket_name in LATENCY_BUCKET_NAMES:
                values = [
                    latency_ms
                    for trace in history
                    if (latency_ms := trace.latency_buckets().get(bucket_name)) is not None
                ]
                provenance_valid_count = sum(
                    1
                    for trace in history
                    if (
                        trace.latency_buckets().get(bucket_name) is not None
                        and trace.latency_bucket_provenance()
                        .get(bucket_name, {})
                        .get("physical_provenance_valid")
                        is True
                    )
                )
                bucket_summaries[bucket_name] = {
                    "latest_ms": latest_buckets.get(bucket_name),
                    "p50_ms": _round_ms(_percentile(values, 0.50)),
                    "p95_ms": (
                        _round_ms(_percentile(values, 0.95))
                        if len(values) >= MIN_P95_SAMPLES
                        else None
                    ),
                    "p99_ms": (
                        _round_ms(_percentile(values, 0.99))
                        if len(values) >= MIN_P99_SAMPLES
                        else None
                    ),
                    "count": len(values),
                    "missing_count": len(history) - len(values),
                    "physical_provenance_valid_count": provenance_valid_count,
                    "physical_provenance_missing_count": (
                        len(values) - provenance_valid_count
                        if bucket_name in PHYSICAL_PROVENANCE_BUCKETS
                        else 0
                    ),
                    "p95_min_samples": MIN_P95_SAMPLES,
                    "p99_min_samples": MIN_P99_SAMPLES,
                }

            slowest_bucket = max(
                (
                    (bucket_name, data["latest_ms"])
                    for bucket_name, data in bucket_summaries.items()
                    if data["latest_ms"] is not None
                ),
                key=lambda item: item[1],
                default=(None, None),
            )

            return {
                "window_size": len(history),
                "bucket_names": list(LATENCY_BUCKET_NAMES),
                "buckets": bucket_summaries,
                "slowest_bucket": slowest_bucket[0],
                "slowest_bucket_ms": slowest_bucket[1],
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest.to_dict() if self._latest is not None else None
            current = self._current.to_dict() if self._current is not None else None
            summary = self.latency_summary()
            return {
                "current": current,
                "latest": latest,
                "latency_summary": summary,
                "slo": evaluate_voice_turn_slo(
                    latest or current,
                    latency_summary=summary,
                ),
                "counters": {
                    "barge_in_count": self._barge_in_count,
                    "timeline_status": self._timeline_status,
                    "timeline_error_count": self._timeline_error_count,
                    "timeline_conflict_count": self._timeline_conflict_count,
                    "timeline_ambiguous_event_count": (self._timeline_ambiguous_event_count),
                    "last_timeline_error_type": self._last_timeline_error_type,
                },
            }


def _timeline_scope(
    trace: VoiceTurnTrace,
    metadata: Mapping[str, Any],
) -> VoiceTimelineScope:
    def stable_identity(field_name: str) -> str | None:
        return _safe_timeline_token(metadata.get(field_name)) or _safe_timeline_token(
            trace.metadata.get(field_name)
        )

    return VoiceTimelineScope(
        voice_turn_id=trace.voice_turn_id,
        thread_id=stable_identity("thread_id"),
        turn_id=stable_identity("turn_id"),
        generation_id=_safe_timeline_token(metadata.get("generation_id")),
        provider_session_id=_safe_timeline_token(metadata.get("provider_session_id")),
        trace_id=stable_identity("trace_id"),
    )


def _timeline_attributes(
    stage: VoiceTimelineStage,
    trace: VoiceTurnTrace,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Project legacy metadata into a small, stage-specific safe schema."""

    attributes: dict[str, object] = {}

    def token(target: str, *source_names: str) -> None:
        value = _first_safe_token(metadata, source_names)
        if value is not None:
            attributes[target] = value

    def number(target: str, *source_names: str) -> None:
        value = _first_finite_number(metadata, source_names)
        if value is not None:
            attributes[target] = value

    def integer(target: str, *source_names: str) -> None:
        value = _first_safe_integer(metadata, source_names)
        if value is not None:
            attributes[target] = value

    def boolean(target: str, *source_names: str) -> None:
        value = _first_boolean(metadata, source_names)
        if value is not None:
            attributes[target] = value

    if stage is VoiceTimelineStage.LISTEN_STARTED:
        source = _safe_timeline_token(trace.source)
        transport = _safe_timeline_token(trace.media_transport)
        if source is not None:
            attributes["source"] = source
        if transport is not None:
            attributes["media_transport"] = transport
        integer("sample_rate_hz", "sample_rate_hz", "sample_rate")
        integer("channels", "channels")
        token("provider", "asr_provider", "provider")
    elif stage is VoiceTimelineStage.FIRST_AUDIO_FRAME:
        integer("frame_samples", "frame_samples", "chunk_samples")
        integer("sample_rate_hz", "sample_rate_hz", "sample_rate")
        integer("channels", "channels")
        integer("byte_count", "byte_count")
    elif stage in {
        VoiceTimelineStage.SPEECH_START,
        VoiceTimelineStage.SPEECH_END,
        VoiceTimelineStage.INTERRUPT_DETECTED,
        VoiceTimelineStage.INTERRUPT_CONFIRMED,
        VoiceTimelineStage.INTERRUPT_DISMISSED,
    }:
        number("peak", "peak")
        number("rms", "rms")
        number("confidence", "confidence")
        token("reason_code", "reason_code")
        token("interrupt_reason", "interrupt_reason")
    elif stage is VoiceTimelineStage.ENDPOINT_COMMITTED:
        token("endpoint_reason", "endpoint_reason", "reason_code")
        number("duration_ms", "duration_ms")
        number("buffered_ms", "buffered_ms")
    elif stage is VoiceTimelineStage.ASR_FINAL:
        token("provider", "provider", "asr_source", "asr_provider")
        number("latency_ms", "latency_ms", "asr_latency_ms")
        explicit_count = _first_safe_integer(metadata, ("character_count",))
        transcript = metadata.get("text", metadata.get("transcript"))
        if explicit_count is not None:
            attributes["character_count"] = explicit_count
        elif isinstance(transcript, str):
            attributes["character_count"] = len(transcript)
    elif stage is VoiceTimelineStage.TURN_CORRELATED:
        token("source", "source")
    elif stage is VoiceTimelineStage.TURN_ADMITTED:
        token("source", "source")
        token("route", "route")
        number("latency_ms", "latency_ms")
    elif stage is VoiceTimelineStage.LLM_REQUESTED:
        token("provider", "provider", "llm_provider")
        token("model", "model")
        token("route", "route")
    elif stage is VoiceTimelineStage.FIRST_LLM_PAYLOAD:
        token("provider", "provider", "llm_provider")
        token("model", "model")
        number("latency_ms", "latency_ms")
        integer("token_count", "token_count")
    elif stage is VoiceTimelineStage.FIRST_SEMANTIC:
        token("provider", "provider", "llm_provider")
        token("model", "model")
        number("latency_ms", "latency_ms")
        integer("character_count", "character_count")
    elif stage is VoiceTimelineStage.FIRST_CLAUSE:
        integer("clause_index", "clause_index")
        integer("character_count", "character_count")
    elif stage is VoiceTimelineStage.TTS_FIRST_PCM:
        token("provider", "provider", "tts_provider")
        token("model", "model")
        number("latency_ms", "latency_ms", "tts_latency_ms")
        number("duration_ms", "duration_ms")
        integer("byte_count", "byte_count")
        _copy_audio_attributes(attributes, metadata)
    elif stage in {
        VoiceTimelineStage.SPEAKER_RENDER_STARTED,
        VoiceTimelineStage.SPEAKER_PHYSICAL_STARTED,
        VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
        VoiceTimelineStage.SPEAKER_PHYSICAL_STOPPED,
    }:
        number("played_ms", "played_ms")
        number("duration_ms", "duration_ms")
        token("evidence_kind", "evidence_kind")
        token("clock_id", "clock_id")
        boolean("instrumented", "instrumented")
        boolean("provenance_verified", "provenance_verified")
        _copy_audio_attributes(attributes, metadata)
    elif stage is VoiceTimelineStage.FALLBACK_SELECTED:
        token("provider", "provider")
        token("model", "model")
        token("fallback_reason", "fallback_reason", "reason_code")
        token("route", "route")
        boolean("degraded", "degraded")
    return attributes


def _copy_audio_attributes(
    target: dict[str, object],
    metadata: Mapping[str, Any],
) -> None:
    audio_class = _first_safe_token(metadata, ("audio_class",))
    segment_id = _first_safe_token(
        metadata,
        ("audio_segment_id", "segment_id"),
    )
    if audio_class is not None and audio_class in AUDIO_CLASSES:
        target["audio_class"] = audio_class
    if segment_id is not None:
        target["audio_segment_id"] = segment_id


def _first_safe_token(
    metadata: Mapping[str, Any],
    source_names: tuple[str, ...],
) -> str | None:
    for source_name in source_names:
        value = _safe_timeline_token(metadata.get(source_name))
        if value is not None:
            return value
    return None


def _first_finite_number(
    metadata: Mapping[str, Any],
    source_names: tuple[str, ...],
) -> int | float | None:
    for source_name in source_names:
        value = metadata.get(source_name)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        ):
            return value
    return None


def _first_safe_integer(
    metadata: Mapping[str, Any],
    source_names: tuple[str, ...],
) -> int | None:
    for source_name in source_names:
        value = metadata.get(source_name)
        if not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= 2**63 - 1:
            return value
    return None


def _first_boolean(
    metadata: Mapping[str, Any],
    source_names: tuple[str, ...],
) -> bool | None:
    for source_name in source_names:
        value = metadata.get(source_name)
        if isinstance(value, bool):
            return value
    return None


def _safe_legacy_snapshot_metadata(
    metadata: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
) -> dict[str, object]:
    """Expose only bounded diagnostic facts, never transcript/error payloads."""

    safe: dict[str, object] = {}
    for key in allowed_fields:
        value = metadata.get(key)
        if key == "asr_source":
            if isinstance(value, str) and value in _SAFE_ASR_SOURCE_LABELS:
                safe[key] = value
            continue
        if key in {"chunk_samples", "peak"}:
            if type(value) is int and 0 <= value <= 2_147_483_647:
                safe[key] = value
            continue
        if key == "rms" and isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and 0.0 <= numeric <= 2_147_483_647.0:
                safe[key] = value
    return safe


def _safe_trace_turn_metadata(
    metadata: Mapping[str, Any],
    *,
    include_scope: bool,
) -> dict[str, object]:
    """Retain only bounded operational facts needed after event projection."""

    safe: dict[str, object] = {}
    if include_scope:
        for field_name in _TRACE_SCOPE_METADATA_FIELDS:
            value = _safe_timeline_token(metadata.get(field_name))
            if value is not None:
                safe[field_name] = value

    asr_source = metadata.get("asr_source")
    if isinstance(asr_source, str) and asr_source in _SAFE_ASR_SOURCE_LABELS:
        safe["asr_source"] = asr_source

    raw_error_type = metadata.get("error_type")
    error_type = (
        raw_error_type.strip()
        if isinstance(raw_error_type, str)
        and _SAFE_ERROR_TYPE.fullmatch(raw_error_type.strip()) is not None
        else None
    )
    if error_type is not None:
        safe["error_type"] = error_type
    return safe


def _safe_trace_stage_metadata(
    stage_name: str,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Retain only stage facts consumed by snapshots or latency provenance."""

    safe = _safe_legacy_snapshot_metadata(
        metadata,
        allowed_fields=_LEGACY_STAGE_SNAPSHOT_FIELDS.get(
            stage_name,
            _NO_LEGACY_SNAPSHOT_FIELDS,
        ),
    )
    if stage_name not in _PHYSICAL_PROVENANCE_STAGE_NAMES:
        return safe

    evidence_kind = _safe_timeline_token(metadata.get("evidence_kind"))
    if evidence_kind is not None:
        safe["evidence_kind"] = evidence_kind
    clock_id = _safe_timeline_token(metadata.get("clock_id"))
    if clock_id is not None:
        safe["clock_id"] = clock_id
    for field_name in ("instrumented", "provenance_verified"):
        value = metadata.get(field_name)
        if isinstance(value, bool):
            safe[field_name] = value
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping) and isinstance(
        validated := provenance.get("validated"),
        bool,
    ):
        safe["provenance"] = {"validated": validated}
    return safe


def _safe_timeline_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if _SAFE_TIMELINE_TOKEN.fullmatch(normalized) is not None else None


def _safe_trace_status(status: object) -> str:
    normalized = str(status or "completed").strip().lower()
    return normalized if normalized in _SAFE_TRACE_STATUSES else "other"


def evaluate_voice_turn_slo(
    turn: dict[str, Any] | None,
    *,
    latency_summary: dict[str, Any] | None = None,
    thresholds_ms: dict[str, float] | None = None,
    required_buckets: tuple[str, ...] = DEFAULT_REQUIRED_VOICE_TURN_BUCKETS,
    required_physical_provenance_buckets: tuple[str, ...] = (
        DEFAULT_REQUIRED_PHYSICAL_PROVENANCE_BUCKETS
    ),
) -> dict[str, Any]:
    """Evaluate whether the latest voice turn is fast enough for live conversation."""
    thresholds = dict(DEFAULT_VOICE_TURN_SLO_MS)
    thresholds.update(thresholds_ms or {})
    if not turn:
        return {
            "status": "no_turn",
            "ready_to_converse": False,
            "thresholds_ms": thresholds,
            "required_buckets": list(required_buckets),
            "missing_buckets": list(required_buckets),
            "missing_provenance_buckets": list(required_physical_provenance_buckets),
            "failed_buckets": [],
            "slowest_bucket": None,
            "slowest_bucket_ms": None,
        }

    raw_buckets = turn.get("latency_buckets")
    buckets: Mapping[str, Any] = raw_buckets if isinstance(raw_buckets, Mapping) else {}
    missing = [name for name in required_buckets if buckets.get(name) is None]
    raw_provenance = turn.get("latency_bucket_provenance")
    provenance = raw_provenance if isinstance(raw_provenance, Mapping) else {}
    missing_provenance = [
        name
        for name in required_physical_provenance_buckets
        if buckets.get(name) is not None
        and not (
            isinstance(provenance.get(name), Mapping)
            and provenance[name].get("physical_provenance_valid") is True
        )
    ]
    failed = [
        {
            "bucket": name,
            "actual_ms": buckets.get(name),
            "threshold_ms": threshold,
        }
        for name, threshold in thresholds.items()
        if buckets.get(name) is not None and float(buckets[name]) > threshold
    ]
    summary = latency_summary or {}
    slowest_bucket = summary.get("slowest_bucket")
    slowest_bucket_ms = summary.get("slowest_bucket_ms")
    if not slowest_bucket:
        slowest_bucket, slowest_bucket_ms = max(
            ((name, value) for name, value in buckets.items() if value is not None),
            key=lambda item: item[1],
            default=(None, None),
        )
    status = "passed"
    if missing or missing_provenance:
        status = "insufficient_evidence"
    if failed:
        status = "failed"
    return {
        "status": status,
        "ready_to_converse": status == "passed",
        "thresholds_ms": thresholds,
        "required_buckets": list(required_buckets),
        "missing_buckets": missing,
        "missing_provenance_buckets": missing_provenance,
        "failed_buckets": failed,
        "slowest_bucket": slowest_bucket,
        "slowest_bucket_ms": slowest_bucket_ms,
    }


def _default_audio_class(stage_name: str) -> str | None:
    if stage_name.startswith("ack_"):
        return "ack"
    if stage_name.startswith(("feedback_", "processing_feedback_")):
        return "feedback"
    if "semantic" in stage_name:
        return "semantic"
    if stage_name.startswith("safety_"):
        return "safety"
    if stage_name.startswith("error_"):
        return "error"
    return None


def _physical_provenance_valid(stage: VoiceTraceStage | None) -> bool:
    if stage is None:
        return False
    metadata = stage.metadata
    if metadata.get("evidence_kind") != "physical_acoustic":
        return False
    if metadata.get("instrumented") is not True:
        return False
    if not str(metadata.get("clock_id") or "").strip():
        return False
    provenance = metadata.get("provenance")
    return bool(
        metadata.get("provenance_verified") is True
        or (isinstance(provenance, Mapping) and provenance.get("validated") is True)
    )
