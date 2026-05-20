"""Structured voice-turn trace helpers.

The realtime media layer is evolving from a local microphone loop toward
pluggable transports such as WebRTC/LiveKit.  This module records media and
turn-detection milestones without depending on a specific transport.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

LATENCY_BUCKET_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
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
    "barge_in_stop_ms": ("barge_in_stop", "barge_in_confirmed"),
}

LATENCY_BUCKET_NAMES: tuple[str, ...] = tuple(LATENCY_BUCKET_STAGE_ALIASES)
_SUMMARY_WINDOW_SIZE = 10
DEFAULT_VOICE_TURN_SLO_MS: dict[str, float] = {
    "asr_first_partial_ms": 500.0,
    "asr_final_ms": 1200.0,
    "llm_ttft_ms": 900.0,
    "tts_first_audio_ms": 900.0,
    "playback_start_ms": 1200.0,
    "barge_in_stop_ms": 250.0,
}
DEFAULT_REQUIRED_VOICE_TURN_BUCKETS: tuple[str, ...] = (
    "asr_final_ms",
    "llm_ttft_ms",
    "tts_first_audio_ms",
    "playback_start_ms",
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset_ms": round(max(self.offset_ms, 0.0), 2),
            "metadata": dict(self.metadata),
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
            voice_turn_id=uuid4().hex[:16],
            source=source,
            media_transport=media_transport,
            started_at_epoch_s=time.time(),
            started_at_monotonic_s=time.monotonic(),
            metadata=dict(metadata or {}),
        )

    def mark(self, name: str, **metadata: Any) -> None:
        """Record the first occurrence of a stage."""
        stage_name = str(name or "").strip()
        if not stage_name or stage_name in self.stages:
            return
        offset_ms = (time.monotonic() - self.started_at_monotonic_s) * 1000.0
        self.stages[stage_name] = VoiceTraceStage(
            name=stage_name,
            offset_ms=offset_ms,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def latency_buckets(self) -> dict[str, float | None]:
        """Return canonical latency buckets for this turn.

        Bucket keys are stable even when the corresponding stage was not
        recorded, which lets dashboards and tests distinguish missing data from
        a zero-latency stage.
        """
        buckets: dict[str, float | None] = {}
        for bucket_name, stage_aliases in LATENCY_BUCKET_STAGE_ALIASES.items():
            stage = next(
                (self.stages[stage_name] for stage_name in stage_aliases if stage_name in self.stages),
                None,
            )
            buckets[bucket_name] = _round_ms(stage.offset_ms) if stage is not None else None
        return buckets

    def missing_latency_buckets(self) -> list[str]:
        return [
            bucket_name
            for bucket_name, latency_ms in self.latency_buckets().items()
            if latency_ms is None
        ]

    def finish(self, status: str = "completed", **metadata: Any) -> None:
        if self.ended_at_epoch_s is not None:
            return
        self.status = str(status or "completed")
        self.ended_at_epoch_s = time.time()
        self.total_ms = (time.monotonic() - self.started_at_monotonic_s) * 1000.0
        self.metadata.update({k: v for k, v in metadata.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_turn_id": self.voice_turn_id,
            "source": self.source,
            "media_transport": self.media_transport,
            "status": self.status,
            "started_at_epoch_s": round(self.started_at_epoch_s, 3),
            "ended_at_epoch_s": (
                round(self.ended_at_epoch_s, 3)
                if self.ended_at_epoch_s is not None
                else None
            ),
            "total_ms": round(self.total_ms, 2) if self.total_ms is not None else None,
            "stages": [stage.to_dict() for stage in self.stages.values()],
            "latency_buckets": self.latency_buckets(),
            "missing_latency_buckets": self.missing_latency_buckets(),
            "metadata": dict(self.metadata),
        }


class VoiceTurnTraceRecorder:
    """Tracks the active and latest completed voice-turn traces."""

    def __init__(self) -> None:
        self._current: VoiceTurnTrace | None = None
        self._latest: VoiceTurnTrace | None = None
        self._history: list[VoiceTurnTrace] = []
        self._barge_in_count = 0

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

    def start(
        self,
        *,
        source: str,
        media_transport: str,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceTurnTrace:
        if self._current is not None and self._current.ended_at_epoch_s is None:
            self._current.finish("superseded")
            self._latest = self._current
            self._remember(self._current)
        self._current = VoiceTurnTrace.start(
            source=source,
            media_transport=media_transport,
            metadata=metadata,
        )
        self._current.mark("listen_started")
        return self._current

    def mark(self, name: str, **metadata: Any) -> None:
        target = self._current or self._latest
        if target is not None:
            target.mark(name, **metadata)

    def finish(self, status: str = "completed", **metadata: Any) -> None:
        if self._current is None:
            return
        self._current.finish(status, **metadata)
        self._latest = self._current
        self._remember(self._current)
        self._current = None

    def mark_barge_in(self, **metadata: Any) -> None:
        self._barge_in_count += 1
        self.mark("barge_in_confirmed", **metadata)

    def latency_summary(self) -> dict[str, Any]:
        history = self._history[-_SUMMARY_WINDOW_SIZE:]
        latest_buckets = self._latest.latency_buckets() if self._latest is not None else {}
        bucket_summaries: dict[str, dict[str, Any]] = {}

        for bucket_name in LATENCY_BUCKET_NAMES:
            values = [
                latency_ms
                for trace in history
                if (latency_ms := trace.latency_buckets().get(bucket_name)) is not None
            ]
            bucket_summaries[bucket_name] = {
                "latest_ms": latest_buckets.get(bucket_name),
                "p50_ms": _round_ms(_percentile(values, 0.50)),
                "p95_ms": _round_ms(_percentile(values, 0.95)),
                "count": len(values),
                "missing_count": len(history) - len(values),
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
        latest = self._latest.to_dict() if self._latest is not None else None
        current = self._current.to_dict() if self._current is not None else None
        summary = self.latency_summary()
        return {
            "current": current,
            "latest": latest,
            "latency_summary": summary,
            "slo": evaluate_voice_turn_slo(latest or current, latency_summary=summary),
            "counters": {
                "barge_in_count": self._barge_in_count,
            },
        }


def evaluate_voice_turn_slo(
    turn: dict[str, Any] | None,
    *,
    latency_summary: dict[str, Any] | None = None,
    thresholds_ms: dict[str, float] | None = None,
    required_buckets: tuple[str, ...] = DEFAULT_REQUIRED_VOICE_TURN_BUCKETS,
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
            "failed_buckets": [],
            "slowest_bucket": None,
            "slowest_bucket_ms": None,
        }

    buckets = turn.get("latency_buckets") if isinstance(turn.get("latency_buckets"), dict) else {}
    missing = [name for name in required_buckets if buckets.get(name) is None]
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
    if missing:
        status = "insufficient_evidence"
    if failed:
        status = "failed"
    return {
        "status": status,
        "ready_to_converse": status == "passed",
        "thresholds_ms": thresholds,
        "required_buckets": list(required_buckets),
        "missing_buckets": missing,
        "failed_buckets": failed,
        "slowest_bucket": slowest_bucket,
        "slowest_bucket_ms": slowest_bucket_ms,
    }
