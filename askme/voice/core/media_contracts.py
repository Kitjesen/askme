"""Protocol contracts for the next-generation voice media layer.

These contracts keep realtime media transport separate from robot cognition.
Local sounddevice, LiveKit/WebRTC, and future SIP adapters should all enter
askme through these shapes before reaching TaskHandoff or runtime control.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class VoiceTurnEventType(StrEnum):
    MEDIA_JOIN = "media_join"
    FIRST_AUDIO_FRAME = "first_audio_frame"
    VAD_START = "vad_start"
    ASR_FIRST_PARTIAL = "asr_first_partial"
    ASR_FINAL = "asr_final"
    SEMANTIC_END = "semantic_end"
    BARGE_IN_CONFIRMED = "barge_in_confirmed"
    PLAYBACK_DONE = "playback_done"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceMediaFrame:
    """PCM audio frame from any media transport."""

    pcm: bytes
    sample_rate: int
    channels: int = 1
    timestamp_ms: float = 0.0
    participant_id: str = ""
    track_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0 or self.channels <= 0:
            return 0.0
        sample_count = len(self.pcm) / 2 / self.channels
        return round(sample_count / self.sample_rate * 1000.0, 2)


@dataclass(frozen=True)
class VoiceMediaStatus:
    """Transport health for Dashboard and readiness checks."""

    media_transport: str
    session_id: str = ""
    room_id: str = ""
    participant_count: int = 0
    packet_loss: float | None = None
    jitter_ms: float | None = None
    input_transport: str = ""
    output_transport: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_transport": self.media_transport,
            "session_id": self.session_id,
            "room_id": self.room_id,
            "participant_count": self.participant_count,
            "packet_loss": self.packet_loss,
            "jitter_ms": self.jitter_ms,
            "input_transport": self.input_transport,
            "output_transport": self.output_transport,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VoiceTurnEvent:
    """Turn-level event emitted after media/VAD/ASR processing."""

    event_type: VoiceTurnEventType
    voice_turn_id: str
    offset_ms: float = 0.0
    transcript: str = ""
    is_final: bool = False
    confidence: float | None = None
    provider: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "voice_turn_id": self.voice_turn_id,
            "offset_ms": round(max(self.offset_ms, 0.0), 2),
            "transcript": self.transcript,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "provider": self.provider,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InterruptRequest:
    """A user or safety interruption request against the active voice turn."""

    reason: str
    voice_turn_id: str = ""
    transcript: str = ""
    source: str = "voice"
    force_stop_playback: bool = True
    cancel_generation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterruptDecision:
    """Result of applying an interruption request."""

    accepted: bool
    reason: str
    stopped_playback: bool = False
    cancelled_generation: bool = False
    requires_runtime_action: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "stopped_playback": self.stopped_playback,
            "cancelled_generation": self.cancelled_generation,
            "requires_runtime_action": self.requires_runtime_action,
            "metadata": dict(self.metadata),
        }


class MediaSession(Protocol):
    """Realtime audio transport boundary.

    Implementations may be local sounddevice, LiveKit/WebRTC, SIP, or test
    fakes. They must not call planning, runtime, or hardware services directly.
    """

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def frames(self) -> AsyncIterator[VoiceMediaFrame]: ...

    async def send_audio(self, frame: VoiceMediaFrame) -> None: ...

    def status(self) -> VoiceMediaStatus: ...


class TurnDetector(Protocol):
    """Converts media frames into structured voice-turn events."""

    async def feed(self, frame: VoiceMediaFrame) -> list[VoiceTurnEvent]: ...

    async def reset(self) -> None: ...


class InterruptController(Protocol):
    """Owns local voice interruption side effects before runtime routing."""

    async def request_interrupt(self, request: InterruptRequest) -> InterruptDecision: ...


class VoiceGateway(Protocol):
    """Routes final transcripts to chat or runtime-control surfaces."""

    async def handle_turn_event(self, event: VoiceTurnEvent) -> dict[str, Any]: ...
