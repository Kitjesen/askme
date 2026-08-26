"""Stable product contract for targeted speech playback jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class SpeechPlaybackState(str, Enum):
    QUEUED = "queued"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            SpeechPlaybackState.COMPLETED,
            SpeechPlaybackState.FAILED,
            SpeechPlaybackState.CANCELLED,
        }


class SpeechDelivery(str, Enum):
    PLAYBACK = "playback"
    SYNTHESIZE_ONLY = "synthesize_only"


class SpeechPriority(str, Enum):
    SAFETY = "safety"
    OPERATOR = "operator"
    NORMAL = "normal"
    LOW = "low"


@dataclass(frozen=True)
class PlaybackTarget:
    robot_id: str
    device_id: str
    site_id: str = ""
    channel: str = "speaker"


@dataclass(frozen=True)
class SpeechActor:
    operator_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    surface: str = "internal"


@dataclass(frozen=True)
class SpeechPlaybackRequest:
    """One literal speech request. Conversational generation is not accepted."""

    text: str
    target: PlaybackTarget
    actor: SpeechActor | None = None
    idempotency_key: str = ""
    delivery: SpeechDelivery = SpeechDelivery.PLAYBACK
    priority: SpeechPriority = SpeechPriority.NORMAL
    queue_policy: str = "enqueue"
    voice_profile_id: str = ""
    speed: float | None = None
    pitch: float | None = None
    volume: float | None = None
    ttl_s: float = 60.0


@dataclass(frozen=True)
class SpeechPlaybackTimestamps:
    queued_at: str
    synthesis_started_at: str | None = None
    playback_started_at: str | None = None
    completed_at: str | None = None
    cancelled_at: str | None = None
    failed_at: str | None = None


@dataclass(frozen=True)
class SpeechPlaybackJob:
    playback_id: str
    state: SpeechPlaybackState
    target: PlaybackTarget
    delivery: SpeechDelivery
    priority: SpeechPriority
    text_chars: int
    request_hash: str
    idempotency_key: str
    timestamps: SpeechPlaybackTimestamps
    operator_id: str = ""
    cache_hit: bool = False
    artifact: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    customer_message: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "playback_id": self.playback_id,
            "state": self.state.value,
            "target": {
                "robot_id": self.target.robot_id,
                "device_id": self.target.device_id,
                "site_id": self.target.site_id,
                "channel": self.target.channel,
            },
            "delivery": self.delivery.value,
            "priority": self.priority.value,
            "text_chars": self.text_chars,
            "idempotency_key": self.idempotency_key,
            "timestamps": {
                "queued_at": self.timestamps.queued_at,
                "synthesis_started_at": self.timestamps.synthesis_started_at,
                "playback_started_at": self.timestamps.playback_started_at,
                "completed_at": self.timestamps.completed_at,
                "cancelled_at": self.timestamps.cancelled_at,
                "failed_at": self.timestamps.failed_at,
            },
            "cache_hit": self.cache_hit,
            "artifact": self.artifact,
            "error": self.error,
            "customer_message": self.customer_message,
        }


class SpeechPlaybackError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    def to_payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


@dataclass(frozen=True)
class SpeechAudioArtifactFile:
    """Server-side artifact handle; paths never appear in public job payloads."""

    path: Path
    filename: str
    media_type: str
    size_bytes: int
    sha256: str


@runtime_checkable
class SpeechPlaybackPort(Protocol):
    async def start(self) -> None: ...

    async def submit(self, request: SpeechPlaybackRequest) -> SpeechPlaybackJob: ...

    async def status(self, playback_id: str) -> SpeechPlaybackJob: ...

    async def cancel(
        self,
        playback_id: str,
        *,
        reason: str,
        actor: SpeechActor | None = None,
    ) -> SpeechPlaybackJob: ...

    async def artifact_file(self, playback_id: str) -> SpeechAudioArtifactFile: ...

    async def shutdown(self) -> None: ...
