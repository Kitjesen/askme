"""Voice profile API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VoiceProfileCatalogResponse(BaseModel):
    """Selectable voice profiles and local sound cues for customer-facing speech."""

    model_config = ConfigDict(extra="allow")

    profiles: list[dict[str, Any]] = Field(default_factory=list)
    active_profile: str = ""
    active_profile_settings: dict[str, Any] = Field(default_factory=dict)
    default_profile: str = ""
    available_sound_cues: list[str] = Field(default_factory=list)
    sound_cues_enabled: bool | None = None
    persistence_status: str = ""
    profile_count: int | None = None


class VoiceProfileUpdateResponse(BaseModel):
    """Result of changing the active TTS voice profile."""

    model_config = ConfigDict(extra="allow")

    updated: bool = False
    reason: str = ""
    requested_profile: str = ""
    resolved_profile: str = ""
    active_profile: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    applied_settings: dict[str, Any] = Field(default_factory=dict)
    persistence_status: str = ""
    sound_cue: dict[str, Any] = Field(default_factory=dict)
    available: list[str] = Field(default_factory=list)


class VoiceSystemControlResponse(BaseModel):
    """Non-secret runtime state exposed to the voice-system console."""

    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    runtime: dict[str, Any] = Field(default_factory=dict)
    catalog: dict[str, Any] = Field(default_factory=dict)
    prompt: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)


class VoiceSystemUpdateResponse(BaseModel):
    """Result of a live model, provider, or prompt update."""

    model_config = ConfigDict(extra="allow")

    updated: bool = False
    component: str = ""
    state: str = ""
    reason: str = ""
    runtime: dict[str, Any] = Field(default_factory=dict)


class VoiceSpeakRequest(BaseModel):
    """Literal text-to-robot speech; this contract never invokes an LLM."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    robot_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    site_id: str = Field(default="", max_length=128)
    semantics: str = "verbatim"
    priority: str = "normal"
    queue_policy: str = "enqueue"
    voice_profile_id: str = Field(default="", max_length=128)
    speed: float | None = Field(default=None, ge=0.75, le=1.5)
    pitch: float | None = Field(default=None, ge=-12.0, le=12.0)
    volume: float | None = Field(default=None, ge=0.05, le=1.0)
    ttl_s: float = Field(default=60.0, ge=1.0, le=300.0)


class VoicePlaybackCancelRequest(BaseModel):
    """Operator reason for cancelling one playback job."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="operator_cancelled", min_length=1, max_length=160)


class VoicePlaybackResponse(BaseModel):
    """Public lifecycle view for one speech playback job."""

    model_config = ConfigDict(extra="allow")

    playback_id: str
    state: str
    target: dict[str, Any]
    delivery: str = "playback"
    priority: str = "normal"
    text_chars: int = 0
    idempotency_key: str = ""
    timestamps: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    artifact: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    customer_message: str = ""
