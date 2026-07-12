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
