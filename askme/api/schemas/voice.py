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
