"""Agent Profile API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentProfileCatalogResponse(BaseModel):
    """Product-reviewable Agent Profile catalog."""

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    mechanism: str | None = None
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    profile_count: int = 0
    inherited_tool_count: int = 0
    profile_locations: list[str] = Field(default_factory=list)
    profile_scopes: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)


class AgentProfileUpsertResponse(BaseModel):
    """Create/update result for one project-level Agent Profile."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    profile: dict[str, Any] = Field(default_factory=dict)
    path: str = ""
    catalog: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    profile_name: str | None = None


class AgentProfilePreviewResponse(BaseModel):
    """Preview result for one Agent Profile."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    profile: dict[str, Any] = Field(default_factory=dict)
    path: str = ""
    raw_body: str = ""
    error: str | None = None
    profile_name: str | None = None
