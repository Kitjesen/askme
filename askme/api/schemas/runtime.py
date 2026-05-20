"""Runtime TaskRun and handoff API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeContextResponse(BaseModel):
    """Current runtime profile, active run, and operator-facing context."""

    model_config = ConfigDict(extra="allow")

    profile: str = ""
    active_run: dict[str, Any] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class RuntimeProfilesResponse(BaseModel):
    """Available runtime profiles and the currently selected profile."""

    model_config = ConfigDict(extra="allow")

    current_profile: str = ""
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class RuntimeRunListResponse(BaseModel):
    """TaskRun list used by dashboards and operator consoles."""

    model_config = ConfigDict(extra="allow")

    runs: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class RuntimeHandoffSubmitResponse(BaseModel):
    """Result of submitting a TaskHandoff plan to runtime arbitration."""

    model_config = ConfigDict(extra="allow")

    accepted: bool = True
    reason: str = ""
    handoff: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    profile: str = ""
    error: str = ""


class RuntimeRunDetailResponse(BaseModel):
    """Single TaskRun state and evidence."""

    model_config = ConfigDict(extra="allow")

    run: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class RuntimeRunReportResponse(BaseModel):
    """Auditable TaskRun report."""

    model_config = ConfigDict(extra="allow")

    report: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class RuntimeRunActionResponse(BaseModel):
    """Pause, resume, cancel, or advance response for one TaskRun."""

    model_config = ConfigDict(extra="allow")

    handled: bool = False
    reason: str = ""
    run: dict[str, Any] = Field(default_factory=dict)
    event: dict[str, Any] = Field(default_factory=dict)
    operator: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
