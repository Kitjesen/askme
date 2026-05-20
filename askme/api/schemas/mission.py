"""Mission draft, submission, and report API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MissionDraftResponse(BaseModel):
    """Drafted high-level mission plan without hardware dispatch."""

    model_config = ConfigDict(extra="allow")

    drafted: bool = False
    mission: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class MissionSubmitResponse(BaseModel):
    """Dry-run or submitted mission handoff result."""

    model_config = ConfigDict(extra="allow")

    accepted: bool = True
    mission: dict[str, Any] = Field(default_factory=dict)
    submission: dict[str, Any] = Field(default_factory=dict)
    runtime_handoff: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class MissionListResponse(BaseModel):
    """Mission records available to the operator surface."""

    model_config = ConfigDict(extra="allow")

    missions: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class MissionDetailResponse(BaseModel):
    """Single mission record and latest state."""

    model_config = ConfigDict(extra="allow")

    mission: dict[str, Any] = Field(default_factory=dict)
    mission_id: str = ""
    error: str = ""


class MissionReportResponse(BaseModel):
    """Customer-readable mission report payload."""

    model_config = ConfigDict(extra="allow")

    report: dict[str, Any] = Field(default_factory=dict)
    mission: dict[str, Any] = Field(default_factory=dict)
    mission_id: str = ""
    error: str = ""
