"""Customer-facing delivery readiness response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SolutionDeliveryReadinessResponse(BaseModel):
    """Solution-provider readiness gates for customer delivery packages."""

    model_config = ConfigDict(extra="allow")

    readiness_type: str = Field(min_length=1)
    overall_status: str = Field(min_length=1)
    production_ready: bool
    customer_status: str = Field(min_length=1)
    release_claim: str = Field(min_length=1)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    manual_checks: list[str] = Field(default_factory=list)
    next_step: str = Field(min_length=1)


class ProductLaunchReadinessResponse(BaseModel):
    """Customer-facing launch decision across product gates."""

    model_config = ConfigDict(extra="allow")

    readiness_type: str = Field(min_length=1)
    overall_status: str = Field(min_length=1)
    launch_stage: str = Field(min_length=1)
    production_ready: bool
    customer_status: str = Field(min_length=1)
    release_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    manual_checks: list[str] = Field(default_factory=list)
    customer_acceptance_snapshot: dict[str, Any] = Field(default_factory=dict)
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    source_snapshots: dict[str, Any] = Field(default_factory=dict)
