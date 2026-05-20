"""Park space cognition API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpaceHealthResponse(BaseModel):
    """Readiness and capability summary for the park-space cognition service."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    park_id: str = ""
    points: int = 0
    service_points: int = 0
    routes: int = 0
    revision: int = 0
    changes: int = 0
    snapshots: int = 0
    pending_changes: int = 0
    interactions: int = 0
    store: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class SpacePointsResponse(BaseModel):
    """Point directory for buildings, merchants, toilets, gates, and service areas."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    points: list[dict[str, Any]] = Field(default_factory=list)


class SpaceServicePointsResponse(BaseModel):
    """Configured active visitor question/help points."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    service_points: list[dict[str, Any]] = Field(default_factory=list)


class SpaceRoutesResponse(BaseModel):
    """Configured voice and escort route catalog."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    routes: list[dict[str, Any]] = Field(default_factory=list)


class SpaceHistoryResponse(BaseModel):
    """Auditable catalog change history and revision inventory."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    revision: int = 0
    changes: list[dict[str, Any]] = Field(default_factory=list)
    available_revisions: list[int] = Field(default_factory=list)
    store: dict[str, Any] = Field(default_factory=dict)


class SpaceProposalsResponse(BaseModel):
    """Pending and reviewed catalog change proposals."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    revision: int = 0
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    pending_count: int = 0


class SpaceInteractionsResponse(BaseModel):
    """Visitor interaction records for prompt, guide, and refusal events."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    park_id: str = ""
    count: int = 0
    total: int = 0
    limit: int = 50
    interactions: list[dict[str, Any]] = Field(default_factory=list)


class SpaceResolveDestinationResponse(BaseModel):
    """Destination parsing result for wayfinding questions."""

    model_config = ConfigDict(extra="allow")

    resolved: bool = False
    reason: str = ""
    query: str = ""
    reply: str = ""
    confidence: float | None = None
    match_reason: str = ""
    point: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_count: int = 0
    confirmation_prompt: str = ""
    requires_confirmation: bool | None = None
    requires_clarification: bool | None = None
    listing_only: bool | None = None
    requires_operator_update: bool | None = None


class SpaceGuideResponse(SpaceResolveDestinationResponse):
    """Voice guidance or robot escort handoff for a resolved destination."""

    ok: bool | None = None
    guide_ready: bool = False
    mode: str = ""
    route: dict[str, Any] | None = None
    speech_text: str = ""
    field_event_payload: dict[str, Any] = Field(default_factory=dict)
    task_handoff: dict[str, Any] = Field(default_factory=dict)
    runtime_handoff: dict[str, Any] = Field(default_factory=dict)
    interaction_id: str = ""


class SpaceServicePointTriggerResponse(BaseModel):
    """Interaction admission result for a visitor standing at a service point."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    should_prompt: bool = False
    reason: str = ""
    admission: str = ""
    speech_text: str = ""
    dwell_seconds: float | None = None
    required_dwell_seconds: float | None = None
    service_point: dict[str, Any] | None = None
    supported_intents: list[str] = Field(default_factory=list)
    next_expected_input: str = ""
    interaction_id: str = ""


class SpaceManageResponse(BaseModel):
    """Space catalog management result for point/service-point/route changes."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    reason: str = ""
    revision: int | None = None
    change: dict[str, Any] = Field(default_factory=dict)
    persisted: dict[str, Any] = Field(default_factory=dict)
    point: dict[str, Any] | None = None
    service_point: dict[str, Any] | None = None
    route: dict[str, Any] | None = None
    allowed_entities: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)


class SpaceProposalCreateResponse(BaseModel):
    """Catalog change proposal created by an operator before approval."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    proposal_created: bool = False
    proposal: dict[str, Any] = Field(default_factory=dict)
    persisted: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class SpaceProposalReviewResponse(SpaceManageResponse):
    """Review result for an approved or rejected space catalog proposal."""

    reviewed: bool | None = None
    proposal: dict[str, Any] = Field(default_factory=dict)
    proposal_id: str = ""


class SpaceRollbackResponse(BaseModel):
    """Rollback result for restoring a previous space-catalog revision."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    reason: str = ""
    revision: int | None = None
    restored_revision: int | None = None
    target_revision: int | None = None
    available_revisions: list[int] = Field(default_factory=list)
    change: dict[str, Any] = Field(default_factory=dict)
    persisted: dict[str, Any] = Field(default_factory=dict)
