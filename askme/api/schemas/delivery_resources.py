"""Delivery-resource registry and governance response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DeliveryResourceRegistryResponse(BaseModel):
    """Shared resources that customer-project managed objects can bind to."""

    model_config = ConfigDict(extra="allow")

    found: bool
    registry_path: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    delivery_resources: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class DeliveryResourceMutationResponse(BaseModel):
    """Create/update/disable response for one shared delivery resource."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    message: str = ""
    created: bool = False
    resource_type: str = ""
    resource_id: str = ""
    publish_status: str = ""
    allowed_publish_statuses: list[str] = Field(default_factory=list)
    resource: dict[str, Any] = Field(default_factory=dict)
    registry_path: str = ""
    revision: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class DeliveryResourceHistoryResponse(BaseModel):
    """Saved delivery-resource registry revisions for rollback review."""

    model_config = ConfigDict(extra="allow")

    found: bool
    registry_path: str = ""
    revision_count: int = 0
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    next_step: str = ""


class DeliveryResourceRollbackResponse(BaseModel):
    """Rollback dry-run or apply response for the shared resource registry."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    message: str = ""
    dry_run: bool = False
    registry_path: str = ""
    revision: dict[str, Any] = Field(default_factory=dict)
    current_registry_sha256: str = ""
    target_registry_sha256: str = ""
    target_summary: dict[str, Any] = Field(default_factory=dict)
    would_write: bool = False
    rollback_snapshot: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class DeliveryResourceGovernanceRequestsResponse(BaseModel):
    """Pending and reviewed shared-resource governance requests."""

    model_config = ConfigDict(extra="allow")

    root: str = ""
    status: str = ""
    action: str = ""
    overdue_only: bool = False
    requests: list[dict[str, Any]] = Field(default_factory=list)
    request_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)


class DeliveryResourceGovernanceMutationResponse(BaseModel):
    """Create/review response for shared-resource governance requests."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    message: str = ""
    dry_run: bool = False
    request_id: str = ""
    allowed_actions: list[str] = Field(default_factory=list)
    allowed_decisions: list[str] = Field(default_factory=list)
    operation: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    apply_result: dict[str, Any] = Field(default_factory=dict)
    request_registry_sha256: str = ""
    current_registry_sha256: str = ""
    next_step: str = ""


class DeliveryResourceGovernanceEscalationResponse(BaseModel):
    """Escalation result for overdue shared-resource governance requests."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    dry_run: bool = False
    operator_id: str = ""
    reason: str = ""
    checked_count: int = 0
    escalated_count: int = 0
    skipped_count: int = 0
    escalations: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    requests: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""
