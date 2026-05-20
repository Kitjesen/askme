"""Field operation API response contracts.

These schemas describe the stable HTTP surface for customer field events,
device ingest, notification checks, and audit/readiness gates. They keep
``extra=allow`` because field evidence and device payloads intentionally grow
with each customer project.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldEventView(BaseModel):
    """One customer-visible field event or event-like record."""

    model_config = ConfigDict(extra="allow")

    event_id: str = ""
    scenario_id: str = ""
    status: str = ""
    location: str = ""
    incident_state: str = ""
    incident_stage: str = ""
    incident_workflow: dict[str, Any] = Field(default_factory=dict)
    evidence_media: list[dict[str, Any]] = Field(default_factory=list)
    delivery_report: list[dict[str, Any]] = Field(default_factory=list)
    action_audit: list[dict[str, Any]] = Field(default_factory=list)
    sla: dict[str, Any] = Field(default_factory=dict)
    close_approval_required: bool | None = None


class FieldScenarioCatalogResponse(BaseModel):
    """Customer-visible field scenario catalog."""

    model_config = ConfigDict(extra="allow")

    scenarios: list[dict[str, Any]] = Field(default_factory=list)


class FieldScenarioAcceptanceResponse(BaseModel):
    """Acceptance matrix for scenario demos, boundaries, and device entrypoints."""

    model_config = ConfigDict(extra="allow")

    matrix_type: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class FieldEventListApiResponse(BaseModel):
    """Recent field event list response."""

    model_config = ConfigDict(extra="allow")

    events: list[FieldEventView] = Field(default_factory=list)
    total: int = 0
    filtered_total: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    filter: dict[str, Any] = Field(default_factory=dict)
    archive_path: str = ""


class FieldEventDetailApiResponse(BaseModel):
    """Single field event detail response."""

    model_config = ConfigDict(extra="allow")

    found: bool = False
    event_id: str = ""
    reason: str = ""
    event: FieldEventView | None = None


class FieldEventTriggerResponse(BaseModel):
    """Manual or normalized field event trigger result."""

    model_config = ConfigDict(extra="allow")

    accepted: bool = False
    status: str = ""
    reason: str = ""
    scenario_id: str = ""
    event: FieldEventView | None = None
    trigger_contract: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)
    ingest_scope_contract: dict[str, Any] = Field(default_factory=dict)


class FieldEventActionResponse(BaseModel):
    """Operator action result for acknowledge, close, request-close, and resend."""

    model_config = ConfigDict(extra="allow")

    acknowledged: bool | None = None
    requested: bool | None = None
    closed: bool | None = None
    resent: bool | None = None
    reason: str = ""
    event_id: str = ""
    event: FieldEventView | dict[str, Any] | None = None
    delivery_report: list[dict[str, Any]] = Field(default_factory=list)
    sent_channels: list[str] = Field(default_factory=list)


class FieldEventReportResponse(BaseModel):
    """Auditable customer-facing field event report."""

    model_config = ConfigDict(extra="allow")

    found: bool = False
    event_id: str = ""
    reason: str = ""
    report: dict[str, Any] = Field(default_factory=dict)
    markdown: str = ""


class FieldRuntimeDeliveryResponse(BaseModel):
    """Runtime arbiter or robot callback recording result."""

    model_config = ConfigDict(extra="allow")

    recorded: bool = False
    duplicate: bool | None = None
    reason: str = ""
    event_id: str = ""
    event: FieldEventView | None = None
    runtime_delivery: dict[str, Any] = Field(default_factory=dict)
    runtime_delivery_receipt: dict[str, Any] = Field(default_factory=dict)
    runtime_callback_trust: dict[str, Any] = Field(default_factory=dict)


class FieldDeviceStatusResponse(BaseModel):
    """Registered and observed field-device trust/online state."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    require_trusted_devices: bool | None = None
    offline_after_s: float | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    devices: list[dict[str, Any]] = Field(default_factory=list)


class FieldDeviceOnboardingResponse(BaseModel):
    """Delivery-facing report for real device onboarding."""

    model_config = ConfigDict(extra="allow")

    report_type: str = ""
    status: str = ""
    customer_message: str = ""
    policy: dict[str, Any] = Field(default_factory=dict)
    require_trusted_devices: bool | None = None
    offline_after_s: float | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    devices: list[dict[str, Any]] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class FieldIngestHelpResponse(BaseModel):
    """Machine integration help for raw camera, sensor, robot, and map payloads."""

    model_config = ConfigDict(extra="allow")

    accepted_sources: list[str] = Field(default_factory=list)
    examples: dict[str, Any] = Field(default_factory=dict)
    bridge_contract: dict[str, Any] = Field(default_factory=dict)
    freshness_contract: dict[str, Any] = Field(default_factory=dict)
    device_trust_contract: dict[str, Any] = Field(default_factory=dict)


class FieldNotificationTestResponse(BaseModel):
    """Responder notification smoke-test result."""

    model_config = ConfigDict(extra="allow")

    sent: bool = False
    status: str = ""
    notification_group: str = ""
    webhook_configured: bool | None = None
    secret_configured: bool | None = None
    sent_channels: list[str] = Field(default_factory=list)
    delivery_report: list[dict[str, Any]] = Field(default_factory=list)
    message: str = ""
    reason: str = ""


class FieldNotificationPreflightResponse(BaseModel):
    """Notification credential readiness result."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    ready: bool = False
    require_secret: bool | None = None
    groups: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class FieldReadinessResponse(BaseModel):
    """Field deployment readiness gates."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    gates: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class FieldActionAuditIntegrityResponse(BaseModel):
    """Append-only field action audit hash-chain verification."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    path: str = ""
    exists: bool = False
    valid: bool = False
    checked_count: int = 0
    expected_count: int | None = None
    latest_hash: str = ""
    hash_alg: str = ""
    signed: bool | None = None
    signature_alg: str = ""
    failures: list[dict[str, Any]] = Field(default_factory=list)


class FieldCustomerProjectFromTemplateResponse(BaseModel):
    """Customer project creation result from an industry template."""

    model_config = ConfigDict(extra="allow")

    accepted: bool = False
    status: str = ""
    reason: str = ""
    project: dict[str, Any] = Field(default_factory=dict)
    profile_path: str = ""
