"""Audit API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillAuditResponse(BaseModel):
    """Recent skill execution audit records."""

    model_config = ConfigDict(extra="allow")

    records: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    limit: int = 0


class AuditEventsResponse(BaseModel):
    """Unified product audit timeline response."""

    model_config = ConfigDict(extra="allow")

    records: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    total: int = 0
    filtered_total: int = 0
    limit: int = 0
    truncated: bool = False
    omitted_record_count: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)
    time_window: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    product_summary: dict[str, Any] = Field(default_factory=dict)
    customer_report: dict[str, Any] = Field(default_factory=dict)
    delivery_dossier: dict[str, Any] = Field(default_factory=dict)
    audit_readiness: dict[str, Any] = Field(default_factory=dict)
    review_queue: list[dict[str, Any]] = Field(default_factory=list)
    sources: dict[str, Any] = Field(default_factory=dict)
    source_health: dict[str, Any] = Field(default_factory=dict)
    review_integrity: dict[str, Any] = Field(default_factory=dict)
    query_engine: dict[str, Any] = Field(default_factory=dict)


class AuditReviewsResponse(BaseModel):
    """Append-only audit review decision list."""

    model_config = ConfigDict(extra="allow")

    records: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    total: int = 0
    path: str = ""
    integrity: dict[str, Any] = Field(default_factory=dict)


class AuditReviewSubmitResponse(BaseModel):
    """Supervisor audit review decision result."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    record: dict[str, Any] = Field(default_factory=dict)
    path: str = ""
    reason: str | None = None
    record_id: str | None = None


class AuditExportResponse(BaseModel):
    """Signed unified audit export creation result."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    export: dict[str, Any] = Field(default_factory=dict)
    delivery: dict[str, Any] | None = None


class AuditExportsResponse(BaseModel):
    """Recent unified audit export manifests."""

    model_config = ConfigDict(extra="allow")

    exports: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    total: int = 0
    invalid: int = 0
    output_dir: str = ""


class AuditExportRetryStatusResponse(BaseModel):
    """Pending audit export delivery retry queue status."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    queue: str = ""
    exists: bool | None = None
    pending: int = 0
    invalid: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class AuditExportRetryResponse(BaseModel):
    """Audit export delivery retry execution result."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    queue: str = ""
    lock: dict[str, Any] = Field(default_factory=dict)
    attempted: int = 0
    sent: int = 0
    failed: int = 0
    remaining: int | None = 0
    invalid: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
