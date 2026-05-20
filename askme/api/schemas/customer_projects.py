"""Customer project and managed-object response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SiteProfileCatalogResponse(BaseModel):
    """Scoped site-profile catalog for customer delivery."""

    model_config = ConfigDict(extra="allow")

    root: str = Field(min_length=1)
    check_env: bool
    summary: dict[str, Any] = Field(default_factory=dict)
    sites: list[dict[str, Any]] = Field(default_factory=list)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class CustomerProjectCatalogResponse(BaseModel):
    """Scoped customer/project catalog for solution delivery."""

    model_config = ConfigDict(extra="allow")

    root: str = Field(min_length=1)
    check_env: bool
    filters: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    customers: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    delivery_acceptance_gate: dict[str, Any] = Field(default_factory=dict)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class ManagedObjectDirectoryResponse(BaseModel):
    """Scoped managed-object directory for delivery review."""

    model_config = ConfigDict(extra="allow")

    directory_type: str = Field(min_length=1)
    root: str = Field(min_length=1)
    check_env: bool
    filters: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    customer_status: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class CustomerProjectAcceptanceRegistryResponse(BaseModel):
    """Acceptance reference registry across projects and templates."""

    model_config = ConfigDict(extra="allow")

    profile_root: str = Field(min_length=1)
    template_root: str = Field(min_length=1)
    summary: dict[str, Any] = Field(default_factory=dict)
    references: list[dict[str, Any]] = Field(default_factory=list)
    consumers: list[dict[str, Any]] = Field(default_factory=list)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class CustomerProjectResourceCatalogResponse(BaseModel):
    """Model, protocol, skill, and test resources used by customer projects."""

    model_config = ConfigDict(extra="allow")

    profile_root: str = Field(min_length=1)
    template_root: str = Field(min_length=1)
    delivery_resource_root: str = Field(min_length=1)
    summary: dict[str, Any] = Field(default_factory=dict)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    consumers: list[dict[str, Any]] = Field(default_factory=list)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class CustomerProjectTemplateCatalogResponse(BaseModel):
    """Reusable industry template catalog for customer project creation."""

    model_config = ConfigDict(extra="allow")

    root: str = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    templates: list[dict[str, Any]] = Field(default_factory=list)
    customer_claim: str = Field(min_length=1)


class CustomerProjectTemplateHistoryResponse(BaseModel):
    """Release-governance history for one reusable customer-project template."""

    model_config = ConfigDict(extra="allow")

    found: bool
    reason: str = ""
    template_id: str = ""
    template_path: str = ""
    template_package: dict[str, Any] = Field(default_factory=dict)
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    revision_count: int = 0


class CustomerProjectTemplateReleaseRequestsResponse(BaseModel):
    """Reusable-template release requests for product-owner review."""

    model_config = ConfigDict(extra="allow")

    root: str = ""
    template_id: str = ""
    status: str = ""
    requests: list[dict[str, Any]] = Field(default_factory=list)
    request_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectTemplateReleaseNotesResponse(BaseModel):
    """Customer-facing release notes for approved reusable templates."""

    model_config = ConfigDict(extra="allow")

    root: str = ""
    notes: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    customer_claim: str = ""


class CustomerProjectTemplateReleaseNotesExportResponse(BaseModel):
    """Portable proposal/handoff bundle for approved template release notes."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    bundle: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class CustomerProjectTemplateReleaseRequestMutationResponse(BaseModel):
    """Create/review response for reusable-template release governance."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    dry_run: bool = False
    request_id: str = ""
    template_id: str = ""
    template_path: str = ""
    allowed_decisions: list[str] = Field(default_factory=list)
    preview: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    template_package: dict[str, Any] = Field(default_factory=dict)
    release_result: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class CustomerProjectTemplateReleaseUpdateResponse(BaseModel):
    """Direct reusable-template release metadata update response."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    dry_run: bool = False
    template_id: str = ""
    template_path: str = ""
    allowed_publish_statuses: list[str] = Field(default_factory=list)
    version: str = ""
    template: dict[str, Any] = Field(default_factory=dict)
    template_package: dict[str, Any] = Field(default_factory=dict)
    delivery_summary: dict[str, Any] = Field(default_factory=dict)
    delivery_checklist: dict[str, Any] = Field(default_factory=dict)
    revision: dict[str, Any] = Field(default_factory=dict)
    errors: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    next_step: str = ""


class CustomerProjectWorkbenchResponse(BaseModel):
    """Solution-provider workbench for reusable customer delivery."""

    model_config = ConfigDict(extra="allow")

    workbench_type: str = Field(min_length=1)
    overall_status: str = Field(min_length=1)
    customer_status: str = Field(min_length=1)
    release_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    scope_filtered: bool
    filters: dict[str, Any] = Field(default_factory=dict)
    delivery_surfaces: list[dict[str, Any]] = Field(default_factory=list)
    delivery_chain: dict[str, Any] = Field(default_factory=dict)
    customer_vocabulary: list[dict[str, Any]] = Field(default_factory=list)
    customer_acceptance_flow: list[dict[str, Any]] = Field(default_factory=list)
    runtime_blueprint_binding: dict[str, Any] = Field(default_factory=dict)
    customer_readable_contract: dict[str, Any] = Field(default_factory=dict)
    solution_delivery_readiness: dict[str, Any] = Field(default_factory=dict)
    customer_projects: dict[str, Any] = Field(default_factory=dict)
    template_market: dict[str, Any] = Field(default_factory=dict)
    managed_object_directory: dict[str, Any] = Field(default_factory=dict)
    delivery_resources: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectDetailResponse(BaseModel):
    """One customer project with managed-object and delivery workflow detail."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    profile: dict[str, Any] = Field(default_factory=dict)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    managed_objects: dict[str, Any] = Field(default_factory=dict)
    delivery_workflow: dict[str, Any] = Field(default_factory=dict)
    implementation_handoff: dict[str, Any] = Field(default_factory=dict)
    object_change_log: list[dict[str, Any]] = Field(default_factory=list)
    next_step: str = Field(min_length=1)


class CustomerProjectMutationResponse(BaseModel):
    """Create/update response for a customer project profile."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    profile_path: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    implementation_handoff: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class CustomerProjectManagedObjectMutationResponse(BaseModel):
    """Create/update/delete response for one managed object."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    message: str = ""
    profile_path: str = ""
    object_id: str = ""
    offline_reason: str = ""
    managed_object: dict[str, Any] = Field(default_factory=dict)
    deleted_object: dict[str, Any] = Field(default_factory=dict)
    object_change: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    implementation_handoff: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class CustomerProjectHistoryResponse(BaseModel):
    """Saved customer project revisions for rollback review."""

    model_config = ConfigDict(extra="allow")

    found: bool
    reason: str = ""
    profile_path: str = ""
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    current_profile_sha256: str = ""
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    next_step: str = ""


class CustomerProjectRollbackResponse(BaseModel):
    """Rollback dry-run or apply response for a customer project profile."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    dry_run: bool = False
    profile_path: str = ""
    revision: dict[str, Any] = Field(default_factory=dict)
    current_profile_sha256: str = ""
    target_profile_sha256: str = ""
    field_changes: Any = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    would_write: bool = False
    rollback_snapshot: dict[str, Any] = Field(default_factory=dict)
    next_step: str = ""


class CustomerProjectArchiveResponse(BaseModel):
    """Archive response for a customer project profile."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    archived_path: str = ""
    original_path: str = ""
    next_step: str = ""


class CustomerProjectExecutionBindingsResponse(BaseModel):
    """Executable ingest/runtime binding plans for one customer project."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    project_scope: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    plans: list[dict[str, Any]] = Field(default_factory=list)
    plans_by_object_id: dict[str, Any] = Field(default_factory=dict)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


class CustomerProjectExecutionRehearsalResponse(BaseModel):
    """Lab-only managed-object ingest rehearsal response."""

    model_config = ConfigDict(extra="allow")

    accepted: bool | None = None
    found: bool | None = None
    status: str = ""
    reason: str = ""
    object_id: str = ""
    allowed_modes: list[str] = Field(default_factory=list)
    rehearsal: dict[str, Any] = Field(default_factory=dict)
    project_scope: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)
    ingest_result: dict[str, Any] = Field(default_factory=dict)
    onsite_evidence_registration: dict[str, Any] = Field(default_factory=dict)
    event_id: str = ""
    production_claim_allowed: bool = False
    production_eligible: bool = False
    evidence_tier: str = ""
    customer_status: str = ""
    release_claim: str = ""
    next_step: str = ""


class CustomerProjectAcceptanceReportResponse(BaseModel):
    """Customer-readable project acceptance report."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    overall_status: str = Field(min_length=1)
    customer_status: str = Field(min_length=1)
    release_claim: str = Field(min_length=1)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    acceptance_summary: dict[str, Any] = Field(default_factory=dict)
    delivery_workflow: dict[str, Any] = Field(default_factory=dict)
    onsite_acceptance_evidence: dict[str, Any] = Field(default_factory=dict)
    field_readiness: dict[str, Any] = Field(default_factory=dict)
    execution_bindings: dict[str, Any] = Field(default_factory=dict)
    launch_readiness: dict[str, Any] = Field(default_factory=dict)
    site_acceptance_checklist: dict[str, Any] = Field(default_factory=dict)
    acceptance_reviews: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    customer_signoffs: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    env_missing: list[Any] = Field(default_factory=list)


class CustomerProjectOnsiteEvidenceResponse(BaseModel):
    """Onsite acceptance evidence receipts for one customer project."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    onsite_acceptance_evidence: dict[str, Any] = Field(default_factory=dict)
    field_readiness: dict[str, Any] = Field(default_factory=dict)
    readiness_auto_included: bool


class CustomerProjectOnsiteEvidenceRegisterResponse(BaseModel):
    """Write response for one onsite acceptance evidence receipt."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    allowed_types: list[str] = Field(default_factory=list)
    allowed_statuses: list[str] = Field(default_factory=list)
    allowed_tiers: list[str] = Field(default_factory=list)
    trust: dict[str, Any] = Field(default_factory=dict)
    profile_path: str = ""
    receipt: dict[str, Any] = Field(default_factory=dict)
    onsite_acceptance_evidence: dict[str, Any] = Field(default_factory=dict)
    field_readiness: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectAcceptanceClosureResponse(BaseModel):
    """Customer-readable acceptance closure summary."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    project_scope: dict[str, Any] = Field(default_factory=dict)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    overall_status: str = Field(min_length=1)
    customer_claim: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    manual_review: dict[str, Any] = Field(default_factory=dict)
    customer_signoff: dict[str, Any] = Field(default_factory=dict)
    onsite_acceptance_evidence: dict[str, Any] = Field(default_factory=dict)
    acceptance_report: dict[str, Any] = Field(default_factory=dict)
    site_acceptance_checklist: dict[str, Any] = Field(default_factory=dict)
    evidence_timeline: list[dict[str, Any]] = Field(default_factory=list)
    artifact_verification: dict[str, Any] = Field(default_factory=dict)
    blocked_uses: list[str] = Field(default_factory=list)


class CustomerProjectAcceptanceReviewRegisterResponse(BaseModel):
    """Write response for one delivery-owner acceptance review."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    allowed_decisions: list[str] = Field(default_factory=list)
    profile_path: str = ""
    review: dict[str, Any] = Field(default_factory=dict)
    closure: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectSignoffResponse(BaseModel):
    """Customer signoff records for one customer project."""

    model_config = ConfigDict(extra="allow")

    found: bool
    profile_path: str = Field(min_length=1)
    project_scope: dict[str, Any] = Field(default_factory=dict)
    customer: dict[str, Any] = Field(default_factory=dict)
    site: dict[str, Any] = Field(default_factory=dict)
    signoff_count: int
    signoffs: list[dict[str, Any]] = Field(default_factory=list)
    latest: dict[str, Any] | None = None


class CustomerProjectCustomerSignoffRegisterResponse(BaseModel):
    """Write response for one customer signoff decision."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    reason: str = ""
    message: str = ""
    allowed_decisions: list[str] = Field(default_factory=list)
    evidence_ref_assessment: dict[str, Any] = Field(default_factory=dict)
    profile_path: str = ""
    signoff: dict[str, Any] = Field(default_factory=dict)
    closure: dict[str, Any] = Field(default_factory=dict)
