"""Governance, operator directory, and IAM readiness API contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OperatorProjectScope(BaseModel):
    """Tenant/customer/project scope attached to an operator identity."""

    model_config = ConfigDict(extra="allow")

    tenant_ids: list[str] = Field(default_factory=list)
    delivery_namespaces: list[str] = Field(default_factory=list)
    customer_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    site_ids: list[str] = Field(default_factory=list)
    unrestricted: bool = False


class OperatorIdentityView(BaseModel):
    """Customer-visible operator identity used by governance endpoints."""

    model_config = ConfigDict(extra="allow")

    operator_id: str = ""
    display_name: str = ""
    roles: list[str] = Field(default_factory=list)
    source: str = ""
    authenticated: bool = False
    known: bool = False
    project_scope: OperatorProjectScope = Field(default_factory=OperatorProjectScope)


class GovernanceDirectoryReadiness(BaseModel):
    """Readiness summary for demo directory versus enterprise identity binding."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    production_ready: bool = False
    finding_count: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)


class IdentityGatewayReadinessResponse(BaseModel):
    """Enterprise IAM/SSO readiness gate for customer delivery."""

    model_config = ConfigDict(extra="allow")

    gate_type: str = ""
    status: str = ""
    production_ready: bool = False
    identity_mode: str = ""
    identity_provider: str = ""
    production_binding_required: bool = True
    production_target: str = ""
    trusted_identity_headers_enabled: bool = False
    trusted_gateway_contract: dict[str, Any] = Field(default_factory=dict)
    demo_operator_directory: dict[str, Any] = Field(default_factory=dict)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    customer_status: str = ""
    release_claim: str = ""
    next_step: str = ""


class OperatorDirectoryResponse(BaseModel):
    """Full operator directory exposed to Dashboard and delivery checks."""

    model_config = ConfigDict(extra="allow")

    mode: str = ""
    identity_provider: str = ""
    production_binding_required: bool = True
    production_target: str = ""
    session_operator_header: str = ""
    operators: list[OperatorIdentityView] = Field(default_factory=list)
    permissions: dict[str, list[str]] = Field(default_factory=dict)
    roles: list[dict[str, Any]] = Field(default_factory=list)
    authorization_matrix: list[dict[str, Any]] = Field(default_factory=list)
    readiness: GovernanceDirectoryReadiness = Field(
        default_factory=GovernanceDirectoryReadiness
    )
    identity_gateway_readiness: IdentityGatewayReadinessResponse = Field(
        default_factory=IdentityGatewayReadinessResponse
    )
    sso: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class CurrentOperatorResponse(BaseModel):
    """Resolved operator identity, permissions, and readiness warnings."""

    model_config = ConfigDict(extra="allow")

    operator: OperatorIdentityView = Field(default_factory=OperatorIdentityView)
    permissions: list[str] = Field(default_factory=list)
    known: bool = False
    authenticated: bool = False
    directory_mode: str = ""
    identity_provider: str = ""
    warnings: list[str] = Field(default_factory=list)
    readiness: GovernanceDirectoryReadiness = Field(
        default_factory=GovernanceDirectoryReadiness
    )
    identity_gateway_readiness: IdentityGatewayReadinessResponse = Field(
        default_factory=IdentityGatewayReadinessResponse
    )


class AuthorizationDecisionResponse(BaseModel):
    """RBAC authorization result for a requested product operation."""

    model_config = ConfigDict(extra="allow")

    allowed: bool = False
    permission: str = ""
    operator: OperatorIdentityView = Field(default_factory=OperatorIdentityView)
    reason: str = ""
    audit: dict[str, Any] = Field(default_factory=dict)
