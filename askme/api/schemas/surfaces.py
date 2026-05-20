"""API surface boundary response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiSurfaceSpecResponse(BaseModel):
    """One declared HTTP API audience boundary."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    package: str = Field(min_length=1)
    registrar: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    owns: list[str] = Field(default_factory=list)
    route_modules: list[str] = Field(default_factory=list)
    must_not_expose: list[str] = Field(default_factory=list)
    customer_visible: bool
    hardware_authority_allowed: bool
    production_claim_allowed: bool
    customer_boundary: str = Field(min_length=1)


class ApiSurfaceReadinessResponse(BaseModel):
    """Customer-readable readiness gate for API surface separation."""

    model_config = ConfigDict(extra="allow")

    readiness_type: str = Field(min_length=1)
    overall_status: str = Field(min_length=1)
    customer_status: str = Field(min_length=1)
    release_claim: str = Field(min_length=1)
    policy: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class ApiRouteInventoryResponse(BaseModel):
    """Route inventory grouped by declared product surface."""

    model_config = ConfigDict(extra="allow")

    summary: dict[str, Any] = Field(default_factory=dict)
    surfaces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    routes: list[dict[str, Any]] = Field(default_factory=list)
    unclassified_routes: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)


class ApiSurfacesResponse(BaseModel):
    """Top-level response for `/api/surfaces`."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    surfaces: list[ApiSurfaceSpecResponse]
    readiness: ApiSurfaceReadinessResponse
    route_inventory: ApiRouteInventoryResponse
    policy: dict[str, Any] = Field(default_factory=dict)
