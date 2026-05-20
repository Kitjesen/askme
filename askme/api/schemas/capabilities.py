"""Capability center and package API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CapabilityCenterResponse(BaseModel):
    """Customer-facing grouped robot capability catalog."""

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    package_readiness: dict[str, Any] = Field(default_factory=dict)
    runtime_blueprints: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class RuntimeCapabilitiesResponse(BaseModel):
    """Runtime capability snapshot exposed for operators and integration tooling."""

    model_config = ConfigDict(extra="allow")

    app: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    components: dict[str, Any] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)
    openapi: dict[str, Any] = Field(default_factory=dict)
    mission_adapter: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class CapabilityPackageCatalogResponse(BaseModel):
    """Customer-visible capability and scenario package catalog."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    summary: dict[str, Any] = Field(default_factory=dict)
    release_summary: dict[str, Any] = Field(default_factory=dict)
    capability_packages: list[dict[str, Any]] = Field(default_factory=list)
    scenario_packages: list[dict[str, Any]] = Field(default_factory=list)
    inventory: dict[str, Any] = Field(default_factory=dict)
    runtime_blueprints: dict[str, Any] = Field(default_factory=dict)
    readiness: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class CapabilityPackageReadinessResponse(BaseModel):
    """Readiness decision for enabling a capability or scenario package."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    kind: str = Field(min_length=1)
    readiness: dict[str, Any] = Field(default_factory=dict)
    inventory: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class ScenarioIntentCatalogResponse(BaseModel):
    """Auditable spoken-scene routing rules for customer scenarios."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    summary: dict[str, Any] = Field(default_factory=dict)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class ScenarioIntentPreviewResponse(BaseModel):
    """Preview-only routing decision for a spoken or typed utterance."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    reason: str = ""
    text: str = ""
    matched: bool = False
    decision: dict[str, Any] | None = None
    space_resolution: dict[str, Any] | None = None
    available_skill_count: int = 0
    policy: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
