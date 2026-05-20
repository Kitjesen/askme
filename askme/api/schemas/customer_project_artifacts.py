"""Customer project handoff artifact response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerProjectPackageExportResponse(BaseModel):
    """Reusable customer project handoff package export."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    package_path: str = Field(min_length=1)
    package: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectAcceptanceDossierExportResponse(BaseModel):
    """Tamper-evident customer acceptance dossier export."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    dossier_path: str = Field(min_length=1)
    html_path: str = Field(min_length=1)
    dossier: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectProposalBundleExportResponse(BaseModel):
    """Customer-facing proposal bundle export."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    package_path: str = Field(min_length=1)
    dossier_path: str = Field(min_length=1)
    proposal_path: str = Field(min_length=1)
    html_path: str = Field(min_length=1)
    proposal: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectPackageVerifyResponse(BaseModel):
    """Integrity verification result for a reusable customer project package."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    verification: dict[str, Any] = Field(default_factory=dict)
    package_scope: dict[str, Any] = Field(default_factory=dict)
    operator_project_scope: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectPackageDiffResponse(BaseModel):
    """Dry-run diff for importing a reusable customer project package."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    verification: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)
    package_scope: dict[str, Any] = Field(default_factory=dict)
    operator_project_scope: dict[str, Any] = Field(default_factory=dict)
    would_write: bool = False


class CustomerProjectPackageImportResponse(BaseModel):
    """Import or dry-run result for a reusable customer project package."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    verification: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)
    package_scope: dict[str, Any] = Field(default_factory=dict)
    operator_project_scope: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    would_write: bool = False
    delivery_gate: dict[str, Any] = Field(default_factory=dict)
    import_gate_result: str = ""
    implementation_handoff: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectAcceptanceDossierVerifyResponse(BaseModel):
    """Integrity verification result for an acceptance dossier."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    verification: dict[str, Any] = Field(default_factory=dict)
    dossier_scope: dict[str, Any] = Field(default_factory=dict)


class CustomerProjectProposalBundleVerifyResponse(BaseModel):
    """Integrity verification result for a customer proposal bundle."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    verification: dict[str, Any] = Field(default_factory=dict)
    proposal_scope: dict[str, Any] = Field(default_factory=dict)
