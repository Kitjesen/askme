"""Skill growth and customer skill-package response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillGrowthBacklogResponse(BaseModel):
    """Reviewable online-growth candidates derived from skill audit evidence."""

    model_config = ConfigDict(extra="allow")

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    backlog: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""


class SkillGrowthMutationResponse(BaseModel):
    """Mark/update result for one online-growth candidate."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    candidate_id: str = ""
    candidate: dict[str, Any] = Field(default_factory=dict)
    backlog: dict[str, Any] = Field(default_factory=dict)
    action: str = ""
    error: str = ""


class SkillGrowthDraftResponse(BaseModel):
    """Generated-skill draft result created from a growth candidate."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    candidate_id: str = ""
    candidate: dict[str, Any] = Field(default_factory=dict)
    draft: dict[str, Any] = Field(default_factory=dict)
    backlog: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class GeneratedSkillsResponse(BaseModel):
    """Generated-skill governance queue shown in the ability center."""

    model_config = ConfigDict(extra="allow")

    records: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    generated_skills: list[dict[str, Any]] = Field(default_factory=list)
    review_queue: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""


class SkillPackageCatalogResponse(BaseModel):
    """Customer or site scoped ability-package catalog."""

    model_config = ConfigDict(extra="allow")

    packages: list[dict[str, Any]] = Field(default_factory=list)
    skill_packages: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""


class SkillPackageMutationResponse(BaseModel):
    """Create, update, release, rollback, or assignment result for one package."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    package_id: str = ""
    skill_name: str = ""
    package: dict[str, Any] = Field(default_factory=dict)
    release: dict[str, Any] = Field(default_factory=dict)
    rollback: dict[str, Any] = Field(default_factory=dict)
    history: Any = Field(default_factory=list)
    error: str = ""


class SkillPackageHistoryResponse(BaseModel):
    """Version snapshots for a customer or site ability package."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    package_id: str = ""
    count: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    snapshots: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class GeneratedSkillValidationResponse(BaseModel):
    """Preflight validation result for one generated skill."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    skill_name: str = ""
    validation: dict[str, Any] = Field(default_factory=dict)
    errors: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    error: str = ""


class GeneratedSkillPreviewResponse(BaseModel):
    """Reviewable generated SKILL.md body and parsed execution policy."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    skill_name: str = ""
    description: str = ""
    voice_trigger: str = ""
    safety_level: str = ""
    execution: str = ""
    enabled: bool = False
    tags: list[str] = Field(default_factory=list)
    path: str = ""
    prompt: str = ""
    tools: str = ""
    raw_body: str = ""
    raw_body_available: bool = False
    raw_body_error: str = ""
    validation: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class GeneratedSkillReviewResponse(BaseModel):
    """Approve, reject, disable, or return a generated skill to review."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    skill_name: str = ""
    skill: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
