"""Memory and Knowledge Console API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _RequestContract(BaseModel):
    """Base request contract that keeps forward-compatible extension fields."""

    model_config = ConfigDict(extra="allow")

    def to_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude_none=True)
        extras = getattr(self, "__pydantic_extra__", None)
        if extras:
            payload.update(extras)
        return payload


class MemoryAnswerContract(BaseModel):
    """Customer-answer rules enforced by the knowledge and memory runtime."""

    model_config = ConfigDict(extra="allow")

    contract_type: Literal["askme.customer_knowledge_answer_contract.v1"]
    evidence_required: bool
    approved_knowledge_only: bool
    current_knowledge_only: bool
    conflict_free_knowledge_only: bool
    show_evidence_in_answer: bool
    refuse_when_no_evidence: bool
    refuse_when_expired: bool
    refuse_when_conflicting: bool
    robot_behavior_memory_enters_customer_prompt: bool


class MemoryHealthResponse(BaseModel):
    """Product-facing readiness contract for customer knowledge answers."""

    model_config = ConfigDict(extra="allow")

    status: str
    ready: bool
    customer_status: str = Field(min_length=1)
    customer_next_step: str = Field(min_length=1)
    catalog_answer_ready: bool | None = None
    retrieval_runtime_ready: bool | None = None
    current_backend: str | None = None
    configured_backend: str | None = None
    selected_backend: str | None = None
    selected_backend_ready: bool | None = None
    selected_backend_installed: bool | None = None
    fallback_backend: str | None = None
    fallback_ready: bool | None = None
    memory_strategy: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)
    answer_contract: MemoryAnswerContract
    warnings: list[str] = Field(default_factory=list)


class MemorySearchRequest(_RequestContract):
    """Search approved memory/RAG evidence."""

    query: str | None = Field(default=None, min_length=1, max_length=2000)
    text: str | None = Field(default=None, min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    operator_id: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def require_query_or_text(self) -> MemorySearchRequest:
        query = (self.query or "").strip()
        text = (self.text or "").strip()
        if not query and not text:
            raise ValueError("query or text is required.")
        if not query and text:
            self.query = text
        return self

    @field_validator("query", "text", "operator_id")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class MemorySearchResponse(BaseModel):
    """Auditable memory/RAG search result used by answer evidence panels."""

    model_config = ConfigDict(extra="allow")

    query: str = ""
    results: list[dict[str, Any]] = Field(default_factory=list)
    rag: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    answer_contract: dict[str, Any] = Field(default_factory=dict)


class KnowledgePreviewRequest(_RequestContract):
    """Preview knowledge content before it can enter customer answers."""

    filename: str | None = Field(default=None, max_length=260)
    source: str | None = Field(default=None, max_length=260)
    content: str = Field(min_length=1, max_length=500_000)
    owner: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=120)
    quality_status: str | None = Field(default=None, max_length=80)
    visibility: Literal["external", "internal", "private"] | None = None
    customer_id: str | None = Field(default=None, max_length=120)
    project_id: str | None = Field(default=None, max_length=120)
    product_area: str | None = Field(default=None, max_length=120)
    workstream: str | None = Field(default=None, max_length=120)
    linked_object_type: str | None = Field(default=None, max_length=120)
    linked_object_id: str | None = Field(default=None, max_length=120)
    document_type: str | None = Field(default=None, max_length=80)
    source_version: str | None = Field(default=None, max_length=120)

    @field_validator(
        "filename",
        "source",
        "owner",
        "category",
        "quality_status",
        "customer_id",
        "project_id",
        "product_area",
        "workstream",
        "linked_object_type",
        "linked_object_id",
        "document_type",
        "source_version",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content is required.")
        return value


class KnowledgePreviewResponse(BaseModel):
    """Preview result before knowledge is allowed into customer answers."""

    model_config = ConfigDict(extra="allow")

    source: str = ""
    parsed: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any] | str] = Field(default_factory=list)
    dry_run: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class KnowledgeImportResponse(BaseModel):
    """Knowledge import result with indexing and evidence readiness details."""

    model_config = ConfigDict(extra="allow")

    source: str = ""
    parsed: int = 0
    imported: int = 0
    skipped: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any] | str] = Field(default_factory=list)
    rag: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeListResponse(BaseModel):
    """Knowledge Console list response for existing customer knowledge."""

    model_config = ConfigDict(extra="allow")

    backend: str = ""
    total: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)
    rag: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeUpdateResponse(BaseModel):
    """Knowledge metadata update result for approval, deletion, and rebuild actions."""

    model_config = ConfigDict(extra="allow")

    updated: bool = False
    record_id: str = ""
    action: str = ""
    patch: dict[str, Any] = Field(default_factory=dict)
    rag: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
