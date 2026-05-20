"""Shared payload helpers for memory and Knowledge Console routes."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from askme.api.schemas.memory import KnowledgePreviewRequest, MemorySearchRequest


_DISPATCH_CONTRACTS: dict[str, type[Any]] = {
    "search_payload": MemorySearchRequest,
    "preview_payload": KnowledgePreviewRequest,
}


def invalid_request_payload(message: str, *, field: str | None = None) -> dict[str, Any]:
    """Build a stable customer-readable invalid request envelope."""

    payload: dict[str, Any] = {
        "ok": False,
        "error": "invalid_request",
        "message": message,
    }
    if field:
        payload["field"] = field
    return payload


def validation_error_message(exc: ValidationError) -> tuple[str, str | None]:
    """Extract the first Pydantic validation message and dotted field path."""

    errors = exc.errors()
    if not errors:
        return "Invalid request body.", None
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "__root__")
    message = str(first.get("msg") or "Invalid request body.")
    return message, location or None


def validate_payload(body: dict[str, Any], contract: type[Any]) -> dict[str, Any]:
    """Validate an API body against a Pydantic contract and return dispatch payload."""

    parsed = contract.model_validate(body)
    return parsed.to_payload()


def validate_memory_dispatch_payload(method_name: str, body: dict[str, Any]) -> dict[str, Any]:
    """Validate route payloads that need typed API contracts before dispatch."""

    contract = _DISPATCH_CONTRACTS.get(method_name)
    if contract is None:
        return body
    return validate_payload(body, contract)


def memory_route_failure(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Return stable public failure status and payload for memory routes."""

    status_code = 503 if "not configured" in str(exc) else 500
    return status_code, {
        "ok": False,
        "error": "memory_route_failed",
        "message": str(exc),
    }


def knowledge_update_permission(action: str) -> str:
    """Map Knowledge Console update actions to explicit operator permissions."""

    normalized = str(action or "").strip().lower()
    if normalized in {"publish", "approve", "reject", "resolve", "resolve_conflict"}:
        return "knowledge:approve"
    if normalized in {"delete", "restore"}:
        return "knowledge:delete"
    if normalized in {"rollback"}:
        return "knowledge:rollback"
    if normalized in {"rebuild", "rebuild_index", "reindex"}:
        return "knowledge:rebuild"
    return "knowledge:approve"


__all__ = [
    "invalid_request_payload",
    "knowledge_update_permission",
    "memory_route_failure",
    "validate_payload",
    "validate_memory_dispatch_payload",
    "validation_error_message",
]
