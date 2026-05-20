"""Evidence file access and project-scope admission checks for field APIs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItemFromEventDetail = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_FIELD_EVIDENCE_ROOT_NAMES = ("artifacts", "output", "data")


def resolve_field_evidence_path(
    raw_path: str,
    *,
    cwd: Path | None = None,
    allowed_root_names: Sequence[str] = DEFAULT_FIELD_EVIDENCE_ROOT_NAMES,
) -> Path | None:
    """Resolve a local evidence file inside approved workspace evidence roots."""
    raw = str(raw_path or "").strip().replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith(("http://", "https://", "data:")):
        return None
    root = (cwd or Path.cwd()).resolve()
    root_names = tuple(str(item) for item in allowed_root_names)
    candidate = Path(raw)
    allowed_roots = [(root / name).resolve() for name in root_names]
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if not candidate.parts or candidate.parts[0] not in root_names:
            return None
        resolved = (root / candidate).resolve()
    if not any(resolved == allowed_root or allowed_root in resolved.parents for allowed_root in allowed_roots):
        return None
    if not resolved.is_file():
        return None
    return resolved


async def field_evidence_scope_allows(
    raw_path: str,
    resolved_path: Path,
    scope: dict[str, list[str]],
    *,
    dispatch_field_operations: Dispatch,
    scope_allows: ScopeAllows,
    scope_item_from_event_detail: ScopeItemFromEventDetail,
    event_id: str = "",
) -> bool:
    """Return whether a scoped operator may access this evidence artifact."""
    if not any(scope.values()):
        return True
    event_key = str(event_id or "").strip()
    if event_key:
        detail = await dispatch_field_operations("detail_payload", event_key)
        return (
            bool(detail.get("found"))
            and scope_allows(scope, scope_item_from_event_detail(detail))
            and field_evidence_detail_references_path(detail, raw_path, resolved_path)
        )

    result = await dispatch_field_operations("list_payload", limit=500, project_scope=scope)
    events = result.get("events") if isinstance(result.get("events"), list) else []
    return any(
        isinstance(event, dict)
        and scope_allows(scope, scope_item_from_event_detail(event))
        and field_evidence_detail_references_path(event, raw_path, resolved_path)
        for event in events
    )


def field_evidence_detail_references_path(
    detail: dict[str, Any],
    raw_path: str,
    resolved_path: Path,
) -> bool:
    """Return whether an event detail payload references the requested evidence."""
    event = detail.get("event") if isinstance(detail.get("event"), dict) else detail
    for candidate in field_evidence_candidate_strings(event):
        if field_evidence_candidate_matches(candidate, raw_path, resolved_path):
            return True
    return False


def field_evidence_candidate_strings(value: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            candidates.append(text)
        return candidates
    if isinstance(value, dict):
        for item in value.values():
            candidates.extend(field_evidence_candidate_strings(item))
        return candidates
    if isinstance(value, list):
        for item in value:
            candidates.extend(field_evidence_candidate_strings(item))
    return candidates


def field_evidence_candidate_matches(
    candidate: str,
    raw_path: str,
    resolved_path: Path,
) -> bool:
    for path in field_evidence_candidate_paths(candidate):
        if field_evidence_path_key(path) == field_evidence_path_key(raw_path):
            return True
        resolved = resolve_field_evidence_path(path)
        if resolved is not None and resolved == resolved_path:
            return True
    return False


def field_evidence_candidate_paths(candidate: str) -> list[str]:
    text = str(candidate or "").strip()
    if not text:
        return []
    paths = [text]
    parsed = urlparse(text)
    if parsed.path.endswith("/api/field/evidence"):
        query = parse_qs(parsed.query)
        for value in query.get("path", []):
            decoded = unquote(str(value or "").strip())
            if decoded:
                paths.append(decoded)
    return paths


def field_evidence_path_key(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("/")


__all__ = [
    "DEFAULT_FIELD_EVIDENCE_ROOT_NAMES",
    "field_evidence_candidate_matches",
    "field_evidence_candidate_paths",
    "field_evidence_candidate_strings",
    "field_evidence_detail_references_path",
    "field_evidence_path_key",
    "field_evidence_scope_allows",
    "resolve_field_evidence_path",
]
