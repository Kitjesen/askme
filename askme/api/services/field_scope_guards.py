"""Reusable project-scope guard checks for field APIs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_projects import get_customer_project_profile

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItem = Callable[[dict[str, Any]], dict[str, Any]]


def customer_project_scope_allows(
    root: Path,
    identifier: str,
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
    scope_item_from_detail: ScopeItem,
) -> bool:
    """Return whether a scoped operator may access a customer project profile."""
    if not any(scope.values()):
        return True
    detail = get_customer_project_profile(root, identifier)
    if not detail.get("found"):
        return True
    return scope_allows(scope, scope_item_from_detail(detail))


async def field_event_scope_allows(
    event_id: str,
    scope: dict[str, list[str]],
    *,
    dispatch_field_operations: Dispatch,
    scope_allows: ScopeAllows,
    scope_item_from_event_detail: ScopeItem,
) -> bool:
    """Return whether a scoped operator may access a field event detail."""
    if not any(scope.values()):
        return True
    detail = await dispatch_field_operations("detail_payload", event_id)
    if not detail.get("found"):
        return True
    return scope_allows(scope, scope_item_from_event_detail(detail))


__all__ = [
    "customer_project_scope_allows",
    "field_event_scope_allows",
]
