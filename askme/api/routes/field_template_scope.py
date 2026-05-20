"""Template visibility helpers shared by field API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.api.routes.field_project_scope import scope_allows
from askme.pipeline.field.customer_project_templates import (
    customer_project_template_summary_from_items,
    list_customer_project_templates,
)

DEFAULT_DELIVERY_NAMESPACE = "default"


def scope_template_catalog(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
) -> dict[str, Any]:
    if not any(scope.values()):
        return payload
    templates = [
        item
        for item in payload.get("templates", [])
        if isinstance(item, dict) and scope_allows_template(scope, item)
    ]
    filtered = dict(payload)
    filtered["templates"] = templates
    summary = customer_project_template_summary_from_items(templates)
    summary["scope_filtered"] = True
    if isinstance(payload.get("summary"), dict) and payload["summary"].get("filtered"):
        summary["filtered"] = True
        summary["filters"] = payload["summary"].get("filters") or payload.get("filters") or {}
    filtered["summary"] = summary
    return filtered


def visible_template_ids(
    template_root: Path,
    scope: dict[str, list[str]],
) -> set[str] | None:
    if not any(scope.values()):
        return None
    catalog = list_customer_project_templates(Path(template_root))
    return {
        str(item.get("template_id") or "")
        for item in catalog.get("templates", [])
        if isinstance(item, dict) and scope_allows_template(scope, item)
    }


def template_visible_for_scope(
    template_root: Path,
    template_id: str,
    scope: dict[str, list[str]],
) -> tuple[bool, bool]:
    catalog = list_customer_project_templates(Path(template_root), template_id=template_id)
    templates = [item for item in catalog.get("templates", []) if isinstance(item, dict)]
    if not templates:
        return False, False
    if not any(scope.values()):
        return True, True
    return True, any(scope_allows_template(scope, item) for item in templates)


def scope_template_release_requests_payload(
    template_root: Path,
    payload: dict[str, Any],
    scope: dict[str, list[str]],
) -> dict[str, Any]:
    visible_ids = visible_template_ids(template_root, scope)
    if visible_ids is None:
        return payload
    requests = [
        item
        for item in payload.get("requests", [])
        if isinstance(item, dict) and str(item.get("template_id") or "") in visible_ids
    ]
    filtered = dict(payload)
    filtered["requests"] = requests
    filtered["request_count"] = len(requests)
    filtered["summary"] = template_release_request_summary(requests, scope_filtered=True)
    return filtered


def template_release_request_summary(
    requests: list[dict[str, Any]],
    *,
    scope_filtered: bool = False,
) -> dict[str, Any]:
    summary = {
        "pending_count": len([item for item in requests if item.get("status") == "pending"]),
        "applying_count": len([item for item in requests if item.get("status") == "applying"]),
        "apply_failed_count": len([item for item in requests if item.get("status") == "apply_failed"]),
        "approved_count": len([item for item in requests if item.get("status") == "approved"]),
        "rejected_count": len([item for item in requests if item.get("status") == "rejected"]),
    }
    if scope_filtered:
        summary["scope_filtered"] = True
    return summary


def scope_allows_template(scope: dict[str, list[str]], template: dict[str, Any]) -> bool:
    tenant_id = str(template.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
    namespace = str(template.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
    if tenant_id == DEFAULT_DELIVERY_NAMESPACE and namespace == DEFAULT_DELIVERY_NAMESPACE:
        return True
    return scope_allows(
        scope,
        {
            "tenant_id": tenant_id,
            "delivery_namespace": namespace,
            "customer_id": str(template.get("customer_id") or ""),
            "project_id": str(template.get("project_id") or ""),
            "site_id": str(template.get("site_id") or ""),
        },
    )


__all__ = [
    "scope_allows_template",
    "scope_template_catalog",
    "scope_template_release_requests_payload",
    "template_release_request_summary",
    "template_visible_for_scope",
    "visible_template_ids",
]
