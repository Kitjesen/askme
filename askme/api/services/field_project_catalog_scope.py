"""Project and site catalog scope filtering for field product APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from askme.pipeline.field.customer_projects import (
    customer_project_catalog_acceptance_gate,
    customer_project_catalog_summary_from_projects,
)

ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItemFromSite = Callable[[dict[str, Any]], dict[str, Any]]

_DEFAULT_DELIVERY_NAMESPACE = "default"


def scope_project_catalog(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
) -> dict[str, Any]:
    """Filter a customer-project catalog to the current operator project scope."""
    if not any(scope.values()):
        return payload
    projects = [
        project
        for project in payload.get("projects", [])
        if isinstance(project, dict) and scope_allows(scope, project)
    ]
    filtered = dict(payload)
    filtered["projects"] = projects
    filtered["customers"] = customer_rows_for_projects(projects)
    summary = customer_project_catalog_summary_from_projects(
        projects,
        base_summary=payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
    )
    summary["scope_filtered"] = True
    filtered["summary"] = summary
    filtered["delivery_acceptance_gate"] = customer_project_catalog_acceptance_gate(projects)
    return filtered


def scope_site_catalog(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
    scope_item_from_site: ScopeItemFromSite,
) -> dict[str, Any]:
    """Filter a site catalog to the current operator project scope."""
    if not any(scope.values()):
        return payload
    sites = [
        site
        for site in payload.get("sites", [])
        if isinstance(site, dict) and scope_allows(scope, scope_item_from_site(site))
    ]
    filtered = dict(payload)
    filtered["sites"] = sites
    summary = dict(payload.get("summary") if isinstance(payload.get("summary"), dict) else {})
    summary.update({
        "site_count": len(sites),
        "configured_count": len([item for item in sites if item.get("status") == "passed"]),
        "blocked_count": len([item for item in sites if item.get("status") != "passed"]),
        "production_ready_count": len([
            item for item in sites if item.get("deployment_stage") == "production_ready"
        ]),
        "scope_filtered": True,
    })
    filtered["summary"] = summary
    return filtered


def customer_rows_for_projects(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build customer summary rows from scoped project rows."""
    rows: dict[str, dict[str, Any]] = {}
    for project in projects:
        customer_id = str(project.get("customer_id") or "")
        if not customer_id:
            continue
        tenant_id = str(project.get("tenant_id") or _DEFAULT_DELIVERY_NAMESPACE)
        delivery_namespace = str(project.get("delivery_namespace") or _DEFAULT_DELIVERY_NAMESPACE)
        row_key = f"{tenant_id}/{delivery_namespace}/{customer_id}"
        row = rows.setdefault(
            row_key,
            {
                "tenant_id": tenant_id,
                "delivery_namespace": delivery_namespace,
                "customer_id": customer_id,
                "customer_name": str(project.get("customer_name") or customer_id),
                "project_count": 0,
                "projects": [],
                "industries": [],
            },
        )
        row["project_count"] += 1
        row["projects"].append(str(project.get("project_id") or ""))
        industry = str(project.get("industry") or "")
        if industry and industry not in row["industries"]:
            row["industries"].append(industry)
    return list(rows.values())


__all__ = [
    "customer_rows_for_projects",
    "scope_project_catalog",
    "scope_site_catalog",
]
