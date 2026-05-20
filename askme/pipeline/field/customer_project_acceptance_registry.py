"""Managed-object acceptance reference registry for customer projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_managed_objects import (
    _acceptance_test_check,
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_profiles import _customer_payload
from askme.pipeline.field.customer_project_scope import _delivery_scope_payload_from_customer_site
from askme.pipeline.field.customer_project_template_support import (
    _mapping,
    _site_profile_paths,
    _string_list,
    load_field_site_profile,
)
from askme.pipeline.field.paths import DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT


def build_customer_project_acceptance_registry(
    profile_root: Path,
    *,
    template_root: Path | None = DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
) -> dict[str, Any]:
    """Return a delivery registry for all managed-object acceptance references."""
    consumers: list[dict[str, Any]] = []
    for path in _site_profile_paths(profile_root, pattern="*.yaml"):
        consumers.extend(_acceptance_registry_consumers_from_profile(path, scope_type="project"))
    if template_root is not None:
        for path in _site_profile_paths(template_root, pattern="*.yaml"):
            consumers.extend(_acceptance_registry_consumers_from_profile(path, scope_type="template"))
    references = _acceptance_registry_references(consumers)
    summary = _acceptance_registry_summary(consumers, references)
    return {
        "profile_root": str(profile_root),
        "template_root": str(template_root) if template_root is not None else "",
        "summary": summary,
        "references": references,
        "consumers": consumers,
        "customer_claim": (
            "Managed-object acceptance references are inspectable across projects and templates."
        ),
        "next_step": _acceptance_registry_next_step(summary),
    }


def _acceptance_registry_consumers_from_profile(path: Path, *, scope_type: str) -> list[dict[str, Any]]:
    try:
        profile = load_field_site_profile(path)
    except Exception as exc:
        return [{
            "scope_type": scope_type,
            "profile_path": str(path),
            "reference": "",
            "status": "read_error",
            "message": str(exc),
        }]
    customer = _customer_payload(profile)
    site = _mapping(profile.get("site"))
    delivery_scope = _delivery_scope_payload_from_customer_site(customer, site)
    template = _mapping(profile.get("template"))
    objects = managed_object_catalog_from_site_profile(profile).get("objects")
    rows: list[dict[str, Any]] = []
    for item in objects if isinstance(objects, list) else []:
        bindings = _mapping(item.get("bindings"))
        references = _string_list(bindings.get("acceptance_tests"))
        if not references:
            rows.append({
                "scope_type": scope_type,
                "profile_path": str(path),
                "template_id": str(template.get("template_id") or path.stem) if scope_type == "template" else "",
                "tenant_id": delivery_scope["tenant_id"],
                "delivery_namespace": delivery_scope["delivery_namespace"],
                "customer_id": str(customer.get("customer_id") or ""),
                "customer_name": str(customer.get("customer_name") or ""),
                "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
                "project_name": str(customer.get("project_name") or site.get("name") or ""),
                "site_id": str(site.get("site_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or ""),
                "reference": "",
                "status": "missing",
                "message": "Managed object has no acceptance test reference.",
            })
            continue
        for reference in references:
            check = _acceptance_test_check(reference)
            rows.append({
                "scope_type": scope_type,
                "profile_path": str(path),
                "template_id": str(template.get("template_id") or path.stem) if scope_type == "template" else "",
                "tenant_id": delivery_scope["tenant_id"],
                "delivery_namespace": delivery_scope["delivery_namespace"],
                "customer_id": str(customer.get("customer_id") or ""),
                "customer_name": str(customer.get("customer_name") or ""),
                "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
                "project_name": str(customer.get("project_name") or site.get("name") or ""),
                "site_id": str(site.get("site_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or ""),
                "reference": reference,
                "status": str(check.get("status") or "unknown"),
                "path": str(check.get("path") or ""),
                "node": str(check.get("node") or ""),
                "resolved_by": str(check.get("resolved_by") or ""),
                "matched": str(check.get("matched") or ""),
                "message": str(check.get("message") or ""),
            })
    return rows


def _acceptance_registry_references(consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for consumer in consumers:
        reference = str(consumer.get("reference") or "").strip()
        if not reference:
            continue
        row = rows.setdefault(
            reference,
            {
                "reference": reference,
                "status": "linked",
                "consumer_count": 0,
                "linked_count": 0,
                "manual_check_count": 0,
                "blocked_count": 0,
                "consumers": [],
            },
        )
        row["consumer_count"] += 1
        status = str(consumer.get("status") or "unknown")
        bucket = _acceptance_registry_status_bucket(status)
        row[f"{bucket}_count"] += 1
        row["status"] = _merge_acceptance_registry_status(str(row.get("status") or "linked"), status)
        row["consumers"].append({
            "scope_type": consumer.get("scope_type"),
            "tenant_id": consumer.get("tenant_id"),
            "delivery_namespace": consumer.get("delivery_namespace"),
            "customer_id": consumer.get("customer_id"),
            "project_id": consumer.get("project_id"),
            "site_id": consumer.get("site_id"),
            "template_id": consumer.get("template_id"),
            "object_id": consumer.get("object_id"),
            "display_name": consumer.get("display_name"),
            "category": consumer.get("category"),
            "status": status,
            "resolved_by": consumer.get("resolved_by"),
            "matched": consumer.get("matched"),
        })
    return sorted(rows.values(), key=lambda item: (item["status"], item["reference"]))


def _acceptance_registry_summary(
    consumers: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {"linked": 0, "manual_check": 0, "blocked": 0}
    project_ids = set()
    template_ids = set()
    object_ids = set()
    for consumer in consumers:
        status = str(consumer.get("status") or "unknown")
        counts[_acceptance_registry_status_bucket(status)] += 1
        if consumer.get("project_id"):
            project_ids.add(str(consumer.get("project_id")))
        if consumer.get("template_id"):
            template_ids.add(str(consumer.get("template_id")))
        if consumer.get("object_id"):
            object_ids.add(str(consumer.get("object_id")))
    if not consumers or counts["blocked"]:
        overall = "blocked"
    elif counts["manual_check"]:
        overall = "manual_check"
    else:
        overall = "ready"
    return {
        "overall_status": overall,
        "reference_count": len(references),
        "consumer_count": len(consumers),
        "linked_count": counts["linked"],
        "manual_check_count": counts["manual_check"],
        "blocked_count": counts["blocked"],
        "project_count": len(project_ids),
        "template_count": len(template_ids),
        "object_count": len(object_ids),
    }


def _acceptance_registry_status_bucket(status: str) -> str:
    if status in {"linked", "passed", "configured"}:
        return "linked"
    if status in {"node_unresolved", "read_error", "manual_check", "not_run"}:
        return "manual_check"
    return "blocked"


def _merge_acceptance_registry_status(current: str, candidate: str) -> str:
    current_bucket = _acceptance_registry_status_bucket(current)
    candidate_bucket = _acceptance_registry_status_bucket(candidate)
    if "blocked" in {current_bucket, candidate_bucket}:
        return "blocked"
    if "manual_check" in {current_bucket, candidate_bucket}:
        return "manual_check"
    return "linked"


def _acceptance_registry_next_step(summary: dict[str, Any]) -> str:
    if summary.get("blocked_count"):
        return "Fix blocked acceptance references before customer signoff."
    if summary.get("manual_check_count"):
        return "Review unresolved acceptance nodes and convert them into scenario aliases or real pytest nodes."
    if summary.get("consumer_count"):
        return "Run onsite acceptance and attach live evidence to each customer project."
    return "Add acceptance tests to managed objects before customer delivery."


__all__ = [
    "build_customer_project_acceptance_registry",
    "_acceptance_registry_consumers_from_profile",
    "_acceptance_registry_next_step",
    "_acceptance_registry_references",
    "_acceptance_registry_status_bucket",
    "_acceptance_registry_summary",
    "_merge_acceptance_registry_status",
]
