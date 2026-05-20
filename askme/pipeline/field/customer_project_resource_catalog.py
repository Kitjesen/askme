"""Customer-project delivery resource catalog.

This module builds the cross-project view that connects managed objects to
shared product resources such as vision models, sensor protocols, skill
packages, and acceptance tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.delivery_resource_registry import (
    DEFAULT_DELIVERY_RESOURCE_ROOT,
    _delivery_resource_catalog,
    _delivery_resource_registry_next_step,
    _delivery_resource_registry_summary,
    _delivery_resource_rows,
    _merge_delivery_resource_catalog,
)
from askme.pipeline.field.paths import DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT


def build_customer_project_resource_catalog(
    profile_root: Path,
    *,
    template_root: Path | None = DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return product resource bindings used by customer projects and templates."""
    (
        load_field_site_profile,
        site_profile_paths,
        _,
        _,
        _,
        _,
    ) = _site_profile_adapters()
    consumers: list[dict[str, Any]] = []
    resource_catalog = _delivery_resource_catalog({}, delivery_resource_root=delivery_resource_root)
    for path in site_profile_paths(profile_root, pattern="*.yaml"):
        consumers.extend(
            _delivery_resource_consumers_from_profile(
                path,
                scope_type="project",
                delivery_resource_root=delivery_resource_root,
            )
        )
        try:
            resource_catalog = _merge_delivery_resource_catalog(
                resource_catalog,
                _delivery_resource_catalog(
                    load_field_site_profile(path),
                    include_defaults=False,
                    include_shared=False,
                ),
            )
        except Exception:
            continue
    if template_root is not None:
        for path in site_profile_paths(template_root, pattern="*.yaml"):
            consumers.extend(
                _delivery_resource_consumers_from_profile(
                    path,
                    scope_type="template",
                    delivery_resource_root=delivery_resource_root,
                )
            )
            try:
                resource_catalog = _merge_delivery_resource_catalog(
                    resource_catalog,
                    _delivery_resource_catalog(
                        load_field_site_profile(path),
                        include_defaults=False,
                        include_shared=False,
                    ),
                )
            except Exception:
                continue
    resources = _delivery_resource_rows(resource_catalog, consumers)
    summary = _delivery_resource_registry_summary(resources, consumers)
    return {
        "profile_root": str(profile_root),
        "template_root": str(template_root) if template_root is not None else "",
        "delivery_resource_root": str(delivery_resource_root) if delivery_resource_root is not None else "",
        "summary": summary,
        "resources": resources,
        "consumers": consumers,
        "customer_claim": (
            "Managed object bindings are checked against a product resource catalog before delivery signoff."
        ),
        "next_step": _delivery_resource_registry_next_step(summary),
    }


def _delivery_resource_consumers_from_profile(
    path: Path,
    *,
    scope_type: str,
    delivery_resource_root: Path | None,
) -> list[dict[str, Any]]:
    (
        load_field_site_profile,
        _,
        customer_payload,
        delivery_scope_payload_from_customer_site,
        managed_object_catalog_from_site_profile,
        managed_object_resource_binding_status,
    ) = _site_profile_adapters()
    try:
        profile = load_field_site_profile(path)
    except Exception as exc:
        return [{
            "scope_type": scope_type,
            "profile_path": str(path),
            "resource_type": "",
            "resource_id": "",
            "status": "read_error",
            "message": str(exc),
        }]
    customer = customer_payload(profile)
    site = _mapping(profile.get("site"))
    delivery_scope = delivery_scope_payload_from_customer_site(customer, site)
    template = _mapping(profile.get("template"))
    resource_catalog = _delivery_resource_catalog(
        profile,
        delivery_resource_root=delivery_resource_root,
    )
    objects = managed_object_catalog_from_site_profile(profile).get("objects")
    rows: list[dict[str, Any]] = []
    for item in objects if isinstance(objects, list) else []:
        binding_status = managed_object_resource_binding_status(
            _mapping(item.get("bindings")),
            resource_catalog,
        )
        checks = binding_status.get("checks")
        for check in checks if isinstance(checks, list) else []:
            if not isinstance(check, dict):
                continue
            resource_type = str(check.get("resource_type") or "")
            resource_id = str(check.get("resource_id") or "")
            if not resource_type or not resource_id:
                continue
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
                "resource_type": resource_type,
                "resource_id": resource_id,
                "status": str(check.get("status") or "unknown"),
                "source": str(check.get("source") or ""),
                "message": str(check.get("message") or ""),
            })
    return rows


def _site_profile_adapters() -> tuple[Any, Any, Any, Any, Any, Any]:
    from askme.pipeline.field.customer_project_managed_objects import (
        _managed_object_resource_binding_status,
        managed_object_catalog_from_site_profile,
    )
    from askme.pipeline.field.customer_project_profiles import _customer_payload
    from askme.pipeline.field.customer_project_scope import (
        _delivery_scope_payload_from_customer_site,
    )
    from askme.pipeline.field.customer_project_template_support import (
        _site_profile_paths,
        load_field_site_profile,
    )

    return (
        load_field_site_profile,
        _site_profile_paths,
        _customer_payload,
        _delivery_scope_payload_from_customer_site,
        managed_object_catalog_from_site_profile,
        _managed_object_resource_binding_status,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["build_customer_project_resource_catalog"]
