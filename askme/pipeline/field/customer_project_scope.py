"""Customer-project delivery scope and profile diff helpers."""

from __future__ import annotations

from typing import Any

from askme.pipeline.field.customer_project_template_support import (
    DEFAULT_DELIVERY_NAMESPACE,
    _delivery_namespace,
    _delivery_tenant_id,
    _mapping,
    _sha256_json,
    _slug,
)


def _delivery_scope_payload(profile: dict[str, Any]) -> dict[str, str]:
    return _delivery_scope_payload_from_customer_site(
        _mapping(profile.get("customer")),
        _mapping(profile.get("site")),
    )


def _delivery_scope_payload_from_customer_site(
    customer: dict[str, Any],
    site: dict[str, Any],
) -> dict[str, str]:
    tenant_id = _delivery_tenant_id(customer)
    delivery_namespace = _delivery_namespace(customer)
    return {
        "tenant_id": tenant_id,
        "delivery_namespace": delivery_namespace,
        "customer_id": str(customer.get("customer_id") or ""),
        "project_id": str(customer.get("project_id") or site.get("site_id") or ""),
        "site_id": str(site.get("site_id") or ""),
    }


def _same_delivery_project_scope(current: dict[str, str], incoming: dict[str, str]) -> bool:
    if str(current.get("tenant_id") or "") != str(incoming.get("tenant_id") or ""):
        return False
    if str(current.get("delivery_namespace") or "") != str(
        incoming.get("delivery_namespace") or ""
    ):
        return False
    return _same_customer_project_identity(current, incoming)


def _same_customer_project_identity(current: dict[str, str], incoming: dict[str, str]) -> bool:
    current_customer = str(current.get("customer_id") or "")
    incoming_customer = str(incoming.get("customer_id") or "")
    current_project = str(current.get("project_id") or "")
    incoming_project = str(incoming.get("project_id") or "")
    current_site = str(current.get("site_id") or "")
    incoming_site = str(incoming.get("site_id") or "")
    if current_customer and incoming_customer and current_customer == incoming_customer:
        if current_project and incoming_project and current_project == incoming_project:
            return True
        if current_site and incoming_site and current_site == incoming_site:
            return True
    if current_project and incoming_project and current_site and incoming_site:
        return current_project == incoming_project and current_site == incoming_site
    return False


def _customer_project_profile_diff(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in (
        ("customer",),
        ("site",),
        ("zones",),
        ("devices",),
        ("responder_groups",),
        ("thresholds",),
        ("managed_objects",),
    ):
        current_value = _get_nested(current, path)
        incoming_value = _get_nested(incoming, path)
        if current_value != incoming_value:
            changes.append(
                {
                    "path": ".".join(path),
                    "current_sha256": _sha256_json(current_value),
                    "incoming_sha256": _sha256_json(incoming_value),
                }
            )
    return changes


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _customer_delivery_filename_parts(customer: dict[str, Any]) -> list[str]:
    tenant = _slug(_delivery_tenant_id(customer))
    namespace = _slug(_delivery_namespace(customer))
    if tenant == DEFAULT_DELIVERY_NAMESPACE and namespace == DEFAULT_DELIVERY_NAMESPACE:
        return []
    if namespace == tenant:
        return [tenant]
    return [tenant, namespace]


__all__ = [
    "_customer_delivery_filename_parts",
    "_customer_project_profile_diff",
    "_delivery_scope_payload",
    "_delivery_scope_payload_from_customer_site",
    "_get_nested",
    "_same_customer_project_identity",
    "_same_delivery_project_scope",
]
