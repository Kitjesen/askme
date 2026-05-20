"""Project-scope helpers shared by field API routes."""

from __future__ import annotations

from typing import Any

DEFAULT_DELIVERY_NAMESPACE = "default"
_SCOPE_FIELDS = ("tenant_id", "delivery_namespace", "customer_id", "project_id", "site_id")


def clean_scope_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def operator_project_scope(auth_body: dict[str, Any]) -> dict[str, list[str]]:
    operator = (
        auth_body.get("operator_auth", {})
        if isinstance(auth_body.get("operator_auth"), dict)
        else {}
    ).get("operator", {})
    scope = operator.get("project_scope") if isinstance(operator, dict) else {}
    if not isinstance(scope, dict) or scope.get("unrestricted"):
        return {}
    return {
        "tenant_ids": clean_scope_values(scope.get("tenant_ids")),
        "delivery_namespaces": clean_scope_values(scope.get("delivery_namespaces")),
        "customer_ids": clean_scope_values(scope.get("customer_ids")),
        "project_ids": clean_scope_values(scope.get("project_ids")),
        "site_ids": clean_scope_values(scope.get("site_ids")),
    }


def scope_allows(scope: dict[str, list[str]], item: dict[str, Any]) -> bool:
    if not any(scope.values()):
        return True
    for scope_key, item_key in (
        ("tenant_ids", "tenant_id"),
        ("delivery_namespaces", "delivery_namespace"),
        ("customer_ids", "customer_id"),
        ("project_ids", "project_id"),
        ("site_ids", "site_id"),
    ):
        allowed = scope.get(scope_key) or []
        if "*" in allowed:
            continue
        value = str(item.get(item_key) or "").strip()
        if allowed and value not in allowed:
            return False
    return True


def scoped_query_value(
    requested: str,
    scope: dict[str, list[str]],
    scope_key: str,
) -> tuple[bool, str]:
    value = str(requested or "").strip()
    allowed = scope.get(scope_key) or []
    if not allowed or "*" in allowed:
        return True, value
    if value:
        return value in allowed, value
    return True, ""


def has_explicit_project_scope(payload: dict[str, Any]) -> bool:
    explicit_scope = payload.get("project_scope")
    if isinstance(explicit_scope, dict) and any(explicit_scope.get(key) for key in _SCOPE_FIELDS):
        return True
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict) and has_explicit_project_scope(nested_payload):
        return True
    return any(payload.get(key) for key in _SCOPE_FIELDS)


def scope_item_from_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested_payload = payload.get("payload")
    source = nested_payload if isinstance(nested_payload, dict) else payload
    explicit_scope = source.get("project_scope") if isinstance(source.get("project_scope"), dict) else {}
    return {
        "tenant_id": source.get("tenant_id") or explicit_scope.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
        "delivery_namespace": (
            source.get("delivery_namespace")
            or explicit_scope.get("delivery_namespace")
            or DEFAULT_DELIVERY_NAMESPACE
        ),
        "customer_id": source.get("customer_id") or explicit_scope.get("customer_id") or "",
        "project_id": source.get("project_id") or explicit_scope.get("project_id") or "",
        "site_id": source.get("site_id") or explicit_scope.get("site_id") or "",
    }


def apply_single_scope_defaults(payload: dict[str, Any], scope: dict[str, list[str]]) -> None:
    for payload_key, scope_key in (
        ("tenant_id", "tenant_ids"),
        ("delivery_namespace", "delivery_namespaces"),
        ("customer_id", "customer_ids"),
        ("project_id", "project_ids"),
        ("site_id", "site_ids"),
    ):
        allowed = scope.get(scope_key) or []
        if "*" in allowed or len(allowed) != 1:
            continue
        payload.setdefault(payload_key, allowed[0])


def scope_item_from_event_detail(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    project_scope = event.get("project_scope") if isinstance(event.get("project_scope"), dict) else {}
    return {
        "tenant_id": event.get("tenant_id") or project_scope.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
        "delivery_namespace": (
            event.get("delivery_namespace")
            or project_scope.get("delivery_namespace")
            or DEFAULT_DELIVERY_NAMESPACE
        ),
        "customer_id": event.get("customer_id") or project_scope.get("customer_id") or "",
        "project_id": event.get("project_id") or project_scope.get("project_id") or "",
        "site_id": event.get("site_id") or project_scope.get("site_id") or "",
    }


def scope_item_from_site(site: dict[str, Any]) -> dict[str, Any]:
    customer = site.get("customer") if isinstance(site.get("customer"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_detail(payload: dict[str, Any]) -> dict[str, Any]:
    customer = payload.get("customer") if isinstance(payload.get("customer"), dict) else {}
    site = payload.get("site") if isinstance(payload.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    customer = profile.get("customer") if isinstance(profile.get("customer"), dict) else {}
    site = profile.get("site") if isinstance(profile.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_create_body(body: dict[str, Any]) -> dict[str, Any]:
    customer = body.get("customer") if isinstance(body.get("customer"), dict) else {}
    site = body.get("site") if isinstance(body.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_package(payload: dict[str, Any]) -> dict[str, Any]:
    package = payload.get("package") if isinstance(payload.get("package"), dict) else {}
    customer = package.get("customer") if isinstance(package.get("customer"), dict) else {}
    site = package.get("site") if isinstance(package.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_dossier(payload: dict[str, Any]) -> dict[str, Any]:
    dossier = payload.get("dossier") if isinstance(payload.get("dossier"), dict) else {}
    customer = dossier.get("customer") if isinstance(dossier.get("customer"), dict) else {}
    site = dossier.get("site") if isinstance(dossier.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else payload
    customer = proposal.get("customer") if isinstance(proposal.get("customer"), dict) else {}
    site = proposal.get("site") if isinstance(proposal.get("site"), dict) else {}
    return _scope_item_from_customer_site(customer, site)


def scope_item_from_resource(payload: dict[str, Any]) -> dict[str, Any]:
    explicit_scope = payload.get("project_scope") if isinstance(payload.get("project_scope"), dict) else {}
    return {
        "tenant_id": payload.get("tenant_id") or explicit_scope.get("tenant_id") or "",
        "delivery_namespace": (
            payload.get("delivery_namespace")
            or explicit_scope.get("delivery_namespace")
            or ""
        ),
        "customer_id": payload.get("customer_id") or explicit_scope.get("customer_id") or "",
        "project_id": payload.get("project_id") or explicit_scope.get("project_id") or "",
        "site_id": payload.get("site_id") or explicit_scope.get("site_id") or "",
    }


def resource_has_explicit_scope(payload: dict[str, Any]) -> bool:
    item = scope_item_from_resource(payload)
    return any(str(item.get(key) or "").strip() for key in item)


def _scope_item_from_customer_site(
    customer: dict[str, Any],
    site: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tenant_id": customer.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
        "delivery_namespace": customer.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE,
        "customer_id": customer.get("customer_id") or "",
        "project_id": customer.get("project_id") or site.get("site_id") or "",
        "site_id": site.get("site_id") or "",
    }


__all__ = [
    "apply_single_scope_defaults",
    "clean_scope_values",
    "has_explicit_project_scope",
    "operator_project_scope",
    "resource_has_explicit_scope",
    "scope_allows",
    "scope_item_from_create_body",
    "scope_item_from_detail",
    "scope_item_from_dossier",
    "scope_item_from_event_detail",
    "scope_item_from_event_payload",
    "scope_item_from_package",
    "scope_item_from_profile",
    "scope_item_from_proposal",
    "scope_item_from_resource",
    "scope_item_from_site",
    "scoped_query_value",
]
