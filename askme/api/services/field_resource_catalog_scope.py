"""Project-scope filtering for field delivery resource catalogs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItemFromResource = Callable[[dict[str, Any]], dict[str, Any]]
ResourceHasExplicitScope = Callable[[dict[str, Any]], bool]


def scope_acceptance_registry(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
) -> dict[str, Any]:
    """Filter an acceptance registry to the current operator project scope."""
    if not any(scope.values()):
        return payload
    consumers = [
        consumer
        for consumer in payload.get("consumers", [])
        if isinstance(consumer, dict)
        and (
            consumer.get("scope_type") == "template"
            or scope_allows(scope, consumer)
        )
    ]
    references = []
    for reference in payload.get("references", []):
        if not isinstance(reference, dict):
            continue
        scoped_consumers = [
            consumer
            for consumer in reference.get("consumers", [])
            if isinstance(consumer, dict)
            and (
                consumer.get("scope_type") == "template"
                or scope_allows(scope, consumer)
            )
        ]
        if not scoped_consumers:
            continue
        row = dict(reference)
        row["consumers"] = scoped_consumers
        row["consumer_count"] = len(scoped_consumers)
        row["linked_count"] = len([
            item for item in scoped_consumers if registry_status_bucket(item.get("status")) == "linked"
        ])
        row["manual_check_count"] = len([
            item for item in scoped_consumers if registry_status_bucket(item.get("status")) == "manual_check"
        ])
        row["blocked_count"] = len([
            item for item in scoped_consumers if registry_status_bucket(item.get("status")) == "blocked"
        ])
        row["status"] = registry_overall_status(row)
        references.append(row)
    filtered = dict(payload)
    filtered["consumers"] = consumers
    filtered["references"] = references
    filtered["summary"] = registry_summary(consumers, references) | {"scope_filtered": True}
    return filtered


def scope_resource_catalog(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
    scope_item_from_resource: ScopeItemFromResource,
    resource_has_explicit_scope: ResourceHasExplicitScope,
) -> dict[str, Any]:
    """Filter resource catalog rows and consumers to the current operator scope."""
    if not any(scope.values()):
        return payload
    consumers = [
        consumer
        for consumer in payload.get("consumers", [])
        if isinstance(consumer, dict)
        and (
            consumer.get("scope_type") == "template"
            or scope_allows(scope, consumer)
        )
    ]
    resources = []
    for resource in payload.get("resources", []):
        if not isinstance(resource, dict):
            continue
        if resource_has_explicit_scope(resource) and not scope_allows(
            scope,
            scope_item_from_resource(resource),
        ):
            continue
        scoped_consumers = [
            consumer
            for consumer in resource.get("consumers", [])
            if isinstance(consumer, dict)
            and (
                consumer.get("scope_type") == "template"
                or scope_allows(scope, consumer)
            )
        ]
        if not scoped_consumers and int(resource.get("consumer_count") or 0) > 0:
            continue
        row = dict(resource)
        row["consumers"] = scoped_consumers
        row["consumer_count"] = len(scoped_consumers)
        row["project_count"] = len([
            item for item in scoped_consumers if item.get("scope_type") == "project"
        ])
        row["template_count"] = len([
            item for item in scoped_consumers if item.get("scope_type") == "template"
        ])
        row["unregistered_consumer_count"] = len([
            item for item in scoped_consumers if item.get("status") == "unregistered"
        ])
        resources.append(row)
    summary = resource_summary(resources, consumers)
    filtered = dict(payload)
    filtered["consumers"] = consumers
    filtered["resources"] = resources
    filtered["summary"] = summary | {"scope_filtered": True}
    return filtered


def scope_delivery_resource_registry(
    payload: dict[str, Any],
    scope: dict[str, list[str]],
    *,
    scope_allows: ScopeAllows,
    scope_item_from_resource: ScopeItemFromResource,
    resource_has_explicit_scope: ResourceHasExplicitScope,
) -> dict[str, Any]:
    """Filter delivery resource registry rows to the current operator scope."""
    if not any(scope.values()):
        return payload
    resources = [
        resource
        for resource in payload.get("resources", [])
        if isinstance(resource, dict)
        and (
            not resource_has_explicit_scope(resource)
            or scope_allows(scope, scope_item_from_resource(resource))
        )
    ]
    filtered = dict(payload)
    filtered["resources"] = resources
    filtered["summary"] = resource_summary(resources, []) | {"scope_filtered": True}
    filtered["delivery_resources"] = delivery_resource_tree_from_rows(resources)
    return filtered


def delivery_resource_tree_from_rows(resources: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the customer-project delivery resource tree from flat registry rows."""
    tree: dict[str, dict[str, dict[str, Any]]] = {
        resource_type: {}
        for resource_type in (
            "vision_models",
            "sensor_protocols",
            "skill_packages",
            "acceptance_tests",
        )
    }
    for resource in resources:
        resource_type = str(resource.get("resource_type") or "")
        resource_id = str(resource.get("resource_id") or "")
        if not resource_type or not resource_id:
            continue
        row = dict(resource)
        row.pop("consumers", None)
        tree.setdefault(resource_type, {})[resource_id] = row
    return tree


def registry_status_bucket(status: Any) -> str:
    text = str(status or "").strip()
    if text in {"linked", "passed", "configured"}:
        return "linked"
    if text in {"node_unresolved", "read_error", "manual_check", "not_run"}:
        return "manual_check"
    return "blocked"


def registry_overall_status(row: dict[str, Any]) -> str:
    if int(row.get("blocked_count") or 0):
        return "blocked"
    if int(row.get("manual_check_count") or 0):
        return "manual_check"
    return "linked"


def registry_summary(
    consumers: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    linked = len([item for item in consumers if registry_status_bucket(item.get("status")) == "linked"])
    manual = len([item for item in consumers if registry_status_bucket(item.get("status")) == "manual_check"])
    blocked = len([item for item in consumers if registry_status_bucket(item.get("status")) == "blocked"])
    return {
        "overall_status": "blocked" if blocked or not consumers else "manual_check" if manual else "ready",
        "reference_count": len(references),
        "consumer_count": len(consumers),
        "linked_count": linked,
        "manual_check_count": manual,
        "blocked_count": blocked,
        "project_count": len({
            str(item.get("project_id") or "")
            for item in consumers
            if item.get("project_id")
        }),
        "template_count": len({
            str(item.get("template_id") or "")
            for item in consumers
            if item.get("template_id")
        }),
        "object_count": len({
            str(item.get("object_id") or "")
            for item in consumers
            if item.get("object_id")
        }),
    }


def resource_summary(
    resources: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
) -> dict[str, Any]:
    unregistered = [item for item in resources if item.get("status") == "unregistered"]
    used = [item for item in resources if int(item.get("consumer_count") or 0) > 0]
    resource_types = sorted({
        str(item.get("resource_type") or "")
        for item in resources
        if item.get("resource_type")
    })
    return {
        "overall_status": "manual_check" if unregistered else "ready",
        "resource_count": len(resources),
        "used_resource_count": len(used),
        "consumer_count": len(consumers),
        "unregistered_resource_count": len(unregistered),
        "project_consumer_count": len([item for item in consumers if item.get("scope_type") == "project"]),
        "template_consumer_count": len([item for item in consumers if item.get("scope_type") == "template"]),
        "resource_types": resource_types,
    }


__all__ = [
    "delivery_resource_tree_from_rows",
    "registry_overall_status",
    "registry_status_bucket",
    "registry_summary",
    "resource_summary",
    "scope_acceptance_registry",
    "scope_delivery_resource_registry",
    "scope_resource_catalog",
]
