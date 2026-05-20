"""Customer-project managed-object catalog and binding evidence helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_template_support import _mapping, _string_list
from askme.pipeline.field.delivery_resource_registry import (
    DELIVERY_RESOURCE_TYPES,
    _delivery_resource_catalog,
    _delivery_resource_catalog_summary,
)
from askme.pipeline.field.paths import DEFAULT_DELIVERY_RESOURCE_ROOT, PROJECT_ROOT

_ACCEPTANCE_TEST_ALIASES: dict[str, tuple[str, ...]] = {
    "crowd_gathering": ("crowd_gathering_records_security_event",),
    "fire_or_smoke": ("fire_sensor_notifies_security",),
    "illegal_parking": ("illegal_parking_camera_ingest",),
    "trash_bin_full": ("trash_bin_full_notifies_cleaning",),
    "visitor_escort": (
        "visitor_escort_is_archived_without_alert",
        "test_guide_returns_voice_text_or_escort_payload",
        "task_type\"] == \"visitor_escort",
    ),
    "visitor_wayfinding": (
        "visitor_wayfinding_grounded",
        "test_rag_trust_scenario_visitor_wayfinding_uses_grounded_evidence",
    ),
    "wayfinding_help_point": ("wayfinding_help_point_does_not_notify_security",),
}


def managed_object_catalog_from_site_profile(
    profile: dict[str, Any],
    *,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return the customer-specific objects this project expects the robot to handle."""
    managed_objects = _mapping(profile.get("managed_objects"))
    resource_catalog = _delivery_resource_catalog(
        profile,
        delivery_resource_root=delivery_resource_root,
    )
    objects = [
        _managed_object_payload(object_id, item, resource_catalog=resource_catalog)
        for object_id, item in sorted(managed_objects.items())
        if isinstance(item, dict)
    ]
    categories = sorted({str(item.get("category") or "uncategorized") for item in objects})
    scenario_ids = sorted({
        str(scenario_id)
        for item in objects
        for scenario_id in item.get("scenario_ids", [])
        if scenario_id
    })
    return {
        "object_type_count": len(objects),
        "categories": categories,
        "scenario_ids": scenario_ids,
        "resource_catalog_summary": _delivery_resource_catalog_summary(resource_catalog),
        "binding_readiness_summary": _managed_object_binding_readiness_summary(objects),
        "acceptance_summary": _managed_object_acceptance_summary(objects),
        "objects": objects,
        "objects_by_id": {str(item["object_id"]): item for item in objects},
        "customer_claim": (
            "Managed object catalog is configured for this customer project."
            if objects
            else "Managed object catalog is not configured yet."
        ),
    }


def _managed_object_binding_readiness_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"ready": 0, "manual_check": 0, "blocked": 0}
    unregistered: list[dict[str, str]] = []
    for item in objects:
        status_payload = _mapping(item.get("resource_binding_status"))
        status = str(status_payload.get("overall_status") or "blocked")
        counts[status if status in counts else "manual_check"] += 1
        for check in status_payload.get("checks", []) if isinstance(status_payload.get("checks"), list) else []:
            if not isinstance(check, dict) or check.get("status") != "unregistered":
                continue
            unregistered.append({
                "object_id": str(item.get("object_id") or ""),
                "resource_type": str(check.get("resource_type") or ""),
                "resource_id": str(check.get("resource_id") or ""),
            })
    if not objects or counts["blocked"]:
        overall_status = "blocked"
    elif counts["manual_check"]:
        overall_status = "manual_check"
    else:
        overall_status = "ready"
    return {
        "overall_status": overall_status,
        "ready_object_count": counts["ready"],
        "manual_check_object_count": counts["manual_check"],
        "blocked_object_count": counts["blocked"],
        "object_count": len(objects),
        "unregistered_resource_count": len(unregistered),
        "unregistered_resources": unregistered[:20],
    }


def _managed_object_acceptance_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {"ready": 0, "manual_check": 0, "blocked": 0}
    for item in objects:
        status = str(_mapping(item.get("acceptance_status")).get("status") or "blocked")
        status_counts[status if status in status_counts else "manual_check"] += 1
    if not objects or status_counts["blocked"]:
        overall_status = "blocked"
    elif status_counts["manual_check"]:
        overall_status = "manual_check"
    else:
        overall_status = "ready"
    return {
        "overall_status": overall_status,
        "ready_object_count": status_counts["ready"],
        "manual_check_object_count": status_counts["manual_check"],
        "blocked_object_count": status_counts["blocked"],
        "object_count": len(objects),
    }


def _managed_object_binding_missing_count(objects: list[dict[str, Any]]) -> int:
    missing = 0
    for item in objects:
        status = _mapping(item.get("acceptance_status"))
        for requirement in status.get("requirements", []):
            if isinstance(requirement, dict) and requirement.get("status") == "missing":
                missing += 1
    return missing


def _managed_object_payload(
    object_id: str,
    item: dict[str, Any],
    *,
    resource_catalog: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    bindings = _managed_object_binding_payload(_mapping(item.get("bindings")))
    resource_binding_status = _managed_object_resource_binding_status(
        bindings,
        resource_catalog or _delivery_resource_catalog({}),
    )
    return {
        "object_id": str(object_id),
        "display_name": str(item.get("display_name") or object_id),
        "category": str(item.get("category") or "uncategorized"),
        "object_labels": _string_list(item.get("object_labels")),
        "scenario_ids": _string_list(item.get("scenario_ids")),
        "zone_types": _string_list(item.get("zone_types")),
        "device_sources": _string_list(item.get("device_sources")),
        "tenant_ids": _string_list(item.get("tenant_ids") or item.get("tenant_id")),
        "delivery_namespaces": _string_list(
            item.get("delivery_namespaces") or item.get("delivery_namespace")
        ),
        "customer_ids": _string_list(item.get("customer_ids") or item.get("customer_id")),
        "project_ids": _string_list(item.get("project_ids") or item.get("project_id")),
        "site_ids": _string_list(item.get("site_ids") or item.get("site_id")),
        "responder_group": str(item.get("responder_group") or ""),
        "evidence_required": _string_list(item.get("evidence_required")),
        "bindings": bindings,
        "resource_binding_status": resource_binding_status,
        "acceptance_status": _managed_object_acceptance_status(bindings),
        "customer_visible": bool(item.get("customer_visible", True)),
    }


def _managed_object_binding_payload(bindings: dict[str, Any]) -> dict[str, Any]:
    return {
        "vision_models": _string_list(bindings.get("vision_models")),
        "sensor_protocols": _string_list(bindings.get("sensor_protocols")),
        "skill_packages": _string_list(bindings.get("skill_packages")),
        "acceptance_tests": _string_list(bindings.get("acceptance_tests")),
    }


def _managed_object_resource_binding_status(
    bindings: dict[str, Any],
    resource_catalog: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blocked_count = 0
    manual_check_count = 0
    linked_count = 0
    for resource_type in DELIVERY_RESOURCE_TYPES:
        values = _string_list(bindings.get(resource_type))
        if not values:
            checks.append({
                "resource_type": resource_type,
                "resource_id": "",
                "status": "missing",
                "message": f"{resource_type} binding is required.",
            })
            blocked_count += 1
            continue
        for value in values:
            if resource_type == "acceptance_tests":
                resource = _mapping(resource_catalog.get(resource_type)).get(value)
                link_status = (
                    _delivery_resource_link_status(_mapping(resource))
                    if resource
                    else {"bucket": "linked", "message": ""}
                )
                check = _acceptance_test_check(value)
                bucket = _acceptance_resource_bucket(str(check.get("status") or "unknown"))
                if str(link_status.get("bucket") or "linked") != "linked":
                    bucket = str(link_status.get("bucket") or bucket)
                checks.append({
                    "resource_type": resource_type,
                    "resource_id": value,
                    "status": bucket,
                    "reference_status": str(check.get("status") or "unknown"),
                    "source": str(check.get("resolved_by") or ""),
                    "publish_status": str(_mapping(resource).get("publish_status") or ""),
                    "message": str(link_status.get("message") or check.get("message") or ""),
                })
            else:
                resource = _mapping(resource_catalog.get(resource_type)).get(value)
                if resource:
                    link_status = _delivery_resource_link_status(_mapping(resource))
                    bucket = str(link_status.get("bucket") or "linked")
                    checks.append({
                        "resource_type": resource_type,
                        "resource_id": value,
                        "status": bucket,
                        "display_name": str(resource.get("display_name") or value),
                        "version": str(resource.get("version") or ""),
                        "source": str(resource.get("source") or ""),
                        "category": str(resource.get("category") or ""),
                        "publish_status": str(resource.get("publish_status") or ""),
                        "message": str(link_status.get("message") or ""),
                    })
                else:
                    bucket = "unregistered"
                    checks.append({
                        "resource_type": resource_type,
                        "resource_id": value,
                        "status": bucket,
                                "message": "资源未登记在交付资源目录中。",
                    })
            if bucket == "linked":
                linked_count += 1
            elif bucket in {"manual_check", "unregistered"}:
                manual_check_count += 1
            else:
                blocked_count += 1
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check"
        if manual_check_count
        else "ready"
    )
    return {
        "overall_status": overall_status,
        "linked_count": linked_count,
        "manual_check_count": manual_check_count,
        "blocked_count": blocked_count,
        "check_count": len(checks),
        "checks": checks,
        "customer_status": {
            "ready": "对象所需资源已全部关联到交付目录。",
            "manual_check": "部分对象资源需要目录复核。",
            "blocked": "部分必需对象资源缺失或无效。",
        }[overall_status],
    }


def _delivery_resource_link_status(resource: dict[str, Any]) -> dict[str, str]:
    publish_status = str(resource.get("publish_status") or "published").strip()
    if publish_status in {"disabled", "blocked"}:
        return {
            "bucket": "blocked",
            "message": f"资源状态为 {publish_status}，不能用于客户交付。",
        }
    if publish_status in {"draft", "pilot", "deprecated"}:
        return {
            "bucket": "manual_check",
            "message": f"资源状态为 {publish_status}，需要交付负责人复核。",
        }
    return {"bucket": "linked", "message": "资源可用于交付绑定。"}


def _acceptance_resource_bucket(status: str) -> str:
    if status in {"linked"}:
        return "linked"
    if status in {"node_unresolved", "read_error"}:
        return "manual_check"
    return "blocked"


def _managed_object_acceptance_status(bindings: dict[str, Any]) -> dict[str, Any]:
    requirements = []
    for key, label in (
        ("vision_models", "视觉模型"),
        ("sensor_protocols", "传感器协议"),
        ("skill_packages", "技能包"),
        ("acceptance_tests", "验收用例"),
    ):
        values = _string_list(bindings.get(key))
        requirements.append({
            "key": key,
            "label": label,
            "status": "configured" if values else "missing",
            "count": len(values),
            "items": values,
        })
    missing = [item["key"] for item in requirements if item["status"] == "missing"]
    acceptance_checks = [
        _acceptance_test_check(reference)
        for reference in _string_list(bindings.get("acceptance_tests"))
    ]
    blocked_checks = [
        item
        for item in acceptance_checks
        if item.get("status") in {"file_missing", "invalid_reference", "outside_project"}
    ]
    manual_checks = [
        item
        for item in acceptance_checks
        if item.get("status") in {"node_unresolved", "read_error"}
    ]
    if not missing and not blocked_checks and not manual_checks:
        status = "ready"
        customer_status = "验收证据已关联。"
        next_step = "在客户现场运行该对象的验收用例。"
    elif blocked_checks:
        status = "blocked"
        customer_status = "验收用例证据缺失。"
        next_step = "客户验收前先修复缺失或不安全的验收用例引用。"
    elif manual_checks:
        status = "manual_check"
        customer_status = "验收用例证据需要复核。"
        next_step = "修复验收用例节点，或补充明确的场景别名。"
    elif len(missing) == len(requirements):
        status = "blocked"
        customer_status = "验收绑定缺失。"
        next_step = "绑定视觉模型、传感器协议、技能包和验收用例。"
    else:
        status = "manual_check"
        customer_status = "验收绑定不完整。"
        next_step = "客户验收前补齐缺失绑定。"
    return {
        "status": status,
        "customer_status": customer_status,
        "missing": missing,
        "requirements": requirements,
        "acceptance_checks": acceptance_checks,
        "next_step": next_step,
    }


def _acceptance_test_check(reference: str) -> dict[str, Any]:
    ref = str(reference or "").strip()
    path_text, separator, node = ref.partition("::")
    if not ref or not path_text:
        return {
            "reference": ref,
            "status": "invalid_reference",
            "path": path_text,
            "node": node,
            "message": "验收引用必须使用本地测试文件路径。",
        }
    path = Path(path_text)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        resolved = resolved.resolve()
    except OSError as exc:
        return {
            "reference": ref,
            "status": "file_missing",
            "path": path_text,
            "node": node,
            "message": str(exc),
        }
    if not resolved.is_relative_to(PROJECT_ROOT):
        return {
            "reference": ref,
            "status": "outside_project",
            "path": path_text,
            "node": node,
            "message": "验收用例证据必须位于项目仓库内。",
        }
    if not resolved.exists() or not resolved.is_file():
        return {
            "reference": ref,
            "status": "file_missing",
            "path": path_text,
            "node": node,
            "message": "未找到验收用例文件。",
        }
    if not separator:
        return {
            "reference": ref,
            "status": "linked",
            "path": path_text,
            "node": "",
            "resolved_by": "file",
            "matched": resolved.name,
        }
    try:
        text = resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {
            "reference": ref,
            "status": "read_error",
            "path": path_text,
            "node": node,
            "message": str(exc),
        }
    match = _acceptance_node_match(text, node)
    if match:
        return {
            "reference": ref,
            "status": "linked",
            "path": path_text,
            "node": node,
            "resolved_by": match["resolved_by"],
            "matched": match["matched"],
        }
    return {
        "reference": ref,
        "status": "node_unresolved",
        "path": path_text,
        "node": node,
        "message": "验收用例文件存在，但未找到引用的节点或场景别名。",
    }


def _acceptance_node_match(text: str, node: str) -> dict[str, str] | None:
    node = str(node or "").strip()
    pytest_candidates = [node, f"test_{node}"] if node else []
    for candidate in [item for item in pytest_candidates if item]:
        if re.search(rf"\bdef\s+{re.escape(candidate)}\s*\(", text):
            return {"resolved_by": "pytest_node", "matched": candidate}
    for candidate in _ACCEPTANCE_TEST_ALIASES.get(node, ()):
        if candidate in text:
            return {"resolved_by": "scenario_alias", "matched": candidate}
    if node and node in text:
        return {"resolved_by": "literal", "matched": node}
    return None


__all__ = [
    "managed_object_catalog_from_site_profile",
    "_ACCEPTANCE_TEST_ALIASES",
    "_acceptance_node_match",
    "_acceptance_resource_bucket",
    "_acceptance_test_check",
    "_delivery_resource_link_status",
    "_managed_object_acceptance_summary",
    "_managed_object_acceptance_status",
    "_managed_object_binding_readiness_summary",
    "_managed_object_binding_missing_count",
    "_managed_object_binding_payload",
    "_managed_object_payload",
    "_managed_object_resource_binding_status",
]
