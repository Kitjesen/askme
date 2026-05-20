"""Customer-project package delivery rules."""

from __future__ import annotations

from typing import Any

from askme.pipeline.field.customer_project_template_support import (
    _mapping,
    _sha256_json,
    _string_list,
)


def _customer_project_package_action_plan(managed_object_catalog: dict[str, Any]) -> dict[str, Any]:
    objects = (
        managed_object_catalog.get("objects")
        if isinstance(managed_object_catalog.get("objects"), list)
        else []
    )
    actions: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        common = {
            "object_id": object_id,
            "display_name": str(item.get("display_name") or object_id),
            "object_type": str(item.get("category") or ""),
            "category": str(item.get("category") or ""),
        }
        resource_status = _mapping(item.get("resource_binding_status"))
        for check in resource_status.get("checks", []) if isinstance(resource_status.get("checks"), list) else []:
            check = _mapping(check)
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            resource_type = str(check.get("resource_type") or "")
            resource_id = str(check.get("resource_id") or "")
            if status == "missing":
                actions.append({
                    **common,
                    "action": "bind_required_resource",
                    "reason_code": "resource_binding_missing",
                    "reason_label": f"{resource_type} binding is missing.",
                    "severity": "blocked",
                    "owner": "delivery_owner",
                    "target": resource_type,
                    "source": "resource_binding_status",
                    "next_step": f"Bind a {resource_type} resource before exporting as deliverable.",
                })
                continue
            if status == "unregistered":
                actions.append({
                    **common,
                    "action": "register_delivery_resource",
                    "reason_code": "resource_not_registered",
                    "reason_label": f"{resource_type} {resource_id} is not in the resource catalog.",
                    "severity": "manual_check",
                    "owner": "delivery_owner",
                    "target": resource_id,
                    "source": "resource_binding_status",
                    "next_step": "Register or replace the delivery resource, then re-evaluate the package.",
                })
                continue
            if status == "manual_check":
                actions.append({
                    **common,
                    "action": "review_delivery_resource",
                    "reason_code": "resource_manual_check_required",
                    "reason_label": str(check.get("message") or f"{resource_id} requires delivery review."),
                    "severity": "manual_check",
                    "owner": "delivery_owner",
                    "target": resource_id,
                    "source": "resource_binding_status",
                    "next_step": "Approve, replace, or publish the resource before customer signoff.",
                })
                continue
            actions.append({
                **common,
                "action": "replace_blocked_resource",
                "reason_code": "resource_binding_blocked",
                "reason_label": str(check.get("message") or f"{resource_id} cannot be used for delivery."),
                "severity": "blocked",
                "owner": "delivery_owner",
                "target": resource_id or resource_type,
                "source": "resource_binding_status",
                "next_step": "Replace the blocked resource before importing this package.",
            })

        acceptance = _mapping(item.get("acceptance_status"))
        acceptance_status = str(acceptance.get("status") or "blocked")
        for requirement in _string_list(acceptance.get("missing")):
            severity = "blocked" if acceptance_status == "blocked" else "manual_check"
            actions.append({
                **common,
                "action": "bind_acceptance_requirement",
                "reason_code": "acceptance_requirement_missing",
                "reason_label": f"Acceptance requirement {requirement} is missing.",
                "severity": severity,
                "owner": "delivery_owner",
                "target": requirement,
                "source": "acceptance_status",
                "next_step": "Complete required model, protocol, skill, and test bindings.",
            })
        for check in acceptance.get("acceptance_checks", []) if isinstance(acceptance.get("acceptance_checks"), list) else []:
            check = _mapping(check)
            status = str(check.get("status") or "")
            if status == "linked":
                continue
            reference = str(check.get("reference") or "")
            if status in {"node_unresolved", "read_error"}:
                actions.append({
                    **common,
                    "action": "resolve_acceptance_test_reference",
                    "reason_code": "acceptance_test_manual_check_required",
                    "reason_label": str(check.get("message") or "Acceptance test needs review."),
                    "severity": "manual_check",
                    "owner": "qa_owner",
                    "target": reference,
                    "source": "acceptance_status",
                    "next_step": "Resolve the pytest node or add a stable scenario alias.",
                })
                continue
            actions.append({
                **common,
                "action": "fix_acceptance_test_reference",
                "reason_code": "acceptance_test_blocked",
                "reason_label": str(check.get("message") or "Acceptance test reference is blocked."),
                "severity": "blocked",
                "owner": "qa_owner",
                "target": reference,
                "source": "acceptance_status",
                "next_step": "Fix the missing or unsafe acceptance test reference.",
            })

    if not objects:
        actions.append({
            "object_id": "",
            "display_name": "",
            "object_type": "",
            "category": "",
            "action": "add_managed_object_scope",
            "reason_code": "managed_object_catalog_empty",
            "reason_label": "Managed object catalog is empty.",
            "severity": "blocked",
            "owner": "delivery_owner",
            "target": "managed_objects",
            "source": "managed_object_catalog",
            "next_step": "Add customer-specific managed objects before exporting a handoff package.",
        })

    blocked_count = len([item for item in actions if item.get("severity") == "blocked"])
    manual_count = len([item for item in actions if item.get("severity") == "manual_check"])
    overall_status = (
        "blocked"
        if blocked_count
        else "manual_check_required"
        if manual_count
        else "deliverable"
    )
    source_version = _sha256_json({
        "objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "resource_binding_status": _mapping(item.get("resource_binding_status")).get("overall_status"),
                "acceptance_status": _mapping(item.get("acceptance_status")).get("status"),
            }
            for item in objects
            if isinstance(item, dict)
        ],
        "actions": [
            {
                "object_id": str(item.get("object_id") or ""),
                "action": str(item.get("action") or ""),
                "severity": str(item.get("severity") or ""),
                "target": str(item.get("target") or ""),
            }
            for item in actions
        ],
    })
    return {
        "plan_type": "askme.customer_project.managed_object_action_plan",
        "overall_status": overall_status,
        "object_count": len(objects),
        "object_action_count": len({str(item.get("object_id") or "") for item in actions if item.get("object_id")}),
        "action_count": len(actions),
        "blocked_action_count": blocked_count,
        "manual_check_action_count": manual_count,
        "delivery_gate_source_version": source_version,
        "actions": actions[:100],
        "next_step": {
            "deliverable": "No managed-object package action is open.",
            "manual_check_required": "Resolve manual checks before customer signoff.",
            "blocked": "Fix blocked object bindings before importing as a customer project.",
        }[overall_status],
    }


def _customer_project_package_delivery_gate(
    *,
    action_plan: dict[str, Any],
    acceptance_summary: dict[str, Any],
    binding_readiness: dict[str, Any],
    reuse_assessment: dict[str, Any],
    checked_at: str = "",
) -> dict[str, Any]:
    reasons = [
        {
            "object_id": str(item.get("object_id") or ""),
            "object_type": str(item.get("object_type") or item.get("category") or ""),
            "reason_code": str(item.get("reason_code") or "action_plan_present"),
            "reason_label": str(item.get("reason_label") or item.get("action") or ""),
            "severity": str(item.get("severity") or "manual_check"),
            "source": str(item.get("source") or "managed_object_action_plan"),
            "owner": str(item.get("owner") or ""),
            "target": str(item.get("target") or ""),
            "next_step": str(item.get("next_step") or ""),
        }
        for item in action_plan.get("actions", [])
        if isinstance(item, dict)
    ]
    blocked_count = int(action_plan.get("blocked_action_count") or 0)
    manual_count = int(action_plan.get("manual_check_action_count") or 0)
    for status_source, status in (
        ("acceptance_summary", str(_mapping(acceptance_summary).get("overall_status") or "")),
        ("binding_readiness_summary", str(_mapping(binding_readiness).get("overall_status") or "")),
        ("reuse_assessment", str(_mapping(reuse_assessment).get("status") or "")),
    ):
        if status == "blocked" and not any(item["source"] == status_source for item in reasons):
            blocked_count += 1
            reasons.append({
                "object_id": "",
                "object_type": "",
                "reason_code": f"{status_source}_blocked",
                "reason_label": f"{status_source} is blocked.",
                "severity": "blocked",
                "source": status_source,
                "owner": "delivery_owner",
                "target": status_source,
                "next_step": "Fix the blocked delivery gate before customer handoff.",
            })
        elif status == "manual_check" and not any(item["source"] == status_source for item in reasons):
            manual_count += 1
            reasons.append({
                "object_id": "",
                "object_type": "",
                "reason_code": f"{status_source}_manual_check_required",
                "reason_label": f"{status_source} requires manual review.",
                "severity": "manual_check",
                "source": status_source,
                "owner": "delivery_owner",
                "target": status_source,
                "next_step": "Review this gate before customer signoff.",
            })
    if blocked_count:
        status = "blocked"
    elif manual_count:
        status = "manual_check_required"
    else:
        status = "deliverable"
    export_allowed = status != "blocked"
    import_allowed = status != "blocked"
    return {
        "gate_type": "askme.customer_project.package_delivery_gate",
        "delivery_gate_status": status,
        "delivery_gate_reasons": reasons,
        "delivery_gate_checked_at": checked_at,
        "delivery_gate_source_version": str(action_plan.get("delivery_gate_source_version") or ""),
        "export_allowed": export_allowed,
        "import_allowed": import_allowed,
        "customer_handoff_ready": status == "deliverable",
        "action_count": len(reasons),
        "blocked_action_count": blocked_count,
        "manual_check_action_count": manual_count,
        "customer_status": {
            "deliverable": "交付包没有未关闭的现场对象交付事项。",
            "manual_check_required": "交付包可导入试点项目，但人工检查关闭前不能签收。",
            "blocked": "交付包存在阻断项，不能作为客户项目交接包导入。",
        }[status],
        "release_claim": "该门禁只控制客户交接就绪度；生产上线仍需要现场设备、通知、语音和机器人运行验收。",
        "next_step": {
            "deliverable": "导入目标客户命名空间后执行现场验收。",
            "manual_check_required": "仅导入受控试点命名空间，并在签收前关闭人工检查。",
            "blocked": "修复阻断的对象绑定或验收证据后重新导出交付包。",
        }[status],
    }


def _customer_project_package_import_gate_result(delivery_gate: dict[str, Any]) -> str:
    status = str(delivery_gate.get("delivery_gate_status") or "blocked")
    if status == "deliverable":
        return "accepted"
    if status == "manual_check_required":
        return "accepted_with_manual_check"
    return "rejected"

__all__ = [
    "_customer_project_package_action_plan",
    "_customer_project_package_delivery_gate",
    "_customer_project_package_import_gate_result",
]
