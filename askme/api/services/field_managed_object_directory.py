"""Customer-project managed object directory service.

Routes should only collect request filters and enforce authorization. This
module owns the product-facing object directory rows, summaries, and action
plans used by the customer project page and delivery tooling.
"""

from __future__ import annotations

from typing import Any

DEFAULT_DELIVERY_NAMESPACE = "default"


def managed_object_delivery_status(item: dict[str, Any]) -> str:
    """Return the delivery status for one managed object."""
    resource_status = item.get("resource_binding_status")
    acceptance_status = item.get("acceptance_status")
    resource = (
        str(resource_status.get("overall_status") or "manual_check")
        if isinstance(resource_status, dict)
        else "manual_check"
    )
    acceptance = (
        str(acceptance_status.get("status") or "manual_check")
        if isinstance(acceptance_status, dict)
        else "manual_check"
    )
    if resource in {"blocked", "failed"} or acceptance in {
        "blocked",
        "failed",
        "file_missing",
        "outside_project",
    }:
        return "blocked"
    if resource == "ready" and acceptance == "ready":
        return "ready"
    return "manual_check"


def managed_object_directory_rows(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten customer projects into customer-readable managed-object rows."""
    rows: list[dict[str, Any]] = []
    for project in projects:
        objects = project.get("managed_objects") if isinstance(project.get("managed_objects"), list) else []
        for item in objects:
            if not isinstance(item, dict):
                continue
            bindings = item.get("bindings") if isinstance(item.get("bindings"), dict) else {}
            resource_status = (
                item.get("resource_binding_status")
                if isinstance(item.get("resource_binding_status"), dict)
                else {}
            )
            acceptance_status = (
                item.get("acceptance_status")
                if isinstance(item.get("acceptance_status"), dict)
                else {}
            )
            row = {
                "tenant_id": project.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
                "delivery_namespace": project.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE,
                "customer_id": project.get("customer_id") or "",
                "customer_name": project.get("customer_name") or project.get("customer_id") or "",
                "project_id": project.get("project_id") or "",
                "project_name": project.get("project_name") or project.get("project_id") or "",
                "site_id": project.get("site_id") or "",
                "site_name": project.get("site_name") or project.get("site_id") or "",
                "industry": project.get("industry") or "",
                "deployment_stage": project.get("deployment_stage") or "",
                "project_status": project.get("status") or "",
                "object_id": item.get("object_id") or "",
                "display_name": item.get("display_name") or item.get("object_id") or "",
                "category": item.get("category") or "",
                "object_labels": item.get("object_labels") if isinstance(item.get("object_labels"), list) else [],
                "scenario_ids": item.get("scenario_ids") if isinstance(item.get("scenario_ids"), list) else [],
                "zone_types": item.get("zone_types") if isinstance(item.get("zone_types"), list) else [],
                "device_sources": (
                    item.get("device_sources") if isinstance(item.get("device_sources"), list) else []
                ),
                "responder_group": item.get("responder_group") or "",
                "evidence_required": (
                    item.get("evidence_required")
                    if isinstance(item.get("evidence_required"), list)
                    else []
                ),
                "customer_visible": item.get("customer_visible") is not False,
                "scope_constraints": {
                    "tenant_ids": item.get("tenant_ids") if isinstance(item.get("tenant_ids"), list) else [],
                    "delivery_namespaces": (
                        item.get("delivery_namespaces")
                        if isinstance(item.get("delivery_namespaces"), list)
                        else []
                    ),
                    "customer_ids": item.get("customer_ids") if isinstance(item.get("customer_ids"), list) else [],
                    "project_ids": item.get("project_ids") if isinstance(item.get("project_ids"), list) else [],
                    "site_ids": item.get("site_ids") if isinstance(item.get("site_ids"), list) else [],
                },
                "bindings": bindings,
                "resource_binding_status": resource_status,
                "acceptance_status": acceptance_status,
                "delivery_status": managed_object_delivery_status(item),
                "resource_check_count": int(resource_status.get("check_count") or 0),
                "acceptance_test_count": len(
                    bindings.get("acceptance_tests")
                    if isinstance(bindings.get("acceptance_tests"), list)
                    else []
                ),
                "acceptance_check_count": len(
                    acceptance_status.get("acceptance_checks")
                    if isinstance(acceptance_status.get("acceptance_checks"), list)
                    else []
                ),
            }
            row["action_plan"] = managed_object_directory_action_plan(row)
            row["action_count"] = len(row["action_plan"])
            row["blocked_action_count"] = sum(
                1 for action in row["action_plan"] if action.get("severity") == "blocked"
            )
            row["manual_check_action_count"] = sum(
                1 for action in row["action_plan"] if action.get("severity") == "manual_check"
            )
            row["next_step"] = (
                str(row["action_plan"][0].get("next_step") or "")
                if row["action_plan"]
                else str(acceptance_status.get("next_step") or "Run object acceptance checks.")
            )
            rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("tenant_id") or ""),
            str(item.get("delivery_namespace") or ""),
            str(item.get("customer_id") or ""),
            str(item.get("project_id") or ""),
            str(item.get("object_id") or ""),
        )
    )
    return rows


def managed_object_directory_action_plan(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return delivery work items needed before this object can be accepted."""
    actions: list[dict[str, Any]] = []
    object_id = str(row.get("object_id") or "")

    def resource_type_label(resource_type: str) -> str:
        return {
            "vision_models": "识别模型",
            "sensor_protocols": "传感器协议",
            "skill_packages": "能力包",
            "acceptance_tests": "验收用例",
        }.get(resource_type, "交付资源")

    def owner_label(owner: str) -> str:
        return {
            "delivery_owner": "交付负责人",
            "qa_owner": "测试/验收负责人",
        }.get(owner, owner or "负责人")

    def action_payload(
        *,
        action_id: str,
        action: str,
        action_label: str,
        reason_label: str,
        severity: str,
        owner: str,
        target: dict[str, Any],
        status: str,
        message: str,
        next_step: str,
    ) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "action": action,
            "action_label": action_label,
            "reason_label": reason_label,
            "severity": severity,
            "owner": owner,
            "owner_label": owner_label(owner),
            "target": target,
            "status": status,
            "message": message,
            "next_step": next_step,
            "customer_next_step": next_step,
        }

    resource_status = row.get("resource_binding_status") if isinstance(row.get("resource_binding_status"), dict) else {}
    resource_checks = resource_status.get("checks") if isinstance(resource_status.get("checks"), list) else []
    for check in resource_checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "")
        if status == "linked":
            continue
        resource_type = str(check.get("resource_type") or "resource")
        resource_id = str(check.get("resource_id") or "")
        resource_label = resource_type_label(resource_type)
        severity = "blocked" if status in {"missing", "blocked", "disabled", "failed"} else "manual_check"
        if status == "missing":
            action = "bind_required_resource"
            action_label = "补齐对象资源绑定"
            reason_label = f"缺少{resource_label}"
            next_step = f"在对象编辑区为 {object_id} 绑定至少一个{resource_label}。"
        elif status == "unregistered":
            action = "register_delivery_resource"
            action_label = "登记交付资源"
            reason_label = f"{resource_label}未登记"
            next_step = f"先在共享资源登记表登记 {resource_type}:{resource_id}，再回到对象目录复核。"
        elif status in {"manual_check", "draft", "pilot", "deprecated"}:
            action = "review_delivery_resource"
            action_label = "复核交付资源"
            reason_label = f"{resource_label}需复核"
            next_step = f"确认 {resource_type}:{resource_id} 是否可用于当前客户项目。"
        else:
            action = "replace_blocked_resource"
            action_label = "替换不可用资源"
            reason_label = f"{resource_label}不可用于交付"
            next_step = f"把 {resource_type}:{resource_id} 替换为已批准的交付资源。"
        actions.append(action_payload(
            action_id=f"{object_id}:{resource_type}:{resource_id or 'missing'}:{status}",
            action=action,
            action_label=action_label,
            reason_label=reason_label,
            severity=severity,
            owner="delivery_owner",
            target={
                "object_id": object_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
            status=status,
            message=str(check.get("message") or ""),
            next_step=next_step,
        ))

    acceptance_status = row.get("acceptance_status") if isinstance(row.get("acceptance_status"), dict) else {}
    missing_requirements = (
        acceptance_status.get("missing")
        if isinstance(acceptance_status.get("missing"), list)
        else []
    )
    for missing in missing_requirements:
        missing_key = str(missing or "")
        actions.append(action_payload(
            action_id=f"{object_id}:acceptance:{missing_key}:missing",
            action="bind_acceptance_requirement",
            action_label="补齐验收要求",
            reason_label="缺少验收要求",
            severity="blocked",
            owner="delivery_owner",
            target={
                "object_id": object_id,
                "requirement": missing_key,
            },
            status="missing",
            message="Acceptance requirement is not configured.",
            next_step=f"在对象编辑区补齐 {missing_key}，否则客户验收前必须阻断。",
        ))

    acceptance_checks = (
        acceptance_status.get("acceptance_checks", [])
        if isinstance(acceptance_status.get("acceptance_checks"), list)
        else []
    )
    for check in acceptance_checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "")
        if status == "linked":
            continue
        severity = "blocked" if status in {"file_missing", "invalid_reference", "outside_project"} else "manual_check"
        reason_label = {
            "file_missing": "验收用例文件缺失",
            "invalid_reference": "验收引用无效",
            "outside_project": "验证证据不在项目范围内",
            "node_unresolved": "验收用例节点待确认",
        }.get(status, "验收证据需复核")
        actions.append(action_payload(
            action_id=f"{object_id}:acceptance_test:{check.get('reference') or check.get('path') or 'unknown'}:{status}",
            action="fix_acceptance_test_reference",
            action_label="修正验收证据引用",
            reason_label=reason_label,
            severity=severity,
            owner="qa_owner",
            target={
                "object_id": object_id,
                "reference": check.get("reference") or "",
                "path": check.get("path") or "",
                "node": check.get("node") or "",
            },
            status=status,
            message=str(check.get("message") or ""),
            next_step=str(
                check.get("next_step")
                or acceptance_status.get("next_step")
                or "把验收用例引用修正到当前项目内可执行、可复核的证据。"
            ),
        ))
    return actions


def managed_object_directory_summary(
    rows: list[dict[str, Any]],
    *,
    projects: list[dict[str, Any]],
    base_summary: dict[str, Any],
    filtered: bool,
) -> dict[str, Any]:
    """Summarize a managed object directory for dashboard and export UX."""
    ready_count = sum(1 for row in rows if row.get("delivery_status") == "ready")
    manual_check_count = sum(1 for row in rows if row.get("delivery_status") == "manual_check")
    blocked_count = sum(1 for row in rows if row.get("delivery_status") == "blocked")
    scoped_object_count = sum(
        1
        for row in rows
        if any(row.get("scope_constraints", {}).get(key) for key in row.get("scope_constraints", {}))
    )
    overall_status = "manual_check"
    if rows and not blocked_count and not manual_check_count:
        overall_status = "ready"
    elif blocked_count:
        overall_status = "blocked"
    return {
        "object_count": len(rows),
        "project_count": len(projects),
        "customer_count": len({str(row.get("customer_id") or "") for row in rows if row.get("customer_id")}),
        "site_count": len({str(row.get("site_id") or "") for row in rows if row.get("site_id")}),
        "ready_count": ready_count,
        "manual_check_count": manual_check_count,
        "blocked_count": blocked_count,
        "customer_visible_count": sum(1 for row in rows if row.get("customer_visible") is not False),
        "acceptance_test_count": sum(int(row.get("acceptance_test_count") or 0) for row in rows),
        "scoped_object_count": scoped_object_count,
        "action_count": sum(int(row.get("action_count") or 0) for row in rows),
        "blocked_action_count": sum(int(row.get("blocked_action_count") or 0) for row in rows),
        "manual_check_action_count": sum(int(row.get("manual_check_action_count") or 0) for row in rows),
        "categories": sorted({str(row.get("category") or "") for row in rows if row.get("category")}),
        "device_sources": sorted(
            {
                str(source)
                for row in rows
                for source in row.get("device_sources", [])
                if str(source).strip()
            }
        ),
        "scenario_ids": sorted(
            {
                str(scenario)
                for row in rows
                for scenario in row.get("scenario_ids", [])
                if str(scenario).strip()
            }
        ),
        "overall_status": overall_status,
        "scope_filtered": bool(base_summary.get("scope_filtered")),
        "filtered": filtered,
    }


def filter_managed_object_directory_rows(
    rows: list[dict[str, Any]],
    *,
    delivery_status: str,
    category: str,
    customer_visible: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply customer-project object directory query filters."""
    normalized_status = str(delivery_status or "").strip().lower()
    normalized_category = str(category or "").strip().lower()
    normalized_customer_visible = str(customer_visible or "").strip().lower()
    filtered = rows
    filters: dict[str, Any] = {}
    if normalized_status:
        filters["delivery_status"] = normalized_status
        filtered = [
            row
            for row in filtered
            if str(row.get("delivery_status") or "").lower() == normalized_status
        ]
    if normalized_category:
        filters["category"] = normalized_category
        filtered = [
            row
            for row in filtered
            if str(row.get("category") or "").lower() == normalized_category
        ]
    if normalized_customer_visible in {"true", "false"}:
        visible = normalized_customer_visible == "true"
        filters["customer_visible"] = visible
        filtered = [row for row in filtered if bool(row.get("customer_visible")) is visible]
    return filtered, filters


__all__ = [
    "filter_managed_object_directory_rows",
    "managed_object_delivery_status",
    "managed_object_directory_action_plan",
    "managed_object_directory_rows",
    "managed_object_directory_summary",
]
