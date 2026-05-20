"""Customer-project implementation handoff contract builder.

This module is intentionally leaf-level: it does not import the large
``field_site_profile`` module or product facades. The handoff payload is a
customer-facing implementation contract used by project creation, managed
object edits, and package import flows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_template_support import _mapping, _string_list
from askme.pipeline.field.delivery_resource_registry import DELIVERY_RESOURCE_TYPES


def _customer_project_implementation_handoff(
    profile: dict[str, Any],
    *,
    template_id: str,
    profile_path: Path | str,
) -> dict[str, Any]:
    """Return the customer-project implementation checklist after profile creation."""
    customer = _mapping(profile.get("customer"))
    site = _mapping(profile.get("site"))
    managed_objects = _mapping(profile.get("managed_objects"))
    binding_labels = {
        "vision_models": "识别模型",
        "sensor_protocols": "传感器协议",
        "skill_packages": "能力包",
        "acceptance_tests": "验收项",
    }
    object_todos: list[dict[str, Any]] = []
    for object_id, item in sorted(managed_objects.items()):
        if not isinstance(item, dict):
            continue
        bindings = _mapping(item.get("bindings"))
        missing = [
            resource_type
            for resource_type in DELIVERY_RESOURCE_TYPES
            if not _string_list(bindings.get(resource_type))
        ]
        object_todos.append(
            {
                "object_id": str(object_id),
                "display_name": str(item.get("display_name") or object_id),
                "category": str(item.get("category") or ""),
                "ready_for_site_acceptance": not missing,
                "missing_binding_types": missing,
                "missing_binding_labels": [binding_labels.get(key, key) for key in missing],
                "customer_next_step": (
                    "补齐识别模型、传感器协议、能力包和验收项后再进入现场验收。"
                    if missing
                    else "对象能力绑定已齐，可继续登记现场证据。"
                ),
            }
        )
    object_needs_binding_count = len([item for item in object_todos if item["missing_binding_types"]])
    status = "needs_object_binding" if object_needs_binding_count else "ready_for_acceptance_evidence"
    return {
        "handoff_schema": "askme.customer_project_implementation_handoff.v1",
        "template_id": str(template_id or ""),
        "project_id": str(customer.get("project_id") or ""),
        "project_name": str(customer.get("project_name") or ""),
        "customer_id": str(customer.get("customer_id") or ""),
        "customer_name": str(customer.get("customer_name") or ""),
        "site_id": str(site.get("site_id") or ""),
        "site_name": str(site.get("name") or ""),
        "profile_path": str(profile_path),
        "status": status,
        "customer_status": (
            "项目已创建，待补齐现场对象能力绑定。"
            if object_needs_binding_count
            else "项目已创建，对象能力绑定已齐，待登记现场验收证据。"
        ),
        "summary": {
            "object_count": len(object_todos),
            "object_ready_count": len(object_todos) - object_needs_binding_count,
            "object_needs_binding_count": object_needs_binding_count,
        },
        "next_steps": [
            {
                "step_id": "review_project_profile",
                "label": "核对项目基础信息",
                "status": "ready",
                "customer_next_step": "确认客户、项目、现场、行业模板和交付边界。",
            },
            {
                "step_id": "complete_object_bindings",
                "label": "补齐对象能力绑定",
                "status": "pending" if object_needs_binding_count else "ready",
                "customer_next_step": "为现场对象绑定识别模型、传感器协议、能力包和验收项。",
            },
            {
                "step_id": "register_onsite_evidence",
                "label": "登记现场验收证据",
                "status": "pending",
                "customer_next_step": "登记设备接入、语音播报、通知送达、客户复核和现场照片证据。",
            },
            {
                "step_id": "export_delivery_package",
                "label": "生成客户交付包",
                "status": "pending",
                "customer_next_step": "先验包和预览差异，再导出可复用客户项目包。",
            },
        ],
        "object_binding_todo": object_todos[:50],
    }


customer_project_implementation_handoff = _customer_project_implementation_handoff

__all__ = [
    "_customer_project_implementation_handoff",
    "customer_project_implementation_handoff",
]
