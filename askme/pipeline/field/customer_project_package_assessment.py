"""Customer-project package acceptance and reuse assessment helpers."""

from __future__ import annotations

from typing import Any

from askme.pipeline.field.customer_project_managed_objects import (
    _managed_object_acceptance_summary,
    _managed_object_binding_readiness_summary,
)
from askme.pipeline.field.customer_project_template_support import _mapping, _string_list


def _customer_project_package_acceptance_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(catalog.get("acceptance_summary"))
    objects = catalog.get("objects") if isinstance(catalog.get("objects"), list) else []
    blocked = []
    manual = []
    for item in objects:
        status = str(_mapping(item.get("acceptance_status")).get("status") or "blocked")
        row = {
            "object_id": str(item.get("object_id") or ""),
            "display_name": str(item.get("display_name") or item.get("object_id") or ""),
            "status": status,
            "next_step": str(_mapping(item.get("acceptance_status")).get("next_step") or ""),
        }
        if status == "blocked":
            blocked.append(row)
        elif status == "manual_check":
            manual.append(row)
    overall = str(summary.get("overall_status") or "blocked")
    return {
        "overall_status": overall,
        "ready_object_count": int(summary.get("ready_object_count") or 0),
        "manual_check_object_count": int(summary.get("manual_check_object_count") or 0),
        "blocked_object_count": int(summary.get("blocked_object_count") or 0),
        "object_count": int(summary.get("object_count") or len(objects)),
        "blocked_objects": blocked,
        "manual_check_objects": manual,
        "customer_status": {
            "ready": "本地验收证据已关联，仍需完成现场验收。",
            "manual_check": "部分验收证据需要交付复核后才能客户签收。",
            "blocked": "验收证据缺失或存在安全风险，不能声明交付就绪。",
        }.get(overall, "验收状态未知，交付前需要复核交付包。"),
        "release_claim": (
            "不能声明生产上线；真实设备、通知、语音和机器人运行通过现场验收前，"
            "该交付包只承载本地交付证据。"
        ),
    }


def _customer_project_reuse_dependencies(
    profile: dict[str, Any],
    objects: list[dict[str, Any]],
    env_references: list[dict[str, Any]],
) -> dict[str, Any]:
    devices = _mapping(profile.get("devices"))
    responders = _mapping(profile.get("responder_groups"))
    bindings = [_mapping(item.get("bindings")) for item in objects]
    return {
        "device_count": len(devices),
        "device_sources": sorted({
            source
            for item in objects
            for source in _string_list(item.get("device_sources"))
        }),
        "responder_groups": sorted({
            str(item.get("responder_group") or "")
            for item in objects
            if str(item.get("responder_group") or "").strip()
        } | set(str(group) for group in responders)),
        "vision_models": sorted({
            value for binding in bindings for value in _string_list(binding.get("vision_models"))
        }),
        "sensor_protocols": sorted({
            value for binding in bindings for value in _string_list(binding.get("sensor_protocols"))
        }),
        "skill_packages": sorted({
            value for binding in bindings for value in _string_list(binding.get("skill_packages"))
        }),
        "acceptance_tests": sorted({
            value for binding in bindings for value in _string_list(binding.get("acceptance_tests"))
        }),
        "binding_readiness": _managed_object_binding_readiness_summary(objects),
        "env_reference_count": len(env_references),
        "required_env_count": len([item for item in env_references if item.get("required")]),
        "missing_env_count": len([
            item for item in env_references if item.get("required") and not item.get("configured")
        ]),
    }


def _customer_project_package_reuse_assessment(
    *,
    profile: dict[str, Any],
    report: dict[str, Any],
    managed_object_catalog: dict[str, Any],
    acceptance_summary: dict[str, Any],
    env_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether a handoff package can be reused for another customer project."""
    objects = (
        managed_object_catalog.get("objects")
        if isinstance(managed_object_catalog.get("objects"), list)
        else []
    )
    missing_env = [
        item for item in env_references if item.get("required") and not item.get("configured")
    ]
    dependencies = _customer_project_reuse_dependencies(profile, objects, env_references)
    blockers: list[str] = []
    manual_checks: list[str] = []
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    binding_readiness = _mapping(managed_object_catalog.get("binding_readiness_summary"))
    if errors:
        blockers.append(f"Site profile has {len(errors)} validation error(s).")
    if str(acceptance_summary.get("overall_status") or "") == "blocked":
        blockers.append("Managed-object acceptance evidence is blocked.")
    if str(binding_readiness.get("overall_status") or "") == "blocked":
        blockers.append("Managed-object resource bindings are blocked.")
    if not objects:
        blockers.append("Managed-object catalog is empty.")
    if missing_env:
        manual_checks.append(f"{len(missing_env)} live credential or device secret value(s) must be configured onsite.")
    if warnings:
        manual_checks.append(f"{len(warnings)} site warning(s) require delivery review.")
    if str(acceptance_summary.get("overall_status") or "") == "manual_check":
        manual_checks.append("Some managed-object acceptance references require manual review.")
    if str(binding_readiness.get("overall_status") or "") == "manual_check":
        manual_checks.append("Some managed-object resource bindings need catalog review.")
    if blockers:
        status = "blocked"
    elif manual_checks:
        status = "manual_check"
    else:
        status = "ready"
    return {
        "status": status,
        "customer_status": {
            "ready": "Package can seed a new customer project after onsite evidence is refreshed.",
            "manual_check": "Package is reusable, but delivery must rebind live credentials and review acceptance evidence.",
            "blocked": "Package should not be reused until profile or acceptance blockers are fixed.",
        }[status],
        "blocker_count": len(blockers),
        "manual_check_count": len(manual_checks),
        "blockers": blockers,
        "manual_checks": manual_checks,
        "dependencies": dependencies,
        "next_step": {
            "ready": "Import the package, then run onsite smoke and acceptance checks for the target customer.",
            "manual_check": "Resolve manual checks after import before customer signoff.",
            "blocked": "Fix blockers in the source project before using this as a reusable template.",
        }[status],
    }

__all__ = [
    "_customer_project_package_acceptance_summary",
    "_customer_project_package_reuse_assessment",
    "_customer_project_reuse_dependencies",
    "_managed_object_acceptance_summary",
    "_managed_object_binding_readiness_summary",
]
