"""Helpers for product blueprint API payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

BlueprintsProvider = Callable[[], dict[str, Any]]


def load_blueprints_payload(provider: BlueprintsProvider | None) -> dict[str, Any]:
    """Load the blueprint catalog from an injected provider or the product catalog."""

    if provider is not None:
        payload = provider()
        return payload if isinstance(payload, dict) else {}

    from askme.blueprints import catalog_payload

    return catalog_payload()


def blueprint_item_from_payload(
    payload: dict[str, Any],
    blueprint_name: str,
) -> dict[str, Any] | None:
    """Return one blueprint item by canonical name or public alias."""

    requested = str(blueprint_name or "").strip()
    if not requested:
        return None
    canonical = requested
    try:
        from askme.blueprints import get_blueprint_spec

        canonical = get_blueprint_spec(requested).name
    except Exception:
        canonical = requested

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") in {requested, canonical}:
            return item
    return None


def available_blueprint_names(payload: dict[str, Any]) -> list[str]:
    """Return sorted canonical blueprint names for API error payloads."""

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    return sorted(
        str(item.get("name"))
        for item in items
        if isinstance(item, dict) and item.get("name")
    )


def blueprint_runtime_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact product summary of runtime blueprints.

    The capability center uses this shape to explain where scenario and
    capability packages can run without forcing the Dashboard to merge several
    backend endpoints on its own.
    """

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []
    normalized = [_runtime_blueprint_item(item) for item in items if isinstance(item, dict)]
    normalized = [item for item in normalized if item]
    return {
        "summary": {
            "blueprint_count": len(normalized),
            "customer_visible_count": sum(1 for item in normalized if item["customer_visible"]),
            "ready_for_validation_count": sum(
                1 for item in normalized if item["status"] == "ready_for_site_validation"
            ),
            "missing_configuration_count": sum(
                1
                for item in normalized
                if item["status"] in {"configuration_incomplete", "missing_configuration"}
            ),
        },
        "items": normalized,
        "policy": {
            "runtime_blueprints_are_delivery_profiles": True,
            "capability_packages_still_require_runtime_blueprint": True,
            "site_validation_required_before_customer_claim": True,
        },
    }


def _runtime_blueprint_item(item: dict[str, Any]) -> dict[str, Any]:
    package = item.get("delivery_package")
    readiness = item.get("readiness")
    if not isinstance(package, dict):
        package = {}
    if not isinstance(readiness, dict):
        readiness = {}
    deliverables = package.get("deliverables")
    if not isinstance(deliverables, dict):
        deliverables = {}
    return {
        "name": str(item.get("name") or ""),
        "title": str(item.get("title") or ""),
        "product_stage": str(item.get("product_stage") or ""),
        "customer_visible": bool(item.get("customer_visible")),
        "status": str(package.get("status") or readiness.get("status") or "unknown"),
        "customer_status": _blueprint_customer_status(package, readiness),
        "package_id": str(package.get("package_id") or ""),
        "primary_loop": str(item.get("primary_loop") or ""),
        "deployment_targets": _string_list(item.get("deployment_targets")),
        "capabilities": _string_list(item.get("capabilities")),
        "scenarios": _string_list(item.get("scenarios")),
        "missing_config": _string_list(readiness.get("missing_config")),
        "external_services": _string_list(item.get("external_services")),
        "safety_boundaries": _string_list(
            item.get("safety_boundaries") or deliverables.get("safety_boundaries")
        ),
        "validation_commands": _string_list(
            item.get("validation_commands") or deliverables.get("validation_commands")
        ),
        "release_boundary": str(package.get("release_boundary") or ""),
        "acceptance_boundary": _blueprint_acceptance_boundary(package, readiness),
        "customer_claim": str(package.get("customer_claim") or readiness.get("customer_claim") or ""),
        "customer_next_step": _blueprint_next_step(package, readiness),
        "delivery_actions": _blueprint_delivery_actions(package, readiness),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _blueprint_next_step(package: dict[str, Any], readiness: dict[str, Any]) -> str:
    package_next_step = str(package.get("customer_next_step") or "").strip()
    if package_next_step:
        return package_next_step
    status = str(package.get("status") or readiness.get("status") or "")
    if status == "ready_for_site_validation":
        return "先做现场验证，再对客户声明可交付。"
    missing = _string_list(readiness.get("missing_config"))
    if missing:
        return "补齐运行配置：" + "、".join(missing)
    if status:
        return f"复核运行蓝图状态：{status}"
    return "客户使用前先复核运行蓝图就绪情况。"


def _blueprint_customer_status(package: dict[str, Any], readiness: dict[str, Any]) -> str:
    package_status = str(package.get("customer_status") or "").strip()
    if package_status:
        return package_status
    status = str(package.get("status") or readiness.get("status") or "")
    if status == "ready_for_site_validation":
        return "可进入现场验证"
    if status == "ready_for_validation":
        return "配置已补齐，等待验证"
    if status in {"configuration_incomplete", "missing_configuration"}:
        return "运行配置未补齐"
    if status:
        return f"需要复核：{status}"
    return "蓝图状态未知"


def _blueprint_acceptance_boundary(
    package: dict[str, Any],
    readiness: dict[str, Any],
) -> str:
    package_boundary = str(package.get("acceptance_boundary") or "").strip()
    if package_boundary:
        return package_boundary
    boundary = str(package.get("release_boundary") or "").strip()
    if boundary:
        return boundary
    status = str(package.get("status") or readiness.get("status") or "")
    if status in {"configuration_incomplete", "missing_configuration"}:
        return "运行配置、外部服务和现场验证证据补齐前，不能作为客户验收依据。"
    return "该运行蓝图只说明系统可运行范围；客户签收仍需要现场验证证据和人工复核。"


def _blueprint_delivery_actions(
    package: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    package_actions = _string_list(package.get("delivery_actions"))
    if package_actions:
        return package_actions

    missing = _string_list(readiness.get("missing_config"))
    if missing:
        return [
            "补齐运行配置：" + "、".join(missing),
            "完成外部服务凭证配置和冒烟测试。",
            "重新生成蓝图交付包并复核验收边界。",
        ]

    status = str(package.get("status") or readiness.get("status") or "")
    if status == "ready_for_site_validation":
        return [
            "运行现场验证用例。",
            "归档语音、通知、机器人运行和客户复核证据。",
            "签收前复核安全边界和人工接管方案。",
        ]
    if status == "ready_for_validation":
        return [
            "运行验证命令并归档输出。",
            "完成外部服务冒烟测试。",
            "通过后再进入客户试点交付。",
        ]
    return ["复核运行蓝图状态、配置缺口和客户验收边界。"]


__all__ = [
    "BlueprintsProvider",
    "available_blueprint_names",
    "blueprint_item_from_payload",
    "blueprint_runtime_summary",
    "load_blueprints_payload",
]
