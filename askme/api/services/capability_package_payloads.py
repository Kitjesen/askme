"""Payload helpers for customer capability and scenario package APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from askme.contracts import PackageRuntimeInventory

CapabilitiesProvider = Callable[[], dict[str, Any]]

_CUSTOMER_TEXT_ALIASES = {
    "Ready for site validation": "可进入现场验证",
    "Run site validation.": "执行现场验证。",
    "Package can enter validation.": "能力包可进入现场验证。",
    "Scenario can enter validation.": "场景包可进入现场验证。",
    "Validate service point trigger.": "验证服务点触发。",
    "Can enter validation.": "可进入现场验证。",
}

_READINESS_STATUS_LABELS = {
    "ready": "可进入现场验证",
    "manual_check": "需要人工验收确认",
    "blocked": "缺少依赖，不能启用",
    "draft": "草稿，尚未启用",
    "unknown": "状态未知",
}


def default_product_capability_center() -> dict[str, Any]:
    """Build the product capability catalog when no live runtime snapshot exists."""

    try:
        from askme.skills.core.skill_manager import SkillManager

        manager = SkillManager()
        manager.load()
        center = manager.get_capability_center()
    except (ImportError, TypeError, AttributeError):
        return {}
    return center if isinstance(center, dict) else {}


def package_readiness_contract() -> dict[str, Any]:
    """Return the public contract for capability package readiness checks."""

    return {
        "endpoint": "/api/capability-packages/readiness",
        "method": "POST",
        "purpose": "评估能力包或场景包是否可以进入客户现场启用前验证。",
        "accepted_kinds": ["capability_package", "scenario_package"],
        "required_payload": {
            "kind": "capability_package | scenario_package",
            "manifest": "CapabilityPackageManifest or ScenarioPackageManifest",
            "inventory": "Optional PackageRuntimeInventory; omitted means derive from runtime capability snapshot.",
        },
        "customer_statuses": {
            "ready": "可进入现场验证或试点发布评审。",
            "manual_check": "客户启用前需要人工验收确认。",
            "blocked": "缺少依赖时不能面向客户启用。",
        },
    }


def readiness_kind(body: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Infer the package readiness evaluator from request and manifest data."""

    raw = str(body.get("kind") or body.get("type") or manifest.get("kind") or "").strip()
    if raw in {"capability", "capability_package"}:
        return "capability_package"
    if raw in {"scenario", "scenario_package"}:
        return "scenario_package"
    if manifest.get("scenario") or manifest.get("capability_packages"):
        return "scenario_package"
    if manifest.get("capability"):
        return "capability_package"
    return raw


def capability_center_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract capability center data from legacy and current runtime snapshots."""

    center = (
        payload.get("skills", {}).get("capability_center")
        if isinstance(payload.get("skills"), dict)
        else None
    )
    if isinstance(center, dict):
        return center
    nested = (
        payload.get("components", {})
        .get("skill", {})
        .get("capabilities", {})
        .get("capability_center")
        if isinstance(payload.get("components"), dict)
        else None
    )
    return nested if isinstance(nested, dict) else {}


def capability_package_catalog(center: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build the customer enablement catalog from capability-center payloads."""

    capability_packages = _capability_package_items(center)
    scenario_packages = _scenario_package_items(center)
    return {
        "capability_packages": capability_packages,
        "scenario_packages": scenario_packages,
    }


def package_catalog_summary(catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return counts used by the Dashboard package overview."""

    capability_items = catalog["capability_packages"]
    scenario_items = catalog["scenario_packages"]
    all_items = [*capability_items, *scenario_items]
    return {
        "capability_package_count": len(capability_items),
        "scenario_package_count": len(scenario_items),
        "ready_count": sum(1 for item in all_items if _item_readiness_status(item) == "ready"),
        "manual_check_count": sum(1 for item in all_items if _item_readiness_status(item) == "manual_check"),
        "blocked_count": sum(1 for item in all_items if _item_readiness_status(item) == "blocked"),
    }


def package_release_summary(catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize what the package catalog may claim to customers."""

    all_items = [*catalog["capability_packages"], *catalog["scenario_packages"]]
    decisions = [_item_enablement_decision(item) for item in all_items]
    return {
        "package_count": len(all_items),
        "controlled_demo_allowed_count": sum(
            1 for item in decisions if item.get("controlled_demo_allowed")
        ),
        "customer_pilot_allowed_count": sum(
            1 for item in decisions if item.get("customer_pilot_allowed")
        ),
        "production_claim_allowed_count": sum(
            1 for item in decisions if item.get("production_claim_allowed")
        ),
        "blocked_count": sum(1 for item in all_items if _item_readiness_status(item) == "blocked"),
        "manual_acceptance_required_count": sum(
            1
            for item in all_items
            if _item_readiness_status(item) == "manual_check"
            or _item_enablement_decision(item).get("decision") == "human_acceptance_required"
        ),
        "claim_policy": "生产上线声明必须另行完成现场验收和人工接管审批。",
    }


def readiness_inventory(
    body: dict[str, Any],
    provider: CapabilitiesProvider | None,
) -> PackageRuntimeInventory:
    """Resolve explicit or runtime-derived inventory for readiness evaluation."""

    if isinstance(body.get("inventory"), dict):
        return PackageRuntimeInventory.from_payload(body["inventory"])
    if provider is None:
        return PackageRuntimeInventory()
    try:
        payload = provider()
    except (TypeError, AttributeError):
        return PackageRuntimeInventory()
    if not isinstance(payload, dict):
        return PackageRuntimeInventory()
    return inventory_from_capabilities_payload(payload)


def inventory_from_capabilities_payload(payload: dict[str, Any]) -> PackageRuntimeInventory:
    """Convert a runtime capability snapshot into readiness inventory."""

    skills: set[str] = set()
    services: set[str] = set()
    capability_packages: set[str] = set()
    _collect_skills_from_catalog(payload, skills)
    _collect_skills_from_capability_center(payload, skills)
    _collect_services_from_components(payload, services)
    _collect_capability_packages(payload, capability_packages)
    capability_packages.update(_capability_package_id(name) for name in skills)
    return PackageRuntimeInventory(
        skills=frozenset(skills),
        services=frozenset(services),
        capability_packages=frozenset(capability_packages),
    )


def _capability_package_items(center: dict[str, Any]) -> list[dict[str, Any]]:
    package_payload = center.get("capability_packages") if isinstance(center, dict) else {}
    manifests = package_payload.get("items") if isinstance(package_payload, dict) else []
    readiness = package_payload.get("readiness") if isinstance(package_payload, dict) else []
    readiness_by_id = {
        str(item.get("package_id") or ""): item
        for item in readiness
        if isinstance(item, dict) and item.get("package_id")
    }
    items: list[dict[str, Any]] = []
    for manifest in manifests if isinstance(manifests, list) else []:
        if not isinstance(manifest, dict):
            continue
        package_id = str(manifest.get("package_id") or "").strip()
        readiness_item = readiness_by_id.get(package_id, {})
        safe_readiness = _customer_safe_readiness(readiness_item)
        items.append(
            {
                "package_id": package_id,
                "display_name": manifest.get("customer_visible_name")
                or manifest.get("display_name")
                or package_id,
                "kind": "capability_package",
                "capability": manifest.get("capability") or "",
                "status": manifest.get("status") or "draft",
                "risk_level": manifest.get("risk_level") or "low",
                "summary": manifest.get("customer_visible_description")
                or manifest.get("summary")
                or "",
                "manifest": manifest,
                "readiness": safe_readiness,
                "enablement_decision": _customer_safe_enablement_decision(
                    readiness_item.get("enablement_decision") or {}
                ),
                "customer_status": _customer_status_label(readiness_item),
                "customer_message": _customer_text(readiness_item.get("customer_message") or ""),
                "customer_next_step": _customer_text(
                    readiness_item.get("customer_next_step") or "",
                    fallback=_customer_next_step_for_status(readiness_item),
                ),
            }
        )
    return items


def _scenario_package_items(center: dict[str, Any]) -> list[dict[str, Any]]:
    blueprints = center.get("scenario_blueprints") if isinstance(center, dict) else {}
    scenarios = blueprints.get("items") if isinstance(blueprints, dict) else []
    items: list[dict[str, Any]] = []
    for scenario in scenarios if isinstance(scenarios, list) else []:
        if not isinstance(scenario, dict):
            continue
        manifest = (
            scenario.get("package_manifest")
            if isinstance(scenario.get("package_manifest"), dict)
            else {}
        )
        readiness = (
            scenario.get("package_readiness")
            if isinstance(scenario.get("package_readiness"), dict)
            else {}
        )
        safe_readiness = _customer_safe_readiness(readiness)
        package_id = str(
            manifest.get("package_id")
            or readiness.get("package_id")
            or f"scenario.{scenario.get('scenario_id') or ''}"
        ).strip()
        items.append(
            {
                "package_id": package_id,
                "display_name": manifest.get("customer_visible_name")
                or scenario.get("display_name")
                or package_id,
                "kind": "scenario_package",
                "scenario_id": scenario.get("scenario_id") or manifest.get("scenario") or "",
                "coverage_status": scenario.get("coverage_status") or "",
                "risk_level": readiness.get("risk_level") or manifest.get("risk_level") or "low",
                "required_capability_packages": list(manifest.get("capability_packages") or []),
                "customer_missing_dependencies": list(
                    readiness.get("customer_missing_dependencies")
                    or readiness.get("missing_required_dependencies")
                    or []
                ),
                "engineering_missing_dependencies": list(
                    readiness.get("engineering_missing_dependencies")
                    or readiness.get("missing_required_dependencies")
                    or []
                ),
                "manifest": manifest,
                "readiness": safe_readiness,
                "enablement_decision": _customer_safe_enablement_decision(
                    readiness.get("enablement_decision") or {}
                ),
                "customer_status": _customer_status_label(readiness),
                "customer_message": _customer_text(readiness.get("customer_message") or ""),
                "customer_next_step": _customer_text(
                    readiness.get("customer_next_step") or scenario.get("next_action") or "",
                    fallback=_customer_next_step_for_status(readiness),
                ),
            }
        )
    return items


def _item_enablement_decision(item: dict[str, Any]) -> dict[str, Any]:
    decision = item.get("enablement_decision")
    if isinstance(decision, dict) and decision:
        normalized = str(decision.get("decision") or "").strip()
        defaults = _enablement_defaults_for_decision(normalized)
        return {**defaults, **decision}
    status = _item_readiness_status(item)
    if status == "ready":
        return _enablement_defaults_for_decision("site_validation_allowed")
    if status == "manual_check":
        return _enablement_defaults_for_decision("human_acceptance_required")
    return _enablement_defaults_for_decision("blocked")


def _enablement_defaults_for_decision(decision: str) -> dict[str, Any]:
    normalized = str(decision or "").strip()
    if normalized == "site_validation_allowed":
        return {
            "decision": normalized,
            "controlled_demo_allowed": True,
            "customer_pilot_allowed": True,
            "production_claim_allowed": False,
        }
    if normalized == "human_acceptance_required":
        return {
            "decision": normalized,
            "controlled_demo_allowed": True,
            "customer_pilot_allowed": False,
            "production_claim_allowed": False,
        }
    return {
        "decision": normalized or "blocked",
        "controlled_demo_allowed": False,
        "customer_pilot_allowed": False,
        "production_claim_allowed": False,
    }


def _item_readiness_status(item: dict[str, Any]) -> str:
    readiness = item.get("readiness") if isinstance(item.get("readiness"), dict) else {}
    return str(readiness.get("status") or "").strip()


def _customer_status_label(readiness: dict[str, Any]) -> str:
    raw = str(readiness.get("status_label") or readiness.get("status") or "unknown").strip()
    status = str(readiness.get("status") or "").strip()
    if raw in _CUSTOMER_TEXT_ALIASES:
        return _CUSTOMER_TEXT_ALIASES[raw]
    if raw == status or raw in {"unknown", ""}:
        return _READINESS_STATUS_LABELS.get(status or "unknown", raw or "状态未知")
    return raw


def _customer_next_step_for_status(readiness: dict[str, Any]) -> str:
    status = str(readiness.get("status") or "").strip()
    if status == "ready":
        return "安排现场验证。"
    if status == "manual_check":
        return "完成主管或现场负责人确认后再启用。"
    if status == "blocked":
        return "补齐缺失依赖后重新评估。"
    return "确认能力包状态和现场启用条件。"


def _customer_text(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return _CUSTOMER_TEXT_ALIASES.get(text, text)


def _customer_safe_readiness(readiness: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(readiness, dict):
        return {}
    safe = dict(readiness)
    safe["status_label"] = _customer_status_label(readiness)
    safe["customer_message"] = _customer_text(readiness.get("customer_message") or "")
    safe["customer_next_step"] = _customer_text(
        readiness.get("customer_next_step") or "",
        fallback=_customer_next_step_for_status(readiness),
    )
    if isinstance(readiness.get("enablement_decision"), dict):
        safe["enablement_decision"] = _customer_safe_enablement_decision(
            readiness["enablement_decision"]
        )
    return safe


def _customer_safe_enablement_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    safe = dict(decision)
    if "release_claim" in safe:
        safe["release_claim"] = _customer_text(safe.get("release_claim") or "")
    if "customer_message" in safe:
        safe["customer_message"] = _customer_text(safe.get("customer_message") or "")
    if "customer_next_step" in safe:
        safe["customer_next_step"] = _customer_text(safe.get("customer_next_step") or "")
    return safe


def _collect_skills_from_catalog(payload: dict[str, Any], skills: set[str]) -> None:
    catalog = (
        payload.get("skills", {}).get("catalog")
        if isinstance(payload.get("skills"), dict)
        else []
    )
    if not isinstance(catalog, list):
        return
    for item in catalog:
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False or str(item.get("status") or "").lower() in {
            "disabled",
            "missing",
        }:
            continue
        name = str(item.get("skill_name") or item.get("name") or "").strip()
        if name:
            skills.add(name)


def _collect_skills_from_capability_center(payload: dict[str, Any], skills: set[str]) -> None:
    center = (
        payload.get("skills", {}).get("capability_center")
        if isinstance(payload.get("skills"), dict)
        else {}
    )
    if not isinstance(center, dict):
        return
    groups = center.get("groups") if isinstance(center.get("groups"), list) else []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_skills = group.get("skills") if isinstance(group.get("skills"), list) else []
        for item in group_skills:
            if not isinstance(item, dict):
                continue
            if not item.get("installed") or not item.get("enabled"):
                continue
            name = str(item.get("skill_name") or "").strip()
            if name:
                skills.add(name)


def _collect_services_from_components(payload: dict[str, Any], services: set[str]) -> None:
    components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
    for name, item in components.items():
        if not isinstance(item, dict):
            continue
        health = item.get("health") if isinstance(item.get("health"), dict) else item
        status = str(health.get("status") or "").lower()
        if status not in {"error", "failed", "blocked", "unavailable"}:
            services.add(str(name))


def _collect_capability_packages(payload: dict[str, Any], package_ids: set[str]) -> None:
    packages = (
        payload.get("skills", {}).get("skill_packages", {})
        if isinstance(payload.get("skills"), dict)
        else {}
    )
    records = packages.get("packages") if isinstance(packages, dict) else []
    if not isinstance(records, list):
        return
    for item in records:
        if not isinstance(item, dict):
            continue
        if not item.get("enabled") or int(item.get("rollout_percent") or 0) <= 0:
            continue
        package_id = str(item.get("package_id") or "").strip()
        if package_id:
            package_ids.add(package_id)


def _capability_package_id(skill_name: str) -> str:
    return f"capability.{skill_name}"


__all__ = [
    "CapabilitiesProvider",
    "capability_center_from_payload",
    "capability_package_catalog",
    "default_product_capability_center",
    "inventory_from_capabilities_payload",
    "package_catalog_summary",
    "package_readiness_contract",
    "package_release_summary",
    "readiness_kind",
    "readiness_inventory",
]
