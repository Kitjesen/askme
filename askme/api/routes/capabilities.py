"""Runtime capability FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from askme.contracts import (
    PackageRuntimeInventory,
    evaluate_capability_package_readiness,
    evaluate_scenario_package_readiness,
)

CapabilitiesProvider = Callable[[], dict[str, Any]]
BlueprintsProvider = Callable[[], dict[str, Any]]

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def register_capability_routes(
    app: FastAPI,
    *,
    capabilities_provider: CapabilitiesProvider | None,
    blueprints_provider: BlueprintsProvider | None = None,
    logger: logging.Logger,
) -> None:
    """Register runtime capability and customer capability-center routes."""

    @app.get("/api/capabilities", tags=["System"])
    async def capabilities() -> JSONResponse:
        """Return the runtime profile, components, and generated contracts."""
        if capabilities_provider is None:
            return JSONResponse(
                {"error": "capabilities not available"},
                status_code=503,
                headers=_CORS_HEADERS,
            )
        try:
            payload = capabilities_provider()
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Capabilities endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/capability-center", tags=["System"])
    async def capability_center() -> JSONResponse:
        """Return customer-facing grouped robot capabilities."""
        if capabilities_provider is None:
            return JSONResponse(
                {"error": "capabilities not available"},
                status_code=503,
                headers=_CORS_HEADERS,
            )
        try:
            payload = capabilities_provider()
            center = (
                payload.get("skills", {}).get("capability_center")
                if isinstance(payload, dict)
                else None
            )
            if not center and isinstance(payload, dict):
                center = (
                    payload.get("components", {})
                    .get("skill", {})
                    .get("capabilities", {})
                    .get("capability_center")
                )
            if isinstance(center, dict):
                center = {
                    **center,
                    "package_readiness": _package_readiness_contract(),
                }
            return JSONResponse(center or {}, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Capability center endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get("/api/capability-packages", tags=["System"])
    async def capability_packages() -> JSONResponse:
        """Return customer-visible capability and scenario package catalogs."""
        if capabilities_provider is None:
            return JSONResponse(
                {"error": "capabilities not available"},
                status_code=503,
                headers=_CORS_HEADERS,
            )
        try:
            payload = capabilities_provider()
            if not isinstance(payload, dict):
                payload = {}
            center = _capability_center_from_payload(payload)
            catalog = _capability_package_catalog(center)
            return JSONResponse(
                {
                    "ok": True,
                    "summary": _package_catalog_summary(catalog),
                    "release_summary": _package_release_summary(catalog),
                    "capability_packages": catalog["capability_packages"],
                    "scenario_packages": catalog["scenario_packages"],
                    "inventory": _inventory_from_capabilities_payload(payload).to_dict(),
                    "readiness": _package_readiness_contract(),
                    "policy": {
                        "capability_packages_are_customer_enablement_units": True,
                        "scenario_packages_are_delivery_scope_units": True,
                        "blocked_packages_must_not_be_enabled": True,
                        "ready_packages_still_require_site_validation": True,
                    },
                },
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.error("Capability package catalog endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/capability-packages", include_in_schema=False)
    async def capability_packages_cors() -> JSONResponse:
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.options("/api/capability-packages/readiness", include_in_schema=False)
    async def capability_package_readiness_cors() -> JSONResponse:
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.post("/api/capability-packages/readiness", tags=["System"])
    async def capability_package_readiness(request: Request) -> JSONResponse:
        """Evaluate whether a capability or scenario package can be enabled."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            return JSONResponse(
                {"ok": False, "reason": "json_object_required"},
                status_code=422,
                headers=_CORS_HEADERS,
            )
        manifest = body.get("manifest") if isinstance(body.get("manifest"), dict) else body
        inventory = _readiness_inventory(body, capabilities_provider)
        kind = _readiness_kind(body, manifest)
        if kind == "scenario_package":
            readiness = evaluate_scenario_package_readiness(manifest, inventory=inventory)
        elif kind == "capability_package":
            readiness = evaluate_capability_package_readiness(manifest, inventory=inventory)
        else:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "unsupported_package_kind",
                    "allowed_kinds": ["capability_package", "scenario_package"],
                },
                status_code=422,
                headers=_CORS_HEADERS,
            )
        return JSONResponse(
            {
                "ok": True,
                "kind": readiness["kind"],
                "readiness": readiness,
                "inventory": inventory.to_dict(),
                "policy": {
                    "blocked_means_do_not_enable_for_customer": True,
                    "manual_check_requires_human_acceptance": True,
                    "ready_still_requires_site_validation_for_robot_hardware": True,
                },
            },
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/api/blueprints", tags=["System"])
    async def blueprints() -> JSONResponse:
        """Return product runtime blueprints with delivery readiness gates."""
        try:
            if blueprints_provider is not None:
                payload = blueprints_provider()
            else:
                from askme.blueprints import catalog_payload

                payload = catalog_payload()
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.error("Blueprint catalog endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)


def _readiness_kind(body: dict[str, Any], manifest: dict[str, Any]) -> str:
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


def _package_readiness_contract() -> dict[str, Any]:
    return {
        "endpoint": "/api/capability-packages/readiness",
        "method": "POST",
        "purpose": "Evaluate whether a capability or scenario package can be enabled for a site.",
        "accepted_kinds": ["capability_package", "scenario_package"],
        "required_payload": {
            "kind": "capability_package | scenario_package",
            "manifest": "CapabilityPackageManifest or ScenarioPackageManifest",
            "inventory": "Optional PackageRuntimeInventory; omitted means derive from runtime capability snapshot.",
        },
        "customer_statuses": {
            "ready": "Can enter site validation or release workflow.",
            "manual_check": "Requires a human acceptance step before customer enablement.",
            "blocked": "Must not be enabled for customer use until missing dependencies are resolved.",
        },
    }


def _capability_center_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    center = payload.get("skills", {}).get("capability_center") if isinstance(payload.get("skills"), dict) else None
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


def _capability_package_catalog(center: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    capability_packages = _capability_package_items(center)
    scenario_packages = _scenario_package_items(center)
    return {
        "capability_packages": capability_packages,
        "scenario_packages": scenario_packages,
    }


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
                "readiness": readiness_item,
                "enablement_decision": readiness_item.get("enablement_decision") or {},
                "customer_status": readiness_item.get("status_label") or readiness_item.get("status") or "unknown",
                "customer_message": readiness_item.get("customer_message") or "",
                "customer_next_step": readiness_item.get("customer_next_step") or "",
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
        manifest = scenario.get("package_manifest") if isinstance(scenario.get("package_manifest"), dict) else {}
        readiness = scenario.get("package_readiness") if isinstance(scenario.get("package_readiness"), dict) else {}
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
                "readiness": readiness,
                "enablement_decision": readiness.get("enablement_decision") or {},
                "customer_status": readiness.get("status_label") or readiness.get("status") or "unknown",
                "customer_message": readiness.get("customer_message") or "",
                "customer_next_step": readiness.get("customer_next_step") or scenario.get("next_action") or "",
            }
        )
    return items


def _package_catalog_summary(catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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


def _package_release_summary(catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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
        "claim_policy": (
            "Production launch claims require separate onsite acceptance and human takeover "
            "approval."
        ),
    }


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


def _readiness_inventory(
    body: dict[str, Any],
    provider: CapabilitiesProvider | None,
) -> PackageRuntimeInventory:
    if isinstance(body.get("inventory"), dict):
        return PackageRuntimeInventory.from_payload(body["inventory"])
    if provider is None:
        return PackageRuntimeInventory()
    try:
        payload = provider()
    except Exception:
        return PackageRuntimeInventory()
    if not isinstance(payload, dict):
        return PackageRuntimeInventory()
    return _inventory_from_capabilities_payload(payload)


def _inventory_from_capabilities_payload(payload: dict[str, Any]) -> PackageRuntimeInventory:
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


def _collect_skills_from_catalog(payload: dict[str, Any], skills: set[str]) -> None:
    catalog = payload.get("skills", {}).get("catalog") if isinstance(payload.get("skills"), dict) else []
    if not isinstance(catalog, list):
        return
    for item in catalog:
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False or str(item.get("status") or "").lower() in {"disabled", "missing"}:
            continue
        name = str(item.get("skill_name") or item.get("name") or "").strip()
        if name:
            skills.add(name)


def _collect_skills_from_capability_center(payload: dict[str, Any], skills: set[str]) -> None:
    center = payload.get("skills", {}).get("capability_center") if isinstance(payload.get("skills"), dict) else {}
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
