"""Runtime capability FastAPI routes."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from askme.api.schemas.blueprints import (
    BlueprintCatalogResponse,
    BlueprintDeliveryPackageResponse,
    BlueprintDetailResponse,
)
from askme.api.schemas.capabilities import (
    CapabilityCenterResponse,
    CapabilityPackageCatalogResponse,
    CapabilityPackageReadinessResponse,
    RuntimeCapabilitiesResponse,
    ScenarioIntentCatalogResponse,
    ScenarioIntentPreviewResponse,
)
from askme.api.services.blueprint_payloads import (
    BlueprintsProvider,
    available_blueprint_names,
    blueprint_item_from_payload,
    blueprint_runtime_summary,
    load_blueprints_payload,
)
from askme.api.services.capability_package_payloads import (
    CapabilitiesProvider,
    capability_center_from_payload,
    capability_package_catalog,
    default_product_capability_center,
    inventory_from_capabilities_payload,
    package_catalog_summary,
    package_readiness_contract,
    package_release_summary,
    readiness_inventory,
    readiness_kind,
)
from askme.api.services.http_helpers import require_json_object
from askme.api.services.scenario_intent_payloads import (
    requested_or_runtime_skills,
    scenario_intent_decision_payload,
    scenario_intent_rule_payload,
)
from askme.api.services.space_preview import SpaceDispatch, space_resolution_preview
from askme.contracts import (
    evaluate_capability_package_readiness,
    evaluate_scenario_package_readiness,
)
from askme.robot_interaction.scenario_intents import (
    SCENARIO_INTENT_RULES,
    classify_scenario_intent,
)

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def _safe_capabilities_payload(provider: CapabilitiesProvider | None) -> dict[str, object]:
    """Return a capabilities payload when runtime is wired, otherwise an empty snapshot."""

    if provider is None:
        return {}
    payload = provider()
    return payload if isinstance(payload, dict) else {}


def _empty_capability_center(blueprints_provider: BlueprintsProvider | None) -> dict[str, object]:
    """Return the product-level capability center shell for dashboard-only mode."""

    return {
        "title": "Robot Capability Center",
        "summary": {
            "group_count": 0,
            "enabled_skill_count": 0,
            "installed_skill_count": 0,
        },
        "groups": [],
        "package_readiness": package_readiness_contract(),
        "runtime_blueprints": blueprint_runtime_summary(load_blueprints_payload(blueprints_provider)),
        "policy": {
            "runtime_capability_snapshot_available": False,
            "runtime_blueprints_remain_visible_without_live_robot": True,
            "capability_packages_require_runtime_wiring_before_enablement": True,
        },
    }


def _capability_center_snapshot(
    capabilities_provider: CapabilitiesProvider | None,
) -> tuple[dict[str, object], dict[str, object], bool]:
    """Return runtime payload, capability center, and whether it came from live runtime."""

    payload = _safe_capabilities_payload(capabilities_provider)
    center = capability_center_from_payload(payload)
    if center:
        return payload, center, True

    fallback = default_product_capability_center()
    if fallback:
        return {"skills": {"capability_center": fallback}}, fallback, False

    return payload, {}, False


def register_capability_routes(
    app: FastAPI,
    *,
    capabilities_provider: CapabilitiesProvider | None,
    blueprints_provider: BlueprintsProvider | None = None,
    space_dispatch: SpaceDispatch | None = None,
    logger: logging.Logger,
) -> None:
    """Register runtime capability and customer capability-center routes."""

    @app.get(
        "/api/capabilities",
        tags=["System"],
        response_model=RuntimeCapabilitiesResponse,
        response_model_exclude_none=True,
    )
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
            response = RuntimeCapabilitiesResponse.model_validate(payload)
            return JSONResponse(response.model_dump(mode="python"), headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.exception("Capabilities endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get(
        "/api/capability-center",
        tags=["System"],
        response_model=CapabilityCenterResponse,
        response_model_exclude_none=True,
    )
    async def capability_center() -> JSONResponse:
        """Return customer-facing grouped robot capabilities."""
        try:
            _payload, center, runtime_available = _capability_center_snapshot(
                capabilities_provider
            )
            if center:
                center = {
                    **center,
                    "package_readiness": package_readiness_contract(),
                    "runtime_blueprints": blueprint_runtime_summary(
                        load_blueprints_payload(blueprints_provider)
                    ),
                    "policy": {
                        **(
                            center.get("policy", {})
                            if isinstance(center.get("policy"), dict)
                            else {}
                        ),
                        "runtime_capability_snapshot_available": runtime_available,
                        "default_product_catalog_used": not runtime_available,
                        "runtime_blueprints_remain_visible_without_live_robot": True,
                        "capability_packages_require_runtime_wiring_before_enablement": True,
                    },
                }
            response = CapabilityCenterResponse.model_validate(
                center if center else _empty_capability_center(blueprints_provider)
            )
            return JSONResponse(response.model_dump(mode="python"), headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.exception("Capability center endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.get(
        "/api/capability-packages",
        tags=["System"],
        response_model=CapabilityPackageCatalogResponse,
        response_model_exclude_none=True,
    )
    async def capability_packages() -> JSONResponse:
        """Return customer-visible capability and scenario package catalogs."""
        try:
            payload, center, runtime_available = _capability_center_snapshot(
                capabilities_provider
            )
            catalog = capability_package_catalog(center)
            runtime_blueprints = blueprint_runtime_summary(
                load_blueprints_payload(blueprints_provider)
            )
            response = CapabilityPackageCatalogResponse.model_validate(
                {
                    "ok": True,
                    "summary": package_catalog_summary(catalog),
                    "release_summary": package_release_summary(catalog),
                    "capability_packages": catalog["capability_packages"],
                    "scenario_packages": catalog["scenario_packages"],
                    "inventory": inventory_from_capabilities_payload(payload).to_dict(),
                    "runtime_blueprints": runtime_blueprints,
                    "readiness": package_readiness_contract(),
                    "policy": {
                        "capability_packages_are_customer_enablement_units": True,
                        "scenario_packages_are_delivery_scope_units": True,
                        "runtime_blueprints_are_delivery_profiles": True,
                        "runtime_capability_snapshot_available": runtime_available,
                        "default_product_catalog_used": not runtime_available,
                        "blocked_packages_must_not_be_enabled": True,
                        "ready_packages_still_require_site_validation": True,
                    },
                }
            )
            return JSONResponse(response.model_dump(mode="python"), headers=_NO_STORE_HEADERS)
        except Exception as exc:
            logger.exception("Capability package catalog endpoint failed")
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

    @app.post(
        "/api/capability-packages/readiness",
        tags=["System"],
        response_model=CapabilityPackageReadinessResponse,
        response_model_exclude_none=True,
    )
    async def capability_package_readiness(request: Request) -> JSONResponse:
        """Evaluate whether a capability or scenario package can be enabled."""
        try:
            body = require_json_object(await request.json())
        except ValueError as exc:
            return JSONResponse(
                {"ok": False, "reason": "json_object_required", "error": str(exc)},
                status_code=400,
                headers=_CORS_HEADERS,
            )
        manifest = body.get("manifest") if isinstance(body.get("manifest"), dict) else body
        inventory = readiness_inventory(body, capabilities_provider)
        kind = readiness_kind(body, manifest)
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
        response = CapabilityPackageReadinessResponse.model_validate(
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
            }
        )
        return JSONResponse(response.model_dump(mode="python"), headers=_NO_STORE_HEADERS)

    @app.get(
        "/api/scenario-intents",
        tags=["System"],
        response_model=ScenarioIntentCatalogResponse,
        response_model_exclude_none=True,
    )
    async def scenario_intents() -> JSONResponse:
        """Return auditable spoken-scene routing rules for product scenarios."""
        try:
            available_skills = requested_or_runtime_skills({}, capabilities_provider)
            rules = [
                scenario_intent_rule_payload(rule, available_skills)
                for rule in SCENARIO_INTENT_RULES
            ]
            response = ScenarioIntentCatalogResponse.model_validate(
                {
                    "ok": True,
                    "summary": {
                        "rule_count": len(rules),
                        "enabled_rule_count": sum(1 for item in rules if item["enabled"]),
                        "available_skill_count": len(available_skills),
                    },
                    "rules": rules,
                    "policy": {
                        "deterministic_before_llm": True,
                        "high_risk_still_requires_skill_gate_and_confirmation": True,
                        "route_evidence_must_be_recorded": True,
                    },
                },
            )
            return JSONResponse(
                response.model_dump(mode="python", exclude_unset=True),
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.exception("Scenario intent catalog endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.post(
        "/api/scenario-intents/preview",
        tags=["System"],
        response_model=ScenarioIntentPreviewResponse,
        response_model_exclude_none=True,
    )
    async def scenario_intent_preview(request: Request) -> JSONResponse:
        """Preview how a spoken or typed utterance would route before execution."""
        try:
            body = require_json_object(await request.json())
        except ValueError as exc:
            response = ScenarioIntentPreviewResponse.model_validate(
                {"ok": False, "reason": "json_object_required", "error": str(exc)}
            )
            return JSONResponse(
                response.model_dump(mode="python", exclude_unset=True),
                status_code=400,
                headers=_CORS_HEADERS,
            )

        text = str(body.get("text") or body.get("utterance") or "").strip()
        if not text:
            response = ScenarioIntentPreviewResponse.model_validate(
                {"ok": False, "reason": "text_required"}
            )
            return JSONResponse(
                response.model_dump(mode="python", exclude_unset=True),
                status_code=422,
                headers=_CORS_HEADERS,
            )
        try:
            available_skills = requested_or_runtime_skills(body, capabilities_provider)
            decision = classify_scenario_intent(text, available_skills=available_skills)
            space_resolution = await space_resolution_preview(
                text=text,
                body=body,
                decision=decision,
                space_dispatch=space_dispatch,
            )
        except Exception as exc:
            logger.exception("Scenario intent preview endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)
        response = ScenarioIntentPreviewResponse.model_validate(
            {
                "ok": True,
                "text": text,
                "matched": decision is not None,
                "decision": scenario_intent_decision_payload(decision),
                "space_resolution": space_resolution,
                "available_skill_count": len(available_skills),
                "policy": {
                    "preview_only": True,
                    "does_not_execute_skill": True,
                    "does_not_start_guide": True,
                    "safe_for_customer_acceptance_testing": True,
                },
            },
        )
        return JSONResponse(
            response.model_dump(mode="python", exclude_unset=True),
            headers=_NO_STORE_HEADERS,
        )

    @app.options("/api/scenario-intents", include_in_schema=False)
    async def scenario_intents_cors() -> JSONResponse:
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.options("/api/scenario-intents/preview", include_in_schema=False)
    async def scenario_intent_preview_cors() -> JSONResponse:
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.get(
        "/api/blueprints",
        tags=["System"],
        response_model=BlueprintCatalogResponse,
        response_model_exclude_none=True,
    )
    async def blueprints() -> JSONResponse:
        """Return product runtime blueprints with delivery readiness gates."""
        try:
            payload = load_blueprints_payload(blueprints_provider)
            return JSONResponse(
                BlueprintCatalogResponse.model_validate(payload).model_dump(mode="python"),
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.exception("Blueprint catalog endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/blueprints/{blueprint_name}/delivery-package", include_in_schema=False)
    async def blueprint_delivery_package_cors(blueprint_name: str) -> JSONResponse:
        _ = blueprint_name
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.get(
        "/api/blueprints/{blueprint_name}/delivery-package",
        tags=["System"],
        response_model=BlueprintDeliveryPackageResponse,
        response_model_exclude_none=True,
    )
    async def blueprint_delivery_package(blueprint_name: str) -> JSONResponse:
        """Return the customer handoff package for one runtime blueprint."""
        try:
            payload = load_blueprints_payload(blueprints_provider)
            item = blueprint_item_from_payload(payload, blueprint_name)
            if item is None:
                return JSONResponse(
                    {
                        "ok": False,
                        "reason": "blueprint_not_found",
                        "blueprint": blueprint_name,
                        "available": available_blueprint_names(payload),
                    },
                    status_code=404,
                    headers=_CORS_HEADERS,
                )
            package = item.get("delivery_package")
            if not isinstance(package, dict) or not package:
                return JSONResponse(
                    {
                        "ok": False,
                        "reason": "delivery_package_not_available",
                        "blueprint": item.get("name") or blueprint_name,
                    },
                    status_code=409,
                    headers=_CORS_HEADERS,
                )
            success = BlueprintDeliveryPackageResponse.model_validate(
                {
                    "ok": True,
                    "blueprint": item.get("name") or blueprint_name,
                    "delivery_package": package,
                    "policy": {
                        "delivery_package_is_customer_handoff": True,
                        "production_ready_is_never_inferred_from_package": True,
                        "site_validation_required_before_customer_claim": True,
                    },
                }
            )
            return JSONResponse(
                success.model_dump(mode="python"),
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.exception("Blueprint delivery package endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/blueprints/{blueprint_name}", include_in_schema=False)
    async def blueprint_detail_cors(blueprint_name: str) -> JSONResponse:
        _ = blueprint_name
        return JSONResponse(
            {},
            status_code=204,
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    @app.get(
        "/api/blueprints/{blueprint_name}",
        tags=["System"],
        response_model=BlueprintDetailResponse,
        response_model_exclude_none=True,
    )
    async def blueprint_detail(blueprint_name: str) -> JSONResponse:
        """Return one product runtime blueprint by name or public alias."""
        try:
            payload = load_blueprints_payload(blueprints_provider)
            item = blueprint_item_from_payload(payload, blueprint_name)
            if item is None:
                return JSONResponse(
                    {
                        "ok": False,
                        "reason": "blueprint_not_found",
                        "blueprint": blueprint_name,
                        "available": available_blueprint_names(payload),
                    },
                    status_code=404,
                    headers=_CORS_HEADERS,
                )
            success = BlueprintDetailResponse.model_validate(
                {
                    "ok": True,
                    "blueprint": item,
                    "policy": {
                        "customer_visible": bool(item.get("customer_visible")),
                        "production_ready_is_never_inferred_from_catalog": True,
                        "site_validation_required_before_customer_claim": True,
                    },
                }
            )
            return JSONResponse(
                success.model_dump(mode="python"),
                headers=_NO_STORE_HEADERS,
            )
        except Exception as exc:
            logger.exception("Blueprint detail endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)
