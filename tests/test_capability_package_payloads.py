from __future__ import annotations

import logging

from fastapi import FastAPI

from askme.api.routes.capabilities import register_capability_routes
from askme.api.schemas.capabilities import (
    CapabilityCenterResponse,
    CapabilityPackageCatalogResponse,
)
from askme.api.services.capability_package_payloads import (
    capability_center_from_payload,
    capability_package_catalog,
    inventory_from_capabilities_payload,
    package_catalog_summary,
    package_readiness_contract,
    package_release_summary,
    readiness_inventory,
    readiness_kind,
)


def _capability_center_payload() -> dict:
    return {
        "capability_packages": {
            "items": [
                {
                    "package_id": "capability.answer_wayfinding",
                    "display_name": "Answer wayfinding",
                    "capability": "answer_wayfinding",
                    "status": "pilot",
                    "customer_visible_description": "Answer route questions.",
                }
            ],
            "readiness": [
                {
                    "package_id": "capability.answer_wayfinding",
                    "status": "ready",
                    "status_label": "Ready for site validation",
                    "customer_next_step": "Run site validation.",
                }
            ],
        },
        "scenario_blueprints": {
            "items": [
                {
                    "scenario_id": "visitor_escort",
                    "display_name": "Visitor escort",
                    "package_manifest": {
                        "package_id": "scenario.visitor_escort",
                        "scenario": "visitor_escort",
                        "capability_packages": ["capability.answer_wayfinding"],
                    },
                    "package_readiness": {
                        "package_id": "scenario.visitor_escort",
                        "status": "manual_check",
                        "enablement_decision": {
                            "decision": "human_acceptance_required",
                        },
                    },
                }
            ]
        },
    }


def test_package_readiness_contract_is_customer_safe() -> None:
    contract = package_readiness_contract()

    assert contract["endpoint"] == "/api/capability-packages/readiness"
    assert contract["purpose"] == "评估能力包或场景包是否可以进入客户现场启用前验证。"
    assert contract["customer_statuses"]["ready"] == "可进入现场验证或试点发布评审。"
    assert contract["customer_statuses"]["blocked"] == "缺少依赖时不能面向客户启用。"
    assert "发布流程" not in str(contract)


def test_readiness_kind_normalizes_aliases_and_manifest_shape() -> None:
    assert readiness_kind({"kind": "capability"}, {}) == "capability_package"
    assert readiness_kind({"type": "scenario"}, {}) == "scenario_package"
    assert readiness_kind({}, {"capability": "answer_wayfinding"}) == "capability_package"
    assert readiness_kind({}, {"scenario": "visitor_escort"}) == "scenario_package"
    assert readiness_kind({}, {"capability_packages": ["capability.answer_wayfinding"]}) == (
        "scenario_package"
    )
    assert readiness_kind({"kind": "unknown"}, {}) == "unknown"


def test_capability_center_from_payload_supports_current_and_legacy_shapes() -> None:
    center = _capability_center_payload()

    assert capability_center_from_payload({"skills": {"capability_center": center}}) == center
    assert (
        capability_center_from_payload(
            {"components": {"skill": {"capabilities": {"capability_center": center}}}}
        )
        == center
    )


def test_capability_package_catalog_and_summaries_are_stable() -> None:
    catalog = capability_package_catalog(_capability_center_payload())

    assert catalog["capability_packages"][0]["package_id"] == "capability.answer_wayfinding"
    assert catalog["capability_packages"][0]["customer_status"] == "可进入现场验证"
    assert catalog["capability_packages"][0]["customer_next_step"] == "执行现场验证。"
    assert catalog["scenario_packages"][0]["package_id"] == "scenario.visitor_escort"
    assert catalog["scenario_packages"][0]["customer_status"] == "需要人工验收确认"
    assert catalog["scenario_packages"][0]["customer_next_step"] == "完成主管或现场负责人确认后再启用。"
    assert catalog["scenario_packages"][0]["required_capability_packages"] == [
        "capability.answer_wayfinding"
    ]
    assert package_catalog_summary(catalog) == {
        "capability_package_count": 1,
        "scenario_package_count": 1,
        "ready_count": 1,
        "manual_check_count": 1,
        "blocked_count": 0,
    }
    release = package_release_summary(catalog)
    assert release["controlled_demo_allowed_count"] == 2
    assert release["customer_pilot_allowed_count"] == 1
    assert release["production_claim_allowed_count"] == 0
    assert release["manual_acceptance_required_count"] == 1
    assert release["claim_policy"] == "生产上线声明必须另行完成现场验收和人工接管审批。"


def test_inventory_from_capabilities_payload_collects_skills_services_and_packages() -> None:
    inventory = inventory_from_capabilities_payload(
        {
            "skills": {
                "catalog": [
                    {"skill_name": "answer_wayfinding", "enabled": True},
                    {"skill_name": "disabled_skill", "enabled": False},
                ],
                "skill_packages": {
                    "packages": [
                        {"package_id": "capability.parking", "enabled": True, "rollout_percent": 100},
                        {"package_id": "capability.hidden", "enabled": True, "rollout_percent": 0},
                    ]
                },
            },
            "components": {
                "voice": {"health": {"status": "ok"}},
                "vision": {"health": {"status": "blocked"}},
            },
        }
    )

    assert inventory.skills == frozenset({"answer_wayfinding"})
    assert inventory.services == frozenset({"voice"})
    assert inventory.capability_packages == frozenset(
        {"capability.answer_wayfinding", "capability.parking"}
    )


def test_readiness_inventory_prefers_explicit_inventory_over_runtime_provider() -> None:
    inventory = readiness_inventory(
        {"inventory": {"services": ["vision_bridge"]}},
        lambda: {"components": {"voice": {"health": {"status": "ok"}}}},
    )

    assert inventory.services == frozenset({"vision_bridge"})


def test_capability_center_and_package_catalog_match_api_schemas() -> None:
    center = {
        "title": "Robot Capability Center",
        "summary": {"group_count": 1},
        "groups": [{"group_id": "visitor_service"}],
        "package_readiness": package_readiness_contract(),
        "runtime_blueprints": {"summary": {"blueprint_count": 1}},
        "policy": {"runtime_blueprints_remain_visible_without_live_robot": True},
    }
    catalog = capability_package_catalog(_capability_center_payload())
    package_payload = {
        "ok": True,
        "summary": package_catalog_summary(catalog),
        "release_summary": package_release_summary(catalog),
        "capability_packages": catalog["capability_packages"],
        "scenario_packages": catalog["scenario_packages"],
        "inventory": {"skills": ["answer_wayfinding"]},
        "runtime_blueprints": {"summary": {"blueprint_count": 1}},
        "readiness": package_readiness_contract(),
        "policy": {"blocked_packages_must_not_be_enabled": True},
    }

    center_payload = CapabilityCenterResponse.model_validate(center)
    package_catalog = CapabilityPackageCatalogResponse.model_validate(package_payload)

    assert center_payload.package_readiness["endpoint"] == "/api/capability-packages/readiness"
    assert package_catalog.ok is True
    assert package_catalog.capability_packages[0]["package_id"] == "capability.answer_wayfinding"
    assert package_catalog.scenario_packages[0]["package_id"] == "scenario.visitor_escort"


def test_capability_routes_expose_product_response_schemas_in_openapi() -> None:
    app = FastAPI()
    register_capability_routes(
        app,
        capabilities_provider=lambda: {"skills": {"capability_center": _capability_center_payload()}},
        blueprints_provider=None,
        logger=logging.getLogger("test"),
    )

    paths = app.openapi()["paths"]

    assert (
        paths["/api/capability-center"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        .endswith("/CapabilityCenterResponse")
    )
    assert (
        paths["/api/capability-packages"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        .endswith("/CapabilityPackageCatalogResponse")
    )
    assert (
        paths["/api/capability-packages/readiness"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        .endswith("/CapabilityPackageReadinessResponse")
    )
