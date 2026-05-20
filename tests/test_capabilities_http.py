"""HTTP tests for capability and scenario-intent routes."""

import pytest
from fastapi.testclient import TestClient

from askme.api.schemas.capabilities import (
    RuntimeCapabilitiesResponse,
    ScenarioIntentCatalogResponse,
    ScenarioIntentPreviewResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestCapabilitiesHttp:
    def test_capabilities_endpoint_returns_runtime_contracts(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "profile": {"name": "voice", "primary_loop": "voice"},
                    "components": {
                        "skills": {
                            "health": {"status": "ok"},
                            "capabilities": {"openapi_generated": True},
                        }
                    },
                    "skills": {"contract_count": 3, "code_contract_count": 2},
                },
            )
        )

        response = client.get("/api/capabilities")

        assert response.status_code == 200
        data = response.json()
        schema_payload = RuntimeCapabilitiesResponse.model_validate(data)
        assert schema_payload.profile["name"] == "voice"
        assert data["profile"]["name"] == "voice"
        assert data["components"]["skills"]["capabilities"]["openapi_generated"] is True
        assert data["skills"]["contract_count"] == 3
        capabilities_schema = client.app.openapi()["paths"]["/api/capabilities"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        assert capabilities_schema["$ref"].endswith("/RuntimeCapabilitiesResponse")


    def test_capability_center_endpoint_returns_customer_catalog(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "skills": {
                        "capability_center": {
                            "title": "园区巡检机器人能力中心",
                            "summary": {"group_count": 1},
                            "groups": [{"display_name": "巡检任务", "skills": []}],
                        }
                    }
                },
            )
        )

        response = client.get("/api/capability-center")

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "园区巡检机器人能力中心"
        assert data["groups"][0]["display_name"] == "巡检任务"
        assert data["package_readiness"]["endpoint"] == "/api/capability-packages/readiness"
        assert data["package_readiness"]["customer_statuses"]["blocked"] == "缺少依赖时不能面向客户启用。"
        assert data["runtime_blueprints"]["summary"]["blueprint_count"] >= 6
        assert data["runtime_blueprints"]["summary"]["customer_visible_count"] >= 3
        assert data["runtime_blueprints"]["policy"]["runtime_blueprints_are_delivery_profiles"] is True


    def test_capability_center_endpoint_keeps_blueprints_without_runtime_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/capability-center")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["group_count"] >= 1
        assert data["summary"]["available_count"] >= 1
        assert data["summary"]["scenario_count"] >= 9
        assert data["runtime_blueprints"]["summary"]["blueprint_count"] >= 6
        assert data["runtime_blueprints"]["summary"]["customer_visible_count"] >= 3
        assert data["policy"]["runtime_capability_snapshot_available"] is False
        assert data["policy"]["default_product_catalog_used"] is True
        assert data["package_readiness"]["endpoint"] == "/api/capability-packages/readiness"


    def test_capability_packages_endpoint_returns_customer_enablement_catalog(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "skills": {
                        "catalog": [{"skill_name": "answer_wayfinding", "enabled": True}],
                        "capability_center": {
                            "capability_packages": {
                                "items": [
                                    {
                                        "package_id": "capability.answer_wayfinding",
                                        "display_name": "Answer wayfinding",
                                        "status": "pilot",
                                        "capability": "answer_wayfinding",
                                        "risk_level": "low",
                                        "customer_visible_description": "Answer route questions.",
                                    }
                                ],
                                "readiness": [
                                    {
                                        "kind": "capability_package",
                                        "package_id": "capability.answer_wayfinding",
                                        "status": "ready",
                                        "status_label": "Ready for site validation",
                                        "customer_message": "Package can enter validation.",
                                        "customer_next_step": "Run site validation.",
                                    }
                                ],
                            },
                            "scenario_blueprints": {
                                "items": [
                                    {
                                        "scenario_id": "wayfinding_help_point",
                                        "display_name": "Wayfinding help point",
                                        "coverage_status": "ready",
                                        "package_manifest": {
                                            "package_id": "scenario.wayfinding_help_point",
                                            "scenario": "wayfinding_help_point",
                                            "capability_packages": [
                                                "capability.answer_wayfinding"
                                            ],
                                        },
                                        "package_readiness": {
                                            "kind": "scenario_package",
                                            "package_id": "scenario.wayfinding_help_point",
                                            "status": "ready",
                                            "status_label": "Ready for site validation",
                                        "customer_missing_dependencies": [],
                                        "engineering_missing_dependencies": [],
                                        "enablement_decision": {
                                            "decision": "site_validation_allowed",
                                            "release_claim": "Can enter validation.",
                                        },
                                        "customer_message": "Scenario can enter validation.",
                                        "customer_next_step": "Validate service point trigger.",
                                    },
                                    }
                                ]
                            },
                        },
                    },
                    "components": {"voice": {"health": {"status": "ok"}}},
                },
            )
        )

        response = client.get("/api/capability-packages")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["summary"] == {
            "capability_package_count": 1,
            "scenario_package_count": 1,
            "ready_count": 2,
            "manual_check_count": 0,
            "blocked_count": 0,
        }
        assert data["release_summary"] == {
            "package_count": 2,
            "controlled_demo_allowed_count": 2,
            "customer_pilot_allowed_count": 2,
            "production_claim_allowed_count": 0,
            "blocked_count": 0,
            "manual_acceptance_required_count": 0,
            "claim_policy": "生产上线声明必须另行完成现场验收和人工接管审批。",
        }
        assert data["capability_packages"][0]["package_id"] == "capability.answer_wayfinding"
        assert data["capability_packages"][0]["customer_status"] == "可进入现场验证"
        assert data["capability_packages"][0]["customer_message"] == "能力包可进入现场验证。"
        assert data["capability_packages"][0]["customer_next_step"] == "执行现场验证。"
        assert data["capability_packages"][0]["enablement_decision"] == {}
        assert data["scenario_packages"][0]["package_id"] == "scenario.wayfinding_help_point"
        assert data["scenario_packages"][0]["required_capability_packages"] == [
            "capability.answer_wayfinding"
        ]
        assert data["scenario_packages"][0]["customer_missing_dependencies"] == []
        assert data["scenario_packages"][0]["engineering_missing_dependencies"] == []
        assert data["scenario_packages"][0]["customer_message"] == "场景包可进入现场验证。"
        assert data["scenario_packages"][0]["customer_next_step"] == "验证服务点触发。"
        assert data["scenario_packages"][0]["enablement_decision"]["decision"] == (
            "site_validation_allowed"
        )
        assert data["inventory"]["capability_packages"] == ["capability.answer_wayfinding"]
        assert data["inventory"]["services"] == ["voice"]
        assert data["runtime_blueprints"]["summary"]["blueprint_count"] >= 6
        assert data["runtime_blueprints"]["policy"]["capability_packages_still_require_runtime_blueprint"] is True
        assert data["policy"]["blocked_packages_must_not_be_enabled"] is True
        assert data["policy"]["runtime_blueprints_are_delivery_profiles"] is True


    def test_capability_packages_endpoint_keeps_blueprints_without_runtime_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/capability-packages")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["summary"]["capability_package_count"] >= 1
        assert data["summary"]["scenario_package_count"] >= 9
        assert data["runtime_blueprints"]["summary"]["blueprint_count"] >= 6
        assert data["runtime_blueprints"]["summary"]["customer_visible_count"] >= 3
        assert "answer_wayfinding" in data["inventory"]["skills"]
        assert data["inventory"]["services"] == []
        assert "capability.answer_wayfinding" in data["inventory"]["capability_packages"]
        assert data["policy"]["runtime_capability_snapshot_available"] is False
        assert data["policy"]["default_product_catalog_used"] is True


    def test_capability_package_readiness_endpoint_evaluates_explicit_inventory(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(
            "/api/capability-packages/readiness",
            json={
                "kind": "capability_package",
                "manifest": {
                    "package_id": "cap-fire-smoke",
                    "display_name": "Fire and smoke detection",
                    "capability": "detect_fire_smoke",
                    "inputs": ["vision"],
                    "outputs": ["field_event"],
                    "dependencies": [
                        {
                            "name": "vision_bridge",
                            "kind": "service",
                            "reason": "Provides fresh camera detections.",
                        }
                    ],
                    "risk_level": "high",
                    "risk_controls": ["Notify human responder before public broadcast."],
                    "customer_visible_description": "Detects fire and smoke risk.",
                },
                "inventory": {"services": ["vision_bridge"]},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["readiness"]["status"] == "ready"
        assert payload["readiness"]["enableable"] is True
        assert payload["inventory"]["services"] == ["vision_bridge"]
        assert payload["policy"]["ready_still_requires_site_validation_for_robot_hardware"] is True


    def test_capability_package_readiness_endpoint_derives_runtime_inventory(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "skills": {
                        "catalog": [{"skill_name": "answer_wayfinding", "enabled": True}],
                        "skill_packages": {
                            "packages": [
                                {
                                    "package_id": "cap-wayfinding",
                                    "enabled": True,
                                    "rollout_percent": 100,
                                }
                            ]
                        },
                    },
                    "components": {"voice": {"health": {"status": "ok"}}},
                },
            )
        )

        response = client.post(
            "/api/capability-packages/readiness",
            json={
                "kind": "scenario_package",
                "manifest": {
                    "package_id": "scenario-visitor-guide",
                    "display_name": "Visitor guide",
                    "scenario": "visitor_wayfinding_and_escort",
                    "capability_packages": ["cap-wayfinding", "cap-escort"],
                    "customer_visible_steps": ["Ask destination", "Confirm", "Guide"],
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["kind"] == "scenario_package"
        assert payload["readiness"]["status"] == "blocked"
        assert payload["readiness"]["releasable"] is False
        assert payload["readiness"]["missing_required_dependencies"] == ["cap-escort"]
        assert payload["inventory"]["capability_packages"] == [
            "cap-wayfinding",
            "capability.answer_wayfinding",
        ]
        assert payload["inventory"]["services"] == ["voice"]


    def test_capability_package_readiness_endpoint_maps_enabled_skills_to_capability_packages(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {
                    "skills": {
                        "capability_center": {
                            "groups": [
                                {
                                    "group_id": "visitor",
                                    "skills": [
                                        {
                                            "skill_name": "answer_wayfinding",
                                            "installed": True,
                                            "enabled": True,
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                },
            )
        )

        response = client.post(
            "/api/capability-packages/readiness",
            json={
                "kind": "scenario_package",
                "manifest": {
                    "package_id": "scenario-wayfinding",
                    "display_name": "Wayfinding",
                    "scenario": "wayfinding_help_point",
                    "capability_packages": ["capability.answer_wayfinding"],
                    "customer_visible_steps": ["Answer route"],
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["readiness"]["status"] == "ready"
        assert payload["inventory"]["capability_packages"] == ["capability.answer_wayfinding"]

    @pytest.mark.parametrize(
        "path",
        [
            "/api/capability-packages/readiness",
            "/api/scenario-intents/preview",
        ],
    )

    def test_capability_write_routes_reject_non_object_json_body(self, path):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(path, json=["not-an-object"])

        assert response.status_code == 400
        payload = response.json()
        assert payload["reason"] == "json_object_required"
        assert payload["error"] == "JSON object body required"


    def test_scenario_intent_routes_return_product_contracts(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        catalog = client.get("/api/scenario-intents")
        preview = client.post(
            "/api/scenario-intents/preview",
            json={"text": "coffee shop"},
        )

        assert catalog.status_code == 200
        catalog_payload = catalog.json()
        ScenarioIntentCatalogResponse.model_validate(catalog_payload)
        assert catalog_payload["ok"] is True
        assert catalog_payload["summary"]["rule_count"] >= 1
        assert catalog_payload["policy"]["deterministic_before_llm"] is True

        assert preview.status_code == 200
        preview_payload = preview.json()
        ScenarioIntentPreviewResponse.model_validate(preview_payload)
        assert preview_payload["ok"] is True
        assert preview_payload["text"] == "coffee shop"
        assert preview_payload["policy"]["does_not_execute_skill"] is True
