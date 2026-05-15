"""Tests for the MCP and HTTP health surfaces."""

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.api.routes.field import register_field_routes
from askme.health_server import AskmeHealthServer, build_health_snapshot, create_health_app
from askme.pipeline.field_site_profile import (
    create_customer_project_from_template,
    export_customer_project_package,
    list_delivery_resource_registry,
    upsert_delivery_resource,
)


def _runtime_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={
            "uptime_seconds": 12.5,
            "conversation_count": 7,
            "llm": {
                "last_latency_ms": 245.0,
                "last_model": "claude-opus-4-6",
            },
            "voice_pipeline": {
                "last_input_at": "2026-03-09T04:00:00Z",
                "last_input_chars": 12,
            },
        },
        active_skills=["dock_charge", "inspect_zone"],
        voice_status={
            "mode": "voice",
            "enabled": True,
            "pipeline_ok": True,
            "input_ready": True,
            "output_ready": True,
            "asr_available": True,
            "vad_available": True,
            "kws_available": True,
            "wake_word_enabled": True,
            "woken_up": True,
            "tts_backend": "edge",
            "tts_busy": False,
        },
        ota_status={
            "enabled": True,
            "registered": True,
            "device_id": "INVX-THUNDER-001",
            "channel": "stable",
            "product": "inovxio-dog",
            "state": "connected",
        },
    )


def _degraded_runtime_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={"uptime_seconds": 12.5, "conversation_count": 7},
        active_skills=[],
        voice_status={
            "mode": "voice",
            "enabled": True,
            "pipeline_ok": False,
            "input_ready": False,
            "output_ready": True,
        },
        ota_status={
            "enabled": True,
            "registered": False,
            "device_id": "INVX-THUNDER-001",
            "channel": "stable",
            "product": "inovxio-dog",
            "state": "degraded",
        },
    )


def _field_route_test_app(site_profile_root: Path) -> FastAPI:
    app = FastAPI()

    async def optional_json_body(request):
        return await request.json()

    def mission_json(payload, status_code=200):
        return JSONResponse(payload, status_code=status_code)

    async def passthrough_result(result):
        return result

    register_field_routes(
        app,
        dispatch_field_operations=lambda *_args, **_kwargs: {},
        mission_json=mission_json,
        optional_json_body=optional_json_body,
        cors_options_response=lambda methods: Response(headers={"Access-Control-Allow-Methods": methods}),
        logger=health_server.logger,
        authorize=lambda _request, _body, _permission: None,
        field_manual_trigger_body=lambda _request, body: body,
        looks_like_device_ingest_without_scenario=lambda _body: False,
        dispatch_field_voice_directive=passthrough_result,
        dispatch_field_runtime_policy=lambda result, **_kwargs: passthrough_result(result),
        runtime_callback_trust=lambda _body, **_kwargs: {"trusted": True},
        runtime_callback_delivery_body=lambda body, **_kwargs: body,
        runtime_callback_secret=None,
        runtime_callback_max_age_s=60.0,
        cors_headers={},
        identity_readiness_payload=lambda: {},
        site_profile_root=site_profile_root,
        config_provider=lambda: {},
    )
    return app


class TestHealthResource:
    def test_health_returns_valid_json(self):
        from askme.mcp.resources.health_resources import health_check

        result = health_check()
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_health_has_version(self):
        from askme.mcp.resources.health_resources import health_check

        data = json.loads(health_check())
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_subsystems(self):
        from askme.mcp.resources.health_resources import health_check

        data = json.loads(health_check())
        assert "subsystems" in data
        assert "brain" in data["subsystems"]
        assert "robot" in data["subsystems"]
        assert "voice" in data["subsystems"]

    def test_health_has_uptime(self):
        from askme.mcp.resources.health_resources import health_check

        data = json.loads(health_check())
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestHealthServer:
    def test_http_health_endpoint_returns_runtime_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] == 12.5
        assert data["model_name"] == "claude-opus-4-6"
        assert data["last_llm_latency_ms"] == 245.0
        assert data["total_conversations"] == 7
        assert data["active_skills"] == ["dock_charge", "inspect_zone"]
        assert data["voice_pipeline_status"]["pipeline_ok"] is True
        assert data["ota_bridge_status"]["registered"] is True

    def test_field_notification_preflight_endpoint_reports_blocked(self):
        class Handler:
            def notification_preflight_payload(self):
                return {
                    "status": "blocked",
                    "ready": False,
                    "groups": {"security": {"ready": False}},
                    "blockers": ["security notification is not fully configured"],
                    "next_actions": ["Configure security"],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                field_operations_handler=Handler(),
            )
        )

        response = client.get("/api/field/notification-preflight")

        assert response.status_code == 409
        data = response.json()
        assert data["status"] == "blocked"
        assert data["ready"] is False

        dashboard_response = client.get("/api/field/notification-preflight?status_as_200=true")

        assert dashboard_response.status_code == 200
        dashboard_data = dashboard_response.json()
        assert dashboard_data["status"] == "blocked"
        assert dashboard_data["ready"] is False

    def test_field_devices_endpoint_returns_status_payload(self):
        class Handler:
            def device_status_payload(self):
                return {
                    "status": "ok",
                    "summary": {"registered": 1, "online": 1},
                    "devices": [{"device_id": "smoke-01", "status": "online"}],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                field_operations_handler=Handler(),
            )
        )

        response = client.get("/api/field/devices")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["registered"] == 1
        assert data["devices"][0]["device_id"] == "smoke-01"

    def test_field_site_profiles_endpoint_returns_multi_site_catalog(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/field/site-profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["site_count"] >= 1
        assert data["summary"]["configured_count"] >= 1
        assert data["sites"][0]["site_id"]
        assert data["sites"][0]["deployment_stage"] in {
            "site_config_ready",
            "production_ready",
            "blocked",
        }
        assert "next_step" in data
        assert data["summary"]["customer_count"] >= 1
        assert data["summary"]["managed_object_type_count"] >= 1
        assert data["sites"][0]["managed_objects"]

    def test_field_customer_projects_endpoint_returns_solution_scope(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/field/customer-projects")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["customer_count"] >= 1
        assert data["summary"]["project_count"] >= 1
        assert data["summary"]["tenant_count"] >= 1
        assert data["summary"]["delivery_namespace_count"] >= 1
        assert data["summary"]["managed_object_type_count"] >= 1
        assert data["summary"]["delivery_acceptance_gate_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert data["delivery_acceptance_gate"]["gate_type"] == (
            "askme.solution_delivery_catalog_acceptance_gate"
        )
        assert data["summary"]["scope_filtered"] is True
        assert data["projects"][0]["managed_objects"]
        assert data["projects"][0]["tenant_id"] == "default"
        assert data["projects"][0]["delivery_namespace"] == "default"
        assert data["projects"][0]["product_acceptance_gate"]["gate_type"] == (
            "askme.solution_delivery_product_acceptance_gate"
        )
        assert data["projects"][0]["product_acceptance_gate"]["gates"]
        filtered = client.get(
            "/api/field/customer-projects",
            params={
                "customer_id": data["projects"][0]["customer_id"],
                "gate_status": data["projects"][0]["product_acceptance_gate"]["overall_status"],
            },
        )
        assert filtered.status_code == 200
        filtered_payload = filtered.json()
        assert filtered_payload["filters"]["customer_id"] == data["projects"][0]["customer_id"]
        assert filtered_payload["summary"]["filtered"] is True
        assert filtered_payload["delivery_acceptance_gate"]["project_count"] == (
            filtered_payload["summary"]["project_count"]
        )
        assert data["customer_claim"].startswith("AskMe is configured")
        acceptance_summary = data["projects"][0]["managed_objects_summary"]["acceptance_summary"]
        assert acceptance_summary["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        first_object = data["projects"][0]["managed_objects"][0]
        assert first_object["acceptance_status"]["acceptance_checks"]
        assert first_object["acceptance_status"]["acceptance_checks"][0]["status"] in {
            "linked",
            "node_unresolved",
            "file_missing",
        }
        directory = client.get("/api/field/customer-projects/managed-object-directory")
        assert directory.status_code == 200
        directory_payload = directory.json()
        assert directory_payload["directory_type"] == (
            "askme.customer_project_managed_object_directory"
        )
        assert directory_payload["summary"]["object_count"] >= 1
        assert directory_payload["summary"]["object_count"] == len(directory_payload["objects"])
        assert directory_payload["summary"]["ready_count"] >= 1
        assert directory_payload["summary"]["scope_filtered"] is True
        first_directory_object = directory_payload["objects"][0]
        assert first_directory_object["delivery_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert first_directory_object["tenant_id"] == "default"
        assert first_directory_object["delivery_namespace"] == "default"
        assert first_directory_object["bindings"]["acceptance_tests"]
        assert first_directory_object["resource_binding_status"]["checks"]
        assert first_directory_object["acceptance_status"]["acceptance_checks"]
        assert isinstance(first_directory_object["action_plan"], list)
        assert "action_count" in first_directory_object
        assert first_directory_object["next_step"]
        assert directory_payload["summary"]["action_count"] >= 0
        assert directory_payload["summary"]["blocked_action_count"] >= 0
        assert directory_payload["summary"]["manual_check_action_count"] >= 0
        filtered_directory = client.get(
            "/api/field/customer-projects/managed-object-directory",
            params={"delivery_status": first_directory_object["delivery_status"]},
        )
        assert filtered_directory.status_code == 200
        filtered_directory_payload = filtered_directory.json()
        assert filtered_directory_payload["filters"]["delivery_status"] == (
            first_directory_object["delivery_status"]
        )
        assert filtered_directory_payload["summary"]["filtered"] is True
        assert {
            row["delivery_status"] for row in filtered_directory_payload["objects"]
        } == {first_directory_object["delivery_status"]}

    def test_managed_object_directory_generates_action_plan_for_blocked_bindings(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]
        shutil.copytree(repo_root / "deploy" / "site-profiles", Path("deploy/site-profiles"))
        profile_path = Path("deploy/site-profiles/park-demo.yaml")
        profile_text = profile_path.read_text(encoding="utf-8")
        profile_text = profile_text.replace("vehicle-detection", "missing-vehicle-model", 1)
        profile_text = profile_text.replace(
            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking",
            "tests/missing_acceptance.py::missing_case",
            1,
        )
        profile_path.write_text(profile_text, encoding="utf-8")
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get(
            "/api/field/customer-projects/managed-object-directory",
            params={"delivery_status": "blocked"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["object_count"] == 1
        assert payload["summary"]["blocked_count"] == 1
        assert payload["summary"]["action_count"] >= 2
        vehicle = payload["objects"][0]
        assert vehicle["object_id"] == "vehicles"
        assert vehicle["delivery_status"] == "blocked"
        assert vehicle["blocked_action_count"] >= 1
        assert vehicle["manual_check_action_count"] >= 1
        assert vehicle["next_step"]
        actions = {item["action"] for item in vehicle["action_plan"]}
        assert "register_delivery_resource" in actions
        assert "fix_acceptance_test_reference" in actions
        owners = {item["owner"] for item in vehicle["action_plan"]}
        assert {"delivery_owner", "qa_owner"} <= owners
        assert all(item["action_label"] for item in vehicle["action_plan"])
        assert all(item["reason_label"] for item in vehicle["action_plan"])
        assert all(item["owner_label"] for item in vehicle["action_plan"])
        assert all(item["customer_next_step"] for item in vehicle["action_plan"])
        assert "交付负责人" in {item["owner_label"] for item in vehicle["action_plan"]}
        assert "测试/验收负责人" in {item["owner_label"] for item in vehicle["action_plan"]}

    def test_field_solution_delivery_readiness_endpoint_returns_product_gate(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/field/solution-delivery-readiness")

        assert response.status_code == 200
        payload = response.json()
        assert payload["readiness_type"] == "askme.solution_delivery_readiness"
        assert payload["overall_status"] in {"ready", "manual_check", "blocked"}
        assert payload["summary"]["project_count"] >= 1
        assert payload["summary"]["template_count"] >= 4
        assert payload["summary"]["resource_count"] >= 10
        assert payload["customer_status"]
        assert payload["release_claim"]
        assert {
            "customer_project_acceptance",
            "template_market",
            "delivery_resource_bindings",
            "delivery_resource_governance",
        } <= {gate["gate_id"] for gate in payload["gates"]}

        workbench = client.get("/api/field/customer-project-workbench", params={"check_env": "true"})
        assert workbench.status_code == 200
        workbench_payload = workbench.json()
        assert workbench_payload["workbench_type"] == (
            "askme.solution_provider_customer_project_workbench.v1"
        )
        assert workbench_payload["overall_status"] in {"ready", "manual_check", "blocked"}
        assert workbench_payload["solution_delivery_readiness"]["readiness_type"] == (
            "askme.solution_delivery_readiness"
        )
        assert {
            "customer_projects",
            "template_market",
            "managed_objects",
            "delivery_resources",
            "package_delivery_gate",
        } <= {surface["surface_id"] for surface in workbench_payload["delivery_surfaces"]}
        assert {
            "客户项目目录",
            "行业模板市场",
            "对象目录",
            "交付资源",
            "交付包准入",
        } <= {surface["customer_label"] for surface in workbench_payload["delivery_surfaces"]}
        assert workbench_payload["customer_readable_contract"]["contract_type"] == (
            "askme.solution_provider_customer_delivery_contract.v1"
        )
        assert "客户只能看到自己项目、对象、证据和交付包。" in {
            item["customer_value"] for item in workbench_payload["customer_acceptance_flow"]
        }
        vocabulary = {
            item["internal"]: item["customer_label"]
            for item in workbench_payload["customer_vocabulary"]
        }
        assert vocabulary["tenant_id"] == "客户空间"
        assert vocabulary["managed_object"] == "现场对象"
        assert vocabulary["runtime"] == "执行服务"
        assert workbench_payload["customer_projects"]["summary"]["project_count"] >= 1
        assert workbench_payload["managed_object_directory"]["summary"]["object_count"] >= 1
        assert workbench_payload["template_market"]["summary"]["template_count"] >= 4

        launch = client.get("/api/field/product-launch-readiness", params={"check_env": "true"})
        assert launch.status_code == 200
        launch_payload = launch.json()
        assert launch_payload["readiness_type"] == "askme.product_launch_readiness.v1"
        assert launch_payload["overall_status"] in {"ready", "manual_check", "blocked"}
        assert launch_payload["launch_stage"] in {
            "production_acceptance_ready",
            "pilot_or_site_trial",
            "demo_or_integration_only",
        }
        assert launch_payload["customer_status"]
        assert launch_payload["release_claim"]
        assert {
            "identity_gateway",
            "field_operations",
            "solution_delivery",
            "customer_project_workbench",
        } <= {gate["gate_id"] for gate in launch_payload["gates"]}
        assert {
            "/api/governance/identity-readiness",
            "/api/field/readiness",
            "/api/field/solution-delivery-readiness",
            "/api/field/customer-project-workbench",
        } <= {source["endpoint"] for source in launch_payload["evidence_sources"]}

    def test_field_customer_project_read_endpoints_require_known_operator(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))
        headers = {"X-Askme-Operator-Id": "ghost.operator"}

        for path in (
            "/api/field/events",
            "/api/field/site-profiles",
            "/api/field/customer-projects",
            "/api/field/customer-projects/managed-object-directory",
            "/api/field/customer-project-templates",
            "/api/field/customer-project-acceptance-registry",
            "/api/field/customer-project-resource-catalog",
            "/api/field/delivery-resource-registry",
            "/api/field/delivery-resource-registry/history",
            "/api/field/customer-projects/demo-field-ops",
            "/api/field/customer-projects/demo-field-ops/acceptance-report",
            "/api/field/customer-projects/demo-field-ops/export",
            "/api/field/customer-projects/demo-field-ops/acceptance-dossier",
            "/api/field/customer-projects/demo-field-ops/history",
        ):
            response = client.get(path, headers=headers)

            assert response.status_code == 403
            assert response.json()["reason"] == "operator_missing_permission"
            assert response.json()["operator_auth"]["permission"] == "field:project:read"

        for path in (
            "/api/field/customer-projects/package/verify",
            "/api/field/customer-projects/package/diff",
        ):
            response = client.post(path, json={"operator_id": "ghost.operator", "package": {}})

            assert response.status_code == 403
            assert response.json()["reason"] == "operator_missing_permission"
            assert response.json()["operator_auth"]["permission"] == "field:project:read"

    def test_field_customer_project_write_endpoints_enforce_project_scope(self, monkeypatch):
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {
                "field_operations": {
                    "operator_directory": {"mode": "demo_config"},
                    "operators": {
                        "supervisor-1": {"roles": ["supervisor"]},
                        "scoped-supervisor": {
                            "roles": ["supervisor"],
                            "project_scope": {
                                "customer_ids": ["other-customer"],
                                "project_ids": ["other-project"],
                                "site_ids": ["other-site"],
                            },
                        },
                        "scoped-owner": {
                            "roles": ["product_owner"],
                            "project_scope": {
                                "customer_ids": ["other-customer"],
                                "project_ids": ["other-project"],
                                "site_ids": ["other-site"],
                            },
                        },
                    },
                }
            },
        )
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        exported = client.get(
            "/api/field/customer-projects/demo-field-ops/export",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert exported.status_code == 200
        package = exported.json()["package"]
        forbidden_operator = {"operator_id": "scoped-supervisor"}

        denied_from_template = client.post(
            "/api/field/customer-projects/from-template",
            json={
                **forbidden_operator,
                "template_id": "park-visitor-service",
                "customer": {
                    "customer_id": "demo-customer",
                    "project_id": "demo-field-ops",
                },
                "site": {"site_id": "inovx-demo-park"},
            },
        )
        assert denied_from_template.status_code == 403
        assert denied_from_template.json()["reason"] == "project_scope_not_allowed"

        denied_upsert = client.post(
            "/api/field/customer-projects",
            json={
                **forbidden_operator,
                "profile": package["profile"],
            },
        )
        assert denied_upsert.status_code == 403
        assert denied_upsert.json()["reason"] == "project_scope_not_allowed"

        denied_import = client.post(
            "/api/field/customer-projects/import",
            json={**forbidden_operator, "dry_run": True, "package": package},
        )
        assert denied_import.status_code == 403
        assert denied_import.json()["reason"] == "project_scope_not_allowed"

        denied_resource = client.post(
            "/api/field/delivery-resource-registry",
            json={
                **forbidden_operator,
                "resource": {
                    "resource_type": "vision_models",
                    "resource_id": "scope-blocked-model",
                    "display_name": "Scope blocked model",
                    "project_id": "demo-field-ops",
                },
            },
        )
        assert denied_resource.status_code == 403
        assert denied_resource.json()["reason"] == "project_scope_not_allowed"

        denied_registry_rollback = client.post(
            "/api/field/delivery-resource-registry/rollback",
            json={
                **forbidden_operator,
                "revision_id": "missing-revision",
                "dry_run": True,
            },
        )
        assert denied_registry_rollback.status_code == 403
        assert denied_registry_rollback.json()["reason"] == "operator_missing_permission"
        assert denied_registry_rollback.json()["operator_auth"]["permission"] == (
            "resource:governance:approve"
        )

        scoped_registry_request = client.post(
            "/api/field/delivery-resource-governance-requests",
            json={
                "operator_id": "scoped-owner",
                "action": "rollback_registry",
                "operation": {"revision_id": "missing-revision"},
            },
        )
        assert scoped_registry_request.status_code == 403
        assert scoped_registry_request.json()["reason"] == (
            "resource_registry_rollback_requires_unrestricted_operator"
        )

        scoped_registry_rollback = client.post(
            "/api/field/delivery-resource-registry/rollback",
            json={
                "operator_id": "scoped-owner",
                "revision_id": "missing-revision",
                "dry_run": True,
            },
        )
        assert scoped_registry_rollback.status_code == 403
        assert scoped_registry_rollback.json()["reason"] == (
            "resource_registry_rollback_requires_unrestricted_operator"
        )

        denied_object = client.post(
            "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
            json={
                **forbidden_operator,
                "managed_object": {
                    "display_name": "Vehicles",
                    "category": "traffic",
                },
            },
        )
        assert denied_object.status_code == 403
        assert denied_object.json()["reason"] == "project_scope_not_allowed"

        denied_delete = client.request(
            "DELETE",
            "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
            json=forbidden_operator,
        )
        assert denied_delete.status_code == 403
        assert denied_delete.json()["reason"] == "project_scope_not_allowed"

        denied_archive = client.post(
            "/api/field/customer-projects/demo-field-ops/archive",
            json=forbidden_operator,
        )
        assert denied_archive.status_code == 403
        assert denied_archive.json()["reason"] == "project_scope_not_allowed"

        denied_rollback = client.post(
            "/api/field/customer-projects/demo-field-ops/rollback",
            json={**forbidden_operator, "revision_id": "missing"},
        )
        assert denied_rollback.status_code == 403
        assert denied_rollback.json()["reason"] == "project_scope_not_allowed"

        denied_onsite = client.post(
            "/api/field/customer-projects/demo-field-ops/onsite-evidence",
            json={
                **forbidden_operator,
                "evidence": {
                    "evidence_type": "device_ingest",
                    "status": "passed",
                    "summary": "scope should block this receipt",
                },
            },
        )
        assert denied_onsite.status_code == 403
        assert denied_onsite.json()["reason"] == "project_scope_not_allowed"

        denied_review = client.post(
            "/api/field/customer-projects/demo-field-ops/acceptance-review",
            json={
                **forbidden_operator,
                "review": {
                    "decision": "accepted",
                    "risk_acknowledgement": True,
                    "reason": "scope should block this review",
                },
            },
        )
        assert denied_review.status_code == 403
        assert denied_review.json()["reason"] == "project_scope_not_allowed"

        denied_execution = client.request(
            "GET",
            "/api/field/customer-projects/demo-field-ops/execution-bindings",
            headers={"X-Askme-Operator-Id": "scoped-supervisor"},
        )
        assert denied_execution.status_code == 403
        assert denied_execution.json()["reason"] == "project_scope_not_allowed"

        denied_rehearsal = client.post(
            "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
            json={**forbidden_operator, "mode": "dry_run"},
        )
        assert denied_rehearsal.status_code == 403
        assert denied_rehearsal.json()["reason"] == "project_scope_not_allowed"

        denied_signoff = client.post(
            "/api/field/customer-projects/demo-field-ops/customer-signoff",
            json={
                **forbidden_operator,
                "signoff": {
                    "decision": "needs_fix",
                    "signatory_name": "Scope blocked customer",
                    "reason": "scope should block this signoff",
                },
            },
        )
        assert denied_signoff.status_code == 403
        assert denied_signoff.json()["reason"] == "project_scope_not_allowed"

    def test_field_delivery_resource_governance_queue_requires_second_review(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]
        shutil.copytree(repo_root / "deploy" / "site-profiles", Path("deploy/site-profiles"))
        shutil.copytree(
            repo_root / "deploy" / "customer-project-templates",
            Path("deploy/customer-project-templates"),
        )
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {
                "field_operations": {
                    "operator_directory": {"mode": "demo_config"},
                    "delivery_resource_governance": {
                        "delivery_owner_notifications": {
                            "enabled": True,
                            "severity_routes": {"warning": ["log"]},
                        }
                    },
                    "operators": {
                        "product.owner": {"roles": ["product_owner"]},
                        "product.reviewer": {"roles": ["product_owner"]},
                    },
                }
            },
        )
        resource_root = Path("deploy/delivery-resources")
        upsert_delivery_resource(
            resource_root,
            "vision_models",
            "vehicle-detection",
            {
                "display_name": "Vehicle detector",
                "version": "v1.0.0",
                "publish_status": "published",
            },
            operator_id="product.owner",
            reason="initial registration",
        )
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        created = client.post(
            "/api/field/delivery-resource-governance-requests",
            json={
                "operator_id": "product.owner",
                "action": "disable_resource",
                "operation": {
                    "resource_type": "vision_models",
                    "resource_id": "vehicle-detection",
                },
                "reason": "bad onsite accuracy",
                "sla_target_s": 60,
            },
        )

        assert created.status_code == 200
        payload = created.json()
        assert payload["accepted"] is True
        assert payload["request"]["status"] == "pending"
        assert payload["request"]["action"] == "disable_resource"
        assert payload["request"]["sla_target_s"] == 60
        assert payload["request"]["due_at"] > payload["request"]["requested_at"]
        assert payload["request"]["review_sla"]["state"] in {"active", "due_soon"}
        assert payload["preview"]["dry_run"] is True
        impact = payload["preview"]["impact"]
        assert impact["analysis_status"] == "complete"
        assert impact["affected_consumer_count"] >= 1
        assert any(
            item["scope_type"] in {"project", "template"}
            for item in impact["affected_consumers"]
        )
        registry = list_delivery_resource_registry(resource_root)
        assert (
            registry["delivery_resources"]["vision_models"]["vehicle-detection"]["publish_status"]
            == "published"
        )

        queue = client.get(
            "/api/field/delivery-resource-governance-requests",
            params={"status": "pending"},
            headers={"X-Askme-Operator-Id": "product.owner"},
        )
        assert queue.status_code == 200
        assert queue.json()["summary"]["pending_count"] == 1
        assert "overdue_count" in queue.json()["summary"]
        assert queue.json()["summary"]["overdue_count"] == 0
        queued = queue.json()["requests"][0]
        assert queued["preview"]["impact"]["affected_consumer_count"] == (
            impact["affected_consumer_count"]
        )
        assert queued["preview"]["impact"]["affected_consumers"]
        assert queued["review_sla"]["state"] in {"active", "due_soon"}
        request_id = payload["request"]["request_id"]
        request_path = Path(payload["request"]["request_path"])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        request_payload["requested_at"] = time.time() - 120
        request_payload["due_at"] = time.time() - 60
        request_payload["sla_target_s"] = 60
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        overdue_queue = client.get(
            "/api/field/delivery-resource-governance-requests",
            params={"status": "pending", "overdue_only": "true"},
            headers={"X-Askme-Operator-Id": "product.owner"},
        )
        assert overdue_queue.status_code == 200
        assert overdue_queue.json()["overdue_only"] is True
        assert overdue_queue.json()["summary"]["overdue_count"] == 1
        assert overdue_queue.json()["request_count"] == 1
        assert overdue_queue.json()["requests"][0]["review_sla"]["state"] == "overdue"
        assert overdue_queue.json()["requests"][0]["review_sla"]["escalation_required"] is True
        escalation = client.post(
            "/api/field/delivery-resource-governance-requests/escalate-overdue",
            json={
                "operator_id": "product.reviewer",
                "reason": "approval SLA missed",
            },
        )
        assert escalation.status_code == 200
        assert escalation.json()["accepted"] is True
        assert escalation.json()["escalated_count"] == 1
        assert escalation.json()["escalations"][0]["notification"]["status"] == "sent"
        assert escalation.json()["escalations"][0]["notification"]["delivery_mode"] == (
            "configured_channels"
        )
        assert escalation.json()["escalations"][0]["notification"]["sent_channels"] == ["log"]
        assert escalation.json()["escalations"][0]["delivery_report"][0]["channel"] == "log"
        assert escalation.json()["requests"][0]["escalation_count"] == 1
        assert escalation.json()["requests"][0]["last_escalation"]["status"] == "sent"

        self_review = client.post(
            f"/api/field/delivery-resource-governance-requests/{request_id}/review",
            json={
                "operator_id": "product.owner",
                "decision": "approve",
                "reason": "self review should fail",
            },
        )
        assert self_review.status_code == 409
        assert self_review.json()["reason"] == (
            "resource_governance_request_requires_second_approver"
        )
        assert self_review.json()["request"]["review_sla"]["state"] == "overdue"
        assert self_review.json()["request"]["escalation_count"] == 1

        approved = client.post(
            f"/api/field/delivery-resource-governance-requests/{request_id}/review",
            json={
                "operator_id": "product.reviewer",
                "decision": "approve",
                "reason": "approve resource disable",
            },
        )
        assert approved.status_code == 200
        assert approved.json()["request"]["status"] == "approved"
        assert approved.json()["request"]["review_sla"]["state"] == "closed"
        assert approved.json()["apply_result"]["accepted"] is True
        disabled = list_delivery_resource_registry(resource_root)
        assert (
            disabled["delivery_resources"]["vision_models"]["vehicle-detection"]["publish_status"]
            == "disabled"
        )

    def test_field_customer_project_scope_enforces_tenant_namespace(self, monkeypatch):
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {
                "field_operations": {
                    "operator_directory": {"mode": "demo_config"},
                    "operators": {
                        "supervisor-1": {"roles": ["supervisor"]},
                        "default-tenant-supervisor": {
                            "roles": ["supervisor"],
                            "project_scope": {
                                "tenant_ids": ["default"],
                                "delivery_namespaces": ["default"],
                                "customer_ids": ["demo-customer"],
                                "project_ids": ["demo-field-ops"],
                                "site_ids": ["inovx-demo-park"],
                            },
                        },
                        "pilot-tenant-supervisor": {
                            "roles": ["supervisor"],
                            "project_scope": {
                                "tenant_ids": ["tenant-a"],
                                "delivery_namespaces": ["pilot"],
                                "customer_ids": ["demo-customer"],
                                "project_ids": ["demo-field-ops"],
                                "site_ids": ["inovx-demo-park"],
                            },
                        },
                    },
                }
            },
        )
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        allowed_detail = client.get(
            "/api/field/customer-projects/demo-field-ops",
            headers={"X-Askme-Operator-Id": "default-tenant-supervisor"},
        )
        denied_detail = client.get(
            "/api/field/customer-projects/demo-field-ops",
            headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
        )

        assert allowed_detail.status_code == 200
        assert allowed_detail.json()["customer"]["tenant_id"] == "default"
        assert denied_detail.status_code == 403
        assert denied_detail.json()["reason"] == "project_scope_not_allowed"
        allowed_directory = client.get(
            "/api/field/customer-projects/managed-object-directory",
            headers={"X-Askme-Operator-Id": "default-tenant-supervisor"},
        )
        denied_directory = client.get(
            "/api/field/customer-projects/managed-object-directory",
            headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
        )
        assert allowed_directory.status_code == 200
        assert allowed_directory.json()["summary"]["object_count"] >= 1
        assert denied_directory.status_code == 200
        assert denied_directory.json()["summary"]["object_count"] == 0
        assert denied_directory.json()["summary"]["scope_filtered"] is True

        exported = client.get(
            "/api/field/customer-projects/demo-field-ops/export",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert exported.status_code == 200
        package = exported.json()["package"]

        allowed_verify = client.post(
            "/api/field/customer-projects/package/verify",
            json={"operator_id": "default-tenant-supervisor", "package": package},
        )
        denied_verify = client.post(
            "/api/field/customer-projects/package/verify",
            json={"operator_id": "pilot-tenant-supervisor", "package": package},
        )

        assert allowed_verify.status_code == 200
        assert allowed_verify.json()["package_scope"]["tenant_id"] == "default"
        assert allowed_verify.json()["package_scope"]["delivery_namespace"] == "default"
        assert allowed_verify.json()["operator_project_scope"]["tenant_ids"] == ["default"]
        assert allowed_verify.json()["operator_project_scope"]["delivery_namespaces"] == ["default"]
        assert denied_verify.status_code == 403
        assert denied_verify.json()["reason"] == "project_scope_not_allowed"

        allowed_diff = client.post(
            "/api/field/customer-projects/package/diff",
            json={"operator_id": "default-tenant-supervisor", "package": package},
        )
        denied_diff = client.post(
            "/api/field/customer-projects/package/diff",
            json={"operator_id": "pilot-tenant-supervisor", "package": package},
        )
        assert allowed_diff.status_code == 200
        assert allowed_diff.json()["package_scope"]["tenant_id"] == "default"
        assert allowed_diff.json()["package_scope"]["delivery_namespace"] == "default"
        assert allowed_diff.json()["operator_project_scope"]["tenant_ids"] == ["default"]
        assert isinstance(allowed_diff.json()["would_write"], bool)
        assert denied_diff.status_code == 403
        assert denied_diff.json()["reason"] == "project_scope_not_allowed"

        allowed_import_dry_run = client.post(
            "/api/field/customer-projects/import",
            json={
                "operator_id": "default-tenant-supervisor",
                "package": package,
                "dry_run": True,
            },
        )
        denied_import_dry_run = client.post(
            "/api/field/customer-projects/import",
            json={
                "operator_id": "pilot-tenant-supervisor",
                "package": package,
                "dry_run": True,
            },
        )
        assert allowed_import_dry_run.status_code == 200
        assert allowed_import_dry_run.json()["package_scope"]["tenant_id"] == "default"
        assert allowed_import_dry_run.json()["package_scope"]["delivery_namespace"] == "default"
        assert allowed_import_dry_run.json()["operator_project_scope"]["tenant_ids"] == ["default"]
        assert allowed_import_dry_run.json()["dry_run"] is True
        assert isinstance(allowed_import_dry_run.json()["would_write"], bool)
        assert denied_import_dry_run.status_code == 403
        assert denied_import_dry_run.json()["reason"] == "project_scope_not_allowed"

        dossier_response = client.get(
            "/api/field/customer-projects/demo-field-ops/acceptance-dossier",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert dossier_response.status_code == 200
        dossier = dossier_response.json()["dossier"]
        allowed_dossier_verify = client.post(
            "/api/field/customer-projects/acceptance-dossier/verify",
            json={"operator_id": "default-tenant-supervisor", "dossier": dossier},
        )
        denied_dossier_verify = client.post(
            "/api/field/customer-projects/acceptance-dossier/verify",
            json={"operator_id": "pilot-tenant-supervisor", "dossier": dossier},
        )
        assert allowed_dossier_verify.status_code == 200
        assert allowed_dossier_verify.json()["dossier_scope"]["tenant_id"] == "default"
        assert allowed_dossier_verify.json()["dossier_scope"]["delivery_namespace"] == "default"
        assert denied_dossier_verify.status_code == 403
        assert denied_dossier_verify.json()["reason"] == "project_scope_not_allowed"

        scoped_templates = client.get(
            "/api/field/customer-project-templates",
            headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
        )
        assert scoped_templates.status_code == 200
        assert scoped_templates.json()["summary"]["scope_filtered"] is True
        assert scoped_templates.json()["summary"]["template_count"] >= 1

    def test_field_customer_project_from_template_returns_implementation_handoff(self, tmp_path):
        client = TestClient(_field_route_test_app(tmp_path / "site-profiles"))

        response = client.post(
            "/api/field/customer-projects/from-template",
            json={
                "operator_id": "delivery.manager",
                "template_id": "factory-inspection",
                "customer": {
                    "tenant_id": "tenant-api",
                    "delivery_namespace": "pilot",
                    "customer_id": "api-customer",
                    "customer_name": "API Customer",
                    "industry": "manufacturing",
                    "project_id": "api-line-one",
                    "project_name": "API Line One",
                },
                "site": {"site_id": "api-site", "name": "API Site"},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["accepted"] is True
        handoff = payload["implementation_handoff"]
        assert handoff["handoff_schema"] == "askme.customer_project_implementation_handoff.v1"
        assert handoff["project_id"] == "api-line-one"
        assert handoff["customer_name"] == "API Customer"
        assert handoff["site_name"] == "API Site"
        assert handoff["summary"]["object_count"] >= 2
        assert handoff["summary"]["object_ready_count"] >= 1
        assert handoff["next_steps"][0]["label"] == "核对项目基础信息"
        assert handoff["next_steps"][1]["label"] == "补齐对象能力绑定"
        assert "项目已创建" in handoff["customer_status"]
        assert Path(payload["profile_path"]).exists()

    def test_field_customer_project_import_route_returns_implementation_handoff(self, tmp_path):
        source_root = tmp_path / "source-profiles"
        import_root = tmp_path / "import-profiles"
        created = create_customer_project_from_template(
            template_root=Path("deploy/customer-project-templates"),
            profile_root=source_root,
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-api-import",
                "delivery_namespace": "pilot",
                "customer_id": "api-import-customer",
                "customer_name": "API Import Customer",
                "industry": "manufacturing",
                "project_id": "api-import-line",
                "project_name": "API Import Line",
            },
            site={"site_id": "api-import-site", "name": "API Import Site"},
        )
        assert created["accepted"] is True
        exported = export_customer_project_package(
            source_root,
            "api-import-line",
            output_root=tmp_path / "packages",
        )
        assert exported["accepted"] is True

        client = TestClient(_field_route_test_app(import_root))
        preview = client.post(
            "/api/field/customer-projects/import",
            json={
                "operator_id": "delivery.manager",
                "dry_run": True,
                "package": exported["package"],
            },
        )
        assert preview.status_code == 200
        preview_payload = preview.json()
        assert preview_payload["dry_run"] is True
        assert preview_payload["would_write"] is True
        assert preview_payload["implementation_handoff"]["project_id"] == "api-import-line"
        assert preview_payload["implementation_handoff"]["summary"]["object_count"] >= 2

        imported = client.post(
            "/api/field/customer-projects/import",
            json={
                "operator_id": "delivery.manager",
                "package": exported["package"],
            },
        )
        assert imported.status_code == 200
        imported_payload = imported.json()
        assert imported_payload["accepted"] is True
        assert imported_payload["implementation_handoff"]["handoff_schema"] == (
            "askme.customer_project_implementation_handoff.v1"
        )
        assert imported_payload["implementation_handoff"]["project_id"] == "api-import-line"
        assert imported_payload["implementation_handoff"]["next_steps"][2]["label"] == "登记现场验收证据"
        assert Path(imported_payload["profile_path"]).exists()

    def test_field_customer_project_upsert_route_returns_implementation_handoff(self, tmp_path):
        created = create_customer_project_from_template(
            template_root=Path("deploy/customer-project-templates"),
            profile_root=tmp_path / "source-upsert-profiles",
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-api-upsert",
                "delivery_namespace": "pilot",
                "customer_id": "api-upsert-customer",
                "customer_name": "API Upsert Customer",
                "industry": "manufacturing",
                "project_id": "api-upsert-line",
                "project_name": "API Upsert Line",
            },
            site={"site_id": "api-upsert-site", "name": "API Upsert Site"},
        )
        assert created["accepted"] is True
        profile = created["profile"]
        profile["customer"]["project_name"] = "API Upsert Line Updated"

        client = TestClient(_field_route_test_app(tmp_path / "upsert-target-profiles"))
        response = client.post(
            "/api/field/customer-projects",
            json={
                "operator_id": "delivery.manager",
                "profile": profile,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["accepted"] is True
        assert payload["implementation_handoff"]["handoff_schema"] == (
            "askme.customer_project_implementation_handoff.v1"
        )
        assert payload["implementation_handoff"]["project_id"] == "api-upsert-line"
        assert payload["implementation_handoff"]["project_name"] == "API Upsert Line Updated"
        assert payload["implementation_handoff"]["summary"]["object_count"] >= 2
        assert Path(payload["profile_path"]).exists()

    def test_field_customer_project_detail_route_returns_implementation_handoff(self, tmp_path):
        profile_root = tmp_path / "detail-profiles"
        created = create_customer_project_from_template(
            template_root=Path("deploy/customer-project-templates"),
            profile_root=profile_root,
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-api-detail",
                "delivery_namespace": "pilot",
                "customer_id": "api-detail-customer",
                "customer_name": "API Detail Customer",
                "industry": "manufacturing",
                "project_id": "api-detail-line",
                "project_name": "API Detail Line",
            },
            site={"site_id": "api-detail-site", "name": "API Detail Site"},
        )
        assert created["accepted"] is True
        client = TestClient(_field_route_test_app(profile_root))

        response = client.get("/api/field/customer-projects/api-detail-line?check_env=true")

        assert response.status_code == 200
        payload = response.json()
        assert payload["found"] is True
        assert payload["implementation_handoff"]["handoff_schema"] == (
            "askme.customer_project_implementation_handoff.v1"
        )
        assert payload["implementation_handoff"]["project_id"] == "api-detail-line"
        assert payload["implementation_handoff"]["customer_name"] == "API Detail Customer"
        assert payload["implementation_handoff"]["summary"]["object_count"] >= 2
        assert payload["next_step"] == payload["implementation_handoff"]["customer_status"]

    def test_field_customer_project_managed_object_routes_return_implementation_handoff(self, tmp_path):
        profile_root = tmp_path / "object-route-profiles"
        created = create_customer_project_from_template(
            template_root=Path("deploy/customer-project-templates"),
            profile_root=profile_root,
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-api-object",
                "delivery_namespace": "pilot",
                "customer_id": "api-object-customer",
                "customer_name": "API Object Customer",
                "industry": "manufacturing",
                "project_id": "api-object-line",
                "project_name": "API Object Line",
            },
            site={"site_id": "api-object-site", "name": "API Object Site"},
        )
        assert created["accepted"] is True
        client = TestClient(_field_route_test_app(profile_root))

        saved = client.post(
            "/api/field/customer-projects/api-object-line/managed-objects/custom-gate",
            json={
                "operator_id": "delivery.manager",
                "reason": "Add customer-specific gate.",
                "managed_object": {
                    "display_name": "Custom Gate",
                    "category": "access",
                    "object_labels": ["gate"],
                    "scenario_ids": ["gate_inspection"],
                    "zone_types": ["main_channel"],
                    "device_sources": ["camera"],
                    "responder_group": "operations",
                    "evidence_required": ["photo", "location"],
                    "bindings": {
                        "vision_models": ["gate-detection"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": [
                            "tests/scenario_tests/test_field_operations_evaluation.py::gate_inspection"
                        ],
                    },
                },
            },
        )
        assert saved.status_code == 200
        saved_payload = saved.json()
        assert saved_payload["accepted"] is True
        assert saved_payload["object_id"] == "custom-gate"
        assert saved_payload["implementation_handoff"]["handoff_schema"] == (
            "askme.customer_project_implementation_handoff.v1"
        )
        assert saved_payload["implementation_handoff"]["project_id"] == "api-object-line"
        assert saved_payload["implementation_handoff"]["summary"]["object_count"] >= 3
        assert saved_payload["next_step"] == saved_payload["implementation_handoff"]["customer_status"]

        deleted = client.request(
            "DELETE",
            "/api/field/customer-projects/api-object-line/managed-objects/custom-gate",
            json={
                "operator_id": "delivery.manager",
                "reason": "Customer removed this gate from scope.",
            },
        )
        assert deleted.status_code == 200
        deleted_payload = deleted.json()
        assert deleted_payload["accepted"] is True
        assert deleted_payload["implementation_handoff"]["handoff_schema"] == (
            "askme.customer_project_implementation_handoff.v1"
        )
        assert deleted_payload["implementation_handoff"]["project_id"] == "api-object-line"
        assert deleted_payload["implementation_handoff"]["summary"]["object_count"] >= 2
        assert deleted_payload["next_step"] == deleted_payload["implementation_handoff"]["customer_status"]

    def test_field_customer_project_templates_and_export_endpoints(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        templates = client.get("/api/field/customer-project-templates")
        assert templates.status_code == 200
        template_payload = templates.json()
        assert template_payload["summary"]["template_count"] >= 4
        assert template_payload["summary"]["valid_count"] >= 4
        first_template = template_payload["templates"][0]
        assert first_template["template_package"]["package_schema"] == "askme.customer_project_template.v1"
        assert first_template["template_package"]["version"] == "0.1.0"
        assert first_template["template_package"]["publish_status"] == "pilot"
        assert first_template["template_package"]["product_status"] == "manual_check"
        assert first_template["delivery_summary"]["default_object_count"] >= 2
        assert first_template["delivery_summary"]["template_version"] == "0.1.0"
        assert first_template["delivery_summary"]["scenario_ids"]
        assert first_template["delivery_summary"]["skill_packages"]
        assert first_template["applicability_scope"]["industries"]
        assert first_template["out_of_scope"]
        assert first_template["customer_prerequisites"]
        assert first_template["scenario_acceptance_criteria"]
        assert first_template["dependency_matrix"]
        assert first_template["delivery_checklist"][0]["step_id"] == "validate_template"

        filtered_templates = client.get(
            "/api/field/customer-project-templates",
            params={
                "industry": first_template["industry"],
                "publish_status": first_template["publish_status"],
                "product_status": first_template["product_status"],
            },
        )
        assert filtered_templates.status_code == 200
        filtered_template_payload = filtered_templates.json()
        assert filtered_template_payload["filters"]["industry"] == first_template["industry"]
        assert filtered_template_payload["filters"]["publish_status"] == first_template["publish_status"]
        assert filtered_template_payload["filters"]["product_status"] == first_template["product_status"]
        assert filtered_template_payload["summary"]["filtered"] is True
        assert filtered_template_payload["summary"]["template_count"] <= (
            template_payload["summary"]["template_count"]
        )

        missing_template_history = client.get("/api/field/customer-project-templates/no-such-template/history")
        assert missing_template_history.status_code == 404
        assert missing_template_history.json()["reason"] == "template_not_found"

        denied_template_release = client.post(
            "/api/field/customer-project-templates/factory-inspection/release",
            json={"operator_id": "unknown.operator", "release": {"publish_status": "published"}},
        )
        assert denied_template_release.status_code == 403

        supervisor_template_publish = client.post(
            "/api/field/customer-project-templates/factory-inspection/release",
            json={"operator_id": "supervisor-1", "release": {"publish_status": "published"}},
        )
        assert supervisor_template_publish.status_code == 403

        direct_product_publish = client.post(
            "/api/field/customer-project-templates/factory-inspection/release",
            json={"operator_id": "product.owner", "release": {"publish_status": "published"}},
        )
        assert direct_product_publish.status_code == 409
        assert direct_product_publish.json()["reason"] == "published_release_requires_approval_request"

        missing_template_release = client.post(
            "/api/field/customer-project-templates/no-such-template/release",
            json={"operator_id": "product.owner", "release": {"publish_status": "pilot"}},
        )
        assert missing_template_release.status_code == 404
        assert missing_template_release.json()["reason"] == "template_not_found"

        release_requests = client.get("/api/field/customer-project-template-release-requests")
        assert release_requests.status_code == 200
        assert "pending_count" in release_requests.json()["summary"]

        release_notes = client.get("/api/field/customer-project-template-release-notes")
        assert release_notes.status_code == 200
        assert "approved_release_count" in release_notes.json()["summary"]
        assert release_notes.json()["customer_claim"] == (
            "Only approved published template packages appear in these release notes."
        )

        release_notes_bundle = client.post(
            "/api/field/customer-project-template-release-notes/export",
            json={
                "operator_id": "dashboard.operator",
                "customer_context": {
                    "customer_name": "Demo Customer",
                    "project_name": "Demo Proposal",
                },
            },
        )
        assert release_notes_bundle.status_code == 200
        assert release_notes_bundle.json()["accepted"] is True
        assert release_notes_bundle.json()["bundle"]["bundle_schema"] == (
            "askme.template_release_notes_bundle.v1"
        )
        assert "proposal_insert" in release_notes_bundle.json()["bundle"]
        assert release_notes_bundle.json()["bundle"]["proposal_insert"]["section_title"] == (
            "Demo Proposal approved reusable capabilities"
        )
        assert "html" in release_notes_bundle.json()["bundle"]
        assert release_notes_bundle.json()["bundle"]["files"]["html_filename"] == (
            "demo-proposal-template-release-notes.html"
        )

        missing_template_request = client.post(
            "/api/field/customer-project-templates/no-such-template/release-requests",
            json={"operator_id": "product.owner", "release": {"publish_status": "published"}},
        )
        assert missing_template_request.status_code == 404
        assert missing_template_request.json()["reason"] == "template_not_found"

        missing_request_review = client.post(
            "/api/field/customer-project-template-release-requests/no-such-request/review",
            json={"operator_id": "product.reviewer", "decision": "approve"},
        )
        assert missing_request_review.status_code == 404
        assert missing_request_review.json()["reason"] == "release_request_not_found"

        registry = client.get("/api/field/customer-project-acceptance-registry")
        assert registry.status_code == 200
        registry_payload = registry.json()
        assert registry_payload["summary"]["reference_count"] >= 4
        assert registry_payload["summary"]["linked_count"] >= 1
        assert registry_payload["references"]
        assert registry_payload["consumers"]
        assert any(
            item["project_id"] == "demo-field-ops" and item["object_id"] == "vehicles"
            for item in registry_payload["consumers"]
        )

        resources = client.get("/api/field/customer-project-resource-catalog")
        assert resources.status_code == 200
        resource_payload = resources.json()
        assert resource_payload["summary"]["resource_count"] >= 10
        assert resource_payload["summary"]["used_resource_count"] >= 4
        assert resource_payload["summary"]["unregistered_resource_count"] == 0
        assert resource_payload["resources"]
        assert resource_payload["consumers"]
        assert any(
            item["resource_type"] == "vision_models"
            and item["resource_id"] == "vehicle-detection"
            for item in resource_payload["resources"]
        )

        detail = client.get("/api/field/customer-projects/demo-field-ops")
        assert detail.status_code == 200
        assert detail.json()["managed_objects"]["object_type_count"] >= 1
        assert detail.json()["managed_objects"]["binding_readiness_summary"]["overall_status"] == "ready"
        assert detail.json()["delivery_workflow"]["steps"]
        assert detail.json()["delivery_workflow"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        vehicle_status = detail.json()["managed_objects"]["objects_by_id"]["vehicles"]["acceptance_status"]
        assert vehicle_status["acceptance_checks"][0]["matched"] == "illegal_parking_camera_ingest"
        vehicle_resources = detail.json()["managed_objects"]["objects_by_id"]["vehicles"]["resource_binding_status"]
        assert vehicle_resources["overall_status"] == "ready"
        assert vehicle_resources["linked_count"] >= 4

        execution = client.get("/api/field/customer-projects/demo-field-ops/execution-bindings")
        assert execution.status_code == 200
        execution_payload = execution.json()
        assert execution_payload["found"] is True
        assert execution_payload["summary"]["object_count"] >= 4
        assert execution_payload["plans_by_object_id"]["vehicles"]["overall_status"] == "ready"
        assert execution_payload["plans_by_object_id"]["vehicles"]["scope_constraints"]["project_ids"] == []
        assert execution_payload["plans_by_object_id"]["vehicles"]["ingest_contract"]["endpoint"] == (
            "/api/field/ingest"
        )
        vehicle_adapter = execution_payload["plans_by_object_id"]["vehicles"]["input_adapters"][0]
        assert vehicle_adapter["adapter_contract"]["bridge"] == "field-ingest-bridge"
        assert vehicle_adapter["adapter_contract"]["device_signature_required"] is True
        assert "ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET" in (
            vehicle_adapter["adapter_contract"]["device_secret_envs"]
        )
        assert "--dry-run" in vehicle_adapter["adapter_contract"]["dry_run_command"]
        assert "--watch" in vehicle_adapter["adapter_contract"]["live_command"]
        assert execution_payload["plans_by_object_id"]["vehicles"]["bridge_contract"][
            "live_post_required_for_customer_signoff"
        ] is True
        assert execution_payload["plans_by_object_id"]["vehicles"]["skill_routes"][0]["resource_id"] == (
            "capability.detect_illegal_parking"
        )
        vehicle_route = execution_payload["plans_by_object_id"]["vehicles"]["skill_routes"][0]
        assert vehicle_route["capability"] == "detect_illegal_parking"
        assert vehicle_route["installed_contract"] is True
        assert vehicle_route["safety_level"] == "dangerous"
        assert vehicle_route["approval_policy"] == "supervisor_required"
        assert vehicle_route["output_contract"] == "field_event"

        rehearsal = client.post(
            "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
            json={"operator_id": "supervisor-1", "mode": "dry_run"},
        )
        assert rehearsal.status_code == 200
        rehearsal_payload = rehearsal.json()
        assert rehearsal_payload["accepted"] is True
        assert rehearsal_payload["status"] == "lab_rehearsed"
        assert rehearsal_payload["production_claim_allowed"] is False
        assert rehearsal_payload["rehearsal"]["mode"] == "dry_run"
        assert "not production go-live evidence" in rehearsal_payload["customer_status"]
        assert rehearsal_payload["plan"]["object_id"] == "vehicles"
        assert rehearsal_payload["normalized"]["managed_object_id"] == "vehicles"
        assert rehearsal_payload["normalized"]["source"] == "camera"
        assert rehearsal_payload["normalized"]["scenario_id"] == "illegal_parking"
        assert rehearsal_payload["normalized"]["project_scope"]["project_id"] == "demo-field-ops"
        assert rehearsal_payload["production_eligible"] is False
        assert rehearsal_payload["evidence_tier"] == "lab_rehearsal"

        dry_run_evidence = client.post(
            "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
            json={
                "operator_id": "supervisor-1",
                "mode": "dry_run",
                "register_onsite_evidence": True,
            },
        )
        assert dry_run_evidence.status_code == 200
        dry_run_registration = dry_run_evidence.json()["onsite_evidence_registration"]
        assert dry_run_registration["registered"] is False
        assert dry_run_registration["reason"] == "dry_run_rehearsal_not_onsite_evidence"
        assert dry_run_registration["production_eligible"] is False

        shadow_needs_confirmation = client.post(
            "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
            json={
                "operator_id": "supervisor-1",
                "mode": "shadow_post",
                "register_onsite_evidence": True,
            },
        )
        assert shadow_needs_confirmation.status_code == 409
        assert shadow_needs_confirmation.json()["reason"] == "shadow_post_requires_explicit_confirmation"
        assert shadow_needs_confirmation.json()["onsite_evidence_registration"]["registered"] is False

        history = client.get("/api/field/customer-projects/demo-field-ops/history")
        assert history.status_code == 200
        assert history.json()["found"] is True
        assert "revisions" in history.json()

        report = client.get("/api/field/customer-projects/demo-field-ops/acceptance-report")
        assert report.status_code == 200
        assert report.json()["overall_status"] in {
            "ready_for_onsite_acceptance",
            "manual_check",
            "blocked",
        }
        assert report.json()["acceptance_summary"]["overall_status"] == "ready"
        assert report.json()["gates"][0]["gate_id"] == "site_profile"
        gate_ids = {gate["gate_id"] for gate in report.json()["gates"]}
        assert "managed_object_execution_bindings" in gate_ids
        assert "field_readiness" in gate_ids
        assert "field_smoke_evidence" in gate_ids
        assert "voice_notification_evidence" in gate_ids
        assert "runtime_audit_trust" in gate_ids
        assert report.json()["field_readiness"]["evidence_reports"]
        assert report.json()["execution_bindings"]["summary"]["overall_status"] == "ready"
        report_launch = report.json()["launch_readiness"]
        assert report_launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
        assert report_launch["overall_status"] in {"ready", "manual_check", "blocked"}
        assert report_launch["launch_stage"] in {
            "production_acceptance_ready",
            "pilot_or_site_trial",
            "demo_or_integration_only",
        }
        assert isinstance(report_launch["production_ready"], bool)
        report_vehicle_contract = next(
            item
            for item in report.json()["execution_bindings"]["object_contracts"]
            if item["object_id"] == "vehicles"
        )
        assert report_vehicle_contract["input_adapters"][0]["bridge"] == "field-ingest-bridge"
        assert report_vehicle_contract["input_adapters"][0]["device_signature_required"] is True
        assert "onsite_acceptance_evidence" in report.json()
        assert report.json()["delivery_workflow"]["steps"]
        assert report.json()["site_acceptance_checklist"]["items"]
        assert report.json()["site_acceptance_checklist"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }

        onsite = client.get("/api/field/customer-projects/demo-field-ops/onsite-evidence")
        assert onsite.status_code == 200
        assert onsite.json()["found"] is True
        assert onsite.json()["readiness_auto_included"] is True
        assert "field_readiness" in onsite.json()
        assert onsite.json()["onsite_acceptance_evidence"]["summary"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }

        closure = client.get("/api/field/customer-projects/demo-field-ops/acceptance-closure")
        assert closure.status_code == 200
        assert closure.json()["found"] is True
        assert closure.json()["overall_status"] in {
            "ready_for_acceptance",
            "ready_for_customer_signoff",
            "accepted_by_customer",
            "manual_check",
            "blocked",
        }
        assert "manual_review" in closure.json()
        assert "customer_signoff" in closure.json()
        assert any(gate["gate_id"] == "customer_signoff" for gate in closure.json()["gates"])
        artifact_verification = closure.json()["artifact_verification"]
        assert "acceptance_dossier" in artifact_verification
        assert "proposal_bundle" in artifact_verification
        assert "audit_export" in artifact_verification

        signoff = client.get("/api/field/customer-projects/demo-field-ops/customer-signoff")
        assert signoff.status_code == 200
        assert signoff.json()["found"] is True
        assert "signoffs" in signoff.json()
        assert signoff.json()["project_scope"]["project_id"] == "demo-field-ops"

        exported = client.get("/api/field/customer-projects/demo-field-ops/export")
        assert exported.status_code == 200
        package = exported.json()["package"]
        assert package["package_type"] == "askme.customer_project"
        assert package["manifest"]["payload_sha256"]
        assert package["manifest"]["managed_object_count"] >= 1
        assert package["manifest"]["acceptance_overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert package["acceptance_summary"]["overall_status"] == (
            package["manifest"]["acceptance_overall_status"]
        )
        assert package["manifest"]["resource_binding_overall_status"] == (
            package["binding_readiness_summary"]["overall_status"]
        )
        assert package["manifest"]["resource_binding_unregistered_resource_count"] == (
            package["binding_readiness_summary"]["unregistered_resource_count"]
        )
        assert package["manifest"]["resource_binding_ready_object_count"] == (
            package["binding_readiness_summary"]["ready_object_count"]
        )
        assert package["manifest"]["resource_binding_manual_check_object_count"] == (
            package["binding_readiness_summary"]["manual_check_object_count"]
        )
        assert package["manifest"]["resource_binding_blocked_object_count"] == (
            package["binding_readiness_summary"]["blocked_object_count"]
        )
        assert package["manifest"]["delivery_resource_count"] == (
            package["resource_catalog_summary"]["resource_count"]
        )
        assert package["deployment_dependencies"]["binding_readiness"]["overall_status"] == (
            package["binding_readiness_summary"]["overall_status"]
        )
        assert package["applicability_scope"]["scenarios"]
        assert package["out_of_scope"]
        assert package["customer_prerequisites"]
        assert package["scenario_acceptance_criteria"]
        assert package["dependency_matrix"]
        assert package["manifest"]["scenario_acceptance_criteria_count"] == len(
            package["scenario_acceptance_criteria"]
        )
        assert package["manifest"]["customer_prerequisite_count"] == len(
            package["customer_prerequisites"]
        )
        assert package["managed_object_action_plan"]["overall_status"] in {
            "deliverable",
            "manual_check_required",
            "blocked",
        }
        assert package["package_delivery_gate"]["delivery_gate_status"] == (
            package["manifest"]["package_delivery_gate_status"]
        )
        assert package["package_delivery_gate"]["import_allowed"] == (
            package["manifest"]["package_delivery_import_allowed"]
        )
        assert package["package_delivery_gate"]["customer_handoff_ready"] == (
            package["manifest"]["package_delivery_customer_handoff_ready"]
        )

        dossier = client.get("/api/field/customer-projects/demo-field-ops/acceptance-dossier")
        assert dossier.status_code == 200
        assert dossier.json()["accepted"] is True
        assert dossier.json()["html_path"].endswith(".html")
        assert dossier.json()["dossier"]["manifest"]["project_id"] == "demo-field-ops"
        assert dossier.json()["dossier"]["manifest"]["payload_sha256"]
        assert "onsite_evidence_status" in dossier.json()["dossier"]["manifest"]
        assert "site_acceptance_checklist_status" in dossier.json()["dossier"]["manifest"]
        dossier_launch = dossier.json()["dossier"]["launch_readiness"]
        assert dossier_launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
        assert dossier.json()["dossier"]["manifest"]["launch_readiness_status"] == (
            dossier_launch["overall_status"]
        )
        assert dossier.json()["dossier"]["manifest"]["launch_stage"] == dossier_launch["launch_stage"]
        assert dossier.json()["dossier"]["manifest"]["production_ready"] is dossier_launch[
            "production_ready"
        ]
        assert dossier.json()["dossier"]["delivery_workflow"]["steps"]
        assert dossier.json()["dossier"]["site_acceptance_checklist"]["items"]
        assert dossier.json()["dossier"]["evidence_inventory"]

        dossier_verification = client.post(
            "/api/field/customer-projects/acceptance-dossier/verify",
            json={
                "operator_id": "supervisor-1",
                "dossier": dossier.json()["dossier"],
            },
        )
        assert dossier_verification.status_code == 200
        assert dossier_verification.json()["accepted"] is True
        assert dossier_verification.json()["verification"]["valid"] is True
        assert dossier_verification.json()["dossier_scope"]["project_id"] == "demo-field-ops"

        tampered_dossier = json.loads(json.dumps(dossier.json()["dossier"]))
        tampered_dossier["customer"]["project_name"] = "tampered customer handoff"
        tampered_verification = client.post(
            "/api/field/customer-projects/acceptance-dossier/verify",
            json={
                "operator_id": "supervisor-1",
                "dossier": tampered_dossier,
            },
        )
        assert tampered_verification.status_code == 422
        assert tampered_verification.json()["accepted"] is False
        assert "manifest.payload_sha256 mismatch" in tampered_verification.json()["verification"]["errors"]

        tampered_launch_dossier = json.loads(json.dumps(dossier.json()["dossier"]))
        tampered_launch_dossier["launch_readiness"]["overall_status"] = "ready"
        tampered_launch_verification = client.post(
            "/api/field/customer-projects/acceptance-dossier/verify",
            json={
                "operator_id": "supervisor-1",
                "dossier": tampered_launch_dossier,
            },
        )
        assert tampered_launch_verification.status_code == 422
        assert tampered_launch_verification.json()["accepted"] is False
        assert "manifest.payload_sha256 mismatch" in tampered_launch_verification.json()[
            "verification"
        ]["errors"]

        proposal = client.get("/api/field/customer-projects/demo-field-ops/proposal-bundle")
        assert proposal.status_code == 200
        assert proposal.json()["accepted"] is True
        assert proposal.json()["proposal"]["proposal_type"] == "askme.customer_project_proposal"
        assert proposal.json()["proposal"]["manifest"]["project_id"] == "demo-field-ops"
        assert proposal.json()["proposal"]["customer_project_package"]["manifest"]["project_id"] == (
            "demo-field-ops"
        )
        assert proposal.json()["proposal"]["acceptance_dossier"]["manifest"]["project_id"] == (
            "demo-field-ops"
        )
        proposal_launch = proposal.json()["proposal"]["launch_readiness"]
        proposal_readable = proposal.json()["proposal"]["customer_readable_delivery"]
        assert proposal_readable["applicability_scope"]["scenarios"]
        assert proposal_readable["customer_prerequisites"]
        assert proposal_readable["scenario_acceptance_criteria"]
        assert proposal_readable["dependency_matrix"]
        assert proposal.json()["proposal"]["manifest"]["proposal_scenario_acceptance_criteria_count"] == len(
            proposal_readable["scenario_acceptance_criteria"]
        )
        assert proposal_launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
        assert proposal.json()["proposal"]["acceptance_dossier"]["launch_readiness"][
            "overall_status"
        ] == proposal_launch["overall_status"]
        assert proposal.json()["proposal"]["manifest"]["launch_readiness_status"] == (
            proposal_launch["overall_status"]
        )
        assert "html" in proposal.json()["proposal"]
        assert "上线准入" in proposal.json()["proposal"]["html"]

        proposal_verification = client.post(
            "/api/field/customer-projects/proposal-bundle/verify",
            json={
                "operator_id": "supervisor-1",
                "proposal": proposal.json()["proposal"],
            },
        )
        assert proposal_verification.status_code == 200
        assert proposal_verification.json()["accepted"] is True
        assert proposal_verification.json()["verification"]["valid"] is True
        assert proposal_verification.json()["proposal_scope"]["project_id"] == "demo-field-ops"

        tampered_launch_proposal = json.loads(json.dumps(proposal.json()["proposal"]))
        tampered_launch_proposal["launch_readiness"]["production_ready"] = True
        tampered_launch_proposal["launch_readiness"]["launch_stage"] = "production_acceptance_ready"
        launch_proposal_verification = client.post(
            "/api/field/customer-projects/proposal-bundle/verify",
            json={
                "operator_id": "supervisor-1",
                "proposal": tampered_launch_proposal,
            },
        )
        assert launch_proposal_verification.status_code == 422
        assert launch_proposal_verification.json()["accepted"] is False
        assert "manifest.payload_sha256 mismatch" in launch_proposal_verification.json()[
            "verification"
        ]["errors"]

        verification = client.post(
            "/api/field/customer-projects/package/verify",
            json={
                "operator_id": "supervisor-1",
                "package": package,
            },
        )
        assert verification.status_code == 200
        assert verification.json()["accepted"] is True
        assert verification.json()["verification"]["valid"] is True
        assert verification.json()["package_scope"]["project_id"] == "demo-field-ops"
        assert verification.json()["verification"]["delivery_gate_status"] == (
            package["package_delivery_gate"]["delivery_gate_status"]
        )
        assert verification.json()["verification"]["import_allowed"] == (
            package["package_delivery_gate"]["import_allowed"]
        )

        diff_preview = client.post(
            "/api/field/customer-projects/package/diff",
            json={
                "operator_id": "supervisor-1",
                "package": package,
            },
        )
        assert diff_preview.status_code == 200
        assert diff_preview.json()["accepted"] is True
        assert diff_preview.json()["verification"]["valid"] is True
        assert diff_preview.json()["diff"]["change_type"] in {"noop", "replace"}
        assert "field_changes" in diff_preview.json()["diff"]
        assert diff_preview.json()["diff"]["incoming_binding_readiness_summary"]["overall_status"] == (
            package["binding_readiness_summary"]["overall_status"]
        )
        assert diff_preview.json()["diff"]["incoming_binding_readiness_summary"]["ready_object_count"] == (
            package["binding_readiness_summary"]["ready_object_count"]
        )
        assert (
            diff_preview.json()["diff"]["incoming_binding_readiness_summary"][
                "manual_check_object_count"
            ]
            == package["binding_readiness_summary"]["manual_check_object_count"]
        )
        assert diff_preview.json()["diff"]["incoming_binding_readiness_summary"]["blocked_object_count"] == (
            package["binding_readiness_summary"]["blocked_object_count"]
        )
        assert (
            diff_preview.json()["diff"]["incoming_binding_readiness_summary"][
                "unregistered_resource_count"
            ]
            == package["binding_readiness_summary"]["unregistered_resource_count"]
        )
        assert diff_preview.json()["diff"]["current_binding_readiness_summary"]["overall_status"] == (
            package["binding_readiness_summary"]["overall_status"]
        )
        assert diff_preview.json()["diff"]["incoming_delivery_gate"]["delivery_gate_status"] == (
            package["package_delivery_gate"]["delivery_gate_status"]
        )

        preview = client.post(
            "/api/field/customer-projects/import",
            json={
                "operator_id": "supervisor-1",
                "dry_run": True,
                "package": package,
            },
        )
        assert preview.status_code == 200
        assert preview.json()["dry_run"] is True
        assert preview.json()["verification"]["valid"] is True
        assert preview.json()["diff"]["change_type"] in {"noop", "replace"}
        assert preview.json()["diff"]["incoming_acceptance_summary"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert preview.json()["diff"]["incoming_binding_readiness_summary"]["overall_status"] == (
            package["binding_readiness_summary"]["overall_status"]
        )
        assert preview.json()["diff"]["incoming_delivery_gate"]["delivery_gate_status"] == (
            package["package_delivery_gate"]["delivery_gate_status"]
        )
        assert preview.json()["import_gate_result"] in {
            "accepted",
            "accepted_with_manual_check",
            "rejected",
        }

        windows_payload = json.dumps(
            {
                "operator_id": "supervisor-1",
                "dry_run": True,
                "package": package,
                "client_note": "Windows 客户端兼容性检查",
            },
            ensure_ascii=False,
        ).encode("gb18030")
        windows_preview = client.post(
            "/api/field/customer-projects/import",
            data=windows_payload,
            headers={"Content-Type": "application/json"},
        )
        assert windows_preview.status_code == 200
        assert windows_preview.json()["verification"]["valid"] is True

    def test_http_healthz_endpoint_matches_health_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "askme"
        assert data["voice_pipeline_status"]["pipeline_ok"] is True

    def test_http_health_endpoint_returns_degraded_snapshot_without_5xx(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "degraded"
        assert data["degraded_reasons"] == ["voice_pipeline", "ota_bridge"]

    def test_http_healthz_endpoint_returns_degraded_snapshot_without_5xx(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        assert data["status"] == "degraded"
        assert data["voice_pipeline_status"]["pipeline_ok"] is False

    def test_http_health_endpoint_reports_provider_exception(self):
        def broken_provider():
            raise RuntimeError("provider failed")

        client = TestClient(create_health_app(broken_provider))

        response = client.get("/health")

        assert response.status_code == 500
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == {
            "status": "error",
            "error": "provider failed",
        }

    def test_metrics_endpoint_returns_prometheus_text(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "askme_uptime_seconds 12.5" in response.text
        assert "askme_conversations_total 7" in response.text
        assert "askme_last_llm_latency_ms 245" in response.text
        assert 'askme_model_info{model_name="claude-opus-4-6"} 1' in response.text
        assert 'askme_active_skill_info{skill="dock_charge"} 1' in response.text
        assert "askme_voice_pipeline_ok 1" in response.text
        assert "askme_ota_bridge_registered 1" in response.text

    def test_metrics_prometheus_endpoint_matches_metrics_contract(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "text/plain" in response.headers["content-type"]
        assert "askme_up 1" in response.text
        assert "askme_health_status 1" in response.text

    def test_metrics_prometheus_endpoint_marks_degraded_snapshot_unhealthy(self):
        client = TestClient(create_health_app(lambda: _degraded_runtime_snapshot()))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert "askme_up 1" in response.text
        assert "askme_health_status 0" in response.text
        assert "askme_voice_pipeline_ok 0" in response.text
        assert "askme_ota_bridge_registered 0" in response.text

    def test_metrics_prometheus_endpoint_marks_provider_exception_unhealthy(self):
        def broken_provider():
            raise RuntimeError("provider failed")

        client = TestClient(create_health_app(broken_provider))

        response = client.get("/metrics/prometheus")

        assert response.status_code == 500
        assert response.headers["Cache-Control"] == "no-store"
        assert "askme_up 1" in response.text
        assert "askme_health_status 0" in response.text

    def test_chat_endpoint_forwards_speak_request_to_handler(self):
        seen: dict[str, object] = {}

        async def chat_handler(text: str, *, speak: bool = False):
            seen["text"] = text
            seen["speak"] = speak
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"text": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert seen == {"text": "hello", "speak": True}
        assert response.json() == {
            "reply": "reply:hello",
            "spoken": True,
            "text": "hello",
            "evidence": [],
        }

    def test_chat_endpoint_reports_timeout_from_config(self, monkeypatch):
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {"conversation": {"chat_timeout_s": 0.001}},
        )

        async def chat_handler(text: str, *, speak: bool = False):
            import asyncio

            await asyncio.sleep(0.05)
            return {"reply": text}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post("/api/chat", json={"text": "slow"})

        assert response.status_code == 504
        assert response.json()["error"] == "chat timed out"

    def test_conversation_diagnostics_endpoint_reports_chat_state(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"text": "hello"},
            headers={"X-Request-Id": "trace-route-1"},
        )
        diagnostics = client.get("/api/conversation/diagnostics")

        assert response.status_code == 200
        assert response.headers["X-Askme-Trace-Id"] == "trace-route-1"
        assert diagnostics.status_code == 200
        payload = diagnostics.json()["chat"]
        assert payload["configured"] is True
        assert payload["total_turns"] == 1
        assert payload["in_flight"] == 0
        assert payload["last_turn"]["status"] == "ok"
        assert payload["last_turn"]["trace_id"] == "trace-route-1"

    def test_metrics_endpoint_exports_chat_runtime_metrics(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        chat = client.post("/api/chat", json={"text": "hello"})
        metrics = client.get("/metrics/prometheus")

        assert chat.status_code == 200
        assert metrics.status_code == 200
        assert "askme_chat_turns_total 1" in metrics.text
        assert "askme_chat_in_flight 0" in metrics.text
        assert "askme_chat_failures_total 0" in metrics.text

    def test_chat_endpoint_accepts_message_alias(self):
        seen: dict[str, object] = {}

        async def chat_handler(text: str, *, speak: bool = False):
            seen["text"] = text
            seen["speak"] = speak
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"message": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert seen == {"text": "hello", "speak": True}
        assert response.json() == {
            "reply": "reply:hello",
            "spoken": True,
            "text": "hello",
            "evidence": [],
        }

    def test_chat_endpoint_returns_voice_transcript_metadata_for_voice_turn(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {"reply": f"reply:{text}", "spoken": speak}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={
                "text": "纭",
                "voice": True,
                "transcript_id": "voice-confirm-1",
                "asr_confidence": 0.87,
            },
        )

        voice_turn = response.json()["voice_turn"]
        assert response.status_code == 200
        assert voice_turn["transcript_id"] == "voice-confirm-1"
        assert voice_turn["recognized_text"] == "纭"
        assert voice_turn["confidence"] == 0.87
        assert voice_turn["safety_bypass_allowed"] is False

    def test_chat_endpoint_keeps_text_only_handler_compatible(self):
        async def chat_handler(text: str):
            return f"reply:{text}"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post(
            "/api/chat",
            json={"text": "hello", "speak": True},
        )

        assert response.status_code == 200
        assert response.json() == {
            "reply": "reply:hello",
            "text": "hello",
            "spoken": False,
            "evidence": [],
        }

    def test_chat_endpoint_preserves_handler_evidence_payload(self):
        async def chat_handler(text: str, *, speak: bool = False):
            return {
                "reply": f"reply:{text}",
                "evidence": [{"text": "site fact", "source": "site.md"}],
                "rag": {"backend": "vector", "used_in_answer": True},
            }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
            )
        )

        response = client.post("/api/chat", json={"text": "hello"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["source"] == "site.md"
        assert payload["rag"]["backend"] == "vector"

    def test_chat_endpoint_attaches_memory_evidence_for_plain_text_handler(self):
        class MemoryHandler:
            def health(self):
                return {
                    "enabled": True,
                    "backend": "vector",
                    "available": True,
                    "last_backend": "vector",
                    "last_retrieve_ms": 12,
                    "last_retrieved_items": 1,
                    "last_evidence": [{
                        "text": "A 区入口在东门",
                        "source": "site.md",
                        "record_id": "rec-a",
                    }],
                    "last_dropped_evidence": [{
                        "text": "expired memory fact",
                        "drop_reason": "expired",
                        "record_id": "rec-old",
                    }],
                    "last_answer_policy": {
                        "state": "grounded",
                        "action": "answer_with_evidence",
                    },
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return f"reply:{text}"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "where is gate A?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["record_id"] == "rec-a"
        assert payload["rag"]["answer_policy"]["state"] == "grounded"
        assert payload["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"

    def test_chat_endpoint_does_not_overwrite_handler_evidence_with_memory_context(self):
        class MemoryHandler:
            def health(self):
                return {
                    "last_evidence": [{"text": "memory fact", "source": "memory.md"}],
                    "last_answer_policy": {"state": "grounded"},
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return {
                "reply": f"reply:{text}",
                "evidence": [{"text": "handler fact", "source": "handler.md"}],
                "rag": {"backend": "handler"},
            }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "hello"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["evidence"][0]["source"] == "handler.md"
        assert payload["rag"]["backend"] == "handler"

    def test_chat_endpoint_forces_refusal_when_rag_policy_blocks_plain_text_reply(self):
        class MemoryHandler:
            def health(self):
                return {
                    "enabled": True,
                    "backend": "vector",
                    "available": True,
                    "last_backend": "vector",
                    "last_evidence": [],
                    "last_dropped_evidence": [{
                        "text": "old route",
                        "drop_reason": "expired",
                        "record_id": "route-old",
                    }],
                    "last_answer_policy": {
                        "state": "stale",
                        "action": "refuse_and_request_update",
                        "reason": "expired",
                        "required_operator_action": "refresh_knowledge",
                    },
                }

        async def chat_handler(text: str, *, speak: bool = False):
            return "go straight to the old gate"

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                chat_handler=chat_handler,
                memory_handler=MemoryHandler(),
            )
        )

        response = client.post("/api/chat", json={"text": "how do I reach the gate?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["reply"] != "go straight to the old gate"
        assert payload["rag_blocked"] is True
        assert payload["rag"]["answer_blocked"] is True
        assert payload["rag"]["forced_reply"] is True
        assert payload["rag"]["block_reason"] == "expired"
        assert payload["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"

    def test_live_endpoint_uses_conversation_provider(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                conversation_provider=lambda: [{"role": "user", "content": "hello"}],
            )
        )

        response = client.get("/api/live")

        assert response.status_code == 200
        assert response.json() == {
            "messages": [{"role": "user", "content": "hello"}],
            "count": 1,
        }

    def test_conversations_endpoint_reads_configured_history_file(self, tmp_path, monkeypatch):
        history = [{"role": "assistant", "content": "ready"}]
        history_path = tmp_path / "conversation-history.json"
        history_path.write_text(json.dumps(history), encoding="utf-8")
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {"conversation": {"history_file": str(history_path)}},
        )

        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/conversations")

        assert response.status_code == 200
        assert response.json() == {"messages": history, "count": 1}

    def test_memory_search_endpoint_dispatches_handler(self):
        class Handler:
            async def search_payload(self, payload):
                return {
                    "query": payload["query"],
                    "results": [{"text": "site fact", "source": "site.md"}],
                    "rag": {"backend": "vector"},
                    "warnings": [],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post("/api/memory/search", json={"query": "site"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "site"
        assert payload["results"][0]["source"] == "site.md"
        assert payload["rag"]["backend"] == "vector"

    def test_knowledge_preview_endpoint_dispatches_handler(self):
        class Handler:
            async def preview_payload(self, payload):
                return {
                    "source": payload["filename"],
                    "parsed": 1,
                    "records": [{"text": "fact", "category": "faq"}],
                    "errors": [],
                    "dry_run": True,
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/preview",
            json={"filename": "faq.md", "content": "- fact"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "faq.md"
        assert payload["parsed"] == 1
        assert payload["records"][0]["text"] == "fact"

    def test_knowledge_import_endpoint_dispatches_handler(self):
        class Handler:
            async def import_payload(self, payload):
                return {
                    "source": payload["filename"],
                    "parsed": 1,
                    "imported": 1,
                    "skipped": 0,
                    "errors": [],
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/import",
            json={"filename": "faq.md", "content": "- fact"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["imported"] == 1
        assert payload["rag"]["backend"] == "vector"

    def test_knowledge_list_endpoint_dispatches_handler(self):
        class Handler:
            async def list_knowledge_payload(self, payload):
                return {
                    "backend": "vector",
                    "total": 1,
                    "records": [{"record_id": "know_1", "text": "fact"}],
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post("/api/knowledge/list", json={"limit": 50})

        assert response.status_code == 200
        payload = response.json()
        assert payload["records"][0]["record_id"] == "know_1"
        assert payload["total"] == 1

    def test_knowledge_update_endpoint_dispatches_handler(self):
        class Handler:
            async def update_knowledge_payload(self, payload):
                return {
                    "updated": True,
                    "record_id": payload["record_id"],
                    "patch": {"approval_status": "deleted"},
                    "rag": {"backend": "vector"},
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/update",
            json={"record_id": "know_1", "action": "delete"},
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["updated"] is True
        assert payload["record_id"] == "know_1"

    def test_knowledge_update_blocks_operator_without_approval_role(self):
        class Handler:
            async def update_knowledge_payload(self, payload):
                return {"updated": True}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                memory_handler=Handler(),
            )
        )

        response = client.post(
            "/api/knowledge/update",
            json={"record_id": "know_1", "action": "delete"},
            headers={"X-Askme-Operator-Id": "dashboard.operator"},
        )

        assert response.status_code == 403
        payload = response.json()
        assert payload["reason"] == "operator_missing_permission"
        assert payload["operator_auth"]["permission"] == "knowledge:delete"

    def test_governance_operator_directory_exposes_demo_boundary(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/governance/operator-directory")

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "demo_config"
        assert payload["identity_provider"] == "local_config"
        assert payload["production_binding_required"] is True
        assert payload["session_operator_header"] == "x-askme-operator-id"
        assert payload["permissions"]["operator"]
        assert "field:project:read" in payload["permissions"]["operator"]
        dashboard_operator = next(
            operator for operator in payload["operators"] if operator["operator_id"] == "dashboard.operator"
        )
        assert dashboard_operator["project_scope"]["project_ids"] == ["demo-field-ops"]
        assert payload["sso"]["configured"] is False
        assert payload["sso"]["trusted_identity_headers_enabled"] is False
        assert any(
            operator["operator_id"] == "dashboard.operator"
            for operator in payload["operators"]
        )
        assert payload["readiness"]["status"] == "demo_or_trial_only"
        assert payload["readiness"]["production_ready"] is False
        assert payload["identity_gateway_readiness"]["gate_type"] == (
            "askme.governance.identity_gateway_readiness.v1"
        )
        assert payload["identity_gateway_readiness"]["status"] == "blocked"
        assert payload["identity_gateway_readiness"]["production_ready"] is False
        assert payload["identity_gateway_readiness"]["release_claim"].startswith("只能承诺演示或试点能力")
        assert any(item["role"] == "supervisor" for item in payload["roles"])
        assert "resource:governance:write" in payload["permissions"]["supervisor"]
        assert "template:release:approve" in payload["permissions"]["product_owner"]
        assert "resource:governance:approve" in payload["permissions"]["product_owner"]
        assert any(item["role"] == "product_owner" for item in payload["roles"])
        assert any(row["scope"] == "knowledge:approve" for row in payload["authorization_matrix"])
        assert any(row["scope"] == "template:release:approve" for row in payload["authorization_matrix"])
        assert any(row["scope"] == "resource:governance:approve" for row in payload["authorization_matrix"])

    def test_governance_current_operator_resolves_permissions(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get(
            "/api/governance/current-operator",
            params={"operator_id": "supervisor-1"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["operator"]["operator_id"] == "supervisor-1"
        assert payload["operator"]["known"] is True
        assert "knowledge:approve" in payload["permissions"]
        assert payload["readiness"]["production_ready"] is False
        assert payload["identity_gateway_readiness"]["identity_mode"] == "demo_operator_directory"

    def test_governance_unknown_operator_is_limited(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        current = client.get(
            "/api/governance/current-operator",
            params={"operator_id": "ghost.operator"},
        )
        authorization = client.post(
            "/api/governance/authorize",
            json={"operator_id": "ghost.operator", "permission": "field:event:create"},
        )

        assert current.status_code == 200
        payload = current.json()
        assert payload["operator"]["known"] is False
        assert payload["permissions"] == []
        assert authorization.status_code == 403
        assert authorization.json()["reason"] == "operator_missing_permission"

    def test_dashboard_contains_cognition_planning_controls(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/dashboard")

        assert response.status_code == 200
        assert 'id="dashboard-nav"' in response.text
        assert 'id="app-page"' in response.text
        for page in (
            "/dashboard/conversation",
            "/dashboard/projects",
            "/dashboard/field",
            "/dashboard/space",
            "/dashboard/knowledge",
            "/dashboard/capabilities",
            "/dashboard/voice",
            "/dashboard/delivery",
        ):
            page_response = client.get(page)
            assert page_response.status_code == 200
            assert 'id="app-page"' in page_response.text

        js_response = client.get("/dashboard/app.js")
        css_response = client.get("/dashboard/app.css")

        assert js_response.status_code == 200
        assert css_response.status_code == 200
        assert "/api/governance/current-operator" in js_response.text
        assert "/dashboard/conversation" in js_response.text
        assert "/dashboard/projects" in js_response.text
        assert "/dashboard/field" in js_response.text
        assert "/dashboard/space" in js_response.text
        assert "/dashboard/knowledge" in js_response.text
        assert "/dashboard/capabilities" in js_response.text
        assert "/dashboard/voice" in js_response.text
        assert "/dashboard/delivery" in js_response.text
        assert "renderOverview" in js_response.text
        assert "机器人现场任务运营台" in js_response.text
        assert "客户现在能验收什么" in js_response.text
        assert "多现场交付" in js_response.text
        assert "服务在线" in js_response.text
        assert "renderConversation" in js_response.text
        assert "renderProjects" in js_response.text
        assert "renderField" in js_response.text
        assert "renderKnowledge" in js_response.text
        assert "renderCapabilities" in js_response.text
        assert "renderVoice" in js_response.text
        assert "renderDelivery" in js_response.text
        assert "/api/chat" in js_response.text
        assert "/api/knowledge/preview" in js_response.text
        assert "/api/knowledge/import" in js_response.text
        assert "/api/knowledge/list" in js_response.text
        assert "/api/memory/search" in js_response.text
        assert "/api/space/service-point-trigger" in js_response.text
        assert "/api/space/guide" in js_response.text
        assert "/api/space/manage" in js_response.text
        assert "/api/space/rollback" in js_response.text
        assert "/api/space/history" in js_response.text
        assert "/api/space/proposals" in js_response.text
        assert "/api/space/proposals/review" in js_response.text
        assert "/api/runtime/handoff" in js_response.text
        assert "/api/blueprints" in js_response.text
        assert "renderSpace" in js_response.text
        assert "renderSpaceChanges" in js_response.text
        assert "renderSpaceProposals" in js_response.text
        assert "renderSpaceGuideResult" in js_response.text
        assert "renderSpaceRuntimeSubmission" in js_response.text
        assert "renderBlueprintReadiness" in js_response.text
        assert "runtime_handoff_ready" in js_response.text
        assert "runtime_handoff_plan" in js_response.text
        assert "space-submit-runtime" in js_response.text
        assert "space-confirmed" in js_response.text
        assert "space-save-service-point" in js_response.text
        assert "space-save-route" in js_response.text
        assert "space-rollback" in js_response.text
        assert "space-propose-point" in js_response.text
        assert "space-form" in css_response.text
        assert "space-handoff" in css_response.text
        assert "/api/capability-center" in js_response.text
        assert "/api/skill-audit" in js_response.text
        assert "/api/audit/events" in js_response.text
        assert "/api/audit/reviews" in js_response.text
        assert "/api/audit/export" in js_response.text
        assert "/api/audit/exports" in js_response.text
        assert "product_summary" in js_response.text
        assert "review_queue" in js_response.text
        assert 'id="audit-since"' in js_response.text
        assert "auditSeverityClass" in js_response.text
        assert 'id="audit-review-panel"' in js_response.text
        assert "renderAuditReviewPanel" in js_response.text
        assert "renderAuditReviewIntegrity" in js_response.text
        assert "renderAuditSourceHealth" in js_response.text
        assert "renderAuditExportHistory" in js_response.text
        assert "renderAuditDeliveryDossier" in js_response.text
        assert "delivery_dossier" in js_response.text
        assert "客户交付审计材料" in js_response.text
        assert "禁止声明" in js_response.text
        assert "无人值守生产上线声明" in js_response.text
        assert "source_health" in js_response.text
        assert "invalid_record_count" in js_response.text
        assert "renderAuditExportResult" in js_response.text
        assert "askme.audit_last_export" in js_response.text
        assert "evidence_summary" in js_response.text
        assert "renderAuditEvidenceRefs" in js_response.text
        assert "auditEvidenceHref" in js_response.text
        assert "isImageEvidence" in js_response.text
        assert "renderAuditReviewHistory" in js_response.text
        assert "wireAuditReviewPanelControls" in js_response.text
        assert "audit-source-health" in css_response.text
        assert "audit-source-grid" in css_response.text
        assert "audit-evidence-thumb" in css_response.text
        assert "audit-export-result" in css_response.text
        assert "audit-export-history" in css_response.text
        assert "audit-delivery-dossier" in css_response.text
        assert "/api/agent-profiles" in js_response.text
        assert "/api/skills/generated" in js_response.text
        assert "/api/skill-packages" in js_response.text
        assert "/api/skill-growth/backlog" in js_response.text
        assert 'id="agent-profile-name"' in js_response.text
        assert "data-agent-preview" in js_response.text
        assert "保存 Agent Profile" in js_response.text
        assert "renderAgentProfilePreview" in js_response.text
        assert "scenario_blueprints" in js_response.text
        assert "场景能力蓝图" in js_response.text
        assert "renderScenarioBlueprint" in js_response.text
        assert "/api/capability-packages" in js_response.text
        assert "renderCapabilityPackageItem" in js_response.text
        assert "renderScenarioPackageItem" in js_response.text
        assert "客户可启用能力包" in js_response.text
        assert "enablement_decision" in js_response.text
        assert "release_summary" in js_response.text
        assert "生产声明" in js_response.text
        assert "生产上线声明必须有现场验收" in js_response.text
        assert "交付声明：" in js_response.text
        assert "package_readiness" in js_response.text
        assert "启用准入" in js_response.text
        assert "重新检查" in js_response.text
        assert "customer_next_step" in js_response.text
        assert "下一步：" in js_response.text
        assert "/draft" in js_response.text
        assert 'id="skill-package-id"' in js_response.text
        assert "data-skill-package" in js_response.text
        assert "data-growth-action" in js_response.text
        assert "生成草稿" in js_response.text
        assert "/preview" in js_response.text
        assert "预检" in js_response.text
        assert "知识管理" in js_response.text
        assert "导入并发布" in js_response.text
        assert "重建索引" in js_response.text
        assert "机器人现场任务运营台" in js_response.text
        assert "/api/governance/operator-directory" in js_response.text
        assert "/api/governance/identity-readiness" in js_response.text
        assert "data-identity-gateway-readiness" in js_response.text
        assert "企业身份准入" in js_response.text
        assert "只能演示或试点" in js_response.text
        assert "identity-header-grid" in css_response.text
        assert "knowledge-operations" in js_response.text
        assert "/api/field/scenarios" in js_response.text
        assert "/api/field/site-profiles" in js_response.text
        assert "/api/field/customer-projects" in js_response.text
        assert "/api/field/customer-project-workbench" in js_response.text
        assert "/api/field/customer-projects/managed-object-directory" in js_response.text
        assert "/api/field/customer-project-templates" in js_response.text
        assert "/api/field/customer-project-template-release-requests" in js_response.text
        assert "/api/field/customer-project-template-release-notes" in js_response.text
        assert "/api/field/customer-projects/import" in js_response.text
        assert "/api/field/customer-projects/package/verify" in js_response.text
        assert "/api/field/customer-projects/package/diff" in js_response.text
        assert "data-project-package-verify" in js_response.text
        assert "data-project-package-diff" in js_response.text
        assert "renderProjectPackageScopeEvidence" in js_response.text
        assert "当前账号可操作范围" in js_response.text
        assert "交付包归属" in js_response.text
        assert "导入演练" in js_response.text
        assert "project-scope-evidence" in css_response.text
        assert "/api/field/customer-projects/proposal-bundle/verify" in js_response.text
        assert "/api/field/customer-projects/acceptance-dossier/verify" in js_response.text
        assert "onsite-evidence" in js_response.text
        assert "include_readiness_auto" in js_response.text
        assert "acceptance-closure" in js_response.text
        assert "acceptance-review" in js_response.text
        assert "/api/field/customer-project-acceptance-registry" in js_response.text
        assert "/api/field/customer-project-resource-catalog" in js_response.text
        assert "/api/field/solution-delivery-readiness" in js_response.text
        assert "/api/field/product-launch-readiness" in js_response.text
        assert "/api/field/delivery-resource-registry" in js_response.text
        assert "/api/field/delivery-resource-registry/history" in js_response.text
        assert "/api/field/delivery-resource-registry/rollback" in js_response.text
        assert "/api/field/delivery-resource-governance-requests" in js_response.text
        assert "fieldDeliveryResourceRegistry" in js_response.text
        assert "fieldDeliveryResourceRegistryHistory" in js_response.text
        assert "fieldDeliveryResourceRegistryRollback" in js_response.text
        assert "fieldDeliveryResourceGovernanceRequests" in js_response.text
        assert "产品验收准入" in js_response.text
        assert "data-delivery-acceptance-gate" in js_response.text
        assert "data-project-product-acceptance-gate" in js_response.text
        assert "data-project-filter-apply" in js_response.text
        assert "customerProjectFilterQuery" in js_response.text
        assert "data-template-filter-apply" in js_response.text
        assert "customerProjectTemplateFilterQuery" in js_response.text
        assert "客户交付总门禁" in js_response.text
        assert "renderSolutionDeliveryReadiness" in js_response.text
        assert "客户上线准入总览" in js_response.text
        assert "renderProductLaunchReadiness" in js_response.text
        assert "data-product-launch-readiness" in js_response.text
        assert "product-launch-readiness" in css_response.text
        assert "product-launch-gates" in css_response.text
        assert "验收引用登记" in js_response.text
        assert "renderAcceptanceRegistrySummary" in js_response.text
        assert "renderProjectResourceCatalogSummary" in js_response.text
        assert "resource_binding_status" in js_response.text
        assert "renderProjectBindingReadiness" in js_response.text
        assert "resource_binding_overall_status" in js_response.text
        assert "action_label" in js_response.text
        assert "owner_label" in js_response.text
        assert "customer_next_step" in js_response.text
        assert "交付资源登记" in js_response.text
        assert "共享交付资源登记表" in js_response.text
        assert "data-resource-register" in js_response.text
        assert "data-resource-history" in js_response.text
        assert "data-resource-disable" in js_response.text
        assert "data-resource-rollback" in js_response.text
        assert "data-resource-governance-requests" in js_response.text
        assert "data-resource-governance-review" in js_response.text
        assert "overdue_only=true" in js_response.text
        assert "review_sla" in js_response.text
        assert "due_at" in js_response.text
        assert "复核时限" in js_response.text
        assert "只看逾期" in js_response.text
        assert "升级逾期" in js_response.text
        assert "escalate-overdue" in js_response.text
        assert "renderDeliveryResourceGovernanceEscalation" in js_response.text
        assert "renderDeliveryResourceGovernanceEscalationResult" in js_response.text
        assert "renderDeliveryResourceGovernanceSla" in js_response.text
        assert "资源治理影响" in js_response.text
        assert "preview?.impact" in js_response.text
        assert "affected_consumers" in js_response.text
        assert "affected_customer_project_count" in js_response.text
        assert "affected_template_count" in js_response.text
        assert 'id="resource-project-id"' in js_response.text
        assert 'id="resource-type"' in js_response.text
        assert "registerDeliveryResourceFromForm" in js_response.text
        assert "renderDeliveryResourceGovernancePanel" in js_response.text
        assert "loadDeliveryResourceHistory" in js_response.text
        assert "disableDeliveryResource" in js_response.text
        assert "requestDeliveryResourceDisable" in js_response.text
        assert "reviewDeliveryResourceGovernanceRequest" in js_response.text
        assert "rollbackDeliveryResourceRegistry" in js_response.text
        assert "delivery-resource-form" in css_response.text
        assert "delivery-resource-governance" in css_response.text
        assert "resource-impact" in css_response.text
        assert "resource-sla" in css_response.text
        assert "resource-escalation" in css_response.text
        assert "资源绑定行动计划" in js_response.text
        assert "renderDeliveryResourceActionPlan" in js_response.text
        assert "data-delivery-resource-action-plan" in js_response.text
        assert "delivery-resource-action-plan" in css_response.text
        assert "incoming_binding_readiness_summary" in js_response.text
        assert "unregistered_resources" in js_response.text
        assert "导入预检" in js_response.text
        assert "renderProjectPackageDeliveryGate" in js_response.text
        assert "data-project-package-delivery-gate" in js_response.text
        assert "package_delivery_gate" in js_response.text
        assert "incoming_delivery_gate" in js_response.text
        assert "project-package-delivery-gate" in css_response.text
        assert "project-page-nav" in css_response.text
        assert "polishProjectWorkspaceCopy" in js_response.text
        assert "data-product-workbench" in js_response.text
        assert "renderProjectGoldenPathWorkbench" in js_response.text
        assert "data-project-golden-path" in js_response.text
        assert "方案商交付路径" in js_response.text
        assert "从行业模板到客户交付包，按验收节点推进" in js_response.text
        assert "project-golden-path" in css_response.text
        assert "project-workspace-explainer" in css_response.text
        assert "客户项目不是配置文件，是交付产品" in js_response.text
        assert "交付准入" in js_response.text
        assert "行业模板市场" in js_response.text
        assert "project-section-projects" in js_response.text
        assert "project-section-templates" in js_response.text
        assert "project-section-objects" in js_response.text
        assert "project-section-package" in js_response.text
        assert "project-section-acceptance" in js_response.text
        assert "project-section-resources" in js_response.text
        assert "project-section-events" in js_response.text
        assert "project-section-sites" in js_response.text
        assert "project-package-workbench" in css_response.text
        assert 'id="project-proposal-json"' in js_response.text
        assert 'id="project-dossier-json"' in js_response.text
        assert 'document.getElementById("project-proposal-json")' in js_response.text
        assert 'document.getElementById("project-dossier-json")' in js_response.text
        assert "客户项目交付包已生成" in js_response.text
        assert "客户方案包校验" in js_response.text
        assert "验收材料校验" in js_response.text
        assert "验收报告" in js_response.text
        assert "客户现场验收清单" in js_response.text
        assert "现场验收证据" in js_response.text
        assert "验收闭环" in js_response.text
        assert "客户签收" in js_response.text
        assert "customer-signoff" in js_response.text
        assert "fieldCustomerProjectCustomerSignoffSuffix" in js_response.text
        assert "fieldCustomerProjectExecutionBindingsSuffix" in js_response.text
        assert "data-project-execution-bindings" in js_response.text
        assert "执行接入计划" in js_response.text
        assert "renderCustomerProjectExecutionBindings" in js_response.text
        assert "renderExecutionScopeConstraints" in js_response.text
        assert "范围约束" in js_response.text
        assert "loadSelectedCustomerProjectExecutionBindings" in js_response.text
        assert "approval_policy" in js_response.text
        assert "output_contract" in js_response.text
        assert "hardware_boundary" in js_response.text
        assert "renderExecutionAdapterContracts" in js_response.text
        assert "adapter_contract" in js_response.text
        assert "field-ingest-bridge" in js_response.text
        assert "dry_run_command" in js_response.text
        assert "live_command" in js_response.text
        assert "data-object-rehearsal" in js_response.text
        assert "rehearseCustomerProjectObject" in js_response.text
        assert "renderObjectExecutionRehearsalResult" in js_response.text
        assert "实验室演示证据，不能作为生产上线验收依据。" in js_response.text
        assert "register_onsite_evidence" in js_response.text
        assert "renderEvidenceBoundaryTags" in js_response.text
        assert "production_eligible" in js_response.text
        assert "\\u9a8c\\u6536\\u5019\\u9009\\u8bc1\\u636e" in js_response.text
        assert "data-project-customer-signoff-load" in js_response.text
        assert "data-project-customer-signoff-submit" in js_response.text
        assert "project-customer-signatory-name" in js_response.text
        assert "project-customer-signoff-credential-ref" in js_response.text
        assert "project-customer-signoff-credential-sha256" in js_response.text
        assert "签收凭证 SHA-256" in js_response.text
        assert "customer_signoff" in js_response.text
        assert "ready_for_customer_signoff" in js_response.text
        assert "accepted_by_customer" in js_response.text
        assert "renderCustomerProjectCustomerSignoff" in js_response.text
        assert "loadSelectedCustomerProjectCustomerSignoff" in js_response.text
        assert "registerSelectedCustomerProjectCustomerSignoff" in js_response.text
        assert "客户验收材料" in js_response.text
        assert "系统自动采信" in js_response.text
        assert "managed-objects" in js_response.text
        assert "acceptance_checks" in js_response.text
        assert "field_changes" in js_response.text
        assert "data-managed-object-summary" in js_response.text
        assert "data-managed-object-export" in js_response.text
        assert "fieldCustomerProjectManagedObjectDirectory" in js_response.text
        assert "当前可见" in js_response.text
        assert "managedObjectDirectorySummary" in js_response.text
        assert "managedObjectExportRows" in js_response.text
        assert "managedObjectDirectoryKey" in js_response.text
        assert "renderManagedObjectActionPlan" in js_response.text
        assert "data-managed-object-action-plan" in js_response.text
        assert "action_plan" in js_response.text
        assert "managed-object-actions" in css_response.text
        assert "导出对象目录" in js_response.text
        assert "导出可交付对象清单" in js_response.text
        assert "managed-object-checks" in css_response.text
        assert "field_readiness" in js_response.text
        assert "evidence_reports" in js_response.text
        assert "onsite_acceptance_evidence" in js_response.text
        assert "data-acceptance-gates" in js_response.text
        assert "data-project-acceptance-report" in js_response.text
        assert "data-project-acceptance-dossier" in js_response.text
        assert "data-project-lifecycle-onsite-evidence" in js_response.text
        assert "data-project-lifecycle-onsite-load" in js_response.text
        assert "data-project-lifecycle-closure" in js_response.text
        assert "data-project-lifecycle-review" in js_response.text
        assert 'id="project-acceptance-evidence-refs"' in js_response.text
        assert 'id="project-acceptance-evidence-picker"' in js_response.text
        assert "data-acceptance-evidence-add" in js_response.text
        assert "setAcceptanceEvidenceOptions" in js_response.text
        assert "addSelectedAcceptanceEvidenceRef" in js_response.text
        assert "onsiteReceiptEvidenceRef" in js_response.text
        assert "acceptance-evidence-picker" in css_response.text
        assert (
            "evidence_refs: commaList(document.getElementById(\"project-acceptance-evidence-refs\")"
            in js_response.text
        )
        assert "renderCustomerProjectOnsiteEvidence" in js_response.text
        assert "renderSiteAcceptanceChecklist" in js_response.text
        assert "site_acceptance_checklist" in js_response.text
        assert "客户现场验收清单" in js_response.text
        assert "onsiteReceiptSourceLabel" in js_response.text
        assert "field_readiness_auto_backfill" in js_response.text
        assert "系统自动采信" in js_response.text
        assert "renderCustomerProjectAcceptanceClosure" in js_response.text
        assert "registerSelectedCustomerProjectOnsiteEvidence" in js_response.text
        assert "registerSelectedCustomerProjectAcceptanceReview" in js_response.text
        assert "data-project-proposal" in js_response.text
        assert "data-project-proposal-verify" in js_response.text
        assert "data-project-dossier-verify" in js_response.text
        assert "data-project-lifecycle-proposal" in js_response.text
        assert "renderCustomerProjectAcceptanceDossier" in js_response.text
        assert "renderCustomerProjectProposalBundle" in js_response.text
        assert "verifyCustomerProjectProposalBundle" in js_response.text
        assert "verifyCustomerProjectAcceptanceDossier" in js_response.text
        assert "renderProjectDossierVerifyResult" in js_response.text
        assert "renderProjectProposalVerifyResult" in js_response.text
        assert "acceptance-dossier" in js_response.text
        assert "proposal-bundle" in js_response.text
        assert "exportCustomerProjectProposalBundle" in js_response.text
        assert "downloadCustomerProjectProposalBundle" in js_response.text
        assert "html_path" in js_response.text
        assert "Printable HTML" in js_response.text
        assert "renderCustomerProjectAcceptanceReport" in js_response.text
        assert "客户项目与对象目录" in js_response.text
        assert "项目交付包导入" in js_response.text
        assert "事件归属检查" in js_response.text
        assert "data-project-lifecycle-export" in js_response.text
        assert "data-project-lifecycle-history" in js_response.text
        assert "data-project-lifecycle-rollback-dry" in js_response.text
        assert "data-project-lifecycle-rollback" in js_response.text
        assert "project-rollback-revision" in js_response.text
        assert "loadSelectedCustomerProjectHistory" in js_response.text
        assert "rollbackSelectedCustomerProject" in js_response.text
        assert "renderProjectRevisionHistory" in js_response.text
        assert "renderProjectRollbackResult" in js_response.text
        assert "data-project-lifecycle-archive" in js_response.text
        assert "data-object-delete" in js_response.text
        assert "renderManagedObjectWriteResult" in js_response.text
        assert "data-managed-object-write-result" in js_response.text
        assert "对象变更后的实施步骤" in js_response.text
        assert "对象已保存" in js_response.text
        assert "对象已下线" in js_response.text
        assert "renderProjectExportResult" in js_response.text
        assert "renderProjectScopeLabel" in js_response.text
        assert "renderProjectCollisionCandidates" in js_response.text
        assert "交付包冲突项" in js_response.text
        assert 'id="project-tenant-id"' in js_response.text
        assert 'id="project-delivery-namespace"' in js_response.text
        assert "tenant_id: document.getElementById(\"project-tenant-id\")" in js_response.text
        assert (
            "delivery_namespace: document.getElementById(\"project-delivery-namespace\")"
            in js_response.text
        )
        assert "项目基础信息" in js_response.text
        assert 'id="project-edit-id"' in js_response.text
        assert "data-project-edit-load" in js_response.text
        assert "data-project-edit-save" in js_response.text
        assert "loadProjectProfileForEdit" in js_response.text
        assert "saveProjectProfileMetadata" in js_response.text
        assert "currentProjectEditProfile" in js_response.text
        assert "对象目录" in js_response.text
        assert "data-managed-object-directory" in js_response.text
        assert "renderManagedObjectDirectory" in js_response.text
        assert "renderManagedObjectCard" in js_response.text
        assert "对象变更记录" in js_response.text
        assert "data-object-change-log" in js_response.text
        assert "renderObjectChangeLog" in js_response.text
        assert "object-change-log" in css_response.text
        assert "交付流程" in js_response.text
        assert "data-project-delivery-workflow" in js_response.text
        assert "renderProjectDeliveryWorkflow" in js_response.text
        assert "project-delivery-workflow" in css_response.text
        assert "project-delivery-step" in css_response.text
        assert "data-object-load" in js_response.text
        assert "loadManagedObjectIntoEditor" in js_response.text
        assert "currentCustomerProjectItems" in js_response.text
        assert "managed-object-directory" in css_response.text
        assert "managed-object-card" in css_response.text
        assert "managed-object-bindings" in css_response.text
        assert "object-delete-impact" in js_response.text
        assert "object-offline-impact" in css_response.text
        assert "renderManagedObjectOfflineImpact" in js_response.text
        assert "updateManagedObjectDeleteImpact" in js_response.text
        assert "移除现场对象前必须填写下线原因" in js_response.text
        assert "data-managed-object-editor" in js_response.text
        assert "基础对象" in js_response.text
        assert "识别范围" in js_response.text
        assert "能力配置" in js_response.text
        assert "绑定交付资源" in js_response.text
        assert "加入绑定" in js_response.text
        assert "data-object-resource-picker" in js_response.text
        assert "data-object-resource-add" in js_response.text
        assert 'id="object-resource-picker"' in js_response.text
        assert "addSelectedObjectResourceBinding" in js_response.text
        assert "objectBindingInputId" in js_response.text
        assert "object-resource-picker" in css_response.text
        assert 'id="object-labels"' in js_response.text
        assert 'id="object-responder-group"' in js_response.text
        assert 'id="object-evidence-required"' in js_response.text
        assert 'id="object-vision-models"' in js_response.text
        assert 'id="object-sensor-protocols"' in js_response.text
        assert "setObjectEditInput(\"object-vision-models\"" in js_response.text
        assert "setObjectEditInput(\"object-sensor-protocols\"" in js_response.text
        assert 'id="object-project-ids"' in js_response.text
        assert 'id="object-site-ids"' in js_response.text
        assert "客户范围保护" in js_response.text
        assert "object-project-ids" in js_response.text
        assert "tenant_ids: commaList(document.getElementById(\"object-tenant-ids\")" in js_response.text
        assert "project_ids: commaList(document.getElementById(\"object-project-ids\")" in js_response.text
        assert (
            "vision_models: commaList(document.getElementById(\"object-vision-models\")"
            in js_response.text
        )
        assert (
            "sensor_protocols: commaList(document.getElementById(\"object-sensor-protocols\")"
            in js_response.text
        )
        assert "验收证据" in js_response.text
        assert "managed-object-editor" in css_response.text
        assert "object-editor-section" in css_response.text
        assert "incoming_delivery_scope" in js_response.text
        assert "collision_candidates" in js_response.text
        assert "delivery_namespace_count" in js_response.text
        assert "project-collision-warning" in css_response.text
        assert "deleteJson" in js_response.text
        assert "多现场交付" in js_response.text
        assert "多现场交付" in js_response.text
        assert "客户项目目录" in js_response.text
        assert "行业模板市场" in js_response.text
        assert "从模板创建项目" in js_response.text
        assert "data-project-template-create-readiness" in js_response.text
        assert "renderProjectTemplateCreateReadiness" in js_response.text
        assert "renderCustomerProjectCreateResult" in js_response.text
        assert "implementation_handoff" in js_response.text
        assert "renderProjectImplementationHandoff" in js_response.text
        assert "data-project-implementation-handoff" in js_response.text
        assert "data-project-detail-handoff" in js_response.text
        assert "项目详情加载后的实施步骤" in js_response.text
        assert "导入演练后的实施步骤" in js_response.text
        assert "保存后的实施步骤" in js_response.text
        assert "项目信息已保存" in js_response.text
        assert "data-created-project-load" in js_response.text
        assert "data-created-object-guide" in js_response.text
        assert "客户项目已创建" in js_response.text
        assert "补齐对象绑定" in js_response.text
        assert "创建准入" in js_response.text
        assert "客户配合项" in js_response.text
        assert "暂不承诺" in js_response.text
        assert "project-create-readiness" in css_response.text
        assert "project-create-result-card" in css_response.text
        assert "project-create-next-steps" in css_response.text
        assert "template-market-card" in css_response.text
        assert "template-delivery-checklist" in css_response.text
        assert "template_package" in js_response.text
        assert "renderTemplatePackageReadiness" in js_response.text
        assert "renderTemplateApplicabilityScope" in js_response.text
        assert "renderTemplateCustomerPrerequisites" in js_response.text
        assert "renderTemplateScenarioAcceptanceCriteria" in js_response.text
        assert "renderTemplateOutOfScope" in js_response.text
        assert "模板交付包" in js_response.text
        assert "renderTemplateDeliveryChecklist" in js_response.text
        assert "renderTemplateObjectPreview" in js_response.text
        assert "data-template-select" in js_response.text
        assert "data-template-release" in js_response.text
        assert "data-template-history" in js_response.text
        assert "data-template-release-request" in js_response.text
        assert "data-template-release-requests" in js_response.text
        assert "updateTemplateRelease" in js_response.text
        assert "createTemplateReleaseRequest" in js_response.text
        assert "reviewTemplateReleaseRequest" in js_response.text
        assert "project-section-template-governance" in js_response.text
        assert "renderTemplateReleaseGovernance" in js_response.text
        assert "renderTemplateReleaseGovernanceRequest" in js_response.text
        assert "template-governance-result" in js_response.text
        assert "templateReleaseGovernanceResultEl" in js_response.text
        assert "wireTemplateReleaseReviewControls" in js_response.text
        assert "template-governance-board" in css_response.text
        assert "renderTemplateReleaseRequests" in js_response.text
        assert "renderTemplateReleaseNotes" in js_response.text
        assert "exportTemplateReleaseNotesBundle" in js_response.text
        assert "downloadTemplateReleaseNotesBundle" in js_response.text
        assert "downloadTextFile" in js_response.text
        assert "URL.createObjectURL" in js_response.text
        assert "proposal_insert" in js_response.text
        assert "data-template-release-notes-export" in js_response.text
        assert "/api/field/customer-project-template-release-notes/export" in js_response.text
        assert "renderTemplateReleaseHistory" in js_response.text
        assert "请输入发布治理原因" in js_response.text
        assert "请输入模板发布申请原因" in js_response.text
        assert "模板发布说明" in js_response.text
        assert "selectTemplateForCreate" in js_response.text
        assert "对象快速维护" in js_response.text
        assert "/api/field/events" in js_response.text
        assert "/api/field/notification-preflight?status_as_200=true" in js_response.text
        assert "renderFieldEventDetail" in js_response.text
        assert "incident_workflow" in js_response.text
        assert "action_audit" in js_response.text
        assert "runtime_delivery" in js_response.text
        assert "resend-notification" in js_response.text
        assert "request-close" in js_response.text
        assert "/api/field/readiness" in js_response.text
        assert "/api/field/devices" in js_response.text
        assert "暂无现场事件" in js_response.text
        assert "保安群" in js_response.text
        assert "/api/voice/profiles" in js_response.text
        assert "/api/voice/profile" in js_response.text
        assert "asr_final_ms" in js_response.text
        assert "llm_ttft_ms" in js_response.text
        assert "tts_first_audio_ms" in js_response.text
        assert "playback_start_ms" in js_response.text
        assert "speak: true" in js_response.text
        assert "play_audio: true" in js_response.text

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
        assert data["profile"]["name"] == "voice"
        assert data["components"]["skills"]["capabilities"]["openapi_generated"] is True
        assert data["skills"]["contract_count"] == 3

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
        assert data["package_readiness"]["customer_statuses"]["blocked"].startswith("Must not")

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
            "claim_policy": (
                "Production launch claims require separate onsite acceptance and human "
                "takeover approval."
            ),
        }
        assert data["capability_packages"][0]["package_id"] == "capability.answer_wayfinding"
        assert data["capability_packages"][0]["customer_next_step"] == "Run site validation."
        assert data["capability_packages"][0]["enablement_decision"] == {}
        assert data["scenario_packages"][0]["package_id"] == "scenario.wayfinding_help_point"
        assert data["scenario_packages"][0]["required_capability_packages"] == [
            "capability.answer_wayfinding"
        ]
        assert data["scenario_packages"][0]["customer_missing_dependencies"] == []
        assert data["scenario_packages"][0]["engineering_missing_dependencies"] == []
        assert data["scenario_packages"][0]["enablement_decision"]["decision"] == (
            "site_validation_allowed"
        )
        assert data["inventory"]["capability_packages"] == ["capability.answer_wayfinding"]
        assert data["inventory"]["services"] == ["voice"]
        assert data["policy"]["blocked_packages_must_not_be_enabled"] is True

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

    def test_blueprints_endpoint_returns_delivery_readiness(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/blueprints")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["blueprint_count"] >= 6
        edge = next(item for item in data["items"] if item["name"] == "edge_robot")
        assert edge["inspection"]["valid"] is True
        assert edge["readiness"]["production_ready"] is False
        assert edge["readiness"]["gates"][0]["gate_id"] == "runtime_composition"

    def test_skill_audit_endpoint_returns_records_shape(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skill-audit?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 3
        assert isinstance(data["records"], list)

    def test_skill_growth_backlog_endpoint_returns_candidates_shape(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skill-growth/backlog?min_occurrences=1&limit=3")

        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        assert data["policy"]["human_product_owner_required"] is True
        assert data["policy"]["auto_create_or_enable_skills"] is False

    def test_skill_growth_backlog_update_requires_rbac_permission(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post(
            "/api/skill-growth/backlog/grow_missing",
            json={"action": "promote"},
        )
        assert denied.status_code == 403

    def test_skill_growth_backlog_draft_creates_pending_generated_skill(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import askme.skills.growth_backlog as growth_backlog_module
        import askme.skills.skill_manager as skill_manager_module
        from askme.skills.audit import SkillAuditLog
        from askme.skills.skill_manager import SkillManager

        audit = SkillAuditLog(tmp_path / "skill-audit.jsonl")
        audit.append(skill_name="unknown", status="failed", user_text="检查喷泉灯", reason="no_skill")
        audit.append(skill_name="unknown", status="blocked", user_text="检查喷泉灯", reason="not_found")
        monkeypatch.setattr(growth_backlog_module, "SkillAuditLog", lambda: audit)
        monkeypatch.setattr(
            growth_backlog_module,
            "default_skill_growth_state_path",
            lambda: tmp_path / "growth.json",
        )
        monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(
            skill_manager_module,
            "_SETTINGS_FILE",
            tmp_path / "skills_settings.json",
        )
        monkeypatch.setattr(
            skill_manager_module,
            "SkillAuditLog",
            lambda: SkillAuditLog(tmp_path / "skill-audit.jsonl"),
        )

        client = TestClient(create_health_app(lambda: _runtime_snapshot()))
        candidate_id = client.get(
            "/api/skill-growth/backlog?min_occurrences=1&limit=3"
        ).json()["candidates"][0]["candidate_id"]

        denied = client.post(f"/api/skill-growth/backlog/{candidate_id}/draft", json={})
        assert denied.status_code == 403

        authorized = client.post(
            f"/api/skill-growth/backlog/{candidate_id}/draft",
            json={"operator_id": "admin-1"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert authorized.status_code == 200
        payload = authorized.json()
        assert payload["ok"] is True
        assert payload["draft"]["enabled"] is False
        assert payload["draft"]["status"] == "pending_approval"
        skill_name = payload["draft"]["skill_name"]
        assert (tmp_path / "skills" / skill_name / "SKILL.md").exists()

        manager = SkillManager(project_dir=tmp_path)
        manager.load()
        skill = manager.get(skill_name)
        assert skill is not None
        assert skill.source == "generated"
        assert skill.enabled is False

    def test_agent_profiles_endpoint_returns_product_roles(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/agent-profiles")

        assert response.status_code == 200
        data = response.json()
        names = {profile["name"] for profile in data["profiles"]}
        assert "field_operator" in names
        assert "skill_growth_manager" in names

    def test_agent_profile_upsert_and_preview_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post("/api/agent-profiles", json={})
        assert denied.status_code == 403

        response = client.post(
            "/api/agent-profiles",
            json={
                "operator_id": "admin-1",
                "name": "Parking PM",
                "display_name": "Parking PM",
                "description": "Plans customer-facing illegal parking detection delivery.",
                "instructions": "Only produce parking detection delivery plans with acceptance criteria.",
                "tools": ["read_file", "robot_api", "temporal_query"],
                "spawnable_profiles": ["safety_reviewer"],
                "skills": ["detect_illegal_parking"],
            },
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["profile"]["name"] == "parking_pm"
        assert (tmp_path / ".askme" / "agents" / "parking_pm.md").exists()

        preview = client.get("/api/agent-profiles/parking_pm/preview")
        assert preview.status_code == 200
        assert preview.json()["profile"]["preloaded_skills"] == ["detect_illegal_parking"]

    def test_agent_profile_upsert_rejects_unknown_tool_without_client_allowlist(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(
            "/api/agent-profiles",
            json={
                "operator_id": "admin-1",
                "name": "Unsafe Agent",
                "description": "This profile tries to expand its own tool allowlist.",
                "instructions": "Use a fake tool to bypass governance.",
                "tools": ["not_a_real_tool"],
                "known_tools": ["not_a_real_tool"],
            },
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert data["error"] == "unknown tools requested"
        assert data["unknown_tools"] == ["not_a_real_tool"]
        assert not (tmp_path / ".askme" / "agents" / "unsafe_agent.md").exists()

    def test_generated_skills_endpoint_returns_review_queue(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated")

        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert data["policy"]["approval_required"] is True
        assert data["policy"]["auto_enable_generated_skills"] is False

    def test_skill_packages_endpoint_returns_package_policy(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skill-packages")

        assert response.status_code == 200
        data = response.json()
        assert "packages" in data
        assert data["policy"]["customer_scoped_enablement"] is True

    def test_skill_package_upsert_requires_rbac_permission(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post(
            "/api/skill-packages",
            json={"package_id": "fanmu-phase-1", "display_name": "Fanmu"},
        )
        assert denied.status_code == 403

        authorized = client.post(
            "/api/skill-packages",
            json={"package_id": "fanmu-phase-1", "display_name": "Fanmu"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )
        assert authorized.status_code == 200
        assert authorized.json()["package"]["package_id"] == "fanmu-phase-1"

    def test_generated_skill_validation_endpoint_for_missing_skill(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated/missing-skill/validation")

        assert response.status_code == 404
        assert response.json()["ok"] is False

    def test_generated_skill_preview_endpoint_for_missing_skill(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated/missing-skill/preview")

        assert response.status_code == 404
        assert response.json()["ok"] is False

    def test_generated_skill_review_requires_rbac_permission(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post(
            "/api/skills/generated/missing-skill/review",
            json={"action": "approve"},
        )
        assert denied.status_code == 403
        assert denied.json()["reason"] == "operator_missing_permission"

        authorized = client.post(
            "/api/skills/generated/missing-skill/review",
            json={"action": "approve"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )
        assert authorized.status_code == 400
        assert authorized.json()["error"] == "generated skill not found"

    def test_skill_package_update_requires_rbac_permission(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post(
            "/api/skill-packages/default-demo/skills/missing-skill",
            json={"action": "assign"},
        )
        assert denied.status_code == 403

        authorized = client.post(
            "/api/skill-packages/default-demo/skills/missing-skill",
            json={"action": "assign"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )
        assert authorized.status_code == 400
        assert authorized.json()["error"] == "generated skill not found"

    def test_skill_package_release_history_and_rollback_endpoints(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import askme.skills.skill_manager as skill_manager_module
        from askme.skills.audit import SkillAuditLog

        monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(
            skill_manager_module,
            "_SETTINGS_FILE",
            tmp_path / "skills_settings.json",
        )
        monkeypatch.setattr(
            skill_manager_module,
            "SkillAuditLog",
            lambda: SkillAuditLog(tmp_path / "skill-audit.jsonl"),
        )
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        denied = client.post(
            "/api/skill-packages/default-demo/release",
            json={"release_channel": "pilot", "rollout_percent": 25},
        )
        assert denied.status_code == 403

        first = client.post(
            "/api/skill-packages/default-demo/release",
            json={"release_channel": "pilot", "rollout_percent": 25},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )
        second = client.post(
            "/api/skill-packages/default-demo/release",
            json={"release_channel": "prod", "rollout_percent": 100},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["package"]["release_version"] == 2

        history = client.get("/api/skill-packages/default-demo/history")
        assert history.status_code == 200
        assert history.json()["count"] == 2

        rollback = client.post(
            "/api/skill-packages/default-demo/rollback",
            json={"target_version": 1, "note": "rollback test"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert rollback.status_code == 200
        package = rollback.json()["package"]
        assert package["release_version"] == 3
        assert package["rollback_of_version"] == 1
        assert package["release_channel"] == "pilot"
        assert package["rollout_percent"] == 25

    def test_control_api_key_protects_non_probe_routes(self):
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                capabilities_provider=lambda: {"profile": {"name": "voice"}},
                control_api_key="secret",
            )
        )

        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

        unauth = client.get("/api/capabilities")
        assert unauth.status_code == 401
        assert unauth.json()["error"] == "control API authentication required"

        wrong = client.get(
            "/api/capabilities",
            headers={"Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401

        bearer = client.get(
            "/api/capabilities",
            headers={"Authorization": "Bearer secret"},
        )
        assert bearer.status_code == 200
        assert bearer.json()["profile"]["name"] == "voice"

        api_key = client.get(
            "/api/capabilities",
            headers={"X-Askme-Api-Key": "secret"},
        )
        assert api_key.status_code == 200

    def test_health_server_defaults_to_loopback_and_requires_remote_auth(self):
        server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
        assert server.host == "127.0.0.1"

        with pytest.raises(ValueError, match="binds outside loopback"):
            AskmeHealthServer(
                {"host": "0.0.0.0"},
                health_provider=lambda: _runtime_snapshot(),
            )

        remote = AskmeHealthServer(
            {"host": "0.0.0.0", "control_api_key": "secret"},
            health_provider=lambda: _runtime_snapshot(),
        )
        assert remote.host == "0.0.0.0"

    def test_cognition_endpoints_delegate_to_handler(self):
        class DummyCognitionHandler:
            def __init__(self):
                self.refresh_seen = None

            async def context_payload(self, *, refresh_perception: bool = False):
                self.refresh_seen = refresh_perception
                return {"world_state": {"fact_count": 1}, "working_memory": {"item_count": 0}}

            async def plan_from_payload(self, payload):
                return {"planned": True, "plan": {"goal": payload["text"]}}

        handler = DummyCognitionHandler()
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                cognition_handler=handler,
            )
        )

        context = client.get("/api/cognition/context?refresh_perception=true")
        assert context.status_code == 200
        assert context.json()["world_state"]["fact_count"] == 1
        assert handler.refresh_seen is True

        plan = client.post("/api/cognition/plan", json={"text": "inspect area-a"})
        assert plan.status_code == 200
        assert plan.json()["planned"] is True
        assert plan.json()["plan"]["goal"] == "inspect area-a"

    def test_runtime_endpoints_delegate_to_handler(self):
        class DummyRuntimeHandler:
            def context_payload(self):
                return {"profile": "sim", "active_run": {"run_id": "run-1", "current_state": "queued"}}

            def profiles_payload(self):
                return {"current_profile": "sim", "profiles": [{"name": "fake"}, {"name": "shadow"}, {"name": "sim"}]}

            def list_payload(self):
                return {"runs": [{"run_id": "run-1"}], "count": 1}

            def submit_plan_payload(self, plan):
                return {
                    "accepted": True,
                    "handoff": {"task_type": plan["mission"]["mission"]["mission_type"]},
                    "run": {"run_id": "run-submitted", "current_state": "completed"},
                }

            def events_payload(self, *, after=None, limit=20):
                return {
                    "profile": "sim",
                    "hardware_dispatch": False,
                    "cursor": 123.0,
                    "events": [
                        {
                            "event_id": "evt-1",
                            "run_id": "run-1",
                            "event_type": "task_queued",
                            "state": "queued",
                            "message": "queued",
                            "created_at": 123.0,
                        }
                    ],
                    "event_count": 1,
                    "active_run": {"run_id": "run-1", "current_state": "queued"},
                }

            def get_payload(self, run_id):
                return {"run": {"run_id": run_id, "current_state": "queued"}}

            def report_payload(self, run_id):
                return {"report": {"run_id": run_id, "status": "queued"}}

            def pause_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "paused"}}

            def resume_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}

            def cancel_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "cancelled"}}

            def advance_payload(self, run_id):
                return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}

            def voice_turn_payload(self, text, **kwargs):
                return {
                    "handled": True,
                    "reply": "TaskRun paused.",
                    "runtime": {"run": {"run_id": "run-1", "current_state": "paused"}},
                    "voice_turn": {
                        "recognized_text": text,
                        "runtime_control_intent": "pause",
                        "safety_bypass_allowed": False,
                        "transcript_id": kwargs.get("transcript_id", ""),
                    },
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                runtime_handler=DummyRuntimeHandler(),
            )
        )

        context = client.get("/api/runtime/context")
        assert context.status_code == 200
        assert context.json()["active_run"]["run_id"] == "run-1"
        assert context.json()["profile"] == "sim"

        profiles = client.get("/api/runtime/profiles")
        assert profiles.status_code == 200
        assert profiles.json()["current_profile"] == "sim"

        runs = client.get("/api/runtime/runs")
        assert runs.status_code == 200
        assert runs.json()["count"] == 1

        submitted = client.post(
            "/api/runtime/handoff",
            json={
                "operator_id": "dashboard.operator",
                "runtime_handoff_plan": {
                    "plan_id": "space-escort-test",
                    "planning_session_id": "space-session-test",
                    "intent": "visitor_escort",
                    "handoff_ready": True,
                    "operator_id": "dashboard.operator",
                    "mission": {"mission": {"mission_type": "visitor_escort"}},
                },
            },
        )
        assert submitted.status_code == 200
        assert submitted.json()["run"]["run_id"] == "run-submitted"
        assert submitted.json()["handoff"]["task_type"] == "visitor_escort"

        events = client.get("/api/runtime/events?once=1")
        assert events.status_code == 200
        assert "text/event-stream" in events.headers["content-type"]
        assert "event: runtime.events" in events.text
        assert '"event_type":"task_queued"' in events.text

        run = client.get("/api/runtime/runs/run-1")
        assert run.status_code == 200
        assert run.json()["run"]["current_state"] == "queued"

        report = client.get("/api/runtime/runs/run-1/report")
        assert report.status_code == 200
        assert report.json()["report"]["status"] == "queued"

        paused = client.post("/api/runtime/runs/run-1/pause")
        assert paused.status_code == 200
        assert paused.json()["run"]["current_state"] == "paused"

        resumed = client.post("/api/runtime/runs/run-1/resume")
        assert resumed.status_code == 200
        assert resumed.json()["run"]["current_state"] == "executing"

        cancelled_forbidden = client.post("/api/runtime/runs/run-1/cancel")
        assert cancelled_forbidden.status_code == 403

        cancelled = client.post(
            "/api/runtime/runs/run-1/cancel",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["run"]["current_state"] == "cancelled"

        advanced = client.post(
            "/api/runtime/runs/run-1/advance",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert advanced.status_code == 200
        assert advanced.json()["run"]["current_state"] == "executing"

        voice = client.post(
            "/api/runtime/voice-turn",
            json={"text": "pause current task", "transcript_id": "voice-1", "confidence": 0.9},
        )
        assert voice.status_code == 200
        assert voice.json()["runtime"]["run"]["current_state"] == "paused"
        assert voice.json()["voice_turn"]["recognized_text"] == "pause current task"
        assert voice.json()["voice_turn"]["safety_bypass_allowed"] is False

    def test_runtime_voice_turn_endpoint_reports_timeout_from_config(self, monkeypatch):
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {"conversation": {"runtime_voice_turn_timeout_s": 0.001}},
        )

        class SlowRuntimeHandler:
            async def voice_turn_payload(self, text, **kwargs):
                import asyncio

                await asyncio.sleep(0.05)
                return {"handled": True, "reply": text}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                runtime_handler=SlowRuntimeHandler(),
            )
        )

        response = client.post("/api/runtime/voice-turn", json={"text": "pause current task"})

        assert response.status_code == 504
        assert response.json()["error"] == "runtime voice-turn timed out"

    def test_runtime_control_endpoint_forwards_operator_context(self):
        class DummyRuntimeHandler:
            def __init__(self):
                self.seen = {}

            def pause_payload(
                self,
                run_id,
                *,
                operator_id="askme.operator",
                reason="",
                risk_acknowledgement=False,
            ):
                self.seen = {
                    "run_id": run_id,
                    "operator_id": operator_id,
                    "reason": reason,
                    "risk_acknowledgement": risk_acknowledgement,
                }
                return {
                    "handled": True,
                    "run": {"run_id": run_id, "current_state": "paused"},
                    "operator": self.seen,
                }

        runtime = DummyRuntimeHandler()
        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                runtime_handler=runtime,
            )
        )

        response = client.post(
            "/api/runtime/runs/run-7/pause",
            json={
                "operator_id": "guard-1",
                "reason": "visitor entered path",
                "risk_acknowledgement": True,
            },
        )

        assert response.status_code == 200
        assert runtime.seen == {
            "run_id": "run-7",
            "operator_id": "guard-1",
            "reason": "visitor entered path",
            "risk_acknowledgement": True,
        }

    def test_mission_endpoints_delegate_to_handler(self):
        class DummyMissionHandler:
            def __init__(self):
                self.mission = {
                    "mission_id": "mission-1",
                    "goal": "inspect area-a",
                    "status": "draft",
                }

            def draft_from_payload(self, payload):
                self.mission["goal"] = payload["text"]
                return {"mission": self.mission, "drafted": True}

            def submit_from_payload(self, payload):
                self.mission["status"] = "dry_run" if payload.get("dry_run", True) else "submitted"
                return {
                    "mission": self.mission,
                    "submission": {"submitted": False, "dry_run": True},
                }

            def list_payload(self):
                return {"missions": [self.mission], "count": 1}

            def get_payload(self, mission_id):
                if mission_id != self.mission["mission_id"]:
                    return {"error": "mission not found", "mission_id": mission_id}
                return {"mission": self.mission}

            def report_payload(self, mission_id):
                if mission_id != self.mission["mission_id"]:
                    return {"error": "mission not found", "mission_id": mission_id}
                return {"report": {"mission_id": mission_id, "status": self.mission["status"]}}

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                mission_handler=DummyMissionHandler(),
            )
        )

        draft = client.post("/api/missions/draft", json={"text": "inspect area-a"})
        assert draft.status_code == 200
        assert draft.json()["drafted"] is True

        submit = client.post("/api/missions", json={"text": "inspect area-a", "dry_run": True})
        assert submit.status_code == 200
        assert submit.json()["submission"]["dry_run"] is True

        mission_list = client.get("/api/missions")
        assert mission_list.status_code == 200
        assert mission_list.json()["count"] == 1

        mission_get = client.get("/api/missions/mission-1")
        assert mission_get.status_code == 200
        assert mission_get.json()["mission"]["mission_id"] == "mission-1"

        report = client.get("/api/missions/mission-1/report")
        assert report.status_code == 200
        assert report.json()["report"]["status"] == "dry_run"

        missing = client.get("/api/missions/missing")
        assert missing.status_code == 404

    def test_mission_endpoint_returns_unconfigured_status(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post("/api/missions/draft", json={"text": "inspect area-a"})

        assert response.status_code == 503
        assert response.json()["error"] == "mission handler not configured"


class TestVoiceBridgeSnapshot:
    """Cover the voice_bridge=None / voice_bridge=<dict> branches of build_health_snapshot."""

    _BASE_KWARGS = dict(
        app_name="askme",
        app_version="4.0.0",
        model_name="claude-opus-4-6",
        metrics_snapshot={"uptime_seconds": 1.0},
        active_skills=[],
        voice_status={"pipeline_ok": True},
    )

    def test_voice_bridge_none_key_absent(self):
        snapshot = build_health_snapshot(**self._BASE_KWARGS)
        assert "voice_bridge" not in snapshot

    def test_voice_bridge_present_key_included(self):
        bridge_payload = {"status": "connected"}
        snapshot = build_health_snapshot(**self._BASE_KWARGS, voice_bridge=bridge_payload)
        assert "voice_bridge" in snapshot
        assert snapshot["voice_bridge"] == bridge_payload
