"""HTTP tests for field and customer-project routes."""

import json
import shutil
import time
from pathlib import Path

from askme.pipeline.field_site_profile import (
    DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    create_customer_project_from_template,
    export_customer_project_package,
    list_delivery_resource_registry,
    upsert_delivery_resource,
)
from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.api.schemas.customer_project_artifacts import (
    CustomerProjectAcceptanceDossierExportResponse,
    CustomerProjectAcceptanceDossierVerifyResponse,
    CustomerProjectPackageDiffResponse,
    CustomerProjectPackageExportResponse,
    CustomerProjectPackageImportResponse,
    CustomerProjectPackageVerifyResponse,
    CustomerProjectProposalBundleExportResponse,
    CustomerProjectProposalBundleVerifyResponse,
)
from askme.api.schemas.customer_projects import (
    CustomerProjectAcceptanceClosureResponse,
    CustomerProjectAcceptanceRegistryResponse,
    CustomerProjectAcceptanceReportResponse,
    CustomerProjectCatalogResponse,
    CustomerProjectDetailResponse,
    CustomerProjectExecutionBindingsResponse,
    CustomerProjectExecutionRehearsalResponse,
    CustomerProjectHistoryResponse,
    CustomerProjectManagedObjectMutationResponse,
    CustomerProjectMutationResponse,
    CustomerProjectOnsiteEvidenceResponse,
    CustomerProjectResourceCatalogResponse,
    CustomerProjectSignoffResponse,
    CustomerProjectTemplateCatalogResponse,
    CustomerProjectTemplateHistoryResponse,
    CustomerProjectTemplateReleaseNotesExportResponse,
    CustomerProjectTemplateReleaseNotesResponse,
    CustomerProjectTemplateReleaseRequestMutationResponse,
    CustomerProjectTemplateReleaseRequestsResponse,
    CustomerProjectTemplateReleaseUpdateResponse,
    CustomerProjectWorkbenchResponse,
    ManagedObjectDirectoryResponse,
    SiteProfileCatalogResponse,
)
from askme.api.schemas.delivery_readiness import (
    ProductLaunchReadinessResponse,
    SolutionDeliveryReadinessResponse,
)
from askme.api.schemas.delivery_resources import (
    DeliveryResourceGovernanceEscalationResponse,
    DeliveryResourceGovernanceMutationResponse,
    DeliveryResourceGovernanceRequestsResponse,
)
from askme.health_server import create_health_app
from tests.support.field_route_app import (
    field_route_test_app as _field_route_test_app,
)
from tests.support.field_route_app import (
    scoped_project_authorize as _scoped_project_authorize,
)
from tests.support.health_snapshots import runtime_snapshot as _runtime_snapshot


def test_field_customer_project_write_preflight_methods_are_explicit(tmp_path: Path) -> None:
    client = TestClient(_field_route_test_app(tmp_path / "site-profiles"))

    expected_methods = {
        "/api/field/readiness": "GET, OPTIONS",
        "/api/field/customer-projects": "GET, POST, OPTIONS",
        "/api/field/customer-projects/from-template": "POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops": "GET, POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/history": "GET, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/rollback": "POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/archive": "POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/acceptance-report": "GET, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/execution-bindings": "GET, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal": (
            "POST, OPTIONS"
        ),
        "/api/field/customer-projects/demo-field-ops/onsite-evidence": "GET, POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/acceptance-closure": "GET, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/acceptance-review": "POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/customer-signoff": "GET, POST, OPTIONS",
        "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles": (
            "POST, DELETE, OPTIONS"
        ),
        "/api/field/customer-projects/managed-object-directory": "GET, OPTIONS",
    }

    for path, methods in expected_methods.items():
        response = client.options(path)

        assert response.status_code == 200
        assert response.headers["access-control-allow-methods"] == methods




class TestFieldHttp:
    def test_field_route_roots_are_project_anchored_when_cwd_changes(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]

        roots = health_server._field_operations_path_roots({})

        assert roots["site_profile_root"] == repo_root / "deploy" / "site-profiles"
        assert roots["customer_project_template_root"] == (
            repo_root / "deploy" / "customer-project-templates"
        )
        assert roots["delivery_resource_root"] == repo_root / "deploy" / "delivery-resources"
        assert roots["customer_project_package_root"] == (
            repo_root / "artifacts" / "customer-project-packages"
        )
        assert roots["customer_project_acceptance_dossier_root"] == (
            repo_root / "artifacts" / "customer-project-acceptance-dossiers"
        )
        assert roots["customer_project_proposal_root"] == (
            repo_root / "artifacts" / "customer-project-proposals"
        )


    def test_field_route_roots_allow_explicit_customer_project_overrides(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]

        roots = health_server._field_operations_path_roots(
            {
                "field_operations": {
                    "site_profile_root": "deploy/customer-a/site-profiles",
                    "customer_project_template_root": str(tmp_path / "templates"),
                    "delivery_resource_root": "deploy/customer-a/delivery-resources",
                    "customer_project_package_root": "artifacts/customer-a/packages",
                    "customer_project_acceptance_dossier_root": str(tmp_path / "dossiers"),
                    "customer_project_proposal_root": "artifacts/customer-a/proposals",
                }
            }
        )

        assert roots["site_profile_root"] == repo_root / "deploy" / "customer-a" / "site-profiles"
        assert roots["customer_project_template_root"] == tmp_path / "templates"
        assert roots["delivery_resource_root"] == (
            repo_root / "deploy" / "customer-a" / "delivery-resources"
        )
        assert roots["customer_project_package_root"] == (
            repo_root / "artifacts" / "customer-a" / "packages"
        )
        assert roots["customer_project_acceptance_dossier_root"] == tmp_path / "dossiers"
        assert roots["customer_project_proposal_root"] == (
            repo_root / "artifacts" / "customer-a" / "proposals"
        )


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


    def test_field_device_onboarding_endpoint_returns_delivery_report(self):
        class Handler:
            def device_onboarding_payload(self):
                return {
                    "report_type": "askme.field.device_onboarding_report.v1",
                    "status": "manual_check",
                    "summary": {"registered": 1, "ready": 0, "manual_check": 1},
                    "devices": [
                        {
                            "device_id": "smoke-01",
                            "onboarding_gate": {"status": "manual_check"},
                        }
                    ],
                    "next_actions": ["Send one signed payload."],
                }

        client = TestClient(
            create_health_app(
                lambda: _runtime_snapshot(),
                field_operations_handler=Handler(),
            )
        )

        response = client.get("/api/field/device-onboarding")

        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "askme.field.device_onboarding_report.v1"
        assert data["summary"]["manual_check"] == 1
        assert data["devices"][0]["device_id"] == "smoke-01"


    def test_field_site_profiles_endpoint_returns_multi_site_catalog(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/field/site-profiles")

        assert response.status_code == 200
        data = response.json()
        schema_data = SiteProfileCatalogResponse.model_validate(data)
        assert schema_data.root
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
        schema_data = CustomerProjectCatalogResponse.model_validate(data)
        assert schema_data.projects
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
        schema_directory = ManagedObjectDirectoryResponse.model_validate(directory_payload)
        assert schema_directory.directory_type == (
            "askme.customer_project_managed_object_directory"
        )
        assert directory_payload["directory_type"] == (
            "askme.customer_project_managed_object_directory"
        )
        assert directory_payload["summary"]["object_count"] >= 1
        assert directory_payload["summary"]["object_count"] == len(directory_payload["objects"])
        assert directory_payload["summary"]["ready_count"] >= 1
        assert directory_payload["summary"]["scope_filtered"] is True
        assert directory_payload["customer_status"] == "对象目录已按当前操作人范围过滤，可用于交付复核。"
        assert directory_payload["next_step"] == "导出客户交付包前，先处理阻断或待复核的对象。"
        assert "Managed object" not in directory_payload["customer_status"]
        assert "Review blocked" not in directory_payload["next_step"]
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
        base_config = health_server.get_config()
        field_config = dict(base_config.get("field_operations", {}))
        field_config["site_profile_root"] = str(tmp_path / "deploy" / "site-profiles")
        monkeypatch.setattr(
            health_server,
            "get_config",
            lambda: {**base_config, "field_operations": field_config},
        )
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
        schema_payload = SolutionDeliveryReadinessResponse.model_validate(payload)
        assert schema_payload.readiness_type == "askme.solution_delivery_readiness"
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
        schema_workbench = CustomerProjectWorkbenchResponse.model_validate(workbench_payload)
        assert schema_workbench.workbench_type == (
            "askme.solution_provider_customer_project_workbench.v1"
        )
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
        assert "客户只能看到自己的项目、对象、证据和交付包。" in {
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
        assert workbench_payload["runtime_blueprint_binding"]["binding_type"] == (
            "askme.customer_project.runtime_blueprint_binding.v1"
        )
        assert workbench_payload["runtime_blueprint_binding"]["summary"]["project_count"] >= 1
        assert (
            workbench_payload["runtime_blueprint_binding"]["summary"][
                "available_customer_blueprint_count"
            ]
            >= 3
        )
        assert workbench_payload["runtime_blueprint_binding"]["policy"][
            "managed_objects_must_bind_resources_before_customer_claim"
        ] is True
        assert workbench_payload["delivery_chain"]["chain_type"] == (
            "askme.customer_project.delivery_chain.v1"
        )
        assert {
            "project_scope",
            "template_market",
            "managed_object_directory",
            "capability_resource_binding",
            "runtime_blueprint",
            "acceptance_package",
        } <= {step["step_id"] for step in workbench_payload["delivery_chain"]["steps"]}

        launch = client.get("/api/field/product-launch-readiness", params={"check_env": "true"})
        assert launch.status_code == 200
        launch_payload = launch.json()
        schema_launch = ProductLaunchReadinessResponse.model_validate(launch_payload)
        assert schema_launch.readiness_type == "askme.product_launch_readiness.v1"
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
            "dashboard_pages",
        } <= {gate["gate_id"] for gate in launch_payload["gates"]}
        assert {
            "/api/governance/identity-readiness",
            "/api/field/readiness",
            "/api/field/solution-delivery-readiness",
            "/api/field/customer-project-workbench",
            "/api/dashboard/pages",
        } <= {source["endpoint"] for source in launch_payload["evidence_sources"]}
        assert launch_payload["summary"]["dashboard_page_count"] >= 1
        assert launch_payload["summary"]["dashboard_endpoint_missing_count"] == 0
        assert launch_payload["summary"]["dashboard_endpoint_internal_count"] == 0
        snapshot = launch_payload["customer_acceptance_snapshot"]
        assert snapshot["snapshot_type"] == "askme.customer_project_acceptance_snapshot.v1"
        assert snapshot["overall_status"] == launch_payload["overall_status"]
        assert snapshot["metrics"]["gate_count"] == launch_payload["summary"]["gate_count"]
        assert snapshot["metrics"]["dashboard_endpoint_missing_count"] == 0
        assert {
            "identity_gateway",
            "field_operations",
            "solution_delivery",
            "customer_project_workbench",
            "dashboard_pages",
        } <= {source["source_id"] for source in snapshot["evidence_sources"]}


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
            "/api/field/customer-projects/demo-field-ops/execution-bindings",
            "/api/field/customer-projects/demo-field-ops/onsite-evidence",
            "/api/field/customer-projects/demo-field-ops/acceptance-closure",
            "/api/field/customer-projects/demo-field-ops/customer-signoff",
            "/api/field/customer-projects/demo-field-ops/export",
            "/api/field/customer-projects/demo-field-ops/acceptance-dossier",
            "/api/field/customer-projects/demo-field-ops/proposal-bundle",
            "/api/field/customer-projects/demo-field-ops/history",
        ):
            response = client.get(path, headers=headers)

            assert response.status_code == 403
            assert response.json()["reason"] == "operator_missing_permission"
            assert response.json()["operator_auth"]["permission"] == "field:project:read"

        for path in (
            "/api/field/customer-projects/package/verify",
            "/api/field/customer-projects/package/diff",
            "/api/field/customer-projects/proposal-bundle/verify",
            "/api/field/customer-projects/acceptance-dossier/verify",
        ):
            response = client.post(path, json={"operator_id": "ghost.operator"})

            assert response.status_code == 403
            assert response.json()["reason"] == "operator_missing_permission"
            assert response.json()["operator_auth"]["permission"] == "field:project:read"

        import_response = client.post(
            "/api/field/customer-projects/import",
            json={"operator_id": "ghost.operator", "package": {}},
        )

        assert import_response.status_code == 403
        assert import_response.json()["reason"] == "operator_missing_permission"
        assert import_response.json()["operator_auth"]["permission"] == "field:project:write"

        for path, payload in (
            (
                "/api/field/customer-projects/demo-field-ops/execution-bindings/vehicles/rehearsal",
                {"operator_id": "ghost.operator", "mode": "dry_run"},
            ),
            (
                "/api/field/customer-projects/demo-field-ops/onsite-evidence",
                {
                    "operator_id": "ghost.operator",
                    "evidence": {
                        "evidence_type": "device_ingest",
                        "status": "passed",
                        "summary": "unknown operator should not register evidence",
                    },
                },
            ),
            (
                "/api/field/customer-projects/demo-field-ops/acceptance-review",
                {
                    "operator_id": "ghost.operator",
                    "review": {
                        "decision": "accepted",
                        "risk_acknowledgement": True,
                        "reason": "unknown operator should not review acceptance",
                    },
                },
            ),
            (
                "/api/field/customer-projects/demo-field-ops/customer-signoff",
                {
                    "operator_id": "ghost.operator",
                    "signoff": {
                        "decision": "accepted",
                        "signatory_name": "Ghost Operator",
                        "reason": "unknown operator should not sign off",
                    },
                },
            ),
        ):
            response = client.post(path, json=payload)

            assert response.status_code == 403
            assert response.json()["reason"] == "operator_missing_permission"
            assert response.json()["operator_auth"]["permission"] == "field:project:write"


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
                    "site_profile_root": str(tmp_path / "deploy" / "site-profiles"),
                    "customer_project_template_root": str(
                        tmp_path / "deploy" / "customer-project-templates"
                    ),
                    "delivery_resource_root": str(tmp_path / "deploy" / "delivery-resources"),
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
        DeliveryResourceGovernanceMutationResponse.model_validate(payload)
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
        queue_payload = queue.json()
        DeliveryResourceGovernanceRequestsResponse.model_validate(queue_payload)
        assert queue_payload["summary"]["pending_count"] == 1
        assert "overdue_count" in queue_payload["summary"]
        assert queue_payload["summary"]["overdue_count"] == 0
        queued = queue_payload["requests"][0]
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
        overdue_queue_payload = overdue_queue.json()
        DeliveryResourceGovernanceRequestsResponse.model_validate(overdue_queue_payload)
        assert overdue_queue_payload["overdue_only"] is True
        assert overdue_queue_payload["summary"]["overdue_count"] == 1
        assert overdue_queue_payload["request_count"] == 1
        assert overdue_queue_payload["requests"][0]["review_sla"]["state"] == "overdue"
        assert overdue_queue_payload["requests"][0]["review_sla"]["escalation_required"] is True
        escalation = client.post(
            "/api/field/delivery-resource-governance-requests/escalate-overdue",
            json={
                "operator_id": "product.reviewer",
                "reason": "approval SLA missed",
            },
        )
        assert escalation.status_code == 200
        escalation_payload = escalation.json()
        DeliveryResourceGovernanceEscalationResponse.model_validate(escalation_payload)
        assert escalation_payload["accepted"] is True
        assert escalation_payload["escalated_count"] == 1
        assert escalation_payload["escalations"][0]["notification"]["status"] == "sent"
        assert escalation_payload["escalations"][0]["notification"]["delivery_mode"] == (
            "configured_channels"
        )
        assert escalation_payload["escalations"][0]["notification"]["sent_channels"] == ["log"]
        assert escalation_payload["escalations"][0]["delivery_report"][0]["channel"] == "log"
        assert escalation_payload["requests"][0]["escalation_count"] == 1
        assert escalation_payload["requests"][0]["last_escalation"]["status"] == "sent"

        self_review = client.post(
            f"/api/field/delivery-resource-governance-requests/{request_id}/review",
            json={
                "operator_id": "product.owner",
                "decision": "approve",
                "reason": "self review should fail",
            },
        )
        assert self_review.status_code == 409
        self_review_payload = self_review.json()
        DeliveryResourceGovernanceMutationResponse.model_validate(self_review_payload)
        assert self_review_payload["reason"] == (
            "resource_governance_request_requires_second_approver"
        )
        assert self_review_payload["request"]["review_sla"]["state"] == "overdue"
        assert self_review_payload["request"]["escalation_count"] == 1

        approved = client.post(
            f"/api/field/delivery-resource-governance-requests/{request_id}/review",
            json={
                "operator_id": "product.reviewer",
                "decision": "approve",
                "reason": "approve resource disable",
            },
        )
        assert approved.status_code == 200
        approved_payload = approved.json()
        DeliveryResourceGovernanceMutationResponse.model_validate(approved_payload)
        assert approved_payload["request"]["status"] == "approved"
        assert approved_payload["request"]["review_sla"]["state"] == "closed"
        assert approved_payload["apply_result"]["accepted"] is True
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

        allowed_history = client.get(
            "/api/field/customer-projects/demo-field-ops/history",
            headers={"X-Askme-Operator-Id": "default-tenant-supervisor"},
        )
        denied_history = client.get(
            "/api/field/customer-projects/demo-field-ops/history",
            headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
        )
        assert allowed_history.status_code == 200
        assert denied_history.status_code == 403
        assert denied_history.json()["reason"] == "project_scope_not_allowed"

        denied_object_write = client.post(
            "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
            json={
                "operator_id": "pilot-tenant-supervisor",
                "managed_object": {"display_name": "Scope leak attempt"},
            },
        )
        denied_object_delete = client.request(
            "DELETE",
            "/api/field/customer-projects/demo-field-ops/managed-objects/vehicles",
            json={"operator_id": "pilot-tenant-supervisor"},
        )
        assert denied_object_write.status_code == 403
        assert denied_object_write.json()["reason"] == "project_scope_not_allowed"
        assert denied_object_delete.status_code == 403
        assert denied_object_delete.json()["reason"] == "project_scope_not_allowed"

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

        for artifact_path in (
            "/api/field/customer-projects/demo-field-ops/export",
            "/api/field/customer-projects/demo-field-ops/acceptance-dossier",
            "/api/field/customer-projects/demo-field-ops/proposal-bundle",
        ):
            allowed_artifact = client.get(
                artifact_path,
                headers={"X-Askme-Operator-Id": "default-tenant-supervisor"},
            )
            denied_artifact = client.get(
                artifact_path,
                headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
            )

            assert allowed_artifact.status_code == 200
            assert denied_artifact.status_code == 403
            assert denied_artifact.json()["reason"] == "project_scope_not_allowed"

        proposal_response = client.get(
            "/api/field/customer-projects/demo-field-ops/proposal-bundle",
            headers={"X-Askme-Operator-Id": "supervisor-1"},
        )
        assert proposal_response.status_code == 200
        proposal = proposal_response.json()["proposal"]
        allowed_proposal_verify = client.post(
            "/api/field/customer-projects/proposal-bundle/verify",
            json={"operator_id": "default-tenant-supervisor", "proposal": proposal},
        )
        denied_proposal_verify = client.post(
            "/api/field/customer-projects/proposal-bundle/verify",
            json={"operator_id": "pilot-tenant-supervisor", "proposal": proposal},
        )
        assert allowed_proposal_verify.status_code == 200
        assert allowed_proposal_verify.json()["proposal_scope"]["tenant_id"] == "default"
        assert allowed_proposal_verify.json()["proposal_scope"]["delivery_namespace"] == "default"
        assert denied_proposal_verify.status_code == 403
        assert denied_proposal_verify.json()["reason"] == "project_scope_not_allowed"

        scoped_templates = client.get(
            "/api/field/customer-project-templates",
            headers={"X-Askme-Operator-Id": "pilot-tenant-supervisor"},
        )
        assert scoped_templates.status_code == 200
        assert scoped_templates.json()["summary"]["scope_filtered"] is True
        assert scoped_templates.json()["summary"]["template_count"] >= 1


    def test_field_customer_project_from_template_returns_implementation_handoff(self, tmp_path):
        client = TestClient(
            _field_route_test_app(
                tmp_path / "site-profiles",
                customer_project_template_root=DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
            )
        )

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


    def test_field_customer_project_from_template_does_not_fallback_to_repo_templates(
        self,
        tmp_path,
    ):
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

        assert response.status_code == 422
        assert response.json()["accepted"] is False
        assert response.json()["reason"] == "template_not_found"
        assert not (tmp_path / "site-profiles").exists()
        assert not (tmp_path / "api-customer-api-line-one.yaml").exists()


    def test_field_customer_project_artifact_exports_use_explicit_roots(self, tmp_path):
        profile_root = tmp_path / "site-profiles"
        package_root = tmp_path / "exports" / "packages"
        dossier_root = tmp_path / "exports" / "dossiers"
        proposal_root = tmp_path / "exports" / "proposals"
        client = TestClient(
            _field_route_test_app(
                profile_root,
                customer_project_template_root=DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
                customer_project_package_root=package_root,
                customer_project_acceptance_dossier_root=dossier_root,
                customer_project_proposal_root=proposal_root,
            )
        )

        created = client.post(
            "/api/field/customer-projects/from-template",
            json={
                "operator_id": "delivery.manager",
                "template_id": "factory-inspection",
                "customer": {
                    "tenant_id": "tenant-artifacts",
                    "delivery_namespace": "pilot",
                    "customer_id": "api-artifact-customer",
                    "customer_name": "API Artifact Customer",
                    "industry": "manufacturing",
                    "project_id": "api-artifact-line",
                    "project_name": "API Artifact Line",
                },
                "site": {"site_id": "api-artifact-site", "name": "API Artifact Site"},
            },
        )
        assert created.status_code == 200

        package = client.get("/api/field/customer-projects/api-artifact-line/export")
        dossier = client.get(
            "/api/field/customer-projects/api-artifact-line/acceptance-dossier",
            params={"check_env": "false"},
        )
        proposal = client.get(
            "/api/field/customer-projects/api-artifact-line/proposal-bundle",
            params={"check_env": "false"},
        )

        assert package.status_code == 200
        assert dossier.status_code == 200
        assert proposal.status_code == 200
        assert Path(package.json()["package_path"]).parent == package_root
        assert Path(dossier.json()["dossier_path"]).parent == dossier_root
        assert Path(dossier.json()["html_path"]).parent == dossier_root
        assert Path(proposal.json()["proposal_path"]).parent == proposal_root
        assert Path(proposal.json()["html_path"]).parent == proposal_root
        assert Path(proposal.json()["package_path"]).parent == package_root
        assert Path(proposal.json()["dossier_path"]).parent == dossier_root
        assert package_root.exists()
        assert dossier_root.exists()
        assert proposal_root.exists()


    def test_field_customer_project_artifact_exports_preflight_scope_before_writes(self, tmp_path):
        profile_root = tmp_path / "site-profiles"
        package_root = tmp_path / "exports" / "packages"
        dossier_root = tmp_path / "exports" / "dossiers"
        proposal_root = tmp_path / "exports" / "proposals"
        created = create_customer_project_from_template(
            template_root=DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
            profile_root=profile_root,
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-artifacts",
                "delivery_namespace": "pilot",
                "customer_id": "api-artifact-customer",
                "customer_name": "API Artifact Customer",
                "industry": "manufacturing",
                "project_id": "api-artifact-line",
                "project_name": "API Artifact Line",
            },
            site={"site_id": "api-artifact-site", "name": "API Artifact Site"},
        )
        assert created["accepted"] is True
        client = TestClient(
            _field_route_test_app(
                profile_root,
                customer_project_template_root=DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
                customer_project_package_root=package_root,
                customer_project_acceptance_dossier_root=dossier_root,
                customer_project_proposal_root=proposal_root,
                authorize_callback=_scoped_project_authorize(
                    {
                        "tenant_ids": ["tenant-other"],
                        "delivery_namespaces": ["pilot"],
                    }
                ),
            )
        )

        package = client.get("/api/field/customer-projects/api-artifact-line/export")
        dossier = client.get(
            "/api/field/customer-projects/api-artifact-line/acceptance-dossier",
            params={"check_env": "false"},
        )
        proposal = client.get(
            "/api/field/customer-projects/api-artifact-line/proposal-bundle",
            params={"check_env": "false"},
        )

        assert package.status_code == 403
        assert dossier.status_code == 403
        assert proposal.status_code == 403
        assert package.json()["reason"] == "project_scope_not_allowed"
        assert dossier.json()["reason"] == "project_scope_not_allowed"
        assert proposal.json()["reason"] == "project_scope_not_allowed"
        assert not package_root.exists()
        assert not dossier_root.exists()
        assert not proposal_root.exists()


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
        CustomerProjectMutationResponse.model_validate(payload)
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
        CustomerProjectManagedObjectMutationResponse.model_validate(saved_payload)
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
        CustomerProjectManagedObjectMutationResponse.model_validate(deleted_payload)
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
        schema_template_payload = CustomerProjectTemplateCatalogResponse.model_validate(
            template_payload
        )
        assert schema_template_payload.templates
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
        assert template_payload["summary"]["runtime_blueprint_bound_count"] >= 4
        assert (
            first_template["runtime_blueprint_binding"]["binding_type"]
            == "askme.customer_project_template.runtime_blueprint_binding.v1"
        )
        assert first_template["runtime_blueprint_binding"]["selected_blueprint"]["name"]
        assert first_template["runtime_blueprint_binding"]["policy"][
            "created_projects_must_recheck_blueprint_binding"
        ] is True

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

        template_history = client.get(
            "/api/field/customer-project-templates/factory-inspection/history"
        )
        assert template_history.status_code == 200
        template_history_payload = template_history.json()
        CustomerProjectTemplateHistoryResponse.model_validate(template_history_payload)
        assert template_history_payload["found"] is True

        missing_template_history = client.get(
            "/api/field/customer-project-templates/no-such-template/history"
        )
        assert missing_template_history.status_code == 404
        missing_template_history_payload = missing_template_history.json()
        CustomerProjectTemplateHistoryResponse.model_validate(
            missing_template_history_payload
        )
        assert missing_template_history_payload["reason"] == "template_not_found"

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
        direct_product_publish_payload = direct_product_publish.json()
        CustomerProjectTemplateReleaseUpdateResponse.model_validate(
            direct_product_publish_payload
        )
        assert (
            direct_product_publish_payload["reason"]
            == "published_release_requires_approval_request"
        )

        missing_template_release = client.post(
            "/api/field/customer-project-templates/no-such-template/release",
            json={"operator_id": "product.owner", "release": {"publish_status": "pilot"}},
        )
        assert missing_template_release.status_code == 404
        missing_template_release_payload = missing_template_release.json()
        CustomerProjectTemplateReleaseUpdateResponse.model_validate(
            missing_template_release_payload
        )
        assert missing_template_release_payload["reason"] == "template_not_found"

        release_requests = client.get("/api/field/customer-project-template-release-requests")
        assert release_requests.status_code == 200
        release_requests_payload = release_requests.json()
        CustomerProjectTemplateReleaseRequestsResponse.model_validate(
            release_requests_payload
        )
        assert "pending_count" in release_requests_payload["summary"]

        release_notes = client.get("/api/field/customer-project-template-release-notes")
        assert release_notes.status_code == 200
        release_notes_payload = release_notes.json()
        CustomerProjectTemplateReleaseNotesResponse.model_validate(release_notes_payload)
        assert "approved_release_count" in release_notes_payload["summary"]
        assert release_notes_payload["customer_claim"] == "发布说明只包含已审批并发布的模板包。"

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
        release_notes_bundle_payload = release_notes_bundle.json()
        CustomerProjectTemplateReleaseNotesExportResponse.model_validate(
            release_notes_bundle_payload
        )
        assert release_notes_bundle_payload["accepted"] is True
        assert release_notes_bundle_payload["bundle"]["bundle_schema"] == (
            "askme.template_release_notes_bundle.v1"
        )
        assert "proposal_insert" in release_notes_bundle_payload["bundle"]
        assert release_notes_bundle_payload["bundle"]["proposal_insert"]["section_title"] == (
            "Demo Proposal approved reusable capabilities"
        )
        assert "html" in release_notes_bundle_payload["bundle"]
        assert release_notes_bundle_payload["bundle"]["files"]["html_filename"] == (
            "demo-proposal-template-release-notes.html"
        )

        missing_template_request = client.post(
            "/api/field/customer-project-templates/no-such-template/release-requests",
            json={"operator_id": "product.owner", "release": {"publish_status": "published"}},
        )
        assert missing_template_request.status_code == 404
        missing_template_request_payload = missing_template_request.json()
        CustomerProjectTemplateReleaseRequestMutationResponse.model_validate(
            missing_template_request_payload
        )
        assert missing_template_request_payload["reason"] == "template_not_found"

        missing_request_review = client.post(
            "/api/field/customer-project-template-release-requests/no-such-request/review",
            json={"operator_id": "product.reviewer", "decision": "approve"},
        )
        assert missing_request_review.status_code == 404
        missing_request_review_payload = missing_request_review.json()
        CustomerProjectTemplateReleaseRequestMutationResponse.model_validate(
            missing_request_review_payload
        )
        assert missing_request_review_payload["reason"] == "release_request_not_found"

        registry = client.get("/api/field/customer-project-acceptance-registry")
        assert registry.status_code == 200
        registry_payload = registry.json()
        schema_registry_payload = CustomerProjectAcceptanceRegistryResponse.model_validate(
            registry_payload
        )
        assert schema_registry_payload.references
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
        schema_resource_payload = CustomerProjectResourceCatalogResponse.model_validate(
            resource_payload
        )
        assert schema_resource_payload.resources
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
        detail_payload = detail.json()
        schema_detail_payload = CustomerProjectDetailResponse.model_validate(detail_payload)
        assert schema_detail_payload.found is True
        assert detail_payload["managed_objects"]["object_type_count"] >= 1
        assert detail_payload["managed_objects"]["binding_readiness_summary"]["overall_status"] == "ready"
        assert detail_payload["delivery_workflow"]["steps"]
        assert detail_payload["delivery_workflow"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        vehicle_status = detail_payload["managed_objects"]["objects_by_id"]["vehicles"]["acceptance_status"]
        assert vehicle_status["acceptance_checks"][0]["matched"] == "illegal_parking_camera_ingest"
        vehicle_resources = detail_payload["managed_objects"]["objects_by_id"]["vehicles"]["resource_binding_status"]
        assert vehicle_resources["overall_status"] == "ready"
        assert vehicle_resources["linked_count"] >= 4

        execution = client.get("/api/field/customer-projects/demo-field-ops/execution-bindings")
        assert execution.status_code == 200
        execution_payload = execution.json()
        schema_execution_payload = CustomerProjectExecutionBindingsResponse.model_validate(
            execution_payload
        )
        assert schema_execution_payload.found is True
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
        CustomerProjectExecutionRehearsalResponse.model_validate(rehearsal_payload)
        assert rehearsal_payload["accepted"] is True
        assert rehearsal_payload["status"] == "lab_rehearsed"
        assert rehearsal_payload["production_claim_allowed"] is False
        assert rehearsal_payload["rehearsal"]["mode"] == "dry_run"
        assert rehearsal_payload["customer_status"] == "仅限实验室演练，不能作为生产上线验收依据。"
        assert "not production go-live evidence" not in rehearsal_payload["customer_status"]
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
        CustomerProjectExecutionRehearsalResponse.model_validate(dry_run_evidence.json())
        dry_run_registration = dry_run_evidence.json()["onsite_evidence_registration"]
        assert dry_run_registration["registered"] is False
        assert dry_run_registration["reason"] == "dry_run_rehearsal_not_onsite_evidence"
        assert dry_run_registration["production_eligible"] is False
        assert dry_run_registration["customer_status"].startswith("干跑演练只证明适配器形态")

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
        assert shadow_needs_confirmation.json()["next_step"].startswith("仅在实验室或现场演练窗口确认")
        assert shadow_needs_confirmation.json()["onsite_evidence_registration"]["registered"] is False

        history = client.get("/api/field/customer-projects/demo-field-ops/history")
        assert history.status_code == 200
        history_payload = history.json()
        CustomerProjectHistoryResponse.model_validate(history_payload)
        assert history_payload["found"] is True
        assert "revisions" in history_payload

        report = client.get("/api/field/customer-projects/demo-field-ops/acceptance-report")
        assert report.status_code == 200
        report_payload = report.json()
        schema_report_payload = CustomerProjectAcceptanceReportResponse.model_validate(
            report_payload
        )
        assert schema_report_payload.found is True
        assert report_payload["overall_status"] in {
            "ready_for_onsite_acceptance",
            "manual_check",
            "blocked",
        }
        assert report_payload["acceptance_summary"]["overall_status"] == "ready"
        assert report_payload["gates"][0]["gate_id"] == "site_profile"
        gate_ids = {gate["gate_id"] for gate in report_payload["gates"]}
        assert "managed_object_execution_bindings" in gate_ids
        assert "field_readiness" in gate_ids
        assert "field_smoke_evidence" in gate_ids
        assert "voice_notification_evidence" in gate_ids
        assert "runtime_audit_trust" in gate_ids
        assert "field_device_onboarding" in gate_ids
        assert "device_onboarding" in report_payload["field_readiness"]
        assert report_payload["field_readiness"]["evidence_reports"]
        assert report_payload["execution_bindings"]["summary"]["overall_status"] == "ready"
        report_launch = report_payload["launch_readiness"]
        assert report_launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
        assert any(
            gate["gate_id"] == "field_device_onboarding"
            for gate in report_launch["gates"]
        )
        assert report_launch["overall_status"] in {"ready", "manual_check", "blocked"}
        assert report_launch["launch_stage"] in {
            "production_acceptance_ready",
            "pilot_or_site_trial",
            "demo_or_integration_only",
        }
        assert isinstance(report_launch["production_ready"], bool)
        report_vehicle_contract = next(
            item
            for item in report_payload["execution_bindings"]["object_contracts"]
            if item["object_id"] == "vehicles"
        )
        assert report_vehicle_contract["input_adapters"][0]["bridge"] == "field-ingest-bridge"
        assert report_vehicle_contract["input_adapters"][0]["device_signature_required"] is True
        assert "onsite_acceptance_evidence" in report_payload
        assert report_payload["delivery_workflow"]["steps"]
        assert report_payload["site_acceptance_checklist"]["items"]
        assert report_payload["site_acceptance_checklist"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }

        onsite = client.get("/api/field/customer-projects/demo-field-ops/onsite-evidence")
        assert onsite.status_code == 200
        onsite_payload = onsite.json()
        schema_onsite_payload = CustomerProjectOnsiteEvidenceResponse.model_validate(
            onsite_payload
        )
        assert schema_onsite_payload.found is True
        assert onsite_payload["found"] is True
        assert onsite_payload["readiness_auto_included"] is True
        assert "field_readiness" in onsite_payload
        assert onsite_payload["onsite_acceptance_evidence"]["summary"]["overall_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }

        closure = client.get("/api/field/customer-projects/demo-field-ops/acceptance-closure")
        assert closure.status_code == 200
        closure_payload = closure.json()
        schema_closure_payload = CustomerProjectAcceptanceClosureResponse.model_validate(
            closure_payload
        )
        assert schema_closure_payload.found is True
        assert closure_payload["found"] is True
        assert closure_payload["overall_status"] in {
            "ready_for_acceptance",
            "ready_for_customer_signoff",
            "accepted_by_customer",
            "manual_check",
            "blocked",
        }
        assert "manual_review" in closure_payload
        assert "customer_signoff" in closure_payload
        assert any(gate["gate_id"] == "customer_signoff" for gate in closure_payload["gates"])
        artifact_verification = closure_payload["artifact_verification"]
        assert "acceptance_dossier" in artifact_verification
        assert "proposal_bundle" in artifact_verification
        assert "audit_export" in artifact_verification

        signoff = client.get("/api/field/customer-projects/demo-field-ops/customer-signoff")
        assert signoff.status_code == 200
        signoff_payload = signoff.json()
        schema_signoff_payload = CustomerProjectSignoffResponse.model_validate(signoff_payload)
        assert schema_signoff_payload.found is True
        assert signoff_payload["found"] is True
        assert "signoffs" in signoff_payload
        assert signoff_payload["project_scope"]["project_id"] == "demo-field-ops"

        exported = client.get("/api/field/customer-projects/demo-field-ops/export")
        assert exported.status_code == 200
        exported_payload = exported.json()
        schema_exported_payload = CustomerProjectPackageExportResponse.model_validate(
            exported_payload
        )
        assert schema_exported_payload.accepted is True
        package = exported_payload["package"]
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
        assert package["delivery_chain"]["chain_type"] == "askme.customer_project.delivery_chain.v1"
        assert package["delivery_chain"]["step_count"] == 6
        assert package["manifest"]["delivery_chain_step_count"] == package["delivery_chain"][
            "step_count"
        ]
        assert {
            "project_scope",
            "template_market",
            "managed_object_directory",
            "capability_resource_binding",
            "runtime_blueprint",
            "acceptance_package",
        } <= {step["step_id"] for step in package["delivery_chain"]["steps"]}

        dossier = client.get("/api/field/customer-projects/demo-field-ops/acceptance-dossier")
        assert dossier.status_code == 200
        dossier_payload = dossier.json()
        schema_dossier_payload = CustomerProjectAcceptanceDossierExportResponse.model_validate(
            dossier_payload
        )
        assert schema_dossier_payload.accepted is True
        assert dossier_payload["accepted"] is True
        assert dossier_payload["html_path"].endswith(".html")
        assert dossier_payload["dossier"]["manifest"]["project_id"] == "demo-field-ops"
        assert dossier_payload["dossier"]["manifest"]["payload_sha256"]
        assert "onsite_evidence_status" in dossier_payload["dossier"]["manifest"]
        assert "site_acceptance_checklist_status" in dossier_payload["dossier"]["manifest"]
        dossier_launch = dossier_payload["dossier"]["launch_readiness"]
        assert dossier_launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
        assert dossier_payload["dossier"]["manifest"]["launch_readiness_status"] == (
            dossier_launch["overall_status"]
        )
        assert dossier_payload["dossier"]["manifest"]["launch_stage"] == dossier_launch["launch_stage"]
        assert dossier_payload["dossier"]["manifest"]["production_ready"] is dossier_launch[
            "production_ready"
        ]
        assert dossier_payload["dossier"]["delivery_workflow"]["steps"]
        assert dossier_payload["dossier"]["delivery_chain"]["chain_type"] == (
            "askme.customer_project.delivery_chain.v1"
        )
        assert dossier.json()["dossier"]["manifest"]["delivery_chain_step_count"] == (
            dossier.json()["dossier"]["delivery_chain"]["step_count"]
        )
        assert "runtime_blueprint" in {
            step["step_id"] for step in dossier.json()["dossier"]["delivery_chain"]["steps"]
        }
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
        schema_dossier_verification = CustomerProjectAcceptanceDossierVerifyResponse.model_validate(
            dossier_verification.json()
        )
        assert schema_dossier_verification.accepted is True
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
        proposal_payload = proposal.json()
        schema_proposal_payload = CustomerProjectProposalBundleExportResponse.model_validate(
            proposal_payload
        )
        assert schema_proposal_payload.accepted is True
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
        assert proposal_readable["delivery_chain"]["chain_type"] == (
            "askme.customer_project.delivery_chain.v1"
        )
        assert proposal.json()["proposal"]["manifest"]["proposal_delivery_chain_step_count"] == (
            proposal_readable["delivery_chain"]["step_count"]
        )
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
        assert "客户交付链路" in proposal.json()["proposal"]["html"]

        proposal_verification = client.post(
            "/api/field/customer-projects/proposal-bundle/verify",
            json={
                "operator_id": "supervisor-1",
                "proposal": proposal.json()["proposal"],
            },
        )
        assert proposal_verification.status_code == 200
        schema_proposal_verification = CustomerProjectProposalBundleVerifyResponse.model_validate(
            proposal_verification.json()
        )
        assert schema_proposal_verification.accepted is True
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
        schema_verification = CustomerProjectPackageVerifyResponse.model_validate(
            verification.json()
        )
        assert schema_verification.accepted is True
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
        schema_diff_preview = CustomerProjectPackageDiffResponse.model_validate(
            diff_preview.json()
        )
        assert schema_diff_preview.accepted is True
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
        schema_preview = CustomerProjectPackageImportResponse.model_validate(preview.json())
        assert schema_preview.accepted is True
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
        schema_windows_preview = CustomerProjectPackageImportResponse.model_validate(
            windows_preview.json()
        )
        assert schema_windows_preview.accepted is True
        assert windows_preview.json()["verification"]["valid"] is True
