"""Tests for the MCP and HTTP health surfaces."""

import pytest
from fastapi.testclient import TestClient

from askme.api.schemas.monitor import (
    HealthSnapshotResponse,
    SystemStatusResponse,
    TraceSnapshotResponse,
)
from askme.api.schemas.surfaces import ApiSurfacesResponse
from askme.health_server import AskmeHealthServer, build_health_snapshot, create_health_app
from tests.support.health_snapshots import (
    degraded_runtime_snapshot as _degraded_runtime_snapshot,
)
from tests.support.health_snapshots import (
    runtime_snapshot as _runtime_snapshot,
)


class TestHealthServer:
    def test_openapi_is_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ASKME_OPENAPI_ENABLED", raising=False)
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        assert client.get("/api/openapi.json").status_code == 404
        assert client.get("/api/docs").status_code == 404
        assert client.get("/api/redoc").status_code == 404

    def test_openapi_can_be_enabled_explicitly(self, monkeypatch):
        monkeypatch.setenv("ASKME_OPENAPI_ENABLED", "1")
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        spec = client.get("/api/openapi.json")
        docs = client.get("/api/docs")

        assert spec.status_code == 200
        payload = spec.json()
        assert payload["openapi"]
        assert "/api/knowledge/preview" in payload["paths"]
        assert "/api/memory/search" in payload["paths"]
        assert docs.status_code == 200

    def test_http_health_endpoint_returns_runtime_snapshot(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        HealthSnapshotResponse.model_validate(data)
        assert data["status"] == "ok"
        assert data["uptime_seconds"] == 12.5
        assert data["model_name"] == "claude-opus-4-6"
        assert data["last_llm_latency_ms"] == 245.0
        assert data["total_conversations"] == 7
        assert data["active_skills"] == ["dock_charge", "inspect_zone"]
        assert data["voice_pipeline_status"]["pipeline_ok"] is True
        assert data["ota_bridge_status"]["registered"] is True

    def test_api_surfaces_endpoint_returns_customer_boundary_map(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/surfaces")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        schema_payload = ApiSurfacesResponse.model_validate(data)
        assert data["ok"] is True
        assert schema_payload.readiness.readiness_type == "askme.api_surface_readiness.v1"
        assert schema_payload.route_inventory.policy["route_modules_are_contract_source"] is True
        assert [item["name"] for item in data["surfaces"]] == [
            "platform",
            "product",
            "admin",
            "internal",
        ]
        by_name = {item["name"]: item for item in data["surfaces"]}
        assert by_name["product"]["audience"] == "customer dashboard and operator workflows"
        assert by_name["product"]["customer_visible"] is True
        assert by_name["product"]["hardware_authority_allowed"] is False
        assert by_name["product"]["production_claim_allowed"] is False
        assert "客户和操作员可见的业务入口" in by_name["product"]["customer_boundary"]
        assert "customer knowledge" in by_name["product"]["owns"]
        assert "direct hardware authority" in by_name["product"]["must_not_expose"]
        assert by_name["internal"]["customer_visible"] is False
        assert by_name["internal"]["hardware_authority_allowed"] is True
        assert "device onboarding evidence" in by_name["internal"]["owns"]
        assert data["policy"]["product_surface_is_customer_visible"] is True
        assert data["policy"]["internal_surface_must_not_drive_customer_ui"] is True
        assert data["readiness"]["readiness_type"] == "askme.api_surface_readiness.v1"
        assert data["readiness"]["overall_status"] == "ready"
        assert data["readiness"]["summary"]["unclassified_count"] == 0
        assert data["readiness"]["summary"]["missing_surface_count"] == 0
        assert data["readiness"]["policy"]["route_inventory_has_no_unclassified_routes"] is True
        assert data["readiness"]["policy"]["all_declared_surfaces_have_routes"] is True
        assert (
            data["readiness"]["policy"]["product_surface_must_not_allow_hardware_authority"]
            is True
        )
        assert (
            data["readiness"]["policy"]["api_surface_must_not_be_production_claim_source"]
            is True
        )
        assert data["readiness"]["policy"]["internal_surface_must_not_be_customer_visible"] is True
        assert "内部机器人控制接口不会驱动客户 UI" in data["readiness"]["release_claim"]
        assert "产品页只调用客户可见接口" not in data["readiness"]["release_claim"]
        assert "按页面和权限使用客户、治理和平台接口" in data["readiness"]["release_claim"]
        inventory = data["route_inventory"]
        assert inventory["policy"]["route_modules_are_contract_source"] is True
        assert inventory["summary"]["total_route_count"] >= 200
        assert inventory["summary"]["unclassified_count"] == 0
        for surface in ("platform", "product", "admin", "internal"):
            assert inventory["surfaces"][surface]["route_count"] > 0
            assert inventory["surfaces"][surface]["module_count"] > 0
        assert inventory["surfaces"]["product"]["customer_visible"] is True
        assert inventory["surfaces"]["product"]["hardware_authority_allowed"] is False
        assert inventory["surfaces"]["internal"]["hardware_authority_allowed"] is True

        routes_by_path: dict[str, list[dict[str, object]]] = {}
        for route in inventory["routes"]:
            routes_by_path.setdefault(route["path"], []).append(route)

        assert any(route["surface"] == "platform" for route in routes_by_path["/api/surfaces"])
        assert any(route["surface"] == "product" for route in routes_by_path["/api/memory/health"])
        assert any(route["surface"] == "admin" for route in routes_by_path["/api/audit/reviews"])
        assert any(route["surface"] == "internal" for route in routes_by_path["/api/runtime/handoff"])
        assert any(route["surface"] == "internal" for route in routes_by_path["/api/field/device-onboarding"])

    def test_api_surfaces_endpoint_exposes_response_schema_in_openapi(self):
        app = create_health_app(lambda: _runtime_snapshot())

        schema = app.openapi()["paths"]["/api/surfaces"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]

        assert schema["$ref"].endswith("/ApiSurfacesResponse")

    def test_openapi_json_response_schema_coverage_allows_only_streaming_and_assets(
        self,
    ):
        app = create_health_app(lambda: _runtime_snapshot())
        allowed_no_json_schema = {
            ("GET", "/api/field/evidence"): "file_download",
            ("GET", "/dashboard"): "html_dashboard_shell",
            ("GET", "/dashboard/{asset_path}"): "html_or_static_asset",
            ("GET", "/api/runtime/events"): "server_sent_events",
        }

        seen_allowed = set()
        missing_or_generic = []
        for path, methods in app.openapi()["paths"].items():
            for method, operation in methods.items():
                if method == "options":
                    continue
                ok_response = (
                    operation.get("responses", {}).get("200")
                    or operation.get("responses", {}).get("201")
                    or operation.get("responses", {}).get("204")
                    or {}
                )
                schema = (
                    (((ok_response.get("content") or {}).get("application/json") or {}))
                    .get("schema")
                )
                route_key = (method.upper(), path)
                if route_key in allowed_no_json_schema:
                    seen_allowed.add(route_key)
                    assert schema in (None, {}), allowed_no_json_schema[route_key]
                    continue
                if not schema:
                    missing_or_generic.append((method.upper(), path, "no-json-schema"))
                    continue
                if (
                    schema == {}
                    or (
                        schema.get("type") == "object"
                        and not schema.get("properties")
                        and not schema.get("$ref")
                    )
                ):
                    missing_or_generic.append((method.upper(), path, "generic-object"))

        assert seen_allowed == set(allowed_no_json_schema)
        assert missing_or_generic == []

    def test_customer_project_product_endpoints_expose_response_schema_in_openapi(self):
        app = create_health_app(lambda: _runtime_snapshot())
        paths = app.openapi()["paths"]

        site_schema = paths["/api/field/site-profiles"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        projects_schema = paths["/api/field/customer-projects"]["get"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]
        templates_schema = paths["/api/field/customer-project-templates"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        template_history_schema = paths[
            "/api/field/customer-project-templates/{template_id}/history"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_requests_schema = paths[
            "/api/field/customer-project-template-release-requests"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_notes_schema = paths[
            "/api/field/customer-project-template-release-notes"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_notes_export_schema = paths[
            "/api/field/customer-project-template-release-notes/export"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_request_create_schema = paths[
            "/api/field/customer-project-templates/{template_id}/release-requests"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_request_review_schema = paths[
            "/api/field/customer-project-template-release-requests/{request_id}/review"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        template_release_update_schema = paths[
            "/api/field/customer-project-templates/{template_id}/release"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        workbench_schema = paths["/api/field/customer-project-workbench"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        directory_schema = paths[
            "/api/field/customer-projects/managed-object-directory"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        acceptance_schema = paths[
            "/api/field/customer-project-acceptance-registry"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        resources_schema = paths["/api/field/customer-project-resource-catalog"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        delivery_registry_schema = paths["/api/field/delivery-resource-registry"][
            "get"
        ]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_register_schema = paths[
            "/api/field/delivery-resource-registry"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_history_schema = paths[
            "/api/field/delivery-resource-registry/history"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_disable_schema = paths[
            "/api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_rollback_schema = paths[
            "/api/field/delivery-resource-registry/rollback"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_governance_schema = paths[
            "/api/field/delivery-resource-governance-requests"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_governance_create_schema = paths[
            "/api/field/delivery-resource-governance-requests"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_governance_review_schema = paths[
            "/api/field/delivery-resource-governance-requests/{request_id}/review"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        delivery_resource_governance_escalate_schema = paths[
            "/api/field/delivery-resource-governance-requests/escalate-overdue"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        solution_schema = paths["/api/field/solution-delivery-readiness"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        launch_schema = paths["/api/field/product-launch-readiness"]["get"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]
        detail_schema = paths["/api/field/customer-projects/{identifier}"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        upsert_schema = paths["/api/field/customer-projects"]["post"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]
        object_upsert_schema = paths[
            "/api/field/customer-projects/{identifier}/managed-objects/{object_id}"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        object_delete_schema = paths[
            "/api/field/customer-projects/{identifier}/managed-objects/{object_id}"
        ]["delete"]["responses"]["200"]["content"]["application/json"]["schema"]
        history_schema = paths[
            "/api/field/customer-projects/{identifier}/history"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        rollback_schema = paths[
            "/api/field/customer-projects/{identifier}/rollback"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        archive_schema = paths[
            "/api/field/customer-projects/{identifier}/archive"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        execution_schema = paths[
            "/api/field/customer-projects/{identifier}/execution-bindings"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        execution_rehearsal_schema = paths[
            "/api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        report_schema = paths[
            "/api/field/customer-projects/{identifier}/acceptance-report"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        evidence_schema = paths[
            "/api/field/customer-projects/{identifier}/onsite-evidence"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        evidence_register_schema = paths[
            "/api/field/customer-projects/{identifier}/onsite-evidence"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        closure_schema = paths[
            "/api/field/customer-projects/{identifier}/acceptance-closure"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        acceptance_review_schema = paths[
            "/api/field/customer-projects/{identifier}/acceptance-review"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        signoff_schema = paths[
            "/api/field/customer-projects/{identifier}/customer-signoff"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        signoff_register_schema = paths[
            "/api/field/customer-projects/{identifier}/customer-signoff"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        package_export_schema = paths[
            "/api/field/customer-projects/{identifier}/export"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        dossier_export_schema = paths[
            "/api/field/customer-projects/{identifier}/acceptance-dossier"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        proposal_export_schema = paths[
            "/api/field/customer-projects/{identifier}/proposal-bundle"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        package_import_schema = paths["/api/field/customer-projects/import"]["post"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        package_verify_schema = paths[
            "/api/field/customer-projects/package/verify"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        package_diff_schema = paths["/api/field/customer-projects/package/diff"][
            "post"
        ]["responses"]["200"]["content"]["application/json"]["schema"]
        proposal_verify_schema = paths[
            "/api/field/customer-projects/proposal-bundle/verify"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        dossier_verify_schema = paths[
            "/api/field/customer-projects/acceptance-dossier/verify"
        ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]

        assert site_schema["$ref"].endswith("/SiteProfileCatalogResponse")
        assert projects_schema["$ref"].endswith("/CustomerProjectCatalogResponse")
        assert templates_schema["$ref"].endswith("/CustomerProjectTemplateCatalogResponse")
        assert template_history_schema["$ref"].endswith(
            "/CustomerProjectTemplateHistoryResponse"
        )
        assert template_release_requests_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseRequestsResponse"
        )
        assert template_release_notes_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseNotesResponse"
        )
        assert template_release_notes_export_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseNotesExportResponse"
        )
        assert template_release_request_create_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseRequestMutationResponse"
        )
        assert template_release_request_review_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseRequestMutationResponse"
        )
        assert template_release_update_schema["$ref"].endswith(
            "/CustomerProjectTemplateReleaseUpdateResponse"
        )
        assert workbench_schema["$ref"].endswith("/CustomerProjectWorkbenchResponse")
        assert directory_schema["$ref"].endswith("/ManagedObjectDirectoryResponse")
        assert acceptance_schema["$ref"].endswith(
            "/CustomerProjectAcceptanceRegistryResponse"
        )
        assert resources_schema["$ref"].endswith("/CustomerProjectResourceCatalogResponse")
        assert delivery_registry_schema["$ref"].endswith(
            "/DeliveryResourceRegistryResponse"
        )
        assert delivery_resource_register_schema["$ref"].endswith(
            "/DeliveryResourceMutationResponse"
        )
        assert delivery_resource_history_schema["$ref"].endswith(
            "/DeliveryResourceHistoryResponse"
        )
        assert delivery_resource_disable_schema["$ref"].endswith(
            "/DeliveryResourceMutationResponse"
        )
        assert delivery_resource_rollback_schema["$ref"].endswith(
            "/DeliveryResourceRollbackResponse"
        )
        assert delivery_resource_governance_schema["$ref"].endswith(
            "/DeliveryResourceGovernanceRequestsResponse"
        )
        assert delivery_resource_governance_create_schema["$ref"].endswith(
            "/DeliveryResourceGovernanceMutationResponse"
        )
        assert delivery_resource_governance_review_schema["$ref"].endswith(
            "/DeliveryResourceGovernanceMutationResponse"
        )
        assert delivery_resource_governance_escalate_schema["$ref"].endswith(
            "/DeliveryResourceGovernanceEscalationResponse"
        )
        assert solution_schema["$ref"].endswith("/SolutionDeliveryReadinessResponse")
        assert launch_schema["$ref"].endswith("/ProductLaunchReadinessResponse")
        assert detail_schema["$ref"].endswith("/CustomerProjectDetailResponse")
        assert upsert_schema["$ref"].endswith("/CustomerProjectMutationResponse")
        assert object_upsert_schema["$ref"].endswith(
            "/CustomerProjectManagedObjectMutationResponse"
        )
        assert object_delete_schema["$ref"].endswith(
            "/CustomerProjectManagedObjectMutationResponse"
        )
        assert history_schema["$ref"].endswith("/CustomerProjectHistoryResponse")
        assert rollback_schema["$ref"].endswith("/CustomerProjectRollbackResponse")
        assert archive_schema["$ref"].endswith("/CustomerProjectArchiveResponse")
        assert execution_schema["$ref"].endswith(
            "/CustomerProjectExecutionBindingsResponse"
        )
        assert execution_rehearsal_schema["$ref"].endswith(
            "/CustomerProjectExecutionRehearsalResponse"
        )
        assert report_schema["$ref"].endswith("/CustomerProjectAcceptanceReportResponse")
        assert evidence_schema["$ref"].endswith("/CustomerProjectOnsiteEvidenceResponse")
        assert evidence_register_schema["$ref"].endswith(
            "/CustomerProjectOnsiteEvidenceRegisterResponse"
        )
        assert closure_schema["$ref"].endswith("/CustomerProjectAcceptanceClosureResponse")
        assert acceptance_review_schema["$ref"].endswith(
            "/CustomerProjectAcceptanceReviewRegisterResponse"
        )
        assert signoff_schema["$ref"].endswith("/CustomerProjectSignoffResponse")
        assert signoff_register_schema["$ref"].endswith(
            "/CustomerProjectCustomerSignoffRegisterResponse"
        )
        assert package_export_schema["$ref"].endswith(
            "/CustomerProjectPackageExportResponse"
        )
        assert dossier_export_schema["$ref"].endswith(
            "/CustomerProjectAcceptanceDossierExportResponse"
        )
        assert proposal_export_schema["$ref"].endswith(
            "/CustomerProjectProposalBundleExportResponse"
        )
        assert package_import_schema["$ref"].endswith(
            "/CustomerProjectPackageImportResponse"
        )
        assert package_verify_schema["$ref"].endswith(
            "/CustomerProjectPackageVerifyResponse"
        )
        assert package_diff_schema["$ref"].endswith("/CustomerProjectPackageDiffResponse")
        assert proposal_verify_schema["$ref"].endswith(
            "/CustomerProjectProposalBundleVerifyResponse"
        )
        assert dossier_verify_schema["$ref"].endswith(
            "/CustomerProjectAcceptanceDossierVerifyResponse"
        )
        skill_schema_refs = {
            ("/api/skill-growth/backlog", "get"): "SkillGrowthBacklogResponse",
            (
                "/api/skill-growth/backlog/{candidate_id}",
                "post",
            ): "SkillGrowthMutationResponse",
            (
                "/api/skill-growth/backlog/{candidate_id}/draft",
                "post",
            ): "SkillGrowthDraftResponse",
            ("/api/skills/generated", "get"): "GeneratedSkillsResponse",
            ("/api/skill-packages", "get"): "SkillPackageCatalogResponse",
            ("/api/skill-packages", "post"): "SkillPackageMutationResponse",
            (
                "/api/skill-packages/{package_id}/skills/{skill_name}",
                "post",
            ): "SkillPackageMutationResponse",
            (
                "/api/skill-packages/{package_id}/history",
                "get",
            ): "SkillPackageHistoryResponse",
            (
                "/api/skill-packages/{package_id}/release",
                "post",
            ): "SkillPackageMutationResponse",
            (
                "/api/skill-packages/{package_id}/rollback",
                "post",
            ): "SkillPackageMutationResponse",
            (
                "/api/skills/generated/{skill_name}/validation",
                "get",
            ): "GeneratedSkillValidationResponse",
            (
                "/api/skills/generated/{skill_name}/preview",
                "get",
            ): "GeneratedSkillPreviewResponse",
            (
                "/api/skills/generated/{skill_name}/review",
                "post",
            ): "GeneratedSkillReviewResponse",
        }
        for (path, method), schema_name in skill_schema_refs.items():
            schema = paths[path][method]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            assert schema["$ref"].endswith(f"/{schema_name}")
        scene_and_vision_schema_refs = {
            ("/api/scenario-intents", "get"): "ScenarioIntentCatalogResponse",
            (
                "/api/scenario-intents/preview",
                "post",
            ): "ScenarioIntentPreviewResponse",
            ("/api/vision/snapshot", "get"): "VisionSnapshotResponse",
            ("/api/vision/analyze", "post"): "VisionAnalyzeResponse",
            ("/api/vision/captures", "get"): "VisionCaptureListResponse",
            (
                "/api/vision/captures/{capture_id}",
                "get",
            ): "VisionCaptureDetailResponse",
            (
                "/api/vision/captures/{capture_id}",
                "delete",
            ): "VisionCaptureDeleteResponse",
        }
        for (path, method), schema_name in scene_and_vision_schema_refs.items():
            schema = paths[path][method]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            assert schema["$ref"].endswith(f"/{schema_name}")
        monitor_schema_refs = {
            ("/health", "get"): "HealthSnapshotResponse",
            ("/trace", "get"): "TraceSnapshotResponse",
            ("/api/status", "get"): "SystemStatusResponse",
            ("/api/dashboard/pages", "get"): "DashboardPageRegistryResponse",
        }
        for (path, method), schema_name in monitor_schema_refs.items():
            schema = paths[path][method]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            assert schema["$ref"].endswith(f"/{schema_name}")

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
        HealthSnapshotResponse.model_validate(response.json())

    def test_status_and_trace_endpoints_return_product_contracts(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        status = client.get("/api/status")
        trace = client.get("/trace")

        assert status.status_code == 200
        status_payload = status.json()
        SystemStatusResponse.model_validate(status_payload)
        assert "timestamp" in status_payload
        assert "perception" in status_payload
        assert "memory" in status_payload

        assert trace.status_code == 200
        trace_payload = trace.json()
        TraceSnapshotResponse.model_validate(trace_payload)
        assert "summary" in trace_payload
        assert "recent" in trace_payload

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
