"""HTTP tests for governance routes."""

from fastapi.testclient import TestClient

from askme.api.schemas.governance import (
    AuthorizationDecisionResponse,
    CurrentOperatorResponse,
    IdentityGatewayReadinessResponse,
    OperatorDirectoryResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestGovernanceHttp:
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
        assert dashboard_operator["project_scope"]["project_ids"] == [
            "demo-field-ops",
            "julong-guide-patrol",
        ]
        assert dashboard_operator["project_scope"]["default_project_ids"] == [
            "julong-guide-patrol"
        ]
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


    def test_governance_authorize_rejects_non_object_json_body(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post("/api/governance/authorize", json=["not-an-object"])

        assert response.status_code == 400
        assert response.json()["error"] == "JSON object body required"


    def test_governance_routes_expose_product_response_schemas(self):
        app = create_health_app(lambda: _runtime_snapshot())
        paths = app.openapi()["paths"]
        expected_refs = {
            ("/api/governance/operator-directory", "get"): "OperatorDirectoryResponse",
            ("/api/governance/identity-readiness", "get"): (
                "IdentityGatewayReadinessResponse"
            ),
            ("/api/governance/current-operator", "get"): "CurrentOperatorResponse",
            ("/api/governance/authorize", "post"): "AuthorizationDecisionResponse",
        }
        for (path, method), schema_name in expected_refs.items():
            schema = paths[path][method]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            assert schema["$ref"].endswith(f"/{schema_name}")

        client = TestClient(app)
        OperatorDirectoryResponse.model_validate(
            client.get("/api/governance/operator-directory").json()
        )
        IdentityGatewayReadinessResponse.model_validate(
            client.get("/api/governance/identity-readiness").json()
        )
        CurrentOperatorResponse.model_validate(
            client.get(
                "/api/governance/current-operator",
                params={"operator_id": "supervisor-1"},
            ).json()
        )
        AuthorizationDecisionResponse.model_validate(
            client.post(
                "/api/governance/authorize",
                json={
                    "operator_id": "supervisor-1",
                    "permission": "knowledge:approve",
                },
            ).json()
        )
