"""HTTP tests for agent profile routes."""

from fastapi.testclient import TestClient

from askme.api.schemas.agent_profiles import (
    AgentProfileCatalogResponse,
    AgentProfilePreviewResponse,
    AgentProfileUpsertResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestAgentProfilesHttp:
    def test_agent_profiles_endpoint_returns_product_roles(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/agent-profiles")

        assert response.status_code == 200
        data = response.json()
        schema_payload = AgentProfileCatalogResponse.model_validate(data)
        assert schema_payload.profile_count >= 1
        names = {profile["name"] for profile in data["profiles"]}
        assert "field_operator" in names
        assert "skill_growth_manager" in names
        paths = client.app.openapi()["paths"]
        assert paths["/api/agent-profiles"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/AgentProfileCatalogResponse")
        assert paths["/api/agent-profiles"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/AgentProfileUpsertResponse")
        assert paths["/api/agent-profiles/{profile_name}/preview"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]["$ref"].endswith("/AgentProfilePreviewResponse")


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
        schema_payload = AgentProfileUpsertResponse.model_validate(payload)
        assert schema_payload.ok is True
        assert payload["ok"] is True
        assert payload["profile"]["name"] == "parking_pm"
        assert (tmp_path / ".askme" / "agents" / "parking_pm.md").exists()

        preview = client.get("/api/agent-profiles/parking_pm/preview")
        assert preview.status_code == 200
        schema_preview = AgentProfilePreviewResponse.model_validate(preview.json())
        assert schema_preview.ok is True
        assert preview.json()["profile"]["preloaded_skills"] == ["detect_illegal_parking"]


    def test_agent_profile_upsert_rejects_non_object_json_body(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(
            "/api/agent-profiles",
            json=["parking_pm"],
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == "JSON object body required"


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
