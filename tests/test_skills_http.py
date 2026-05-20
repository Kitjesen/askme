"""HTTP tests for skill growth, generated skills, and skill packages."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askme.api.schemas.skills import (
    GeneratedSkillPreviewResponse,
    GeneratedSkillReviewResponse,
    GeneratedSkillsResponse,
    GeneratedSkillValidationResponse,
    SkillGrowthBacklogResponse,
    SkillGrowthDraftResponse,
    SkillPackageCatalogResponse,
    SkillPackageHistoryResponse,
    SkillPackageMutationResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestSkillsHttp:
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
        SkillGrowthBacklogResponse.model_validate(data)
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
        SkillGrowthDraftResponse.model_validate(payload)
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


    def test_generated_skills_endpoint_returns_review_queue(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated")

        assert response.status_code == 200
        data = response.json()
        GeneratedSkillsResponse.model_validate(data)
        assert "records" in data
        assert data["policy"]["approval_required"] is True
        assert data["policy"]["auto_enable_generated_skills"] is False


    def test_skill_packages_endpoint_returns_package_policy(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skill-packages")

        assert response.status_code == 200
        data = response.json()
        SkillPackageCatalogResponse.model_validate(data)
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
        authorized_payload = authorized.json()
        SkillPackageMutationResponse.model_validate(authorized_payload)
        assert authorized_payload["package"]["package_id"] == "fanmu-phase-1"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/skill-growth/backlog/candidate-1",
            "/api/skill-growth/backlog/candidate-1/draft",
            "/api/skill-packages",
            "/api/skill-packages/default-demo/skills/missing-skill",
            "/api/skill-packages/default-demo/release",
            "/api/skill-packages/default-demo/rollback",
            "/api/skills/generated/missing-skill/review",
        ],
    )

    def test_skill_write_routes_reject_non_object_json_body(self, path):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(
            path,
            json=["not-an-object"],
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == "JSON object body required"


    def test_generated_skill_validation_endpoint_for_missing_skill(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated/missing-skill/validation")

        assert response.status_code == 404
        payload = response.json()
        GeneratedSkillValidationResponse.model_validate(payload)
        assert payload["ok"] is False


    def test_generated_skill_preview_endpoint_for_missing_skill(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/skills/generated/missing-skill/preview")

        assert response.status_code == 404
        payload = response.json()
        GeneratedSkillPreviewResponse.model_validate(payload)
        assert payload["ok"] is False


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
        authorized_payload = authorized.json()
        GeneratedSkillReviewResponse.model_validate(authorized_payload)
        assert authorized_payload["error"] == "generated skill not found"


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
        authorized_payload = authorized.json()
        SkillPackageMutationResponse.model_validate(authorized_payload)
        assert authorized_payload["error"] == "generated skill not found"


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
        first_payload = first.json()
        second_payload = second.json()
        SkillPackageMutationResponse.model_validate(first_payload)
        SkillPackageMutationResponse.model_validate(second_payload)
        assert second_payload["package"]["release_version"] == 2

        history = client.get("/api/skill-packages/default-demo/history")
        assert history.status_code == 200
        history_payload = history.json()
        SkillPackageHistoryResponse.model_validate(history_payload)
        assert history_payload["count"] == 2

        rollback = client.post(
            "/api/skill-packages/default-demo/rollback",
            json={"target_version": 1, "note": "rollback test"},
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert rollback.status_code == 200
        rollback_payload = rollback.json()
        SkillPackageMutationResponse.model_validate(rollback_payload)
        package = rollback_payload["package"]
        assert package["release_version"] == 3
        assert package["rollback_of_version"] == 1
        assert package["release_channel"] == "pilot"
        assert package["rollout_percent"] == 25
