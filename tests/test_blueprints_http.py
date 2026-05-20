"""HTTP tests for blueprint delivery-package routes."""

from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestBlueprintsHttp:
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


    def test_blueprint_detail_endpoint_returns_one_delivery_package_by_alias(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/blueprints/park")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["blueprint"]["name"] == "edge_robot"
        assert data["blueprint"]["delivery_package"]["package_id"] == "blueprint.edge_robot"
        assert data["policy"]["site_validation_required_before_customer_claim"] is True


    def test_blueprint_detail_endpoint_reports_available_names_for_unknown_blueprint(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/blueprints/not-a-blueprint")

        assert response.status_code == 404
        data = response.json()
        assert data["reason"] == "blueprint_not_found"
        assert "edge_robot" in data["available"]


    def test_blueprint_delivery_package_endpoint_returns_customer_handoff_by_alias(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/blueprints/park/delivery-package")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["blueprint"] == "edge_robot"
        assert data["delivery_package"]["package_id"] == "blueprint.edge_robot"
        assert (
            data["delivery_package"]["operator_runbook"]["start"]
            == "python -m askme.blueprints.presets.edge_robot"
        )
        assert data["policy"]["delivery_package_is_customer_handoff"] is True
        assert data["policy"]["site_validation_required_before_customer_claim"] is True


    def test_blueprint_delivery_package_endpoint_reports_available_names_for_unknown_blueprint(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/blueprints/not-a-blueprint/delivery-package")

        assert response.status_code == 404
        data = response.json()
        assert data["reason"] == "blueprint_not_found"
        assert "edge_robot" in data["available"]
