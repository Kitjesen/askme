"""HTTP tests for audit routes."""

import pytest
from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestAuditHttp:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/audit/reviews",
            "/api/audit/export",
            "/api/audit/export/retry",
        ],
    )
    def test_audit_write_routes_reject_non_object_json_body(self, path):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.post(
            path,
            json=["not-an-object"],
            headers={"X-Askme-Operator-Id": "admin-1"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == "JSON object body required"
