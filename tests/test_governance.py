from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.governance import OperatorDirectory
from askme.health_server import create_health_app


def _runtime_snapshot() -> dict:
    return {
        "status": "ok",
        "components": {},
    }


def _enterprise_config() -> dict:
    return {
        "field_operations": {
            "operator_directory": {
                "mode": "enterprise",
                "identity_provider": "oidc",
                "production_binding_required": True,
                "trusted_identity_headers": {
                    "enabled": True,
                    "operator_id": "x-askme-iam-operator-id",
                    "roles": "x-askme-iam-roles",
                    "display_name": "x-askme-iam-display-name",
                    "source": "x-askme-iam-source",
                    "roles_required": True,
                },
                "permissions": {
                    "operator": ["runtime:pause"],
                    "supervisor": ["runtime:pause", "knowledge:approve"],
                },
            },
            "operators": {},
        }
    }


def test_enterprise_trusted_headers_resolve_authenticated_identity() -> None:
    directory = OperatorDirectory(_enterprise_config())

    identity = directory.resolve_context(
        headers={
            "x-askme-iam-operator-id": "iam.supervisor",
            "x-askme-iam-roles": "supervisor,operator",
            "x-askme-iam-display-name": "IAM Supervisor",
            "x-askme-iam-source": "corp-oidc",
        },
        body={"operator_id": "spoofed.local"},
    )
    decision = directory.authorize(
        None,
        "knowledge:approve",
        headers={
            "x-askme-iam-operator-id": "iam.supervisor",
            "x-askme-iam-roles": "supervisor",
        },
        body={"operator_id": "spoofed.local"},
    )

    assert identity.operator_id == "iam.supervisor"
    assert identity.display_name == "IAM Supervisor"
    assert identity.roles == ("supervisor", "operator")
    assert identity.authenticated is True
    assert decision["allowed"] is True
    assert decision["operator"]["operator_id"] == "iam.supervisor"


def test_enterprise_mode_rejects_body_spoofing_without_trusted_headers() -> None:
    directory = OperatorDirectory(_enterprise_config())

    decision = directory.authorize(
        "spoofed.supervisor",
        "knowledge:approve",
        body={"operator_id": "spoofed.supervisor"},
    )

    assert decision["allowed"] is False
    assert decision["operator"]["known"] is False
    assert decision["operator"]["authenticated"] is False


def test_enterprise_readiness_accepts_external_directory_with_trusted_headers() -> None:
    directory = OperatorDirectory(_enterprise_config())

    readiness = directory.directory_readiness()

    assert readiness["production_ready"] is True
    assert readiness["status"] == "production_ready"
    assert readiness["findings"] == []


def test_http_permission_uses_trusted_iam_identity_over_body_operator(monkeypatch) -> None:
    class RuntimeHandler:
        def __init__(self) -> None:
            self.seen = {}

        def pause_payload(self, run_id: str, *, operator_id: str = "", reason: str = "", risk_acknowledgement: bool = False) -> dict:
            self.seen = {
                "run_id": run_id,
                "operator_id": operator_id,
                "reason": reason,
                "risk_acknowledgement": risk_acknowledgement,
            }
            return {"run": {"run_id": run_id, "current_state": "paused"}}

    runtime = RuntimeHandler()
    monkeypatch.setattr(health_server, "get_config", lambda: _enterprise_config())
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    response = client.post(
        "/api/runtime/runs/run-iam/pause",
        json={
            "operator_id": "spoofed.local",
            "reason": "verified upstream identity",
            "risk_acknowledgement": True,
        },
        headers={
            "x-askme-iam-operator-id": "iam.operator",
            "x-askme-iam-roles": "operator",
        },
    )

    assert response.status_code == 200
    assert runtime.seen["operator_id"] == "iam.operator"
