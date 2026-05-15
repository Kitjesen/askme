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
    gateway = directory.identity_gateway_readiness()

    assert readiness["production_ready"] is True
    assert readiness["status"] == "production_ready"
    assert readiness["findings"] == []
    assert gateway["gate_type"] == "askme.governance.identity_gateway_readiness.v1"
    assert gateway["status"] == "production_ready"
    assert gateway["production_ready"] is True
    assert gateway["identity_mode"] == "enterprise_gateway"
    assert gateway["trusted_gateway_contract"]["required_headers"][0]["claim"] == "operator_id"


def test_demo_directory_identity_gateway_blocks_production_claim() -> None:
    directory = OperatorDirectory({})

    readiness = directory.identity_gateway_readiness()

    assert readiness["status"] == "blocked"
    assert readiness["production_ready"] is False
    assert readiness["identity_mode"] == "demo_operator_directory"
    assert readiness["demo_operator_directory"]["allowed_for"] == ["demo", "lab", "customer_pilot"]
    assert readiness["release_claim"].startswith("只能承诺演示或试点能力")
    assert any(item["code"] == "enterprise_identity_provider_missing" for item in readiness["blockers"])


def test_enterprise_gateway_requires_trusted_identity_headers() -> None:
    config = _enterprise_config()
    config["field_operations"]["operator_directory"]["trusted_identity_headers"]["enabled"] = False
    directory = OperatorDirectory(config)

    readiness = directory.identity_gateway_readiness()

    assert readiness["status"] == "blocked"
    assert readiness["production_ready"] is False
    assert any(item["code"] == "trusted_gateway_headers_disabled" for item in readiness["blockers"])


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


def test_identity_readiness_endpoint_exposes_customer_handoff_gate(monkeypatch) -> None:
    monkeypatch.setattr(health_server, "get_config", lambda: _enterprise_config())
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.get("/api/governance/identity-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["gate_type"] == "askme.governance.identity_gateway_readiness.v1"
    assert payload["status"] == "production_ready"
    assert payload["customer_status"].startswith("企业身份网关已接入")
