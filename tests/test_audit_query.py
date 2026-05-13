import json
from pathlib import Path

from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.audit import AuditExportService, AuditQueryService
from askme.audit.query import AuditPaths
from askme.health_server import create_health_app
from askme.skills.audit import SkillAuditLog


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )


def _runtime_snapshot() -> dict:
    return {"status": "ok", "components": {}}


def test_audit_query_unifies_skill_field_and_runtime_records(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    runtime_path = tmp_path / "runtime-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="approved",
        event_type="governance",
        operator_id="supervisor-1",
        action="approve",
        result_preview="approved wayfinding skill",
    )
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-1",
                "audit": {
                    "at": 10,
                    "action": "close",
                    "outcome": "accepted",
                    "operator_id": "security-1",
                    "reason": "handled",
                },
            }
        ],
    )
    _write_jsonl(
        runtime_path,
        [
            {
                "kind": "operator_action",
                "created_at": 20,
                "run_id": "run-1",
                "action": {
                    "action": "pause",
                    "outcome": "accepted",
                    "operator_id": "guard-1",
                    "reason": "visitor in path",
                },
            }
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(
            skill_audit=skill_path,
            field_action_audit=field_path,
            runtime_audit=runtime_path,
        )
    ).query(limit=10)

    assert payload["count"] == 3
    assert payload["summary"]["by_source"] == {"field": 1, "runtime": 1, "skill": 1}
    assert {record["source"] for record in payload["records"]} == {"field", "runtime", "skill"}
    assert any(record["action"] == "approve" for record in payload["records"])


def test_audit_query_filters_by_operator(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="one",
        status="blocked",
        operator_id="operator-a",
        reason="disabled",
    )
    SkillAuditLog(skill_path).append(
        skill_name="two",
        status="ok",
        operator_id="operator-b",
    )

    payload = AuditQueryService(paths=AuditPaths(skill_audit=skill_path)).query(
        operator_id="operator-a",
    )

    assert payload["count"] == 1
    assert payload["records"][0]["operator_id"] == "operator-a"


def test_audit_events_endpoint_requires_and_uses_rbac(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="ok",
        operator_id="dashboard.operator",
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {"operator": ["audit:read"]},
            },
            "operators": {"dashboard.operator": {"roles": ["operator"]}},
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied = client.get("/api/audit/events", params={"actor_id": "ghost.operator"})
    allowed = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["records"][0]["source"] == "skill"


def test_audit_export_writes_signed_manifest_and_jsonl(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="approved",
        operator_id="supervisor-1",
        action="approve",
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {
            "export": {
                "output_dir": str(tmp_path / "exports"),
                "hmac_secret": "export-secret",
                "signature_key_id": "unit-test-key",
            }
        },
    }

    payload = AuditExportService(config).create_export(actor_id="supervisor-1")

    assert payload["ok"] is True
    export = payload["export"]
    assert export["record_count"] == 1
    assert export["sha256"]
    assert export["signature"]
    assert export["signature_key_id"] == "unit-test-key"
    assert Path(export["records_path"]).exists()
    assert Path(export["manifest_path"]).exists()


def test_audit_export_delivery_posts_manifest(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    seen = {}

    def fake_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        seen["url"] = url
        seen["body"] = body
        seen["headers"] = headers
        seen["timeout_s"] = timeout_s
        return {"sent": True, "status_code": 200}

    payload = AuditExportService(
        {
            "skills": {"audit_path": str(skill_path)},
            "audit": {
                "export": {
                    "output_dir": str(tmp_path / "exports"),
                    "hmac_secret": "export-secret",
                    "webhook_url": "http://siem.local/audit",
                }
            },
        },
        post_json=fake_post,
    ).create_export(actor_id="supervisor-1", deliver=True)

    assert payload["delivery"]["sent"] is True
    assert seen["url"] == "http://siem.local/audit"
    assert seen["body"]["type"] == "askme.unified_audit_export"
    assert seen["body"]["manifest"]["record_count"] == 1
    assert seen["headers"]["X-Askme-Audit-Signature"]


def test_audit_export_delivery_failure_queues_and_replays_retry(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    retry_queue = tmp_path / "exports" / "retry.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {
            "export": {
                "output_dir": str(tmp_path / "exports"),
                "retry_queue_path": str(retry_queue),
                "webhook_url": "http://siem.local/audit",
            }
        },
    }

    def failing_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        raise RuntimeError("siem temporarily unavailable")

    failed = AuditExportService(config, post_json=failing_post).create_export(
        actor_id="supervisor-1",
        deliver=True,
    )

    assert failed["delivery"]["sent"] is False
    assert failed["delivery"]["reason"] == "webhook_delivery_failed"
    assert retry_queue.exists()

    seen = {}

    def successful_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        seen["url"] = url
        seen["body"] = body
        return {"sent": True, "status_code": 200}

    service = AuditExportService(config, post_json=successful_post)
    status = service.retry_status()
    replay = service.retry_queued_deliveries()

    assert status["pending"] == 1
    assert status["items"][0]["export_id"] == failed["export"]["export_id"]
    assert replay["attempted"] == 1
    assert replay["sent"] == 1
    assert replay["remaining"] == 0
    assert not retry_queue.exists()
    assert seen["url"] == "http://siem.local/audit"
    assert seen["body"]["manifest"]["export_id"] == failed["export"]["export_id"]


def test_audit_export_endpoint_requires_export_permission(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read"],
                    "supervisor": ["audit:read", "audit:export"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied = client.post("/api/audit/export", json={"operator_id": "dashboard.operator"})
    allowed = client.post("/api/audit/export", json={"operator_id": "supervisor-1"})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert Path(allowed.json()["export"]["manifest_path"]).exists()


def test_audit_export_retry_endpoint_requires_export_permission(tmp_path: Path, monkeypatch) -> None:
    config = {
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read"],
                    "supervisor": ["audit:read", "audit:export"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied_get = client.get("/api/audit/export/retry", params={"actor_id": "dashboard.operator"})
    allowed_get = client.get("/api/audit/export/retry", params={"actor_id": "supervisor-1"})
    denied_post = client.post("/api/audit/export/retry", json={"operator_id": "dashboard.operator"})
    allowed_post = client.post("/api/audit/export/retry", json={"operator_id": "supervisor-1"})

    assert denied_get.status_code == 403
    assert allowed_get.status_code == 200
    assert allowed_get.json()["status"] == "empty"
    assert denied_post.status_code == 403
    assert allowed_post.status_code == 200
    assert allowed_post.json()["status"] == "empty"
