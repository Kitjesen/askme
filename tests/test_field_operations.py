"""Executable field operation workflow tests."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askme.cognition import WorldStateService
from askme.health_server import (
    build_health_snapshot,
    create_health_app,
    sign_field_runtime_callback_payload,
)
from askme.pipeline.field_operations import FieldOperationsService, sign_field_device_payload
from askme.runtime.field_callbacks import build_field_runtime_callback_sequence
from askme.runtime.handoff import RuntimeHandoffService


class _FakeDispatcher:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_delivery_report = []

    def dispatch(self, message, *, severity="info", topic="", payload=None):
        self.calls.append(
            {
                "message": message,
                "severity": severity,
                "topic": topic,
                "payload": payload or {},
                "config": self.kwargs.get("config", {}),
            }
        )
        self.last_delivery_report = [
            {"channel": "dingtalk", "status": "sent", "reason": ""},
            {"channel": "log", "status": "sent", "reason": ""},
        ]
        return ["dingtalk", "log"]


class _FakeLLM:
    async def chat(self, messages):
        return "你好，我可以带你去北门停车场，请跟在我侧后方。"


class _UnsafeLLM:
    async def chat(self, messages):
        return "危险！请立即撤离并报警处理。"


class _FakeIncidentMemory:
    def __init__(self):
        self.anomalies = []
        self.observations = []
        self.saved = False

    def record_anomaly(self, location, description, coords=None):
        self.anomalies.append(
            {"location": location, "description": description, "coords": coords}
        )

    def record_observation(self, location, description, coords=None):
        self.observations.append(
            {"location": location, "description": description, "coords": coords}
        )

    def save(self):
        self.saved = True


def _health_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="test",
        model_name="test-model",
        metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
        active_skills=[],
        voice_status={"enabled": True, "pipeline_ok": True},
    )


def _service(tmp_path: Path, **config) -> FieldOperationsService:
    _FakeDispatcher.calls = []
    cfg = {
        "archive_path": str(tmp_path / "field-events.jsonl"),
        "dingtalk_webhooks": {
            "security": "http://security.local/ding",
            "cleaning": "http://cleaning.local/ding",
        },
    }
    cfg.update(config)
    return FieldOperationsService(
        config=cfg,
        alert_dispatcher_factory=_FakeDispatcher,
        llm_client=config.get("llm_client"),
    )


def _runtime_world() -> WorldStateService:
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 86,
            "estop_active": False,
            "localized": True,
        },
        stale_after_s=60.0,
    )
    return world


@pytest.mark.asyncio
async def test_fire_event_dispatches_and_archives(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "配电间门口",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert result["accepted"] is True
    event = result["event"]
    assert event["incident_topic"] == "safety.fire_or_smoke"
    assert event["notification_group"] == "security"
    assert event["evidence_media"] == [
        {
            "type": "image",
            "source_key": "image_path",
            "path": "artifacts/evidence/smoke.jpg",
            "preview_url": "/api/field/evidence?path=artifacts%2Fevidence%2Fsmoke.jpg",
            "label": "现场照片",
        }
    ]
    assert event["sent_channels"] == ["dingtalk", "log"]
    assert event["delivery_report"] == [
        {"channel": "dingtalk", "status": "sent", "reason": ""},
        {"channel": "log", "status": "sent", "reason": ""},
    ]
    assert event["incident_state"] == "active"
    assert event["incident_stage"] == "operator"
    workflow = event["incident_workflow"]
    assert workflow["state"] == "active"
    assert {stage["stage"]: stage["status"] for stage in workflow["stages"]} == {
        "admission": "not_required",
        "assessment": "accepted",
        "notification": "sent",
        "voice": "queued",
        "robot_motion": "policy_ready",
        "operator": "pending",
        "archive": "written",
        "memory": "not_connected",
    }
    assert _FakeDispatcher.calls[0]["config"]["dingtalk_webhook"] == "http://security.local/ding"
    listed = service.list_payload()
    assert listed["total"] == 1
    assert listed["filtered_total"] == 1
    assert listed["filter"] == {
        "status": "",
        "notification_group": "",
        "needs_attention": False,
    }
    assert listed["events"][0]["evidence_media"][0]["label"] == "现场照片"
    assert listed["events"][0]["incident_workflow"]["stage"] == "operator"
    assert listed["events"][0]["sla"]["state"] in {"active", "due_soon"}
    assert listed["summary"]["needs_attention"] == 1
    assert listed["summary"]["overdue"] == 0
    assert listed["summary"]["by_notification_group"]["security"] == 1
    assert service.list_payload(needs_attention=True)["filtered_total"] == 1
    assert service.list_payload(notification_group="cleaning")["filtered_total"] == 0


@pytest.mark.asyncio
async def test_field_event_sla_marks_overdue_open_events(tmp_path: Path):
    service = _service(tmp_path)

    await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "image_path": "artifacts/evidence/car.jpg",
            "created_at": time.time() - 7200,
        }
    )

    listed = service.list_payload()

    assert listed["summary"]["overdue"] == 1
    assert listed["events"][0]["sla"]["state"] == "overdue"
    assert listed["events"][0]["sla"]["remaining_s"] < 0


@pytest.mark.asyncio
async def test_p0_event_close_requires_supervisor_approval(tmp_path: Path):
    service = _service(tmp_path)

    created = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "配电间门口",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    blocked = service.close_payload(event_id, {"operator_id": "guard-1", "note": "误报"})
    assert blocked["closed"] is False
    assert blocked["reason"] == "close_requires_supervisor_approval"
    assert blocked["event"]["close_approval_required"] is True
    assert blocked["event"]["action_audit"][-1]["reason"] == "close_requires_supervisor_approval"

    requested = service.request_close_payload(
        event_id,
        {"operator_id": "guard-1", "note": "ready for supervisor review"},
    )
    assert requested["requested"] is True
    assert requested["event"]["status"] == "pending_close_approval"
    assert requested["event"]["close_requested_by"] == "guard-1"

    closed = service.close_payload(
        event_id,
        {
            "operator_id": "guard-1",
            "note": "现场排查后关闭",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
            "approval_note": "确认无火情",
        },
    )
    assert closed["closed"] is True
    assert closed["event"]["incident_state"] == "closed"
    assert closed["event"]["incident_workflow"]["stage"] == "closed"
    assert closed["event"]["close_approval"]["supervisor_id"] == "supervisor-1"
    report = service.event_report_payload(event_id)["report"]
    assert report["close_approval"]["approved"] is True
    assert "close_requested" in {item["type"] for item in report["timeline"]}


@pytest.mark.asyncio
async def test_closed_incident_writes_memory_when_enabled(tmp_path: Path):
    memory = _FakeIncidentMemory()
    service = _service(
        tmp_path,
        incident_memory_enabled=True,
        incident_memory_service=memory,
    )

    created = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "power-room-door",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    closed = service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "confirmed and handed over",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    assert closed["closed"] is True
    delivery = closed["event"]["memory_delivery"]
    assert delivery["status"] == "written"
    assert delivery["target"] == "site_knowledge"
    assert delivery["kind"] == "anomaly"
    assert memory.saved is True
    assert memory.anomalies
    assert memory.anomalies[0]["location"] == "power-room-door"
    assert event_id in memory.anomalies[0]["description"]
    assert closed["event"]["incident_workflow"]["stages"][-1]["status"] == "written"


@pytest.mark.asyncio
async def test_high_risk_close_requires_authorized_operator_and_supervisor(tmp_path: Path):
    service = _service(tmp_path)

    created = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "power-room-door",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    denied_request = service.request_close_payload(
        event_id,
        {"operator_id": "visitor-1", "note": "not staff"},
    )
    assert denied_request["requested"] is False
    assert denied_request["reason"] == "operator_not_authorized"
    assert denied_request["required_roles"] == ["operator", "supervisor", "admin"]
    assert denied_request["event"]["action_audit"][-1]["outcome"] == "denied"
    assert denied_request["event"]["action_audit"][-1]["operator_id"] == "visitor-1"

    requested = service.request_close_payload(
        event_id,
        {"operator_id": "security-1", "note": "ready for approval"},
    )
    assert requested["requested"] is True

    non_supervisor = service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "security-1",
        },
    )
    assert non_supervisor["closed"] is False
    assert non_supervisor["reason"] == "supervisor_not_authorized"
    assert non_supervisor["required_roles"] == ["supervisor", "admin"]
    assert non_supervisor["event"]["action_audit"][-1]["reason"] == "supervisor_not_authorized"

    closed = service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )
    assert closed["closed"] is True
    assert closed["event"]["closed_by"] == "security-1"
    assert closed["event"]["close_approval"]["supervisor_id"] == "supervisor-1"
    assert [item["outcome"] for item in closed["event"]["action_audit"]] == [
        "denied",
        "accepted",
        "denied",
        "accepted",
    ]


@pytest.mark.asyncio
async def test_field_event_action_audit_writes_append_only_jsonl(tmp_path: Path):
    audit_path = tmp_path / "field-action-audit.jsonl"
    service = _service(
        tmp_path,
        action_audit={"enabled": True, "path": str(audit_path), "swallow_errors": False},
    )

    created = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "power-room-door",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    service.request_close_payload(event_id, {"operator_id": "visitor-1", "note": "not staff"})
    service.request_close_payload(event_id, {"operator_id": "security-1", "note": "ready"})
    service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert [line["kind"] for line in lines] == ["field_event_action"] * 3
    assert [line["hash_alg"] for line in lines] == ["sha256"] * 3
    assert [line["sequence"] for line in lines] == [1, 2, 3]
    assert lines[0]["prev_hash"] == "GENESIS"
    assert lines[1]["prev_hash"] == lines[0]["record_hash"]
    assert lines[2]["prev_hash"] == lines[1]["record_hash"]
    assert all(line["record_hash"] for line in lines)
    assert [line["audit"]["action"] for line in lines] == [
        "request_close",
        "request_close",
        "close",
    ]
    assert [line["audit"]["outcome"] for line in lines] == ["denied", "accepted", "accepted"]
    assert lines[0]["audit"]["reason"] == "operator_not_authorized"
    assert lines[0]["event_id"] == event_id
    assert lines[-1]["audit"]["supervisor_id"] == "supervisor-1"

    integrity = service.action_audit_integrity_payload()
    assert integrity["valid"] is True
    assert integrity["checked_count"] == 3
    assert integrity["expected_count"] == 3
    assert integrity["latest_hash"] == lines[-1]["record_hash"]


@pytest.mark.asyncio
async def test_field_event_action_audit_detects_tampering(tmp_path: Path):
    audit_path = tmp_path / "field-action-audit.jsonl"
    service = _service(
        tmp_path,
        action_audit={"enabled": True, "path": str(audit_path), "swallow_errors": False},
    )

    created = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "power-room-door",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    service.request_close_payload(event_id, {"operator_id": "security-1", "note": "ready"})
    service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["audit"]["operator_id"] = "attacker"
    lines[0] = json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    integrity = service.action_audit_integrity_payload()
    assert integrity["valid"] is False
    assert any(item["reason"] == "record_hash_mismatch" for item in integrity["failures"])


@pytest.mark.asyncio
async def test_field_event_action_audit_hmac_detects_recomputed_hash_tampering(tmp_path: Path):
    audit_path = tmp_path / "field-action-audit.jsonl"
    service = _service(
        tmp_path,
        action_audit={
            "enabled": True,
            "path": str(audit_path),
            "swallow_errors": False,
            "hmac_secret": "unit-test-secret",
            "signature_key_id": "unit-test-key",
        },
    )

    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main road",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )
    event_id = created["event"]["event_id"]
    service.acknowledge_payload(event_id, {"operator_id": "security-1", "note": "seen"})

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["signature_alg"] == "hmac-sha256"
    assert records[0]["signature_key_id"] == "unit-test-key"
    assert records[0]["record_signature"]
    assert service.action_audit_integrity_payload()["signed"] is True

    tampered = dict(records[0])
    tampered["audit"] = dict(tampered["audit"], note="forged")
    hash_payload = dict(tampered)
    hash_payload.pop("record_hash", None)
    hash_payload.pop("record_signature", None)
    tampered["record_hash"] = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit_path.write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    integrity = service.action_audit_integrity_payload()
    assert integrity["valid"] is False
    assert any(item["reason"] == "record_signature_mismatch" for item in integrity["failures"])


@pytest.mark.asyncio
async def test_field_event_action_audit_detects_truncated_chain(tmp_path: Path):
    audit_path = tmp_path / "field-action-audit.jsonl"
    service = _service(
        tmp_path,
        action_audit={"enabled": True, "path": str(audit_path), "swallow_errors": False},
    )

    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main road",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )
    event_id = created["event"]["event_id"]
    service.acknowledge_payload(event_id, {"operator_id": "security-1", "note": "seen"})
    service.resend_notification_payload(event_id, {"operator_id": "security-1", "note": "retry"})

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_path.write_text(lines[0] + "\n", encoding="utf-8")

    integrity = service.action_audit_integrity_payload()
    assert integrity["valid"] is False
    assert integrity["checked_count"] == 1
    assert integrity["expected_count"] == 2
    assert any(item["reason"] == "audit_count_mismatch" for item in integrity["failures"])


@pytest.mark.asyncio
async def test_field_event_action_requires_explicit_operator_id(tmp_path: Path):
    service = _service(tmp_path)
    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main road",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    denied = service.acknowledge_payload(created["event"]["event_id"], {"note": "anonymous"})

    assert denied["acknowledged"] is False
    assert denied["reason"] == "operator_not_authorized"
    assert denied["operator_id"] == "anonymous"
    assert denied["authorization_reason"] == "operator_identity_required"
    assert denied["event"]["action_audit"][-1]["operator_id"] == "anonymous"


@pytest.mark.asyncio
async def test_field_event_action_audit_failure_blocks_state_change(tmp_path: Path):
    service = _service(
        tmp_path,
        action_audit={"enabled": True, "path": str(tmp_path), "swallow_errors": False},
    )
    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main road",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )
    event_id = created["event"]["event_id"]

    with pytest.raises(OSError):
        service.acknowledge_payload(event_id, {"operator_id": "security-1", "note": "seen"})

    event = service.list_payload()["events"][0]
    assert event["event_id"] == event_id
    assert event["status"] != "acknowledged"
    assert event["action_audit"] == []


@pytest.mark.asyncio
async def test_field_event_can_be_acknowledged_before_close(tmp_path: Path):
    service = _service(tmp_path)

    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    event_id = created["event"]["event_id"]
    ack = service.acknowledge_payload(
        event_id,
        {"operator_id": "security-1", "note": "保安已接单"},
    )

    assert ack["acknowledged"] is True
    assert ack["event"]["status"] == "acknowledged"
    assert ack["event"]["acknowledged_by"] == "security-1"
    assert service.list_payload()["summary"]["acknowledged"] == 1
    assert service.list_payload(needs_attention=True)["filtered_total"] == 0

    close = service.close_payload(event_id, {"operator_id": "security-1", "note": "已处理"})
    close = service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )
    assert close["closed"] is True
    assert close["event"]["status"] == "closed"
    assert service.list_payload()["summary"]["closed"] == 1


@pytest.mark.asyncio
async def test_field_event_notification_can_be_resent_with_audit(tmp_path: Path):
    service = _service(tmp_path)

    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    event_id = created["event"]["event_id"]
    resend = service.resend_notification_payload(
        event_id,
        {"operator_id": "security-1", "note": "钉钉未读，重发一次"},
    )

    assert resend["resent"] is True
    assert resend["sent_channels"] == ["dingtalk", "log"]
    assert len(_FakeDispatcher.calls) == 2
    assert _FakeDispatcher.calls[-1]["payload"]["event_id"] == event_id
    event = service.list_payload()["events"][0]
    assert event["notification_resends"][0]["resent_by"] == "security-1"
    assert event["notification_resends"][0]["delivery_report"][0]["status"] == "sent"
    assert event["action_audit"][-1]["action"] == "resend_notification"
    assert event["action_audit"][-1]["outcome"] == "accepted"


@pytest.mark.asyncio
async def test_field_event_report_summarizes_delivery_sla_and_evidence(tmp_path: Path):
    service = _service(tmp_path)

    created = await service.trigger_payload(
        {
            "scenario_id": "illegal_parking",
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "image_path": "artifacts/evidence/car.jpg",
        }
    )
    event_id = created["event"]["event_id"]
    service.acknowledge_payload(event_id, {"operator_id": "security-1", "note": "已接单"})
    service.close_payload(event_id, {"operator_id": "security-1", "note": "已挪车"})

    service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    report = service.event_report_payload(event_id)

    assert report["found"] is True
    assert report["report"]["title"].endswith("处置报告")
    assert report["report"]["closed_by"] == "security-1"
    assert report["report"]["close_approval"]["supervisor_id"] == "supervisor-1"
    assert report["report"]["sla_met"] is True
    assert report["report"]["resolution_latency_s"] is not None
    assert "created" in {item["type"] for item in report["report"]["timeline"]}
    assert "close_approved" in {item["type"] for item in report["report"]["timeline"]}
    assert report["report"]["notification_attempts"][0]["kind"] == "initial"
    assert report["report"]["evidence_count"] == 1
    assert report["report"]["delivery_statuses"][0]["status"] == "sent"
    assert {item["action"] for item in report["report"]["action_audit"]} >= {
        "acknowledge",
        "close",
    }
    assert "## 通知送达" in report["markdown"]
    assert "## 处置时间线" in report["markdown"]
    assert "## 通知尝试" in report["markdown"]
    assert "action_audit" in report["report"]
    assert "artifacts/evidence/car.jpg" in report["markdown"]


@pytest.mark.asyncio
async def test_p0_event_missing_evidence_is_not_dispatched(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.trigger_payload(
        {
            "scenario_id": "night_stranger_photo",
            "location": "北侧窗户",
            "zone_name": "北侧窗户",
        }
    )

    assert result["accepted"] is False
    assert result["status"] == "needs_evidence"
    assert "image_path" in result["missing_evidence"]
    assert _FakeDispatcher.calls == []
    assert service.list_payload()["events"][0]["status"] == "needs_evidence"


@pytest.mark.asyncio
async def test_needs_evidence_event_cannot_be_closed_by_supervisor(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.trigger_payload(
        {
            "scenario_id": "night_stranger_photo",
            "location": "north-window",
            "zone_name": "north-window",
        }
    )

    event_id = result["event"]["event_id"]
    closed = service.close_payload(
        event_id,
        {
            "operator_id": "security-1",
            "note": "cannot verify",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    assert closed["closed"] is False
    assert closed["reason"] == "event_not_closable"
    assert closed["status"] == "needs_evidence"


@pytest.mark.asyncio
async def test_acknowledge_needs_evidence_preserves_review_status(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.trigger_payload(
        {
            "scenario_id": "night_stranger_photo",
            "location": "north-window",
            "zone_name": "north-window",
        }
    )
    event_id = result["event"]["event_id"]

    ack = service.acknowledge_payload(
        event_id,
        {"operator_id": "security-1", "note": "received but evidence missing"},
    )

    assert ack["acknowledged"] is True
    assert ack["event"]["status"] == "needs_evidence"
    assert ack["event"]["acknowledged_by"] == "security-1"
    assert service.list_payload(needs_attention=True)["filtered_total"] == 1


@pytest.mark.asyncio
async def test_trash_event_routes_to_cleaning_webhook(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.trigger_payload(
        {
            "scenario_id": "trash_bin_full",
            "location": "C区西门",
            "bin_id": "bin-c-02",
            "fill_ratio": "92%",
            "image_path": "artifacts/evidence/trash.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["event"]["notification_group"] == "cleaning"
    assert result["event"]["playbook"]["tts_profile"] == "cleaning_notice"
    assert result["event"]["voice_directive"]["resolved_profile"] == "cleaning_soft"
    assert _FakeDispatcher.calls[0]["config"]["dingtalk_webhook"] == "http://cleaning.local/ding"


@pytest.mark.asyncio
async def test_event_routes_group_specific_dingtalk_secret(tmp_path: Path):
    service = _service(
        tmp_path,
        dingtalk_secrets={
            "security": "SEC-security",
            "cleaning": "SEC-cleaning",
        },
    )

    await service.trigger_payload(
        {
            "scenario_id": "trash_bin_full",
            "location": "C区西门",
            "bin_id": "bin-c-02",
            "fill_ratio": "92%",
            "image_path": "artifacts/evidence/trash.jpg",
        }
    )

    assert _FakeDispatcher.calls[0]["config"]["dingtalk_webhook"] == "http://cleaning.local/ding"
    assert _FakeDispatcher.calls[0]["config"]["dingtalk_secret"] == "SEC-cleaning"


@pytest.mark.asyncio
async def test_event_uses_security_dingtalk_secret_fallback(tmp_path: Path):
    service = _service(
        tmp_path,
        dingtalk_secrets={"security": "SEC-security"},
    )

    await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "配电间门口",
            "temperature_c": 68,
            "smoke_level": "high",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert _FakeDispatcher.calls[0]["config"]["dingtalk_webhook"] == "http://security.local/ding"
    assert _FakeDispatcher.calls[0]["config"]["dingtalk_secret"] == "SEC-security"


@pytest.mark.asyncio
async def test_notification_smoke_test_uses_group_config(tmp_path: Path):
    service = _service(
        tmp_path,
        dingtalk_secrets={"security": "SEC-security"},
    )

    result = await service.test_notification_payload(
        {
            "notification_group": "security",
            "operator_id": "op-1",
            "message": "测试保安群通知",
        }
    )

    assert result["sent"] is True
    assert result["status"] == "sent"
    assert result["secret_configured"] is True
    assert _FakeDispatcher.calls[0]["topic"] == "field.notification_test"
    assert _FakeDispatcher.calls[0]["config"]["dingtalk_secret"] == "SEC-security"


@pytest.mark.asyncio
async def test_notification_smoke_test_rejects_unknown_group(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.test_notification_payload({"notification_group": "sales"})

    assert result["sent"] is False
    assert result["status"] == "invalid_group"
    assert _FakeDispatcher.calls == []


def test_notification_preflight_blocks_placeholder_config_without_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_SECRET", raising=False)
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "dingtalk_webhooks": {"security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}"},
            "dingtalk_secrets": {"security": "${ASKME_DINGTALK_SECURITY_SECRET}"},
        }
    )

    payload = service.notification_preflight_payload(groups=["security"])

    assert payload["status"] == "blocked"
    assert payload["groups"]["security"]["webhook_configured"] is False
    assert payload["groups"]["security"]["secret_configured"] is False
    assert "ASKME_DINGTALK_SECURITY_WEBHOOK" in payload["groups"]["security"]["missing_env"]
    assert "ASKME_DINGTALK_SECURITY_SECRET" in payload["groups"]["security"]["missing_env"]


def test_notification_preflight_resolves_env_placeholders(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASKME_DINGTALK_SECURITY_WEBHOOK", "https://oapi.dingtalk.com/robot/send?access_token=x")
    monkeypatch.setenv("ASKME_DINGTALK_SECURITY_SECRET", "SEC-test")
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "dingtalk_webhooks": {"security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}"},
            "dingtalk_secrets": {"security": "${ASKME_DINGTALK_SECURITY_SECRET}"},
        }
    )

    payload = service.notification_preflight_payload(groups=["security"])

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["groups"]["security"]["webhook_configured"] is True
    assert payload["groups"]["security"]["secret_configured"] is True
    assert payload["groups"]["security"]["missing_env"] == []


@pytest.mark.asyncio
async def test_camera_vehicle_ingest_uses_map_zone_for_illegal_parking(tmp_path: Path):
    service = _service(
        tmp_path,
        site_map={
            "zones": {
                "main-road-1": {
                    "name": "B区主通道",
                    "type": "main_channel",
                    "parking_allowed": False,
                }
            }
        },
    )

    result = await service.ingest_payload(
        {
            "source": "camera",
            "observed_at": time.time(),
            "zone_id": "main-road-1",
            "detections": [{"label": "vehicle", "confidence": 0.93}],
            "duration_s": 180,
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "illegal_parking"
    assert result["event"]["location"] == "B区主通道"


@pytest.mark.asyncio
async def test_sensor_ingest_triggers_fire_or_smoke(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.ingest_payload(
        {
            "source": "sensor",
            "observed_at": time.time(),
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "配电间",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "fire_or_smoke"
    assert result["event"]["incident_topic"] == "safety.fire_or_smoke"
    assert result["event"]["playbook"]["robot_motion_policy"] == "retreat_to_safe_distance"
    assert result["event"]["playbook"]["tts_profile"] == "emergency_alert"
    assert "temperature_c" in result["event"]["playbook"]["evidence_policy"]
    assert result["event"]["voice_directive"]["resolved_profile"] == "emergency_short"
    assert result["event"]["voice_directive"]["playback_mode"] == "immediate"


@pytest.mark.asyncio
async def test_domestic_camera_event_name_triggers_illegal_parking(tmp_path: Path):
    service = _service(
        tmp_path,
        site_map={
            "zones": {
                "main-road-1": {
                    "name": "主通道",
                    "type": "main_channel",
                    "parking_allowed": False,
                }
            }
        },
    )

    result = await service.ingest_payload(
        {
            "eventType": "车辆违停",
            "cameraIndexCode": "cam-main-road-01",
            "timestamp": time.time(),
            "zone_id": "main-road-1",
            "duration_s": 180,
            "pictureUrl": "artifacts/evidence/domestic-parking.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "illegal_parking"
    assert result["normalized"]["camera_id"] == "cam-main-road-01"
    assert result["normalized"]["detections"][0]["label"] == "vehicle"
    assert result["event"]["location"] == "主通道"


@pytest.mark.asyncio
async def test_night_stranger_photo_uses_photo_evidence(tmp_path: Path):
    service = _service(
        tmp_path,
        site_map={
            "zones": {
                "window-corner-1": {
                    "name": "北侧窗边",
                    "type": "window",
                    "parking_allowed": False,
                }
            }
        },
    )

    result = await service.ingest_payload(
        {
            "alarmType": "夜间陌生人拍照",
            "cameraIndexCode": "cam-window-01",
            "timestamp": time.time(),
            "zone_id": "window-corner-1",
            "is_night": True,
            "known_person": False,
            "duration_s": 4,
            "snapshotUrl": "artifacts/evidence/night-photo.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "night_stranger_photo"
    assert result["normalized"]["taking_photo"] is True
    labels = {item["label"] for item in result["normalized"]["detections"]}
    assert {"person", "phone"}.issubset(labels)
    assert result["event"]["notification_group"] == "security"


@pytest.mark.asyncio
async def test_domestic_trash_bin_alarm_triggers_cleaning_workflow(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.ingest_payload(
        {
            "eventType": "垃圾桶满溢",
            "cameraIndexCode": "cam-bin-17",
            "timestamp": time.time(),
            "binId": "bin-17",
            "location": "花园出口",
            "imageUrl": "artifacts/evidence/bin-full.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "trash_bin_full"
    assert result["normalized"]["bin_id"] == "bin-17"
    assert result["event"]["notification_group"] == "cleaning"
    assert result["event"]["incident_topic"] == "sanitation.trash_bin_full"


@pytest.mark.asyncio
async def test_domestic_crowd_alarm_triggers_security_workflow(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.ingest_payload(
        {
            "eventType": "人员聚集",
            "cameraIndexCode": "cam-plaza-01",
            "timestamp": time.time(),
            "personCount": 8,
            "duration_min": 35,
            "location": "中心广场",
            "imageUrl": "artifacts/evidence/crowd-domestic.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["scenario_id"] == "crowd_gathering"
    assert result["normalized"]["person_count"] == 8
    assert result["event"]["notification_group"] == "security"


@pytest.mark.asyncio
async def test_signed_registered_device_ingest_is_trusted(tmp_path: Path):
    service = _service(
        tmp_path,
        require_trusted_devices=True,
        device_registry={
            "smoke-01": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "device-secret",
                "require_signature": True,
            }
        },
    )
    body = {
        "source": "sensor",
        "device_id": "smoke-01",
        "observed_at": time.time(),
        "device_signature_timestamp": time.time(),
        "sensor": {"temperature_c": 72, "smoke_level": 0.9},
        "location": "Power Room",
        "image_path": "artifacts/evidence/smoke.jpg",
    }
    body["device_signature"] = sign_field_device_payload(body, secret="device-secret")

    result = await service.ingest_payload(body)

    assert result["accepted"] is True
    assert result["normalized"]["device_trust"]["status"] == "trusted"
    assert result["normalized"]["device_trust"]["signature_verified"] is True
    assert result["normalized"]["scenario_id"] == "fire_or_smoke"


@pytest.mark.asyncio
async def test_device_status_payload_reports_online_and_never_seen_devices(tmp_path: Path):
    service = _service(
        tmp_path,
        require_trusted_devices=True,
        device_offline_after_s=600,
        device_registry={
            "smoke-01": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "device-secret",
                "require_signature": True,
            },
            "camera-01": {
                "allowed_sources": ["camera"],
                "hmac_secret": "camera-secret",
                "require_signature": True,
            },
        },
    )
    body = {
        "source": "sensor",
        "device_id": "smoke-01",
        "observed_at": time.time(),
        "device_signature_timestamp": time.time(),
        "sensor": {"temperature_c": 72, "smoke_level": 0.9},
        "location": "Power Room",
        "image_path": "artifacts/evidence/smoke.jpg",
    }
    body["device_signature"] = sign_field_device_payload(body, secret="device-secret")

    await service.ingest_payload(body)
    payload = service.device_status_payload()

    devices = {item["device_id"]: item for item in payload["devices"]}
    assert payload["summary"]["registered"] == 2
    assert payload["summary"]["observed"] == 1
    assert devices["smoke-01"]["status"] == "online"
    assert devices["smoke-01"]["signature_verified"] is True
    assert devices["camera-01"]["status"] == "never_seen"


@pytest.mark.asyncio
async def test_trusted_device_ingest_rejects_missing_signature(tmp_path: Path):
    service = _service(
        tmp_path,
        require_trusted_devices=True,
        device_registry={
            "smoke-01": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "device-secret",
                "require_signature": True,
            }
        },
    )

    result = await service.ingest_payload(
        {
            "source": "sensor",
            "device_id": "smoke-01",
            "observed_at": time.time(),
            "device_signature_timestamp": time.time(),
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "Power Room",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert result["accepted"] is False
    assert result["reason"] == "device_not_trusted"
    assert result["normalized"]["device_trust"]["reason"] == "missing_device_signature"


@pytest.mark.asyncio
async def test_trusted_device_ingest_rejects_wrong_source(tmp_path: Path):
    service = _service(
        tmp_path,
        require_trusted_devices=True,
        device_registry={
            "smoke-01": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "device-secret",
                "require_signature": True,
            }
        },
    )
    body = {
        "source": "camera",
        "device_id": "smoke-01",
        "observed_at": time.time(),
        "device_signature_timestamp": time.time(),
        "detections": [{"label": "smoke", "confidence": 0.9}],
        "location": "Power Room",
        "image_path": "artifacts/evidence/smoke.jpg",
    }
    body["device_signature"] = sign_field_device_payload(body, secret="device-secret")

    result = await service.ingest_payload(body)

    assert result["accepted"] is False
    assert result["normalized"]["device_trust"]["reason"] == "device_source_not_allowed"


@pytest.mark.asyncio
async def test_low_risk_service_can_use_llm_narrative(tmp_path: Path):
    service = _service(
        tmp_path,
        llm_narrative_enabled=True,
        llm_client=_FakeLLM(),
    )

    result = await service.trigger_payload(
        {
            "scenario_id": "visitor_escort",
            "location": "游客中心",
            "destination": "北门停车场",
        }
    )

    assert result["accepted"] is True
    assert result["event"]["llm_narrative_used"] is True
    assert result["event"]["llm_narrative_status"] == "used"
    assert result["event"]["llm_narrative_reason"] == "accepted_low_risk_narrative"
    assert "北门停车场" in result["event"]["voice"]
    assert result["event"]["playbook"]["robot_motion_policy"] == "low_speed_escort"
    assert result["event"]["playbook"]["allow_llm_narrative"] is True
    assert result["event"]["voice_directive"]["resolved_profile"] == "visitor_friendly"
    assert result["event"]["voice_directive"]["text"] == result["event"]["voice"]


@pytest.mark.asyncio
async def test_playbook_allowed_incident_can_use_llm_narrative(tmp_path: Path):
    service = _service(
        tmp_path,
        llm_narrative_enabled=True,
        llm_client=_FakeLLM(),
    )

    result = await service.trigger_payload(
        {
            "scenario_id": "trash_bin_full",
            "location": "游客中心门口",
            "bin_id": "bin-17",
            "fill_ratio": "92%",
            "image_path": "artifacts/evidence/bin.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["event"]["playbook"]["allow_llm_narrative"] is True
    assert result["event"]["llm_narrative_status"] == "used"
    assert result["event"]["llm_narrative_used"] is True


@pytest.mark.asyncio
async def test_unsafe_llm_narrative_is_rejected_and_fixed_voice_remains(tmp_path: Path):
    service = _service(
        tmp_path,
        llm_narrative_enabled=True,
        llm_client=_UnsafeLLM(),
    )

    result = await service.trigger_payload(
        {
            "scenario_id": "trash_bin_full",
            "location": "游客中心门口",
            "bin_id": "bin-17",
            "fill_ratio": "92%",
            "image_path": "artifacts/evidence/bin.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["event"]["llm_narrative_used"] is False
    assert result["event"]["llm_narrative_status"] == "rejected"
    assert result["event"]["llm_narrative_reason"] == "adds_high_risk_claim"
    assert "撤离" not in result["event"]["voice"]


@pytest.mark.asyncio
async def test_high_risk_event_skips_llm_narrative_even_when_enabled(tmp_path: Path):
    service = _service(
        tmp_path,
        llm_narrative_enabled=True,
        llm_client=_FakeLLM(),
    )

    result = await service.trigger_payload(
        {
            "scenario_id": "fire_or_smoke",
            "location": "配电间门口",
            "temperature_c": 68,
            "smoke_level": 0.9,
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["event"]["llm_narrative_used"] is False
    assert result["event"]["llm_narrative_status"] == "skipped"
    assert result["event"]["llm_narrative_reason"] == "high_risk_event_uses_fixed_playbook"


@pytest.mark.asyncio
async def test_stale_sensor_ingest_is_archived_without_dispatch(tmp_path: Path):
    service = _service(tmp_path, max_input_age_s=5)

    result = await service.ingest_payload(
        {
            "source": "sensor",
            "observed_at": time.time() - 30,
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "配电间",
            "image_path": "artifacts/evidence/smoke.jpg",
        }
    )

    assert result["accepted"] is False
    assert result["reason"] == "sensor_input_not_trusted"
    assert result["event"]["status"] == "needs_review"
    assert result["event"]["freshness_status"] == "stale"
    assert "freshness 不合格" in result["event"]["operator_action"]
    assert _FakeDispatcher.calls == []


@pytest.mark.asyncio
async def test_low_confidence_camera_ingest_requires_review(tmp_path: Path):
    service = _service(
        tmp_path,
        min_detection_confidence=0.75,
        site_map={
            "zones": {
                "main-road-1": {
                    "name": "B区主通道",
                    "type": "main_channel",
                    "parking_allowed": False,
                }
            }
        },
    )

    result = await service.ingest_payload(
        {
            "source": "camera",
            "observed_at": time.time(),
            "zone_id": "main-road-1",
            "detections": [{"label": "vehicle", "confidence": 0.52}],
            "duration_s": 180,
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    assert result["accepted"] is False
    assert result["event"]["status"] == "needs_review"
    assert result["event"]["confidence"] == 0.52
    assert "检测置信度" in result["event"]["operator_action"]
    assert _FakeDispatcher.calls == []


@pytest.mark.asyncio
async def test_duplicate_ingest_does_not_notify_twice(tmp_path: Path):
    service = _service(
        tmp_path,
        dedupe_window_s=120,
        site_map={
            "zones": {
                "main-road-1": {
                    "name": "B区主通道",
                    "type": "main_channel",
                    "parking_allowed": False,
                }
            }
        },
    )
    payload = {
        "source": "camera",
        "observed_at": time.time(),
        "zone_id": "main-road-1",
        "detections": [{"label": "vehicle", "confidence": 0.93}],
        "duration_s": 180,
        "image_path": "artifacts/evidence/car.jpg",
    }

    first = await service.ingest_payload(dict(payload))
    second = await service.ingest_payload(dict(payload, observed_at=time.time()))

    assert first["accepted"] is True
    assert second["accepted"] is True
    assert second["status"] == "duplicate"
    assert second["duplicate_of"] == first["event"]["event_id"]
    assert len(_FakeDispatcher.calls) == 1


def test_field_operations_http_endpoints(tmp_path: Path):
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=_service(tmp_path),
        )
    )

    scenarios = client.get("/api/field/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) >= 9
    help_payload = client.get("/api/field/ingest")
    assert help_payload.status_code == 200
    assert "field-ingest-bridge" in help_payload.json()["bridge_contract"]["dry_run"]
    assert help_payload.json()["examples"]["camera_vehicle"]["detections"][0]["class_id"] == "2"
    assert help_payload.json()["examples"]["domestic_camera_parking"]["eventType"] == "车辆违停"
    assert help_payload.json()["examples"]["domestic_night_photo"]["alarmType"] == "夜间陌生人拍照"
    assert help_payload.json()["examples"]["trash_bin_alarm"]["eventType"] == "垃圾桶满溢"
    readiness = client.get("/api/field/readiness")
    assert readiness.status_code == 200
    assert "gates" in readiness.json()

    trigger = client.post(
        "/api/field/events",
        json={
            "scenario_id": "illegal_parking",
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "image_path": "artifacts/evidence/car.jpg",
        },
    )
    assert trigger.status_code == 200
    event_id = trigger.json()["event"]["event_id"]
    assert trigger.json()["trigger_contract"]["admission_path"] == "field_events_manual"
    assert trigger.json()["event"]["payload"]["trigger_source"] == "operator_manual"
    assert trigger.json()["event"]["payload"]["operator_id"] == "dashboard.operator"

    ingest = client.post(
        "/api/field/ingest",
        json={
            "source": "robot",
            "observed_at": time.time(),
            "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
            "location": "A区东侧",
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["normalized"]["scenario_id"] == "robot_abnormal_incident"
    assert ingest.json()["event"]["incident_topic"] == "actuator.joint_motor_fault"
    assert ingest.json()["event"]["playbook"]["tts_profile"] == "robot_fault"
    assert ingest.json()["event"]["voice_directive"]["resolved_profile"] == "fault_urgent"
    assert ingest.json()["event"]["voice_directive"]["interrupt_current_speech"] is True

    rejected_device_trigger = client.post(
        "/api/field/events",
        json={
            "source": "sensor",
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "配电间",
        },
    )
    assert rejected_device_trigger.status_code == 422
    assert rejected_device_trigger.json()["reason"] == "device_payload_must_use_field_ingest"

    events = client.get("/api/field/events")
    assert events.status_code == 200
    assert events.json()["total"] == 2
    assert events.json()["summary"]["needs_attention"] >= 1
    security_events = client.get("/api/field/events?notification_group=security")
    assert security_events.status_code == 200
    assert security_events.json()["filter"]["notification_group"] == "security"
    assert security_events.json()["filtered_total"] >= 1
    attention_events = client.get("/api/field/events?needs_attention=true")
    assert attention_events.status_code == 200
    assert attention_events.json()["filter"]["needs_attention"] is True

    ack = client.post(
        f"/api/field/events/{event_id}/acknowledge",
        json={"operator_id": "security-1", "note": "已派人查看"},
    )
    assert ack.status_code == 200
    assert ack.json()["event"]["status"] == "acknowledged"

    resend = client.post(
        f"/api/field/events/{event_id}/resend-notification",
        json={"operator_id": "security-1", "note": "重发到保安群"},
    )
    assert resend.status_code == 200
    assert resend.json()["resent"] is True

    requested = client.post(
        f"/api/field/events/{event_id}/request-close",
        json={"operator_id": "security-1", "note": "ready for supervisor"},
    )
    assert requested.status_code == 200
    assert requested.json()["event"]["status"] == "pending_close_approval"

    detail = client.get(f"/api/field/events/{event_id}")
    assert detail.status_code == 200
    detail_event = detail.json()["event"]
    assert detail.json()["found"] is True
    assert detail_event["event_id"] == event_id
    assert detail_event["incident_workflow"]["stages"]
    assert detail_event["incident_stage"] == "operator"
    assert detail_event["sla"]["state"] in {"active", "due_soon", "overdue", "closed"}
    assert detail_event["close_approval_required"] is True
    assert detail_event["evidence_media"][0]["path"] == "artifacts/evidence/car.jpg"
    assert [item["action"] for item in detail_event["action_audit"]][-3:] == [
        "acknowledge",
        "resend_notification",
        "request_close",
    ]

    missing_detail = client.get("/api/field/events/not-a-real-event")
    assert missing_detail.status_code == 404
    assert missing_detail.json()["reason"] == "event_not_found"

    unauthorized_close = client.post(
        f"/api/field/events/{event_id}/close",
        json={
            "operator_id": "visitor-1",
            "note": "not staff",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )
    assert unauthorized_close.status_code == 403
    assert unauthorized_close.json()["reason"] == "operator_missing_permission"

    bad_supervisor = client.post(
        f"/api/field/events/{event_id}/close",
        json={
            "operator_id": "security-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "security-1",
        },
    )
    assert bad_supervisor.status_code == 403
    assert bad_supervisor.json()["reason"] == "operator_missing_permission"

    report = client.get(f"/api/field/events/{event_id}/report")
    assert report.status_code == 200
    assert report.json()["report"]["event_id"] == event_id
    assert "处置报告" in report.json()["markdown"]

    close = client.post(
        f"/api/field/events/{event_id}/close",
        json={"operator_id": "security-1", "note": "已通知车主挪车"},
    )
    assert close.status_code == 403
    assert close.json()["reason"] == "operator_missing_permission"
    close = client.post(
        f"/api/field/events/{event_id}/close",
        json={
            "operator_id": "supervisor-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )
    assert close.status_code == 200
    assert close.json()["event"]["status"] == "closed"

    integrity = client.get("/api/field/audit/integrity")
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
    assert integrity.json()["checked_count"] >= 1
    assert integrity.json()["hash_alg"] == "sha256"

    audit_path = tmp_path / "field-action-audit.jsonl"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_path.write_text(lines[0] + "\n", encoding="utf-8")
    broken_integrity = client.get("/api/field/audit/integrity")
    assert broken_integrity.status_code == 409
    assert broken_integrity.json()["valid"] is False

    notification = client.post(
        "/api/field/notification-test",
        json={"notification_group": "security", "message": "测试保安群通知"},
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert notification.status_code == 200
    assert notification.json()["status"] == "sent"

    invalid = client.post(
        "/api/field/notification-test",
        json={"notification_group": "sales"},
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    assert invalid.status_code == 422


def test_field_evidence_endpoint_serves_only_approved_roots():
    artifact = Path("artifacts/evidence/unit-test-evidence.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("field-evidence", encoding="utf-8")
    client = TestClient(create_health_app(lambda: _health_snapshot()))

    try:
        ok = client.get("/api/field/evidence", params={"path": str(artifact).replace("\\", "/")})
        assert ok.status_code == 200
        assert ok.text == "field-evidence"

        denied = client.get("/api/field/evidence", params={"path": "askme/health_server.py"})
        assert denied.status_code == 404

        traversal = client.get(
            "/api/field/evidence",
            params={"path": "artifacts/../askme/health_server.py"},
        )
        assert traversal.status_code == 404
    finally:
        artifact.unlink(missing_ok=True)


def test_field_event_endpoint_dispatches_voice_directive(tmp_path: Path):
    class Voice:
        def __init__(self):
            self.profiles: list[str] = []
            self.spoken: list[str] = []
            self.playback_started = False

        def set_voice_profile_payload(self, body):
            self.profiles.append(body["profile_id"])
            return {
                "updated": True,
                "active_profile": body["profile_id"],
                "profile": {"profile_id": body["profile_id"]},
            }

        def speak(self, text):
            self.spoken.append(text)

        def start_playback(self):
            self.playback_started = True

    voice = Voice()
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            voice_handler=voice,
        )
    )

    response = client.post(
        "/api/field/events",
        json={
            "scenario_id": "fire_or_smoke",
            "location": "配电间门口",
            "temperature_c": 68,
            "smoke_level": 0.9,
            "image_path": "artifacts/evidence/smoke.jpg",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["voice_delivery"]["status"] == "queued"
    assert payload["voice_delivery_record"]["recorded"] is True
    assert payload["event"]["voice_directive"]["resolved_profile"] == "emergency_short"
    assert payload["event"]["voice_delivery"]["status"] == "queued"
    assert payload["event"]["incident_workflow"]["stages"][3]["status"] == "queued"
    assert voice.profiles == ["emergency_short"]
    assert voice.spoken == [payload["event"]["voice_directive"]["text"]]
    assert voice.playback_started is True

    archived = service.list_payload()["events"][0]
    assert archived["voice_delivery"]["status"] == "queued"
    assert archived["incident_workflow"]["stages"][3]["status"] == "queued"
    assert archived["runtime_delivery"]["status"] == "policy_ready"
    assert archived["runtime_delivery"]["reason"] == "runtime_handler_not_configured"
    assert archived["runtime_delivery"]["hardware_dispatch"] is False


def test_field_event_endpoint_submits_runtime_policy_when_handler_is_configured(tmp_path: Path):
    class Runtime:
        def __init__(self):
            self.plans: list[dict] = []

        def submit_plan_payload(self, plan):
            self.plans.append(plan)
            return {
                "accepted": True,
                "profile": "fake",
                "run": {
                    "run_id": "run-field-1",
                    "profile": "fake",
                    "current_state": "completed",
                    "handoff": {"handoff_id": "handoff-field-1"},
                },
            }

    runtime = Runtime()
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            runtime_handler=runtime,
        )
    )

    response = client.post(
        "/api/field/events",
        json={
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main channel",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    delivery = payload["runtime_delivery"]
    assert delivery["status"] == "completed"
    assert delivery["dispatch_mode"] == "task_handoff"
    assert delivery["hardware_dispatch"] is False
    assert delivery["run_id"] == "run-field-1"
    assert delivery["handoff_id"] == "handoff-field-1"
    assert payload["runtime_delivery_record"]["recorded"] is True
    assert runtime.plans
    plan = runtime.plans[0]
    assert plan["handoff_ready"] is True
    assert plan["mission"]["mission"]["mission_type"] == "field_incident_response"
    assert plan["mission"]["mission"]["field_event"]["scenario_id"] == "illegal_parking"
    archived = service.list_payload()["events"][0]
    assert archived["runtime_delivery"]["status"] == "completed"
    assert archived["incident_workflow"]["stages"][4]["status"] == "completed"


def test_field_event_shadow_runtime_callback_roundtrip_updates_archive(tmp_path: Path):
    service = _service(tmp_path)
    runtime = RuntimeHandoffService(world_state=_runtime_world(), profile="shadow")
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            runtime_handler=runtime,
            field_runtime_callback_secret="runtime-secret",
        )
    )

    created = client.post(
        "/api/field/events",
        json={
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main channel",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        },
    )

    assert created.status_code == 200
    body = created.json()
    event_id = body["event"]["event_id"]
    runtime_result = body["runtime_handoff_result"]
    callbacks = build_field_runtime_callback_sequence(
        runtime_result,
        secret="runtime-secret",
    )

    responses = [
        client.post(
            f"/api/field/events/{event_id}/runtime-delivery",
            json=callback,
        )
        for callback in callbacks
    ]

    callback_statuses = [item["status"] for item in callbacks]
    assert [item.status_code for item in responses] == [200] * len(callbacks)
    archived = service.list_payload()["events"][0]
    assert [item["status"] for item in archived["runtime_delivery_receipts"]] == callback_statuses
    assert callback_statuses[-1] == "shadowed"
    assert archived["runtime_delivery"]["status"] == "shadowed"
    assert archived["runtime_delivery"]["runtime_callback_trust"]["status"] == "trusted"
    assert archived["incident_workflow"]["stages"][4]["status"] == "shadowed"


def test_field_event_runtime_delivery_callback_updates_archive(tmp_path: Path):
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
        )
    )
    created = client.post(
        "/api/field/events",
        json={
            "scenario_id": "fire_or_smoke",
            "location": "power room",
            "temperature_c": 68,
            "smoke_level": 0.9,
            "image_path": "artifacts/evidence/smoke.jpg",
        },
    )
    event_id = created.json()["event"]["event_id"]

    callback = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json={
            "status": "failed",
            "dispatch_mode": "task_handoff",
            "robot_motion_policy": "retreat_to_safe_distance",
            "hardware_dispatch": False,
            "run_id": "run-callback-1",
            "handoff_id": "handoff-callback-1",
            "reason": "simulated_runtime_fault",
        },
    )

    assert callback.status_code == 200
    payload = callback.json()
    assert payload["recorded"] is True
    assert payload["runtime_delivery"]["status"] == "failed"
    assert payload["event"]["incident_workflow"]["stages"][4]["status"] == "failed"
    archived = service.list_payload()["events"][0]
    assert archived["runtime_delivery"]["run_id"] == "run-callback-1"
    assert archived["runtime_delivery"]["runtime_callback_trust"]["status"] == "unsigned"
    assert archived["incident_workflow"]["open_gaps"]


def test_field_event_runtime_delivery_callback_requires_valid_signature(tmp_path: Path):
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            field_runtime_callback_secret="runtime-secret",
        )
    )
    created = client.post(
        "/api/field/events",
        json={
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main channel",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        },
    )
    event_id = created.json()["event"]["event_id"]

    unsigned = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json={
            "status": "completed",
            "dispatch_mode": "task_handoff",
            "robot_motion_policy": "retreat_to_safe_distance",
            "hardware_dispatch": False,
            "run_id": "run-unsigned",
            "runtime_signature_timestamp": time.time(),
        },
    )

    assert unsigned.status_code == 403
    assert unsigned.json()["reason"] == "missing_runtime_signature"

    signed_body = {
        "status": "completed",
        "dispatch_mode": "task_handoff",
        "robot_motion_policy": "retreat_to_safe_distance",
        "hardware_dispatch": False,
        "run_id": "run-signed",
        "handoff_id": "handoff-signed",
        "runtime_signature_timestamp": time.time(),
    }
    signed_body["runtime_signature"] = sign_field_runtime_callback_payload(
        signed_body,
        secret="runtime-secret",
    )
    signed = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json=signed_body,
    )

    assert signed.status_code == 200
    payload = signed.json()
    assert payload["runtime_delivery"]["run_id"] == "run-signed"
    trust = payload["runtime_delivery"]["runtime_callback_trust"]
    assert trust["trusted"] is True
    assert trust["signature_verified"] is True
    archived = service.list_payload()["events"][0]
    assert archived["runtime_delivery"]["runtime_callback_trust"]["status"] == "trusted"


def test_field_event_runtime_delivery_callback_rejects_invalid_status(tmp_path: Path):
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            field_runtime_callback_secret="runtime-secret",
        )
    )
    created = client.post(
        "/api/field/events",
        json={
            "scenario_id": "illegal_parking",
            "location": "main road",
            "zone_name": "main channel",
            "plate_number": "A12345",
            "image_path": "artifacts/evidence/car.jpg",
        },
    )
    event_id = created.json()["event"]["event_id"]
    before = service.list_payload()["events"][0]["runtime_delivery"]
    body = {
        "status": "motor_override_now",
        "dispatch_mode": "task_handoff",
        "robot_motion_policy": "retreat_to_safe_distance",
        "run_id": "run-invalid",
        "runtime_signature_timestamp": time.time(),
    }
    body["runtime_signature"] = sign_field_runtime_callback_payload(
        body,
        secret="runtime-secret",
    )

    rejected = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json=body,
    )

    assert rejected.status_code == 422
    assert rejected.json()["reason"] == "invalid_runtime_delivery_status"
    archived = service.list_payload()["events"][0]
    assert archived["runtime_delivery"] == before


def test_field_event_runtime_delivery_callback_is_idempotent(tmp_path: Path):
    service = _service(tmp_path)
    client = TestClient(
        create_health_app(
            lambda: _health_snapshot(),
            field_operations_handler=service,
            field_runtime_callback_secret="runtime-secret",
        )
    )
    created = client.post(
        "/api/field/events",
        json={
            "scenario_id": "fire_or_smoke",
            "location": "power room",
            "temperature_c": 68,
            "smoke_level": 0.9,
            "image_path": "artifacts/evidence/smoke.jpg",
        },
    )
    event_id = created.json()["event"]["event_id"]
    body = {
        "runtime_callback_id": "runtime-callback-001",
        "status": "executing",
        "dispatch_mode": "task_handoff",
        "robot_motion_policy": "retreat_to_safe_distance",
        "hardware_dispatch": False,
        "run_id": "run-idempotent",
        "handoff_id": "handoff-idempotent",
        "runtime_signature_timestamp": time.time(),
    }
    body["runtime_signature"] = sign_field_runtime_callback_payload(
        body,
        secret="runtime-secret",
    )

    first = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json=body,
    )
    second = client.post(
        f"/api/field/events/{event_id}/runtime-delivery",
        json=body,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    duplicate = second.json()
    assert duplicate["duplicate"] is True
    assert duplicate["reason"] == "runtime_callback_already_recorded"
    archived = service.list_payload()["events"][0]
    assert archived["runtime_delivery"]["runtime_callback_id"] == "runtime-callback-001"
    assert len(archived["runtime_delivery_receipts"]) == 1
