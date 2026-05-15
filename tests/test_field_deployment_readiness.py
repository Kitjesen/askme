from __future__ import annotations

import asyncio
import json
from pathlib import Path

from askme.audit import AuditReviewService
from askme.pipeline.field_deployment_readiness import build_field_deployment_readiness
from askme.pipeline.field_operations import FieldOperationsService


class _FakeDispatcher:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_delivery_report = [
            {"channel": "dingtalk", "status": "sent", "reason": ""},
            {"channel": "log", "status": "sent", "reason": ""},
        ]

    def dispatch(self, message, *, severity="info", topic="", payload=None):
        _ = message, severity, topic, payload
        return ["dingtalk", "log"]


def _write_runtime_roundtrip_report(
    path: Path,
    *,
    mode: str = "live",
    final_status: str = "completed",
) -> None:
    path.write_text(
        json.dumps({
            "ok": True,
            "mode": mode,
            "callback_status_codes": [200, 200, 200, 200, 200],
            "receipt_count": 5,
            "runtime_statuses": [
                "created",
                "submitted",
                "validating",
                "preflight",
                final_status,
            ],
            "final_runtime_delivery": {
                "status": final_status,
                "runtime_callback_trust": {
                    "trusted": True,
                    "status": "trusted",
                },
            },
        }),
        encoding="utf-8",
    )


def test_field_deployment_readiness_blocks_without_reports(tmp_path: Path) -> None:
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "blocked"
    assert "field scenario evaluation has not passed" in payload["blockers"]
    assert "field ingest HTTP smoke has not passed" in payload["blockers"]
    assert "field voice smoke has not passed" in payload["blockers"]
    assert "field runtime roundtrip smoke has not passed" in payload["blockers"]
    assert "field event archive has no events" in payload["blockers"]
    assert payload["next_actions"]


def test_field_deployment_readiness_blocks_unresolved_unified_audit_review(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "events.jsonl"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "fire_or_smoke"}) + "\n",
        encoding="utf-8",
    )

    payload = build_field_deployment_readiness(
        config={
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
        },
        archive_path=archive,
        webhooks={},
        webhook_secrets={},
        unified_audit={
            "product_summary": {
                "status": "needs_review",
                "record_count": 3,
                "requires_review_count": 1,
                "high_or_critical_count": 1,
            },
            "filtered_total": 3,
            "review_queue": [
                {
                    "record_id": "field:1",
                    "customer_label": "Field event action audit",
                    "severity": "high",
                    "action": "close",
                    "outcome": "denied",
                    "operator_id": "guard-1",
                    "resource_type": "field_event",
                    "resource_id": "field-1",
                    "timestamp": "2026-05-14T10:00:00Z",
                }
            ],
        },
    )

    assert payload["gates"]["unified_audit_review_clear"] is False
    assert "unified audit review queue has unresolved high-risk records" in payload["blockers"]
    assert payload["unified_audit"]["status"] == "needs_review"
    assert payload["unified_audit"]["requires_review_count"] == 1
    assert payload["unified_audit"]["review_queue"][0]["record_id"] == "field:1"
    assert any("Unified Audit" in item for item in payload["next_actions"])


def test_field_deployment_readiness_uses_audit_review_decisions(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    field_audit = tmp_path / "field-action-audit.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "fire_or_smoke"}) + "\n",
        encoding="utf-8",
    )
    field_audit.write_text(
        json.dumps({
            "kind": "field_event_action",
            "sequence": 1,
            "event_id": "field-1",
            "audit": {
                "at": 10,
                "action": "close",
                "outcome": "denied",
                "operator_id": "guard-1",
                "reason": "supervisor_not_authorized",
            },
        })
        + "\n",
        encoding="utf-8",
    )
    service = FieldOperationsService(
        config={
            "archive_path": str(archive),
            "action_audit": {"path": str(field_audit), "enabled": False},
            "audit_review_path": str(review_path),
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
        }
    )
    blocked = service.readiness_payload()

    AuditReviewService(path=review_path).submit(
        record_id="field:1",
        reviewer_id="supervisor-1",
        decision="accepted",
        note="expected denial; close retried with supervisor approval",
        created_at=20,
    )
    cleared = service.readiness_payload()

    assert blocked["gates"]["unified_audit_review_clear"] is False
    assert blocked["unified_audit"]["requires_review_count"] == 1
    assert cleared["gates"]["unified_audit_review_clear"] is True
    assert cleared["gates"]["unified_audit_review_integrity_verified"] is True
    assert cleared["unified_audit"]["requires_review_count"] == 0

    lines = review_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["note"] = "forged"
    review_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    tampered_payload = service.readiness_payload()

    assert tampered_payload["gates"]["unified_audit_review_integrity_verified"] is False
    assert "unified audit review log integrity has not passed" in tampered_payload["blockers"]


def test_field_deployment_readiness_blocks_unhealthy_unified_audit_sources(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "events.jsonl"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "fire_or_smoke"}) + "\n",
        encoding="utf-8",
    )

    payload = build_field_deployment_readiness(
        config={
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
        },
        archive_path=archive,
        webhooks={},
        webhook_secrets={},
        unified_audit={
            "product_summary": {
                "status": "auditable",
                "record_count": 3,
                "requires_review_count": 0,
                "high_or_critical_count": 0,
            },
            "filtered_total": 3,
            "review_integrity": {"valid": True, "exists": True, "checked_count": 1, "failures": []},
            "source_health": {
                "field_action_audit": {
                    "exists": True,
                    "readable": True,
                    "valid_record_count": 1,
                    "invalid_record_count": 2,
                    "path": str(tmp_path / "field-action-audit.jsonl"),
                },
                "runtime_audit": {
                    "exists": True,
                    "readable": False,
                    "valid_record_count": 0,
                    "invalid_record_count": 0,
                    "path": str(tmp_path / "runtime-audit.jsonl"),
                    "error": "permission denied",
                },
            },
        },
    )

    assert payload["gates"]["unified_audit_sources_healthy"] is False
    assert "unified audit sources have unreadable or invalid records" in payload["blockers"]
    assert payload["unified_audit"]["invalid_source_count"] == 1
    assert payload["unified_audit"]["unreadable_source_count"] == 1
    assert payload["unified_audit"]["unhealthy_sources"][0]["source"] == "field_action_audit"
    assert any("Audit Source Health" in item for item in payload["next_actions"])


def test_field_deployment_readiness_reports_lab_ready_with_sample_smoke(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    scenario_report = tmp_path / "scenario.json"
    smoke_report = tmp_path / "smoke.json"
    voice_smoke_report = tmp_path / "voice-smoke.json"
    notification_smoke_report = tmp_path / "notification-smoke.json"
    runtime_roundtrip_report = tmp_path / "runtime-roundtrip.json"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "illegal_parking", "payload": {"source": "camera"}})
        + "\n",
        encoding="utf-8",
    )
    scenario_report.write_text(
        json.dumps({
            "status": "passed",
            "scenario_count": 10,
            "passed": 10,
            "failed": 0,
            "external_services": False,
            "hardware_dispatch": False,
        }),
        encoding="utf-8",
    )
    smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "event_count": 3,
            "local_server": True,
        }),
        encoding="utf-8",
    )
    voice_smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "local_server": True,
            "live_tts": False,
            "voice_delivery": {"status": "queued"},
            "voice_directive": {
                "requested_profile": "emergency_alert",
                "resolved_profile": "emergency_short",
            },
        }),
        encoding="utf-8",
    )
    notification_smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "local_server": True,
            "external_services": False,
            "sent_groups": ["security", "cleaning", "operations"],
            "collector_request_count": 3,
        }),
        encoding="utf-8",
    )
    _write_runtime_roundtrip_report(
        runtime_roundtrip_report,
        mode="local_server",
        final_status="shadowed",
    )
    service = FieldOperationsService(
        config={
            "archive_path": str(archive),
            "scenario_report_path": str(scenario_report),
            "smoke_report_path": str(smoke_report),
            "voice_smoke_report_path": str(voice_smoke_report),
            "notification_smoke_report_path": str(notification_smoke_report),
            "runtime_roundtrip_report_path": str(runtime_roundtrip_report),
            "dingtalk_webhooks": {"security": "http://security.local/ding"},
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "ready_for_lab"
    assert payload["blockers"] == []
    assert payload["gates"]["scenario_eval_passed"] is True
    assert payload["gates"]["http_smoke_passed"] is True
    assert payload["gates"]["voice_smoke_passed"] is True
    assert payload["gates"]["voice_smoke_uses_live_tts"] is False
    assert payload["gates"]["notification_smoke_passed"] is True
    assert payload["gates"]["notification_smoke_uses_external_services"] is False
    assert payload["gates"]["runtime_roundtrip_smoke_passed"] is True
    assert payload["gates"]["runtime_roundtrip_final_status_verified"] is True
    assert payload["gates"]["close_approval_workflow_verified"] is False
    assert payload["gates"]["event_report_timeline_verified"] is False
    assert payload["voice_smoke_report"]["voice_delivery_status"] == "queued"
    assert payload["voice_smoke_report"]["voice_profile"] == "emergency_short"
    assert payload["notification_smoke_report"]["collector_request_count"] == 3
    assert payload["runtime_roundtrip_report"]["final_status"] == "shadowed"
    assert "illegal_parking" in payload["archive"]["scenario_ids"]
    assert "camera" in payload["archive"]["sources"]
    assert any("temporary local server" in item for item in payload["warnings"])
    assert any("recorded voice handler" in item for item in payload["warnings"])
    assert any("local webhook collector" in item for item in payload["warnings"])
    assert any("close approval workflow" in item for item in payload["warnings"])
    assert any("report timeline evidence" in item for item in payload["warnings"])
    assert any("runtime callback HMAC secret" in item for item in payload["warnings"])
    assert any("request close approval" in item for item in payload["next_actions"])
    assert any("closed field event report" in item for item in payload["next_actions"])


def test_field_deployment_readiness_production_mode_blocks_lab_evidence(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    scenario_report = tmp_path / "scenario.json"
    smoke_report = tmp_path / "smoke.json"
    voice_smoke_report = tmp_path / "voice-smoke.json"
    notification_smoke_report = tmp_path / "notification-smoke.json"
    runtime_roundtrip_report = tmp_path / "runtime-roundtrip.json"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "illegal_parking", "payload": {"source": "camera"}})
        + "\n",
        encoding="utf-8",
    )
    scenario_report.write_text(
        json.dumps({"status": "passed", "external_services": False, "hardware_dispatch": False}),
        encoding="utf-8",
    )
    smoke_report.write_text(json.dumps({"status": "passed", "local_server": True}), encoding="utf-8")
    voice_smoke_report.write_text(
        json.dumps({"status": "passed", "local_server": True, "live_tts": False}),
        encoding="utf-8",
    )
    notification_smoke_report.write_text(
        json.dumps({"status": "passed", "local_server": True, "external_services": False}),
        encoding="utf-8",
    )
    _write_runtime_roundtrip_report(
        runtime_roundtrip_report,
        mode="local_server",
        final_status="shadowed",
    )
    service = FieldOperationsService(
        config={
            "deployment_mode": "production",
            "archive_path": str(archive),
            "site_profile_path": "deploy/site-profiles/park-demo.yaml",
            "scenario_report_path": str(scenario_report),
            "smoke_report_path": str(smoke_report),
            "voice_smoke_report_path": str(voice_smoke_report),
            "notification_smoke_report_path": str(notification_smoke_report),
            "runtime_roundtrip_report_path": str(runtime_roundtrip_report),
            "dingtalk_webhooks": {"security": "http://security.local/ding"},
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "blocked"
    assert payload["deployment_mode"] == "production"
    assert "production requires real robot hardware dispatch evidence" in payload["blockers"]
    assert "production requires live TTS on the target audio device" in payload["blockers"]
    assert "production requires real notification service smoke" in payload["blockers"]
    assert "production requires cleaning DingTalk webhook" in payload["blockers"]
    assert "production requires security DingTalk signing secret" in payload["blockers"]
    assert "production requires at least one trusted field-device event" in payload["blockers"]
    assert "production requires signed runtime delivery callbacks" in payload["blockers"]
    assert "production requires runtime roundtrip smoke against a running deployment" in payload["blockers"]
    assert payload["gates"]["site_profile_valid"] is True


def test_field_deployment_readiness_can_be_production_ready(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    audit_path = tmp_path / "field-action-audit.jsonl"
    scenario_report = tmp_path / "scenario.json"
    smoke_report = tmp_path / "smoke.json"
    voice_smoke_report = tmp_path / "voice-smoke.json"
    notification_smoke_report = tmp_path / "notification-smoke.json"
    runtime_roundtrip_report = tmp_path / "runtime-roundtrip.json"
    scenario_report.write_text(
        json.dumps({
            "status": "passed",
            "scenario_count": 10,
            "passed": 10,
            "failed": 0,
            "external_services": True,
            "hardware_dispatch": True,
        }),
        encoding="utf-8",
    )
    smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "event_count": 3,
            "local_server": False,
        }),
        encoding="utf-8",
    )
    voice_smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "local_server": False,
            "live_tts": True,
            "voice_delivery": {"status": "queued"},
            "voice_directive": {"resolved_profile": "security_clear"},
        }),
        encoding="utf-8",
    )
    notification_smoke_report.write_text(
        json.dumps({
            "status": "passed",
            "local_server": False,
            "external_services": True,
            "sent_groups": ["security", "cleaning", "operations"],
            "collector_request_count": 0,
        }),
        encoding="utf-8",
    )
    _write_runtime_roundtrip_report(runtime_roundtrip_report)
    config = {
        "archive_path": str(archive),
        "site_profile_path": "deploy/site-profiles/park-demo.yaml",
        "scenario_report_path": str(scenario_report),
        "smoke_report_path": str(smoke_report),
        "voice_smoke_report_path": str(voice_smoke_report),
        "notification_smoke_report_path": str(notification_smoke_report),
        "runtime_roundtrip_report_path": str(runtime_roundtrip_report),
        "action_audit": {
            "enabled": True,
            "path": str(audit_path),
            "swallow_errors": False,
            "hmac_secret": "readiness-test-secret",
        },
        "dingtalk_webhooks": {
            "security": "http://security.local/ding",
            "cleaning": "http://cleaning.local/ding",
            "operations": "http://ops.local/ding",
        },
        "dingtalk_secrets": {
            "security": "sec",
            "cleaning": "clean",
            "operations": "ops",
        },
        "device_registry": {
            "smoke-01": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "device-secret",
                "require_signature": True,
            }
        },
        "runtime_callback_hmac_secret": "runtime-callback-secret",
    }
    service = FieldOperationsService(config=config, alert_dispatcher_factory=_FakeDispatcher)
    created = asyncio.run(
        service.trigger_payload(
            {
                "scenario_id": "fire_or_smoke",
                "source": "sensor",
                "device_id": "smoke-01",
                "device_trust": {
                    "required": True,
                    "device_id": "smoke-01",
                    "source": "sensor",
                    "registered": True,
                    "signature_verified": True,
                    "trusted": True,
                    "status": "trusted",
                    "reason": "",
                },
                "location": "power-room-door",
                "temperature_c": 68,
                "smoke_level": "high",
                "image_path": "artifacts/evidence/smoke.jpg",
                "created_at": 1000.0,
            }
        )
    )
    event_id = created["event"]["event_id"]
    service.request_close_payload(event_id, {"operator_id": "guard-1", "note": "ready"})
    service.close_payload(
        event_id,
        {
            "operator_id": "guard-1",
            "note": "handled",
            "supervisor_approved": True,
            "supervisor_id": "supervisor-1",
        },
    )

    payload = service.readiness_payload()

    assert payload["status"] == "production_ready"
    assert payload["blockers"] == []
    assert payload["warnings"] == []
    assert payload["delivery_brief"]["stage_code"] == "site_launch_ready"
    assert payload["delivery_brief"]["release_scope"] == "site_launch_acceptance"
    assert payload["delivery_brief"]["customer_status"] == "已达到现场上线验收标准"
    assert payload["delivery_brief"]["stakeholder_messages"]["customer"] == "已达到现场上线验收标准"
    assert payload["delivery_brief"]["checklist"][0]["owner"] == "研发"
    assert payload["delivery_brief"]["checklist"][0]["status"] == "ready"
    assert payload["gates"]["close_approval_workflow_verified"] is True
    assert payload["gates"]["event_report_timeline_verified"] is True
    assert payload["gates"]["action_audit_integrity_verified"] is True
    assert payload["gates"]["action_audit_signed"] is True
    assert payload["gates"]["audit_delivery_retry_queue_empty"] is True
    assert payload["gates"]["trusted_device_events_observed"] is True
    assert payload["gates"]["runtime_callback_signature_configured"] is True
    assert payload["gates"]["runtime_roundtrip_smoke_passed"] is True
    assert payload["gates"]["runtime_roundtrip_against_existing_server"] is True
    assert payload["gates"]["all_registered_devices_signature_ready"] is True
    assert payload["gates"]["site_profile_valid"] is True
    assert payload["gates"]["site_profile_wayfinding_configured"] is True
    assert payload["site_profile"]["summary"]["site_id"] == "inovx-demo-park"
    assert payload["archive"]["trusted_device_ids"] == ["smoke-01"]
    assert payload["device_trust"]["unsigned_device_count"] == 0
    assert payload["device_trust"]["signature_ready_device_ids"] == ["smoke-01"]
    assert payload["action_audit_integrity"]["signed"] is True
    assert payload["audit_delivery_retry_queue"]["pending"] == 0


def test_field_deployment_readiness_blocks_production_when_any_registered_device_is_unsigned(
    tmp_path: Path,
) -> None:
    service = FieldOperationsService(
        config={
            "deployment_mode": "production",
            "archive_path": str(tmp_path / "events.jsonl"),
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
            "device_registry": {
                "camera-main-road-1": {
                    "allowed_sources": ["camera"],
                    "hmac_secret": "camera-secret",
                    "require_signature": True,
                },
                "bin-17": {
                    "allowed_sources": ["camera"],
                    "require_signature": True,
                },
                "legacy-smoke-1": {
                    "allowed_sources": ["sensor"],
                    "hmac_secret": "legacy-secret",
                    "require_signature": False,
                },
            },
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "blocked"
    assert payload["gates"]["device_registry_configured"] is True
    assert payload["gates"]["device_signatures_required"] is True
    assert payload["gates"]["all_registered_devices_signature_ready"] is False
    assert payload["delivery_brief"]["stage_code"] == "delivery_blocked"
    assert payload["delivery_brief"]["release_scope"] == "pilot_demo_and_site_integration"
    assert payload["delivery_brief"]["customer_status"] == "暂未达到交付验收标准，需先处理关键阻塞项"
    assert "正式上线项已在交付清单中跟踪" in payload["delivery_brief"]["release_claim"]
    assert payload["delivery_brief"]["stakeholder_messages"]["delivery"]
    assert (
        "production requires every registered field device to require signatures "
        "and have a signing secret"
    ) in payload["blockers"]
    assert "field device registry has unsigned or unsecreted devices: bin-17, legacy-smoke-1" in payload["warnings"]
    assert payload["device_trust"]["registered_device_count"] == 3
    assert payload["device_trust"]["signed_device_count"] == 1
    assert payload["device_trust"]["unsigned_device_count"] == 2
    assert payload["device_trust"]["signature_ready_device_ids"] == ["camera-main-road-1"]
    assert payload["device_trust"]["missing_secret_device_ids"] == ["bin-17"]
    assert payload["device_trust"]["signature_disabled_device_ids"] == ["legacy-smoke-1"]
    assert "Require signatures and configure HMAC secrets for every registered field device" in payload["next_actions"]


def test_field_deployment_readiness_treats_unresolved_device_secret_env_as_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET", raising=False)
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "scenario_report_path": str(tmp_path / "missing-scenario.json"),
            "smoke_report_path": str(tmp_path / "missing-smoke.json"),
            "voice_smoke_report_path": str(tmp_path / "missing-voice-smoke.json"),
            "notification_smoke_report_path": str(tmp_path / "missing-notification-smoke.json"),
            "runtime_roundtrip_report_path": str(tmp_path / "missing-runtime-roundtrip.json"),
            "device_registry": {
                "camera-main-road-1": {
                    "allowed_sources": ["camera"],
                    "secret": "${ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET}",
                    "require_signature": True,
                }
            },
        }
    )

    payload = service.readiness_payload()

    assert payload["gates"]["device_registry_configured"] is True
    assert payload["gates"]["device_signatures_required"] is False
    assert payload["gates"]["all_registered_devices_signature_ready"] is False
    assert payload["device_trust"]["unsigned_device_ids"] == ["camera-main-road-1"]
    assert payload["device_trust"]["missing_secret_device_ids"] == ["camera-main-road-1"]


def test_field_deployment_readiness_blocks_pending_audit_delivery_retry_queue(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    scenario_report = tmp_path / "scenario.json"
    smoke_report = tmp_path / "smoke.json"
    voice_smoke_report = tmp_path / "voice-smoke.json"
    notification_smoke_report = tmp_path / "notification-smoke.json"
    runtime_roundtrip_report = tmp_path / "runtime-roundtrip.json"
    retry_queue = tmp_path / "audit-delivery-retry.jsonl"
    archive.write_text(
        json.dumps({"event_id": "field-1", "scenario_id": "illegal_parking", "payload": {"source": "camera"}})
        + "\n",
        encoding="utf-8",
    )
    scenario_report.write_text(
        json.dumps({
            "status": "passed",
            "external_services": True,
            "hardware_dispatch": True,
        }),
        encoding="utf-8",
    )
    smoke_report.write_text(json.dumps({"status": "passed", "local_server": False}), encoding="utf-8")
    voice_smoke_report.write_text(
        json.dumps({"status": "passed", "local_server": False, "live_tts": True}),
        encoding="utf-8",
    )
    notification_smoke_report.write_text(
        json.dumps({"status": "passed", "local_server": False, "external_services": True}),
        encoding="utf-8",
    )
    _write_runtime_roundtrip_report(runtime_roundtrip_report)
    retry_queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"checkpoint": {"latest_hash": "hash-pending"}},
        })
        + "\n",
        encoding="utf-8",
    )
    service = FieldOperationsService(
        config={
            "archive_path": str(archive),
            "scenario_report_path": str(scenario_report),
            "smoke_report_path": str(smoke_report),
            "voice_smoke_report_path": str(voice_smoke_report),
            "notification_smoke_report_path": str(notification_smoke_report),
            "runtime_roundtrip_report_path": str(runtime_roundtrip_report),
            "action_audit": {"retry_queue_path": str(retry_queue), "enabled": False},
            "dingtalk_webhooks": {
                "security": "http://security.local/ding",
                "cleaning": "http://cleaning.local/ding",
                "operations": "http://ops.local/ding",
            },
            "dingtalk_secrets": {
                "security": "sec",
                "cleaning": "clean",
                "operations": "ops",
            },
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "blocked"
    assert payload["gates"]["audit_delivery_retry_queue_empty"] is False
    assert payload["audit_delivery_retry_queue"]["pending"] == 1
    assert payload["audit_delivery_retry_queue"]["latest_hashes"] == ["hash-pending"]
    assert "field audit delivery retry queue still has pending or invalid items" in payload["blockers"]
    assert any("field-audit-retry-delivery" in item for item in payload["next_actions"])


def test_field_deployment_readiness_blocks_operator_actions_without_audit_chain(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl"
    scenario_report = tmp_path / "scenario.json"
    smoke_report = tmp_path / "smoke.json"
    voice_smoke_report = tmp_path / "voice-smoke.json"
    notification_smoke_report = tmp_path / "notification-smoke.json"
    runtime_roundtrip_report = tmp_path / "runtime-roundtrip.json"
    archive.write_text(
        json.dumps(
            {
                "event_id": "field-1",
                "scenario_id": "fire_or_smoke",
                "priority": "P0",
                "severity": "error",
                "status": "closed",
                "created_at": 1000.0,
                "close_requested_at": 1010.0,
                "close_requested_by": "guard-1",
                "closed_at": 1040.0,
                "closed_by": "guard-1",
                "close_approval": {"approved": True, "supervisor_id": "supervisor-1"},
                "delivery_report": [{"channel": "dingtalk", "status": "sent"}],
                "payload": {"source": "sensor"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for path in (scenario_report, smoke_report, voice_smoke_report, notification_smoke_report):
        payload = {"status": "passed", "local_server": False, "external_services": True}
        if path == scenario_report:
            payload["hardware_dispatch"] = True
        if path == voice_smoke_report:
            payload["live_tts"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
    _write_runtime_roundtrip_report(runtime_roundtrip_report)
    service = FieldOperationsService(
        config={
            "archive_path": str(archive),
            "scenario_report_path": str(scenario_report),
            "smoke_report_path": str(smoke_report),
            "voice_smoke_report_path": str(voice_smoke_report),
            "notification_smoke_report_path": str(notification_smoke_report),
            "runtime_roundtrip_report_path": str(runtime_roundtrip_report),
            "action_audit": {
                "enabled": True,
                "path": str(tmp_path / "missing-audit.jsonl"),
                "swallow_errors": False,
                "hmac_secret": "readiness-test-secret",
            },
            "dingtalk_webhooks": {
                "security": "http://security.local/ding",
                "cleaning": "http://cleaning.local/ding",
                "operations": "http://ops.local/ding",
            },
            "dingtalk_secrets": {
                "security": "sec",
                "cleaning": "clean",
                "operations": "ops",
            },
        }
    )

    payload = service.readiness_payload()

    assert payload["status"] == "blocked"
    assert payload["gates"]["action_audit_integrity_verified"] is False
    assert "field action audit integrity has not passed" in payload["blockers"]
    assert any("audit chain" in item for item in payload["next_actions"])
