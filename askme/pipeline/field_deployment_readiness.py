"""Deployment readiness checks for Askme field operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def build_field_deployment_readiness(
    *,
    config: dict[str, Any],
    archive_path: str | Path,
    webhooks: dict[str, str],
    webhook_secrets: dict[str, str],
    action_audit_integrity: dict[str, Any] | None = None,
    unified_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an operator-facing readiness report for field operations."""
    archive = _archive_summary(Path(archive_path))
    deployment_mode = _deployment_mode(config)
    production_mode = deployment_mode == "production"
    scenario_report_path = _path_from_config(
        config,
        "scenario_report_path",
        "artifacts/field_operations/scenario-evaluation.json",
    )
    smoke_report_path = _path_from_config(
        config,
        "smoke_report_path",
        "artifacts/field_operations/smoke/field-ingest-smoke.json",
    )
    voice_smoke_report_path = _path_from_config(
        config,
        "voice_smoke_report_path",
        "artifacts/field_operations/smoke/field-voice-smoke.json",
    )
    notification_smoke_report_path = _path_from_config(
        config,
        "notification_smoke_report_path",
        "artifacts/field_operations/smoke/field-notification-smoke.json",
    )
    runtime_roundtrip_report_path = _path_from_config(
        config,
        "runtime_roundtrip_report_path",
        "artifacts/runtime_handoff/field-runtime-roundtrip-live-smoke.json",
    )
    scenario_report = _report_summary(
        scenario_report_path,
        status_key="status",
        pass_value="passed",
    )
    smoke_report = _report_summary(
        smoke_report_path,
        status_key="status",
        pass_value="passed",
    )
    voice_smoke_report = _report_summary(
        voice_smoke_report_path,
        status_key="status",
        pass_value="passed",
    )
    notification_smoke_report = _report_summary(
        notification_smoke_report_path,
        status_key="status",
        pass_value="passed",
    )
    runtime_roundtrip_report = _runtime_roundtrip_report_summary(
        runtime_roundtrip_report_path
    )
    audit_integrity = (
        action_audit_integrity
        if isinstance(action_audit_integrity, dict)
        else {"valid": False, "signed": False, "checked_count": 0, "failures": []}
    )
    audit_retry_queue = _audit_retry_queue_summary(
        _action_audit_retry_queue_path(
            config,
            default=str(Path(archive_path).with_name("audit-delivery-retry.jsonl")),
        )
    )
    unified_audit_review = _unified_audit_review_summary(unified_audit)
    notifications = _notification_summary(webhooks, webhook_secrets)
    device_trust = _device_trust_summary(config)
    site_profile = _site_profile_summary(config)
    runtime_callbacks = _runtime_callback_summary(config)
    action_audit_required = (
        archive["action_audit_count"] > 0
        or archive["operator_action_event_count"] > 0
    )
    gates = {
        "scenario_eval_passed": scenario_report["passed"],
        "http_smoke_passed": smoke_report["passed"],
        "voice_smoke_passed": voice_smoke_report["passed"],
        "archive_has_events": archive["event_count"] > 0,
        "action_audit_integrity_verified": (
            not action_audit_required or audit_integrity.get("valid") is True
        ),
        "action_audit_signed": (
            action_audit_required and audit_integrity.get("signed") is True
        ),
        "audit_delivery_retry_queue_empty": (
            audit_retry_queue["pending"] == 0 and audit_retry_queue["invalid"] == 0
        ),
        "security_notification_configured": notifications["groups"]["security"]["webhook_configured"],
        "cleaning_notification_configured": notifications["groups"]["cleaning"]["webhook_configured"],
        "operations_notification_configured": notifications["groups"]["operations"]["webhook_configured"],
        "security_notification_secret_configured": notifications["groups"]["security"]["secret_configured"],
        "cleaning_notification_secret_configured": notifications["groups"]["cleaning"]["secret_configured"],
        "operations_notification_secret_configured": notifications["groups"]["operations"]["secret_configured"],
        "uses_external_services": bool(scenario_report.get("external_services") or False),
        "uses_real_hardware": bool(scenario_report.get("hardware_dispatch") or False),
        "smoke_against_existing_server": smoke_report.get("local_server") is False,
        "voice_smoke_uses_live_tts": voice_smoke_report.get("live_tts") is True,
        "voice_smoke_against_existing_server": voice_smoke_report.get("local_server") is False,
        "notification_smoke_passed": notification_smoke_report["passed"],
        "notification_smoke_uses_external_services": notification_smoke_report.get("external_services") is True,
        "notification_smoke_against_existing_server": notification_smoke_report.get("local_server") is False,
        "runtime_roundtrip_smoke_passed": runtime_roundtrip_report["passed"],
        "runtime_roundtrip_against_existing_server": runtime_roundtrip_report.get("local_server") is False,
        "runtime_roundtrip_trusted_callbacks": runtime_roundtrip_report["trusted_callbacks"],
        "runtime_roundtrip_final_status_verified": runtime_roundtrip_report["final_status_verified"],
        "device_registry_configured": device_trust["registered_device_count"] > 0,
        "device_signatures_required": device_trust["signed_device_count"] > 0,
        "all_registered_devices_signature_ready": device_trust[
            "all_registered_devices_signature_ready"
        ],
        "trusted_device_events_observed": archive["trusted_device_event_count"] > 0,
        "site_profile_configured": site_profile["configured"],
        "site_profile_valid": site_profile["valid"],
        "site_profile_map_configured": site_profile["readiness"]["map_configured"],
        "site_profile_parking_policy_configured": site_profile["readiness"]["parking_policy_configured"],
        "site_profile_wayfinding_configured": site_profile["readiness"]["wayfinding_configured"],
        "site_profile_responder_groups_configured": site_profile["readiness"]["responder_groups_configured"],
        "site_profile_device_registry_configured": site_profile["readiness"]["device_registry_configured"],
        "runtime_callback_signature_configured": runtime_callbacks["secret_configured"],
        "close_approval_workflow_verified": archive["close_approval_count"] > 0
        and archive["close_request_count"] > 0,
        "event_report_timeline_verified": archive["report_timeline_ready_count"] > 0,
        "unified_audit_review_clear": unified_audit_review["requires_review_count"] == 0,
        "unified_audit_review_integrity_verified": unified_audit_review["integrity_valid"],
        "unified_audit_sources_healthy": unified_audit_review["source_health_healthy"],
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not gates["scenario_eval_passed"]:
        blockers.append("field scenario evaluation has not passed")
    if not gates["http_smoke_passed"]:
        blockers.append("field ingest HTTP smoke has not passed")
    if not gates["voice_smoke_passed"]:
        blockers.append("field voice smoke has not passed")
    if not gates["archive_has_events"]:
        blockers.append("field event archive has no events")
    if not gates["action_audit_integrity_verified"]:
        blockers.append("field action audit integrity has not passed")
    if action_audit_required and not gates["action_audit_signed"]:
        warnings.append("field action audit is not HMAC-signed")
    if not gates["audit_delivery_retry_queue_empty"]:
        blockers.append("field audit delivery retry queue still has pending or invalid items")
    if not gates["unified_audit_review_clear"]:
        blockers.append("unified audit review queue has unresolved high-risk records")
    if not gates["unified_audit_review_integrity_verified"]:
        blockers.append("unified audit review log integrity has not passed")
    if not gates["unified_audit_sources_healthy"]:
        blockers.append("unified audit sources have unreadable or invalid records")
    for group in ("security", "cleaning", "operations"):
        group_report = notifications["groups"][group]
        if not group_report["webhook_configured"]:
            warnings.append(f"{group} DingTalk webhook is not configured")
        elif not group_report["secret_configured"]:
            warnings.append(f"{group} DingTalk secret is not configured")
    if not gates["smoke_against_existing_server"]:
        warnings.append("last HTTP smoke used a temporary local server, not a running deployment")
    if gates["voice_smoke_passed"] and not gates["voice_smoke_against_existing_server"]:
        warnings.append("last voice smoke used a temporary local server, not a running deployment")
    if gates["voice_smoke_passed"] and not gates["voice_smoke_uses_live_tts"]:
        warnings.append("last voice smoke used recorded voice handler, not live TTS")
    if not gates["notification_smoke_passed"]:
        warnings.append("field DingTalk notification smoke has not passed")
    elif not gates["notification_smoke_against_existing_server"]:
        warnings.append("last notification smoke used a temporary local server, not a running deployment")
    if gates["notification_smoke_passed"] and not gates["notification_smoke_uses_external_services"]:
        warnings.append("last notification smoke used a local webhook collector, not real DingTalk")
    if not gates["runtime_roundtrip_smoke_passed"]:
        blockers.append("field runtime roundtrip smoke has not passed")
    elif not gates["runtime_roundtrip_against_existing_server"]:
        warnings.append("last runtime roundtrip smoke used a temporary local server, not a running deployment")
    if not gates["close_approval_workflow_verified"]:
        warnings.append("field close approval workflow has not been verified in the archive")
    if not gates["event_report_timeline_verified"]:
        warnings.append("field event report timeline evidence has not been verified in the archive")
    if (
        gates["device_registry_configured"]
        and not gates["all_registered_devices_signature_ready"]
    ):
        sample = ", ".join(device_trust["unsigned_device_ids"][:5])
        suffix = f": {sample}" if sample else ""
        warnings.append(f"field device registry has unsigned or unsecreted devices{suffix}")
    if not gates["site_profile_configured"]:
        warnings.append("field site profile is not configured")
    elif not gates["site_profile_valid"]:
        warnings.append("field site profile is not valid")
    else:
        for warning in site_profile.get("warnings", []):
            warnings.append(f"field site profile: {warning}")
        if not gates["site_profile_parking_policy_configured"]:
            warnings.append("field site profile has no parking-restricted main channel")
        if not gates["site_profile_wayfinding_configured"]:
            warnings.append("field site profile has no wayfinding help point")
        if not gates["site_profile_device_registry_configured"]:
            warnings.append("field site profile has no device registry")
    if not gates["runtime_callback_signature_configured"]:
        warnings.append("field runtime callback HMAC secret is not configured")
    if not gates["uses_real_hardware"]:
        warnings.append("scenario report still declares hardware_dispatch=false")
    if not gates["uses_external_services"]:
        warnings.append("scenario report still declares external_services=false")
    if production_mode:
        _add_production_blockers(blockers, gates)

    status = "blocked" if blockers else ("ready_for_lab" if warnings else "production_ready")
    next_actions = _next_actions(blockers, warnings)
    return {
        "status": status,
        "deployment_mode": deployment_mode,
        "blockers": blockers,
        "warnings": warnings,
        "gates": gates,
        "scenario_report": scenario_report,
        "smoke_report": smoke_report,
        "voice_smoke_report": voice_smoke_report,
        "notification_smoke_report": notification_smoke_report,
        "runtime_roundtrip_report": runtime_roundtrip_report,
        "action_audit_integrity": audit_integrity,
        "audit_delivery_retry_queue": audit_retry_queue,
        "unified_audit": unified_audit_review,
        "notifications": notifications,
        "device_trust": device_trust,
        "site_profile": site_profile,
        "runtime_callbacks": runtime_callbacks,
        "archive": archive,
        "next_actions": next_actions,
        "delivery_brief": _delivery_brief(
            status=status,
            blockers=blockers,
            warnings=warnings,
            gates=gates,
            next_actions=next_actions,
        ),
    }


def _deployment_mode(config: dict[str, Any]) -> str:
    raw = str(
        config.get("deployment_mode")
        or config.get("readiness_mode")
        or config.get("field_deployment_mode")
        or "lab"
    ).strip().lower()
    return "production" if raw in {"prod", "production"} else "lab"


def _add_production_blockers(blockers: list[str], gates: dict[str, bool]) -> None:
    required_gates = {
        "security_notification_configured": "production requires security DingTalk webhook",
        "cleaning_notification_configured": "production requires cleaning DingTalk webhook",
        "operations_notification_configured": "production requires operations DingTalk webhook",
        "security_notification_secret_configured": "production requires security DingTalk signing secret",
        "cleaning_notification_secret_configured": "production requires cleaning DingTalk signing secret",
        "operations_notification_secret_configured": "production requires operations DingTalk signing secret",
        "uses_external_services": "production requires scenario evidence from external services",
        "uses_real_hardware": "production requires real robot hardware dispatch evidence",
        "smoke_against_existing_server": "production requires HTTP smoke against a running deployment",
        "voice_smoke_uses_live_tts": "production requires live TTS on the target audio device",
        "voice_smoke_against_existing_server": "production requires voice smoke against a running deployment",
        "notification_smoke_uses_external_services": "production requires real notification service smoke",
        "notification_smoke_against_existing_server": "production requires notification smoke against a running deployment",
        "runtime_roundtrip_smoke_passed": "production requires runtime roundtrip smoke",
        "runtime_roundtrip_against_existing_server": "production requires runtime roundtrip smoke against a running deployment",
        "runtime_roundtrip_trusted_callbacks": "production requires trusted runtime callback receipts",
        "runtime_roundtrip_final_status_verified": "production requires completed or shadowed runtime delivery",
        "device_registry_configured": "production requires a field device registry",
        "device_signatures_required": "production requires signed field-device ingest",
        "all_registered_devices_signature_ready": (
            "production requires every registered field device to require signatures "
            "and have a signing secret"
        ),
        "trusted_device_events_observed": "production requires at least one trusted field-device event",
        "site_profile_configured": "production requires a field site profile",
        "site_profile_valid": "production requires a valid field site profile",
        "site_profile_map_configured": "production requires a configured site map",
        "site_profile_parking_policy_configured": "production requires parking policy zones",
        "site_profile_wayfinding_configured": "production requires wayfinding help points",
        "site_profile_responder_groups_configured": "production requires responder groups in the site profile",
        "site_profile_device_registry_configured": "production requires devices in the site profile",
        "runtime_callback_signature_configured": "production requires signed runtime delivery callbacks",
        "close_approval_workflow_verified": "production requires verified close approval workflow",
        "event_report_timeline_verified": "production requires verified event report timeline",
        "action_audit_signed": "production requires HMAC-signed field action audit",
        "unified_audit_review_clear": "production requires unified audit review queue to be cleared",
        "unified_audit_review_integrity_verified": "production requires valid unified audit review log integrity",
        "unified_audit_sources_healthy": "production requires readable unified audit sources without invalid records",
    }
    for gate, message in required_gates.items():
        if not gates.get(gate):
            blockers.append(message)


def _delivery_brief(
    *,
    status: str,
    blockers: list[str],
    warnings: list[str],
    gates: dict[str, bool],
    next_actions: list[str],
) -> dict[str, Any]:
    customer_status = {
        "production_ready": "已达到现场上线验收标准",
        "ready_for_lab": "已达到试点演示标准，待完成现场上线项",
        "blocked": "暂未达到交付验收标准，需先处理关键阻塞项",
    }.get(status, "状态待确认")
    stage_code = {
        "production_ready": "site_launch_ready",
        "ready_for_lab": "pilot_ready_pending_site_launch",
        "blocked": "delivery_blocked",
    }.get(status, "unknown")
    top_issue = blockers[0] if blockers else (warnings[0] if warnings else "")
    release_claim = (
        "可对外说明：现场异常识别、语音播报、通知、归档和运行闭环已具备上线验收证据"
        if status == "production_ready"
        else "可对外说明：当前版本适合试点演示和现场联调，正式上线项已在交付清单中跟踪"
    )
    release_scope = (
        "site_launch_acceptance"
        if status == "production_ready"
        else "pilot_demo_and_site_integration"
    )
    return {
        "stage_code": stage_code,
        "customer_status": customer_status,
        "business_value": "让园区机器狗把异常发现、语音提醒、群通知、事件归档和任务运行闭环串成可验收产品",
        "release_scope": release_scope,
        "release_claim": release_claim,
        "top_issue": top_issue,
        "stakeholder_messages": {
            "engineering": _engineering_message(gates),
            "testing": _testing_message(blockers, warnings),
            "delivery": next_actions[0] if next_actions else "保持当前上线证据，准备客户验收",
            "sales": release_claim,
            "customer": customer_status,
            "executive": _executive_message(status, blockers, warnings),
        },
        "checklist": [
            _brief_checklist_item(
                owner="研发",
                title="事件入口和运行闭环",
                gates=gates,
                required=[
                    "scenario_eval_passed",
                    "http_smoke_passed",
                    "runtime_roundtrip_smoke_passed",
                    "runtime_roundtrip_final_status_verified",
                ],
            ),
            _brief_checklist_item(
                owner="测试",
                title="可重复验收证据",
                gates=gates,
                required=[
                    "scenario_eval_passed",
                    "close_approval_workflow_verified",
                    "event_report_timeline_verified",
                    "action_audit_integrity_verified",
                ],
            ),
            _brief_checklist_item(
                owner="交付",
                title="现场配置和真实服务",
                gates=gates,
                required=[
                    "site_profile_valid",
                    "notification_smoke_against_existing_server",
                    "smoke_against_existing_server",
                    "runtime_callback_signature_configured",
                ],
            ),
            _brief_checklist_item(
                owner="安全/运维",
                title="设备可信和审计签名",
                gates=gates,
                required=[
                    "device_registry_configured",
                    "all_registered_devices_signature_ready",
                    "trusted_device_events_observed",
                    "action_audit_signed",
                    "unified_audit_review_clear",
                    "unified_audit_review_integrity_verified",
                    "unified_audit_sources_healthy",
                ],
            ),
            _brief_checklist_item(
                owner="销售/客户成功",
                title="可承诺边界",
                gates=gates,
                required=[
                    "uses_external_services",
                    "uses_real_hardware",
                    "voice_smoke_uses_live_tts",
                ],
            ),
        ],
    }


def _engineering_message(gates: dict[str, bool]) -> str:
    if gates.get("runtime_roundtrip_smoke_passed") and gates.get("http_smoke_passed"):
        return "核心事件入口和任务运行闭环已有自动化证据"
    return "优先补齐事件入口、运行闭环和异常播报 smoke 证据"


def _testing_message(blockers: list[str], warnings: list[str]) -> str:
    if blockers:
        return f"先复测阻塞项：{blockers[0]}"
    if warnings:
        return f"生产验收前复测提醒项：{warnings[0]}"
    return "按生产回归清单复测场景、通知、语音、审计和运行闭环"


def _executive_message(status: str, blockers: list[str], warnings: list[str]) -> str:
    if status == "production_ready":
        return "版本已具备现场上线验收证据，可安排客户验收"
    if blockers:
        return f"版本仍有 {len(blockers)} 个关键阻塞项，需先完成交付修复"
    return f"版本已可试点演示，仍有 {len(warnings)} 个现场上线项待完成"


def _brief_checklist_item(
    *,
    owner: str,
    title: str,
    gates: dict[str, bool],
    required: list[str],
) -> dict[str, Any]:
    missing = [name for name in required if not gates.get(name)]
    return {
        "owner": owner,
        "title": title,
        "status": "ready" if not missing else "blocked",
        "required_gates": required,
        "missing_gates": missing,
    }


def _site_profile_summary(config: dict[str, Any]) -> dict[str, Any]:
    profile = config.get("site_profile") if isinstance(config.get("site_profile"), dict) else {}
    readiness = profile.get("readiness") if isinstance(profile.get("readiness"), dict) else {}
    configured = bool(config.get("site_profile_path") or profile)
    valid = str(profile.get("status") or "").lower() == "passed"
    return {
        "configured": configured,
        "valid": valid if configured else False,
        "profile_path": str(config.get("site_profile_path") or ""),
        "summary": profile.get("summary") if isinstance(profile.get("summary"), dict) else {},
        "warnings": profile.get("warnings") if isinstance(profile.get("warnings"), list) else [],
        "readiness": {
            "map_configured": bool(readiness.get("map_configured", False)),
            "parking_policy_configured": bool(readiness.get("parking_policy_configured", False)),
            "wayfinding_configured": bool(readiness.get("wayfinding_configured", False)),
            "responder_groups_configured": bool(readiness.get("responder_groups_configured", False)),
            "device_registry_configured": bool(readiness.get("device_registry_configured", False)),
        },
    }


def _path_from_config(config: dict[str, Any], key: str, default: str) -> Path:
    raw = config.get(key) or default
    return Path(str(raw))


def _action_audit_retry_queue_path(config: dict[str, Any], *, default: str) -> Path:
    action_audit = config.get("action_audit") if isinstance(config.get("action_audit"), dict) else {}
    raw = (
        action_audit.get("retry_queue_path")
        or action_audit.get("delivery_retry_queue")
        or config.get("audit_delivery_retry_queue_path")
        or config.get("field_audit_retry_queue_path")
        or default
    )
    return Path(str(raw))


def _unified_audit_review_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "configured": False,
            "status": "not_configured",
            "requires_review_count": 0,
            "high_or_critical_count": 0,
            "integrity_valid": True,
            "record_count": 0,
            "filtered_total": 0,
            "review_queue": [],
            "time_window": {},
            "source_health": {},
            "source_health_healthy": True,
            "invalid_source_count": 0,
            "unreadable_source_count": 0,
            "unhealthy_sources": [],
            "review_integrity": {"valid": True, "exists": False, "checked_count": 0, "failures": []},
        }
    product_summary = (
        payload.get("product_summary")
        if isinstance(payload.get("product_summary"), dict)
        else {}
    )
    review_queue = payload.get("review_queue") if isinstance(payload.get("review_queue"), list) else []
    requires_review_count = _int_value(
        product_summary.get("requires_review_count"),
        default=len(review_queue),
    )
    high_or_critical_count = _int_value(
        product_summary.get("high_or_critical_count"),
        default=_high_or_critical_count(review_queue),
    )
    status = str(product_summary.get("status") or ("needs_review" if requires_review_count else "auditable"))
    review_integrity = (
        payload.get("review_integrity")
        if isinstance(payload.get("review_integrity"), dict)
        else {"valid": True, "exists": False, "checked_count": 0, "failures": []}
    )
    source_health = payload.get("source_health") if isinstance(payload.get("source_health"), dict) else {}
    source_health_summary = _unified_audit_source_health_summary(source_health)
    return {
        "configured": True,
        "status": status,
        "customer_status": str(product_summary.get("customer_status") or ""),
        "requires_review_count": requires_review_count,
        "high_or_critical_count": high_or_critical_count,
        "integrity_valid": review_integrity.get("valid") is True,
        "record_count": _int_value(product_summary.get("record_count"), default=_int_value(payload.get("total"), default=0)),
        "filtered_total": _int_value(payload.get("filtered_total"), default=0),
        "review_queue": [_audit_review_item(item) for item in review_queue[:10] if isinstance(item, dict)],
        "time_window": payload.get("time_window") if isinstance(payload.get("time_window"), dict) else {},
        "source_health": source_health,
        **source_health_summary,
        "review_integrity": review_integrity,
    }


def _unified_audit_source_health_summary(source_health: dict[str, Any]) -> dict[str, Any]:
    unhealthy_sources: list[dict[str, Any]] = []
    invalid_source_count = 0
    unreadable_source_count = 0
    for source_name, item in source_health.items():
        if not isinstance(item, dict):
            continue
        exists = item.get("exists") is True
        readable = item.get("readable") is not False
        invalid_count = _int_value(item.get("invalid_record_count"), default=0)
        if exists and not readable:
            unreadable_source_count += 1
        if invalid_count > 0:
            invalid_source_count += 1
        if (exists and not readable) or invalid_count > 0:
            unhealthy_sources.append({
                "source": str(source_name),
                "path": str(item.get("path") or ""),
                "readable": readable,
                "invalid_record_count": invalid_count,
                "error": str(item.get("error") or ""),
            })
    return {
        "source_health_healthy": not unhealthy_sources,
        "invalid_source_count": invalid_source_count,
        "unreadable_source_count": unreadable_source_count,
        "unhealthy_sources": unhealthy_sources,
    }


def _high_or_critical_count(records: list[Any]) -> int:
    return sum(
        1
        for item in records
        if isinstance(item, dict) and item.get("severity") in {"critical", "high"}
    )


def _audit_review_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": str(item.get("record_id") or ""),
        "customer_label": str(item.get("customer_label") or ""),
        "severity": str(item.get("severity") or ""),
        "action": str(item.get("action") or ""),
        "outcome": str(item.get("outcome") or ""),
        "operator_id": str(item.get("operator_id") or ""),
        "resource_type": str(item.get("resource_type") or ""),
        "resource_id": str(item.get("resource_id") or item.get("subject") or ""),
        "timestamp": str(item.get("timestamp") or ""),
    }


def _audit_retry_queue_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "pending": 0,
            "invalid": 0,
            "latest_hashes": [],
        }
    pending = 0
    invalid = 0
    latest_hashes: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(record, dict):
            invalid += 1
            continue
        pending += 1
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
        latest_hash = checkpoint.get("latest_hash")
        if latest_hash:
            latest_hashes.append(str(latest_hash))
    return {
        "path": str(path),
        "exists": True,
        "pending": pending,
        "invalid": invalid,
        "latest_hashes": latest_hashes[-10:],
    }


def _report_summary(path: Path, *, status_key: str, pass_value: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "passed": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "passed": False,
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "passed": False}
    status = str(payload.get(status_key) or "unknown")
    return {
        "status": status,
        "path": str(path),
        "passed": status == pass_value,
        "scenario_count": payload.get("scenario_count"),
        "passed_count": payload.get("passed"),
        "failed_count": payload.get("failed"),
        "event_count": payload.get("event_count"),
        "external_services": bool(payload.get("external_services", False)),
        "hardware_dispatch": bool(payload.get("hardware_dispatch", False)),
        "local_server": payload.get("local_server"),
        "live_tts": payload.get("live_tts"),
        "voice_delivery_status": _nested_get(payload, ("voice_delivery", "status")),
        "voice_profile": _nested_get(payload, ("voice_directive", "resolved_profile")),
        "requested_voice_profile": _nested_get(payload, ("voice_directive", "requested_profile")),
        "collector_request_count": payload.get("collector_request_count"),
        "sent_groups": payload.get("sent_groups"),
    }


def _runtime_roundtrip_report_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "passed": False,
            "ok": False,
            "mode": "",
            "local_server": None,
            "final_status": "",
            "receipt_count": 0,
            "callback_status_codes": [],
            "runtime_statuses": [],
            "trusted_callbacks": False,
            "final_status_verified": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "passed": False,
            "ok": False,
            "mode": "",
            "local_server": None,
            "final_status": "",
            "receipt_count": 0,
            "callback_status_codes": [],
            "runtime_statuses": [],
            "trusted_callbacks": False,
            "final_status_verified": False,
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "passed": False,
            "ok": False,
            "mode": "",
            "local_server": None,
            "final_status": "",
            "receipt_count": 0,
            "callback_status_codes": [],
            "runtime_statuses": [],
            "trusted_callbacks": False,
            "final_status_verified": False,
        }
    mode = str(payload.get("mode") or "")
    final_delivery = (
        payload.get("final_runtime_delivery")
        if isinstance(payload.get("final_runtime_delivery"), dict)
        else {}
    )
    final_status = str(final_delivery.get("status") or payload.get("final_status") or "")
    final_trust = (
        final_delivery.get("runtime_callback_trust")
        if isinstance(final_delivery.get("runtime_callback_trust"), dict)
        else {}
    )
    receipt_count = _int_value(payload.get("receipt_count"), default=0)
    callback_status_codes = _int_list(payload.get("callback_status_codes"))
    runtime_statuses = _str_list(payload.get("runtime_statuses"))
    local_server = mode in {"inprocess", "local_server"}
    final_status_verified = final_status in {"shadowed", "completed"}
    trusted_callbacks = (
        receipt_count > 0
        and final_trust.get("status") in {None, "trusted"}
        and final_trust.get("trusted") is not False
    )
    callback_codes_ok = bool(callback_status_codes) and all(
        code == 200 for code in callback_status_codes
    )
    passed = (
        payload.get("ok") is True
        and final_status_verified
        and trusted_callbacks
        and callback_codes_ok
    )
    return {
        "status": "passed" if passed else "failed",
        "path": str(path),
        "passed": passed,
        "ok": payload.get("ok") is True,
        "mode": mode,
        "local_server": local_server,
        "final_status": final_status,
        "receipt_count": receipt_count,
        "callback_status_codes": callback_status_codes,
        "runtime_statuses": runtime_statuses,
        "trusted_callbacks": trusted_callbacks,
        "final_status_verified": final_status_verified,
    }


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    items: list[int] = []
    for item in value:
        try:
            items.append(int(item))
        except (TypeError, ValueError):
            continue
    return items


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _archive_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "event_count": 0,
            "scenario_ids": [],
            "sources": [],
            "close_approval_count": 0,
            "close_request_count": 0,
            "report_timeline_ready_count": 0,
            "action_audit_count": 0,
            "operator_action_event_count": 0,
            "trusted_device_event_count": 0,
            "trusted_device_ids": [],
        }
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    scenario_ids = sorted({str(item.get("scenario_id") or "") for item in events if item.get("scenario_id")})
    sources = sorted({
        str((item.get("payload") or {}).get("source") or "")
        for item in events
        if isinstance(item.get("payload"), dict) and (item.get("payload") or {}).get("source")
    })
    close_approval_count = sum(
        1
        for item in events
        if isinstance(item.get("close_approval"), dict)
        and item.get("close_approval", {}).get("approved") is True
    )
    close_request_count = sum(1 for item in events if item.get("close_requested_at"))
    report_timeline_ready_count = sum(
        1
        for item in events
        if item.get("created_at")
        and item.get("closed_at")
        and item.get("closed_by")
        and isinstance(item.get("delivery_report"), list)
    )
    action_audit_count = sum(
        len(item.get("action_audit"))
        for item in events
        if isinstance(item.get("action_audit"), list)
    )
    operator_action_event_count = sum(1 for item in events if _event_has_operator_action(item))
    trusted_device_ids = sorted({
        str((item.get("payload") or {}).get("device_trust", {}).get("device_id") or "")
        for item in events
        if isinstance(item.get("payload"), dict)
        and isinstance((item.get("payload") or {}).get("device_trust"), dict)
        and (item.get("payload") or {}).get("device_trust", {}).get("trusted") is True
        and (item.get("payload") or {}).get("device_trust", {}).get("device_id")
    })
    return {
        "path": str(path),
        "exists": True,
        "event_count": len(events),
        "scenario_ids": scenario_ids,
        "sources": sources,
        "close_approval_count": close_approval_count,
        "close_request_count": close_request_count,
        "report_timeline_ready_count": report_timeline_ready_count,
        "action_audit_count": action_audit_count,
        "operator_action_event_count": operator_action_event_count,
        "trusted_device_event_count": len(trusted_device_ids),
        "trusted_device_ids": trusted_device_ids,
        "latest_event_id": str(events[-1].get("event_id") or "") if events else "",
        "latest_status": str(events[-1].get("status") or "") if events else "",
    }


def _event_has_operator_action(event: dict[str, Any]) -> bool:
    if event.get("acknowledged_at") or event.get("close_requested_at") or event.get("closed_at"):
        return True
    resends = event.get("notification_resends")
    return isinstance(resends, list) and bool(resends)


def _notification_summary(webhooks: dict[str, str], secrets: dict[str, str]) -> dict[str, Any]:
    groups: dict[str, dict[str, bool]] = {}
    for group in ("security", "cleaning", "operations"):
        groups[group] = {
            "webhook_configured": bool(webhooks.get(group)),
            "secret_configured": bool(secrets.get(group)),
        }
    return {"groups": groups}


def _device_trust_summary(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("device_registry") or config.get("field_devices") or config.get("devices")
    devices = raw if isinstance(raw, dict) else {}
    signature_ready_device_ids: list[str] = []
    unsigned_device_ids: list[str] = []
    missing_secret_device_ids: list[str] = []
    signature_disabled_device_ids: list[str] = []
    for key, value in devices.items():
        item = value if isinstance(value, dict) else {}
        device_id = str(key)
        require_signature = bool(item.get("require_signature", True))
        has_secret = bool(_configured_secret(item.get("hmac_secret") or item.get("secret")))
        if require_signature and has_secret:
            signature_ready_device_ids.append(device_id)
            continue
        unsigned_device_ids.append(device_id)
        if require_signature and not has_secret:
            missing_secret_device_ids.append(device_id)
        if not require_signature:
            signature_disabled_device_ids.append(device_id)
    return {
        "registered_device_count": len(devices),
        "signed_device_count": len(signature_ready_device_ids),
        "unsigned_device_count": len(unsigned_device_ids),
        "signature_ready_device_ids": sorted(signature_ready_device_ids),
        "unsigned_device_ids": sorted(unsigned_device_ids),
        "missing_secret_device_ids": sorted(missing_secret_device_ids),
        "signature_disabled_device_ids": sorted(signature_disabled_device_ids),
        "all_registered_devices_signature_ready": bool(devices) and not unsigned_device_ids,
        "require_trusted_devices": bool(config.get("require_trusted_devices")) or _deployment_mode(config) == "production",
    }


def _configured_secret(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null", "false", "0"}:
        return ""
    if raw.startswith("${") and raw.endswith("}"):
        return os.getenv(raw[2:-1].strip(), "").strip()
    return raw


def _runtime_callback_summary(config: dict[str, Any]) -> dict[str, Any]:
    callbacks = (
        config.get("runtime_callbacks")
        if isinstance(config.get("runtime_callbacks"), dict)
        else {}
    )
    secret = str(
        callbacks.get("hmac_secret")
        or callbacks.get("secret")
        or config.get("runtime_callback_hmac_secret")
        or config.get("field_runtime_callback_hmac_secret")
        or os.getenv("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET")
        or ""
    ).strip()
    return {
        "signature_alg": "hmac-sha256",
        "secret_configured": bool(secret),
        "env": "ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET",
    }


def _next_actions(blockers: list[str], warnings: list[str]) -> list[str]:
    actions: list[str] = []
    if any("scenario evaluation" in item for item in blockers):
        actions.append("Run: python -m askme runtime field-eval --json")
    if any("HTTP smoke" in item for item in blockers):
        actions.append("Run: python -m askme runtime field-ingest-smoke --json")
    if any("voice smoke" in item for item in blockers):
        actions.append("Run: python -m askme runtime field-voice-smoke --json")
    if any("archive" in item for item in blockers):
        actions.append("Trigger or ingest one field event and verify /api/field/events")
    if any("action audit integrity" in item for item in blockers):
        actions.append("Check /api/field/audit/integrity and repair or export the audit chain")
    if any("audit delivery retry queue" in item for item in blockers):
        actions.append("Run: python -m askme runtime field-audit-retry-delivery --json")
        actions.append("Inspect: python -m askme runtime field-audit-retry-status --fail-on-pending")
    if any("unified audit review queue" in item for item in blockers):
        actions.append("Open Dashboard Delivery > Unified Audit and resolve the review queue")
        actions.append("Export a reviewed audit package after high-risk records are resolved")
    if any("unified audit review log integrity" in item for item in blockers):
        actions.append("Inspect artifacts/audit/reviews.jsonl and restore the review log from backup")
    if any("unified audit sources" in item for item in blockers):
        actions.append("Open Dashboard Delivery > Audit Source Health and repair invalid audit JSONL records")
        actions.append("Regenerate the unified audit export after all audit sources are readable")
    if any("audit is not HMAC-signed" in item for item in warnings):
        actions.append("Set ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET and rerun the operator action flow")
    if any("webhook" in item for item in warnings):
        actions.append("Configure DingTalk webhooks for security, cleaning, and operations")
    if any("secret" in item for item in warnings):
        actions.append("Configure DingTalk signing secrets for production robots")
    if any("HTTP smoke" in item and "temporary local server" in item for item in warnings):
        actions.append("Run field-ingest-smoke against the deployed service with --server")
    if any("voice smoke" in item and "temporary local server" in item for item in warnings):
        actions.append("Run field-voice-smoke against the deployed service with --server")
    if any("recorded voice handler" in item for item in warnings):
        actions.append("Run field-voice-smoke with --live-tts on the target audio device")
    if any("notification smoke has not passed" in item for item in warnings):
        actions.append("Run: python -m askme runtime field-notification-smoke --json")
    if any("notification smoke" in item and "temporary local server" in item for item in warnings):
        actions.append("Run field-notification-smoke against the deployed service with --server")
    if any("local webhook collector" in item for item in warnings):
        actions.append("Run field-notification-smoke against real DingTalk credentials")
    if any("runtime roundtrip smoke" in item for item in blockers):
        actions.append(
            "Run: python scripts\\eval\\smoke_field_runtime_roundtrip.py "
            "--start-local-server --secret <secret>"
        )
    if any("runtime roundtrip smoke" in item and "temporary local server" in item for item in warnings):
        actions.append(
            "Run field runtime roundtrip smoke against the deployed service with --base-url"
        )
    if any("runtime callback HMAC secret" in item for item in warnings):
        actions.append("Set ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET for lab/prod runtime callbacks")
    if any("field site profile:" in item and "environment variable" in item for item in warnings):
        actions.append("Set site profile environment variables for DingTalk responders and field devices")
    if any("every registered field device" in item for item in blockers) or any(
        "unsigned or unsecreted devices" in item for item in warnings
    ):
        actions.append("Require signatures and configure HMAC secrets for every registered field device")
    if any("close approval workflow" in item for item in warnings):
        actions.append("Create a P0 field event, request close approval, then close it as a supervisor")
    if any("report timeline evidence" in item for item in warnings):
        actions.append("Generate one closed field event report and verify its timeline")
    if any("hardware_dispatch=false" in item for item in warnings):
        actions.append("Run the same bridge with real camera, sensor, and robot diagnostic JSONL")
    if any("external_services=false" in item for item in warnings):
        actions.append("Run a real DingTalk notification-test with production credentials")
    return actions
