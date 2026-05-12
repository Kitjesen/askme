"""Evaluate deterministic field-operation scenarios and write an auditable artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.pipeline.field_operations import FieldOperationsService  # noqa: E402

DEFAULT_REPORT_PATH = Path("artifacts/field_operations/scenario-evaluation.json")

CUSTOMER_SCENARIO_METADATA: dict[str, dict[str, Any]] = {
    "robot_immobilized_notifies_security": {
        "customer_name": "机器人摔倒/卡住无法恢复",
        "trigger_source": "机器人诊断事件",
        "expected_robot_action": "立即停止运动，语音提示现场人员，并等待保安或运维到场处理。",
        "expected_notification": "通知保安群",
        "expected_archive": True,
    },
    "night_stranger_photo_archived": {
        "customer_name": "夜间陌生人拍照",
        "trigger_source": "夜间摄像头识别 + 停留时长",
        "expected_robot_action": "拍照取证，记录地点，通知保安复核。",
        "expected_notification": "通知保安群",
        "expected_archive": True,
    },
    "illegal_parking_camera_ingest": {
        "customer_name": "车辆违停检测",
        "trigger_source": "摄像头车辆识别 + 园区停车区域规则",
        "expected_robot_action": "拍照记录违停位置，播报提醒，通知保安处理。",
        "expected_notification": "通知保安群",
        "expected_archive": True,
    },
    "fire_sensor_notifies_security": {
        "customer_name": "火灾及烟雾监测",
        "trigger_source": "温度/烟雾传感器",
        "expected_robot_action": "高优先级告警，拍照上传，提示远离风险区域。",
        "expected_notification": "通知保安群",
        "expected_archive": True,
    },
    "trash_bin_full_notifies_cleaning": {
        "customer_name": "垃圾桶满溢监测",
        "trigger_source": "定点垃圾桶图像识别",
        "expected_robot_action": "记录垃圾桶编号和照片，通知保洁处理。",
        "expected_notification": "通知保洁群",
        "expected_archive": True,
    },
    "urgent_patrol_dispatch_notifies_operations": {
        "customer_name": "突发任务巡检",
        "trigger_source": "管理员派遣任务",
        "expected_robot_action": "中断当前巡检，生成突发巡检事件，准备交给运行调度。",
        "expected_notification": "通知运营群",
        "expected_archive": True,
    },
    "crowd_gathering_records_security_event": {
        "customer_name": "人群聚集检测",
        "trigger_source": "人数识别 + 停留时长",
        "expected_robot_action": "记录现场照片和人数，必要时语音提醒并通知保安。",
        "expected_notification": "通知保安群",
        "expected_archive": True,
    },
    "wayfinding_help_point_does_not_notify_security": {
        "customer_name": "路人指路",
        "trigger_source": "固定路引帮助点位 + 停留检测",
        "expected_robot_action": "主动询问是否需要指路，只回答地点路线，不误触发巡检任务。",
        "expected_notification": "不通知保安",
        "expected_archive": True,
    },
    "visitor_escort_is_archived_without_alert": {
        "customer_name": "路人带路",
        "trigger_source": "游客目的地请求 + 园区地图路线",
        "expected_robot_action": "按园区路线带路，记录服务过程，不当成安全告警。",
        "expected_notification": "不通知保安",
        "expected_archive": True,
    },
    "notification_smoke_test_reports_delivery": {
        "customer_name": "钉钉通知链路试发",
        "trigger_source": "运维测试按钮",
        "expected_robot_action": "不触发机器人运动，只验证保安群通知配置。",
        "expected_notification": "通知保安群",
        "expected_archive": False,
    },
}

REAL_INTEGRATION_GAPS = [
    "真实摄像头/VMS 事件流",
    "真实烟雾/温度/MQTT 传感器",
    "生产钉钉机器人 Webhook 与签名密钥",
    "真实机器狗 runtime / 硬件回调",
    "MiniMax 线上语音播报 key、音频设备和现场声学验证",
    "园区地图、停车区域、路引点位的客户现场配置",
]


class _ScenarioDispatcher:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.last_delivery_report: list[dict[str, Any]] = []

    def dispatch(
        self,
        message: str,
        *,
        severity: str = "info",
        topic: str = "",
        payload: dict[str, Any] | None = None,
    ) -> list[str]:
        self.calls.append({
            "message": message,
            "severity": severity,
            "topic": topic,
            "payload": payload or {},
            "config": self.kwargs.get("config", {}),
        })
        webhook = str((self.kwargs.get("config") or {}).get("dingtalk_webhook") or "")
        sent = bool(webhook)
        self.last_delivery_report = [
            {
                "channel": "dingtalk",
                "status": "sent" if sent else "not_sent",
                "reason": "" if sent else "not_configured",
            },
            {"channel": "log", "status": "sent", "reason": ""},
        ]
        return ["dingtalk", "log"] if sent else ["log"]


async def evaluate_scenarios() -> dict[str, Any]:
    """Run deterministic field scenarios without external services or hardware."""
    with tempfile.TemporaryDirectory(prefix="askme-field-ops-") as temp_dir:
        root = Path(temp_dir)
        service = _service(root)
        scenarios = [
            await _scenario_robot_immobilized(service),
            await _scenario_night_stranger_photo(service),
            await _scenario_illegal_parking_from_camera(service),
            await _scenario_fire_sensor(service),
            await _scenario_trash_bin_full(service),
            await _scenario_urgent_patrol_dispatch(service),
            await _scenario_crowd_gathering(service),
            await _scenario_wayfinding_help_point(service),
            await _scenario_visitor_escort(service),
            await _scenario_notification_smoke_test(service),
        ]
    passed = sum(1 for item in scenarios if item["passed"])
    product_demo = _product_demo_summary(scenarios)
    return {
        "suite": "askme-field-operations",
        "external_services": False,
        "hardware_dispatch": False,
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": len(scenarios) - passed,
        "status": "passed" if passed == len(scenarios) else "failed",
        "scenarios": scenarios,
        "product_demo": product_demo,
        "generated_at": time.time(),
    }


def write_report(payload: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _product_demo_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    customer_scenarios = [_customer_scenario_row(item) for item in scenarios]
    demo_ready = all(item.get("passed") for item in customer_scenarios)
    return {
        "suite_name": "园区机器狗场景演示包",
        "demo_ready": demo_ready,
        "real_integration_ready": False,
        "customer_scenario_count": len(customer_scenarios),
        "passed": sum(1 for item in customer_scenarios if item.get("passed")),
        "failed": sum(1 for item in customer_scenarios if not item.get("passed")),
        "customer_scenarios": customer_scenarios,
        "blocked_on_real_integrations": REAL_INTEGRATION_GAPS,
        "how_to_run": (
            "python scripts/eval/evaluate_field_operations_scenarios.py "
            "--output artifacts/field_operations/scenario-evaluation.json"
        ),
        "evidence": {
            "external_services": False,
            "hardware_dispatch": False,
            "report_path": str(DEFAULT_REPORT_PATH),
            "runtime": "deterministic fake dispatcher; no physical robot command is sent",
        },
        "customer_claim_boundary": (
            "当前可证明的是场景决策、播报文案、通知分组、归档和审计链路；"
            "真实摄像头、传感器、钉钉生产群、MiniMax 播报和机器狗运动仍需现场接入验收。"
        ),
    }


def _customer_scenario_row(scenario: dict[str, Any]) -> dict[str, Any]:
    name = str(scenario.get("name") or "")
    meta = CUSTOMER_SCENARIO_METADATA.get(name, {})
    observed = scenario.get("observed") if isinstance(scenario.get("observed"), dict) else {}
    event = observed.get("event") if isinstance(observed.get("event"), dict) else {}
    normalized = observed.get("normalized") if isinstance(observed.get("normalized"), dict) else {}
    delivery_report = event.get("delivery_report") or observed.get("delivery_report") or []
    runtime_delivery = event.get("runtime_delivery") if isinstance(event.get("runtime_delivery"), dict) else {}
    evidence_media = event.get("evidence_media") if isinstance(event.get("evidence_media"), list) else []
    workflow = event.get("incident_workflow") if isinstance(event.get("incident_workflow"), dict) else {}
    return {
        "name": name,
        "customer_name": str(meta.get("customer_name") or name),
        "passed": bool(scenario.get("passed")),
        "trigger_source": str(meta.get("trigger_source") or normalized.get("source") or "manual"),
        "expected_robot_action": str(meta.get("expected_robot_action") or ""),
        "expected_notification": str(meta.get("expected_notification") or ""),
        "expected_archive": bool(meta.get("expected_archive", False)),
        "actual": {
            "accepted": bool(observed.get("accepted") or observed.get("sent")),
            "status": str(observed.get("status") or event.get("status") or ""),
            "location": str(event.get("location") or normalized.get("location") or ""),
            "voice": str(event.get("voice") or ""),
            "notification_group": str(event.get("notification_group") or observed.get("notification_group") or ""),
            "sent_channels": list(event.get("sent_channels") or observed.get("sent_channels") or []),
            "delivery_status": _delivery_status(delivery_report),
            "archive_required": bool(event.get("archive_required", False)),
            "robot_motion_policy": str(
                (event.get("playbook") or {}).get("robot_motion_policy")
                if isinstance(event.get("playbook"), dict)
                else ""
            ),
            "runtime_submission_status": str(runtime_delivery.get("status") or "not_connected"),
            "workflow_state": str(workflow.get("state") or ""),
            "workflow_open_gaps": list(workflow.get("open_gaps") or []),
        },
        "evidence": {
            "event_id": str(event.get("event_id") or ""),
            "incident_topic": str(event.get("incident_topic") or ""),
            "scenario_id": str(event.get("scenario_id") or normalized.get("scenario_id") or ""),
            "priority": str(event.get("priority") or ""),
            "severity": str(event.get("severity") or ""),
            "media_count": len(evidence_media),
            "delivery_report": delivery_report,
        },
    }


def _delivery_status(delivery_report: Any) -> str:
    if not isinstance(delivery_report, list) or not delivery_report:
        return "not_required"
    statuses = [
        str(item.get("status") or "")
        for item in delivery_report
        if isinstance(item, dict)
    ]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "sent" for status in statuses):
        return "sent"
    if any(status == "not_sent" for status in statuses):
        return "not_sent"
    return "unknown"


def _service(root: Path) -> FieldOperationsService:
    _ScenarioDispatcher.calls = []
    return FieldOperationsService(
        config={
            "archive_path": str(root / "field-events.jsonl"),
            "dingtalk_webhooks": {
                "security": "http://security.local/ding",
                "cleaning": "http://cleaning.local/ding",
                "operations": "http://operations.local/ding",
            },
            "dingtalk_secrets": {
                "security": "SEC-security",
                "cleaning": "SEC-cleaning",
                "operations": "SEC-operations",
            },
            "site_map": {
                "zones": {
                    "main-road-1": {
                        "name": "B区主通道",
                        "type": "main_channel",
                        "parking_allowed": False,
                    },
                    "guide-01": {
                        "name": "游客中心路引点",
                        "location": "游客中心",
                        "help_point_id": "guide-01",
                    },
                }
            },
        },
        alert_dispatcher_factory=_ScenarioDispatcher,
    )


async def _scenario_robot_immobilized(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "robot_abnormal_incident",
        "location": "A区东侧主通道",
        "fault_type": "immobilized",
        "duration_s": 18,
        "image_path": "artifacts/evidence/robot-stuck.jpg",
    })
    event = result.get("event", {})
    return _verdict(
        "robot_immobilized_notifies_security",
        result.get("accepted") is True
        and event.get("incident_topic") == "navigation.immobilized"
        and event.get("notification_group") == "security"
        and "dingtalk" in event.get("sent_channels", []),
        observed=result,
    )


async def _scenario_night_stranger_photo(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "night_stranger_photo",
        "location": "北侧一层窗户",
        "zone_name": "北侧窗户",
        "duration_s": 42,
        "image_path": "artifacts/evidence/night-stranger.jpg",
        "confidence": 0.86,
    })
    event = result.get("event", {})
    return _verdict(
        "night_stranger_photo_archived",
        result.get("accepted") is True
        and event.get("incident_topic") == "security.night_stranger_photo"
        and event.get("archive_required") is True,
        observed=result,
    )


async def _scenario_illegal_parking_from_camera(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.ingest_payload({
        "source": "camera",
        "observed_at": time.time(),
        "zone_id": "main-road-1",
        "detections": [{"label": "vehicle", "confidence": 0.92}],
        "duration_s": 180,
        "image_path": "artifacts/evidence/car.jpg",
    })
    event = result.get("event", {})
    return _verdict(
        "illegal_parking_camera_ingest",
        result.get("accepted") is True
        and result.get("normalized", {}).get("scenario_id") == "illegal_parking"
        and event.get("incident_topic") == "traffic.illegal_parking"
        and event.get("location") == "B区主通道",
        observed=result,
    )


async def _scenario_fire_sensor(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.ingest_payload({
        "source": "sensor",
        "observed_at": time.time(),
        "sensor": {"temperature_c": 68, "smoke_level": 0.8},
        "location": "配电间门口",
        "image_path": "artifacts/evidence/smoke.jpg",
    })
    event = result.get("event", {})
    return _verdict(
        "fire_sensor_notifies_security",
        result.get("accepted") is True
        and result.get("normalized", {}).get("scenario_id") == "fire_or_smoke"
        and event.get("incident_topic") == "safety.fire_or_smoke",
        observed=result,
    )


async def _scenario_trash_bin_full(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "trash_bin_full",
        "location": "C区西门",
        "bin_id": "bin-c-02",
        "fill_ratio": "92%",
        "image_path": "artifacts/evidence/trash-full.jpg",
    })
    event = result.get("event", {})
    return _verdict(
        "trash_bin_full_notifies_cleaning",
        result.get("accepted") is True
        and event.get("notification_group") == "cleaning"
        and event.get("incident_topic") == "sanitation.trash_bin_full",
        observed=result,
    )


async def _scenario_urgent_patrol_dispatch(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "urgent_patrol_dispatch",
        "target_location": "A区北门",
        "operator_id": "admin-1",
        "interrupted_mission_id": "patrol-night-01",
        "reason": "保安要求复核",
    })
    event = result.get("event", {})
    return _verdict(
        "urgent_patrol_dispatch_notifies_operations",
        result.get("accepted") is True
        and event.get("notification_group") == "operations"
        and event.get("incident_topic") == "patrol.urgent_dispatch",
        observed=result,
    )


async def _scenario_crowd_gathering(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "crowd_gathering",
        "location": "游客中心门口",
        "person_count": 8,
        "duration_min": 35,
        "image_path": "artifacts/evidence/crowd.jpg",
    })
    event = result.get("event", {})
    return _verdict(
        "crowd_gathering_records_security_event",
        result.get("accepted") is True
        and event.get("incident_topic") == "security.crowd_gathering"
        and event.get("priority") == "P1",
        observed=result,
    )


async def _scenario_wayfinding_help_point(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "wayfinding_help_point",
        "help_point_id": "guide-01",
        "location": "游客中心路引牌",
        "destination": "停车场",
        "dwell_s": 12,
    })
    event = result.get("event", {})
    return _verdict(
        "wayfinding_help_point_does_not_notify_security",
        result.get("accepted") is True
        and event.get("notification_group") == "none"
        and event.get("sent_channels") == [],
        observed=result,
    )


async def _scenario_visitor_escort(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.trigger_payload({
        "scenario_id": "visitor_escort",
        "location": "游客中心",
        "destination": "北门停车场",
        "route_id": "route-visitor-north-parking",
    })
    event = result.get("event", {})
    return _verdict(
        "visitor_escort_is_archived_without_alert",
        result.get("accepted") is True
        and event.get("notification_group") == "none"
        and event.get("archive_required") is True,
        observed=result,
    )


async def _scenario_notification_smoke_test(service: FieldOperationsService) -> dict[str, Any]:
    result = await service.test_notification_payload({
        "notification_group": "security",
        "operator_id": "scenario",
        "message": "现场通知试发",
    })
    return _verdict(
        "notification_smoke_test_reports_delivery",
        result.get("sent") is True
        and result.get("secret_configured") is True
        and result.get("delivery_report", [{}])[0].get("status") == "sent",
        observed=result,
    )


def _verdict(name: str, passed: bool, *, observed: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    payload = asyncio.run(evaluate_scenarios())
    report = write_report(payload, args.output)
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
