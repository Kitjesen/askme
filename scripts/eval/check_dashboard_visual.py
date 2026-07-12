"""Browser-level visual smoke check for the product dashboard."""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests
import uvicorn
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.pipeline.field_operations import FieldOperationsService
from askme.skills.skill_manager import SkillManager
from askme.voice.tts import TTSEngine

from askme.health_server import build_health_snapshot, create_health_app

_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c49444154789c63606060000000040001f61738550000000049454e44ae426082"
)


class _VisualRuntimeHandler:
    def context_payload(self) -> dict[str, Any]:
        return {
            "profile": "sim",
            "current_profile": "sim",
            "active_run": {"run_id": "visual-run-1", "current_state": "executing"},
        }

    def events_payload(self, *, after: Any = None, limit: int = 20) -> dict[str, Any]:
        return {
            "profile": "sim",
            "cursor": time.time(),
            "events": [
                {
                    "event_id": "visual-runtime-event-1",
                    "run_id": "visual-run-1",
                    "event_type": "task_executing",
                    "state": "executing",
                    "message": "visual smoke runtime event",
                    "created_at": time.time(),
                }
            ],
            "event_count": 1,
            "active_run": {"run_id": "visual-run-1", "current_state": "executing"},
        }

    def profiles_payload(self) -> dict[str, Any]:
        return {"current_profile": "sim", "profiles": [{"name": "fake"}, {"name": "sim"}]}

    def list_payload(self) -> dict[str, Any]:
        return {"runs": [{"run_id": "visual-run-1"}], "count": 1}

    def get_payload(self, run_id: str) -> dict[str, Any]:
        return {"run": {"run_id": run_id, "current_state": "executing"}}

    def report_payload(self, run_id: str) -> dict[str, Any]:
        return {"report": {"run_id": run_id, "status": "executing"}}

    def pause_payload(self, run_id: str, **_: Any) -> dict[str, Any]:
        return {"handled": True, "run": {"run_id": run_id, "current_state": "paused"}}

    def resume_payload(self, run_id: str, **_: Any) -> dict[str, Any]:
        return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}

    def cancel_payload(self, run_id: str, **_: Any) -> dict[str, Any]:
        return {"handled": True, "run": {"run_id": run_id, "current_state": "cancelled"}}

    def advance_payload(self, run_id: str, **_: Any) -> dict[str, Any]:
        return {"handled": True, "run": {"run_id": run_id, "current_state": "executing"}}


class _VisualCognitionHandler:
    def context_payload(self, *, refresh_perception: bool = False) -> dict[str, Any]:
        return {
            "world_state": {"fact_count": 1, "facts": [{"key": "area", "value": "B main road"}]},
            "working_memory": {"items": []},
            "planning_sessions": [],
            "refresh_perception": refresh_perception,
        }


def _free_loopback_port() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
    return str(host), int(port)


def _health_snapshot() -> dict[str, Any]:
    return build_health_snapshot(
        app_name="askme",
        app_version="visual-smoke",
        model_name="visual-smoke-model",
        metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
        active_skills=[],
        voice_status={"enabled": True, "pipeline_ok": True},
    )


def _capabilities_payload() -> dict[str, Any]:
    manager = SkillManager()
    manager.load()
    return {
        "profile": {"name": "visual-smoke", "primary_loop": "dashboard"},
        "components": {
            "skills": {
                "health": {"status": "ok"},
                "capabilities": {"openapi_generated": True},
            }
        },
        "skills": {
            "catalog": manager.get_contract_catalog(),
            "capability_center": manager.get_capability_center(),
            "skill_packages": manager.get_skill_packages(),
        },
    }


def _start_server(archive_path: Path) -> dict[str, Any]:
    host, port = _free_loopback_port()
    voice = TTSEngine(
        {
            "backend": "edge",
            "minimax_voice_id": "male-qn-qingse",
            "voice_profile_state_path": str(archive_path.with_name("voice-profile.json")),
        }
    )
    voice.speak = lambda text: None  # type: ignore[method-assign]
    voice.start_playback = lambda: None  # type: ignore[method-assign]
    field_ops = FieldOperationsService(
        config={
            "archive_path": str(archive_path),
            "customer_project": {
                "tenant_id": "default",
                "delivery_namespace": "default",
                "customer_id": "demo-customer",
                "project_id": "demo-field-ops",
                "site_id": "inovx-demo-park",
                "site_name": "Visual Smoke Park",
            },
        }
    )
    app = create_health_app(
        _health_snapshot,
        capabilities_provider=_capabilities_payload,
        field_operations_handler=field_ops,
        cognition_handler=_VisualCognitionHandler(),
        runtime_handler=_VisualRuntimeHandler(),
        voice_handler=voice,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    deadline = time.time() + 8
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            requests.get(f"{base_url}/health", timeout=0.5).raise_for_status()
            return {"server": server, "thread": thread, "voice": voice, "base_url": base_url}
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    voice.shutdown()
    raise RuntimeError(f"dashboard visual smoke server did not start: {last_error}")


def _seed_field_event(base_url: str, output_dir: Path) -> None:
    demo_evidence_path = ROOT / "artifacts" / "evidence" / "dashboard-field-demo.jpg"
    demo_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    demo_evidence_path.write_bytes(_ONE_PIXEL_PNG)
    evidence_path = output_dir / "visual-illegal-parking.png"
    evidence_path.write_bytes(_ONE_PIXEL_PNG)
    payload = {
        "scenario_id": "illegal_parking",
        "location": "B main road",
        "zone_name": "main road",
        "plate_number": "A12345",
        "duration_s": 180,
        "image_path": str(evidence_path).replace("\\", "/"),
        "created_at": time.time(),
    }
    response = requests.post(f"{base_url}/api/field/events", json=payload, timeout=5)
    response.raise_for_status()
    blocked_payload = {
        "scenario_id": "night_stranger_photo",
        "location": "visual north window",
        "zone_name": "visual north window",
        "operator_id": "dashboard.operator",
    }
    blocked_response = requests.post(f"{base_url}/api/field/events", json=blocked_payload, timeout=5)
    if blocked_response.status_code not in {200, 422}:
        blocked_response.raise_for_status()
    blocked_body = blocked_response.json()
    if "event" not in blocked_body:
        raise RuntimeError(f"blocked field event did not archive an event: {blocked_body}")
    stale_evidence_path = output_dir / "visual-stale-smoke.png"
    stale_evidence_path.write_bytes(_ONE_PIXEL_PNG)
    stale_payload = {
        "source": "sensor",
        "observed_at": time.time() - 120,
        "sensor": {"temperature_c": 72, "smoke_level": 0.9},
        "location": "visual transformer room",
        "image_path": str(stale_evidence_path).replace("\\", "/"),
    }
    stale_response = requests.post(f"{base_url}/api/field/ingest", json=stale_payload, timeout=5)
    if stale_response.status_code not in {200, 422}:
        stale_response.raise_for_status()
    stale_body = stale_response.json()
    if "event" not in stale_body:
        raise RuntimeError(f"stale ingest did not archive an event: {stale_body}")


def _check_viewport(page: Any, *, name: str, width: int, height: int, output_dir: Path) -> dict[str, Any]:
    page.set_viewport_size({"width": width, "height": height})
    page.goto("/dashboard", wait_until="domcontentloaded")
    page.wait_for_selector("#dashboard-nav", timeout=5000)
    page.wait_for_selector(".product-shell", timeout=5000)
    page.wait_for_timeout(600)
    screenshot_path = output_dir / f"askme-dashboard-{name}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    body_text = page.locator("body").inner_text(timeout=5000)
    required = [
        "现场任务平台",
        "客户验收视角",
        "客户项目",
        "知识库",
        "现场事件",
        "语音音色",
        "交付检查",
    ]
    missing = [text for text in required if text not in body_text]
    overflow = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
    )
    has_horizontal_overflow = int(overflow["scrollWidth"]) > int(overflow["clientWidth"]) + 2
    bad_text = any(marker in body_text for marker in ("????", "\ufffd"))
    return {
        "name": name,
        "viewport": {"width": width, "height": height},
        "screenshot": str(screenshot_path),
        "missing_required_text": missing,
        "has_bad_text_marker": bad_text,
        "has_horizontal_overflow": has_horizontal_overflow,
        "scroll_width": overflow["scrollWidth"],
        "client_width": overflow["clientWidth"],
    }


def _exercise_interactions(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Exercise one real product interaction through the browser page."""

    page.goto("/dashboard/field", wait_until="domcontentloaded")
    page.wait_for_selector("#field-submit", timeout=5000)
    page.fill("#field-location", "B区主通道")
    page.fill("#field-note", "浏览器实际交互 smoke：车辆停在主通道")
    page.click("#field-submit")
    page.wait_for_selector(".field-detail-card", timeout=8000)
    page.wait_for_timeout(800)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-field-interaction.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "field_event_create",
        "screenshot": str(screenshot_path),
        "has_submit_result": "操作已提交" in body_text,
        "has_event_context": "B区主通道" in body_text,
        "has_customer_language": "现场事件处置" in body_text,
    }


_FIELD_SCENARIO_MATRIX = (
    {
        "scenario_id": "wayfinding_help_point",
        "location": "西门问询点",
        "note": "矩阵验收：游客问咖啡店在哪",
        "markers": ("路人指路", "问路", "guide-point-01"),
    },
    {
        "scenario_id": "visitor_escort",
        "location": "主入口服务点",
        "note": "矩阵验收：游客请求带路去服务中心",
        "markers": ("路人带路", "带路", "demo-route-01"),
    },
    {
        "scenario_id": "illegal_parking",
        "location": "B区主通道",
        "note": "矩阵验收：车辆停在主通道",
        "markers": ("车辆违停", "违停", "DEMO-123"),
    },
    {
        "scenario_id": "fire_or_smoke",
        "location": "3号楼一层",
        "note": "矩阵验收：检测到烟雾和高温",
        "markers": ("火灾", "烟雾", "72"),
    },
    {
        "scenario_id": "trash_bin_full",
        "location": "西门垃圾桶",
        "note": "矩阵验收：垃圾桶满溢",
        "markers": ("垃圾桶", "满溢", "trash-bin-demo"),
    },
    {
        "scenario_id": "night_stranger_photo",
        "location": "北侧窗边",
        "note": "矩阵验收：夜间陌生人在窗边拍照",
        "markers": ("陌生人", "夜间", "北侧窗边"),
    },
    {
        "scenario_id": "crowd_gathering",
        "location": "中央广场",
        "note": "矩阵验收：人数超过阈值并长时间停留",
        "markers": ("人群聚集", "聚集", "8"),
    },
    {
        "scenario_id": "urgent_patrol_dispatch",
        "location": "A区北门",
        "note": "矩阵验收：管理员临时派遣巡检",
        "markers": ("突发任务巡检", "派遣", "dashboard-task"),
    },
    {
        "scenario_id": "robot_abnormal_incident",
        "location": "A区坡道",
        "note": "矩阵验收：机器狗卡住无法运动",
        "markers": ("机器人异常", "卡住", "immobilized"),
    },
)


def _exercise_field_scenario_matrix(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Submit the customer-critical field scenarios through the real Dashboard form."""

    page.goto("/dashboard/field", wait_until="domcontentloaded")
    page.wait_for_selector("#field-submit", timeout=8000)
    page.wait_for_function(
        "() => document.querySelectorAll('#field-scenario option').length > 0"
    )
    option_values = set(
        page.eval_on_selector_all(
            "#field-scenario option",
            "(options) => options.map((option) => option.value)",
        )
    )
    covered: list[str] = []
    missing_options: list[str] = []
    scenario_results: dict[str, bool] = {}

    for item in _FIELD_SCENARIO_MATRIX:
        scenario_id = item["scenario_id"]
        if scenario_id not in option_values:
            missing_options.append(scenario_id)
            scenario_results[scenario_id] = False
            continue

        page.select_option("#field-scenario", scenario_id)
        page.fill("#field-location", item["location"])
        page.fill("#field-note", item["note"])
        page.click("#field-submit")
        page.wait_for_selector(".field-detail-card", timeout=8000)
        page.wait_for_timeout(500)
        body_text = page.locator("body").inner_text(timeout=5000)
        markers = item["markers"]
        passed = item["location"] in body_text and any(marker in body_text for marker in markers)
        scenario_results[scenario_id] = passed
        if passed:
            covered.append(scenario_id)
        page.wait_for_selector("#field-submit", timeout=8000)

    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-field-scenario-matrix.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "field_scenario_matrix",
        "screenshot": str(screenshot_path),
        "scenario_count": len(_FIELD_SCENARIO_MATRIX),
        "covered_scenarios": covered,
        "missing_scenarios": [
            item["scenario_id"]
            for item in _FIELD_SCENARIO_MATRIX
            if not scenario_results.get(item["scenario_id"], False)
        ],
        "missing_options": missing_options,
        "has_submit_result": "操作已提交" in body_text,
        "has_event_context": all(item["location"] in body_text for item in _FIELD_SCENARIO_MATRIX),
        "has_customer_language": "现场事件处置" in body_text and "最近现场事件" in body_text,
        "has_illegal_parking": scenario_results.get("illegal_parking", False),
        "has_fire_or_smoke": scenario_results.get("fire_or_smoke", False),
        "has_trash_bin_full": scenario_results.get("trash_bin_full", False),
        "has_night_stranger_photo": scenario_results.get("night_stranger_photo", False),
        "has_wayfinding_help_point": scenario_results.get("wayfinding_help_point", False),
        "has_visitor_escort": scenario_results.get("visitor_escort", False),
        "has_crowd_gathering": scenario_results.get("crowd_gathering", False),
        "has_urgent_patrol_dispatch": scenario_results.get("urgent_patrol_dispatch", False),
        "has_robot_abnormal_incident": scenario_results.get("robot_abnormal_incident", False),
        "has_notification_or_delivery": any(marker in body_text for marker in ("通知", "送达", "保安", "保洁")),
        "has_evidence_or_archive": any(marker in body_text for marker in ("证据", "归档", "照片", "事件")),
    }


def _exercise_scenario_acceptance_page(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Verify the customer scenario page exposes acceptance boundaries and site gaps."""

    page.goto("/dashboard/scenarios", wait_until="domcontentloaded")
    page.wait_for_selector(".scenario-acceptance-strip", timeout=8000)
    page.wait_for_selector(".scenario-product-grid", timeout=8000)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-scenario-acceptance.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "scenario_acceptance_page",
        "screenshot": str(screenshot_path),
        "has_acceptance_boundary": "验收边界" in body_text and "演示与集成验收" in body_text,
        "has_production_boundary": "无人值守生产上线" in body_text or "生产上线" in body_text,
        "has_real_dependency_copy": "真实接入还缺什么" in body_text,
        "has_device_entrypoint_copy": "设备入口" in body_text,
        "has_all_customer_scenarios": all(
            marker in body_text
            for marker in (
                "路人指路",
                "路人带路",
                "车辆违停",
                "火灾",
                "垃圾桶",
                "夜间陌生人",
                "突发任务巡检",
                "机器人异常",
            )
        ),
    }


def _exercise_field_admission_decision(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Verify blocked/non-escalated events show a customer-readable reason."""

    page.goto("/dashboard/field", wait_until="domcontentloaded")
    page.wait_for_selector(".field-detail-card", timeout=8000)
    stale_row = page.locator(".field-event-row", has_text="visual transformer room").first
    if stale_row.count():
        stale_row.locator(".field-event-select").click()
        page.wait_for_function(
            "() => document.body && document.body.innerText.includes('visual transformer room')"
        )
    page.wait_for_selector(".field-admission-card", timeout=8000)
    page.wait_for_selector(".field-admission-facts span", timeout=8000)
    page.wait_for_selector(".field-ingest-scope-card", timeout=8000)
    page.wait_for_timeout(500)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-field-admission-decision.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "field_admission_decision",
        "screenshot": str(screenshot_path),
        "has_admission_card": page.locator(".field-admission-card").count() > 0,
        "has_admission_facts": page.locator(".field-admission-facts span").count() > 0,
        "has_block_reason_fact": "freshness" in body_text and "stale" in body_text,
        "has_next_step": page.locator(".field-admission-card .muted-line").count() > 0,
        "has_blocked_event_context": "visual transformer room" in body_text,
        "has_ingest_scope_card": page.locator(".field-ingest-scope-card").count() > 0,
        "has_ingest_scope_grid": page.locator(".field-ingest-scope-grid div").count() >= 4,
        "has_ingest_scope_gate": (
            "managed_object_binding_required" in body_text
            or "设备接入还不能作为生产验收证据" in body_text
            or "生产验收证据" in body_text
        ),
    }


def _exercise_conversation_wayfinding(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Exercise the customer chat page with a deterministic park-space answer."""

    page.goto("/dashboard/conversation", wait_until="domcontentloaded")
    page.wait_for_selector("#chat-service-point", timeout=8000)
    page.wait_for_function(
        "() => document.querySelectorAll('#chat-service-point option').length > 0"
    )
    if page.locator('#chat-service-point option[value="guide-west-gate"]').count():
        page.select_option("#chat-service-point", "guide-west-gate")
    page.fill("#chat-input", "咖啡店在哪")
    page.click("#chat-send")
    page.wait_for_selector(".chat-message.assistant", timeout=8000)
    page.wait_for_timeout(700)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-conversation-wayfinding.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "conversation_wayfinding",
        "screenshot": str(screenshot_path),
        "has_context_selector": "现场上下文" in body_text and "当前问询点" in body_text,
        "has_space_answer": "梵木咖啡" in body_text,
        "has_evidence": "回答依据" in body_text and "园区空间认知库" in body_text,
        "has_no_guide_policy": "只回答，不启动带路" in body_text,
        "has_no_chat_503": "chat not available" not in body_text and "服务没有返回可展示内容" not in body_text,
    }


def _exercise_capability_readiness(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Exercise scenario package readiness from the customer capability page."""

    page.goto("/dashboard/capabilities", wait_until="domcontentloaded")
    page.wait_for_selector("[data-scenario-readiness]", timeout=8000)
    page.wait_for_timeout(800)
    body_before = page.locator("body").inner_text(timeout=5000)
    page.locator("[data-scenario-readiness]").first.click()
    page.wait_for_selector(".scenario-readiness-panel", timeout=8000)
    page.wait_for_timeout(500)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-capability-readiness.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "capability_scenario_readiness",
        "screenshot": str(screenshot_path),
        "has_customer_package_catalog": "客户可启用能力包" in body_before,
        "has_capability_page": "场景能力蓝图" in body_before,
        "has_inline_gate": "启用准入" in body_before,
        "has_release_summary": "生产声明" in body_before and "发布声明规则" in body_before,
        "has_release_claim_copy": "交付声明：" in body_before or "交付声明：" in body_text,
        "has_recheck_button": "重新检查" in body_before,
        "has_readiness_panel": "启用检查" in body_text,
        "has_package_status": "交付包" in body_before or "交付包" in body_text,
        "has_missing_dependency_copy": "缺失" in body_text,
        "has_next_step_copy": "下一步：" in body_text,
    }


def _exercise_audit_delivery_dossier(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Exercise the customer-facing audit delivery contract on the audit page."""

    page.goto("/dashboard/audit", wait_until="domcontentloaded")
    page.wait_for_selector(".audit-delivery-dossier", timeout=8000)
    page.wait_for_timeout(500)
    body_text = page.locator("body").inner_text(timeout=5000)
    screenshot_path = output_dir / "askme-dashboard-audit-dossier.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "audit_delivery_dossier",
        "screenshot": str(screenshot_path),
        "has_audit_dossier": "Customer Delivery Audit Dossier" in body_text
        or "客户交付审计材料" in body_text,
        "has_allowed_uses": "允许用途" in body_text,
        "has_blocked_uses": "禁止声明" in body_text,
        "has_production_claim_boundary": "无人值守生产上线声明" in body_text
        or "unattended production launch claim" in body_text,
    }


def _exercise_project_workspace(page: Any, *, output_dir: Path) -> dict[str, Any]:
    """Exercise the customer-project workspace information architecture."""

    _install_resource_governance_route_mocks(page)
    page.goto("/dashboard/projects", wait_until="domcontentloaded")
    page.wait_for_selector(".project-page-nav", timeout=8000)
    page.wait_for_selector("#project-section-package", timeout=8000)
    page.wait_for_timeout(700)
    governance_history_loaded = False
    governance_rollback_previewed = False
    governance_disable_handled = False
    governance_requests_loaded = False
    resource_governance_sla_visible = False
    resource_governance_escalated = False
    if page.locator("[data-resource-history]").count():
        page.click("[data-resource-history]")
        page.wait_for_timeout(200)
        governance_text = page.locator("#resource-governance-result").inner_text(timeout=5000)
        governance_history_loaded = "visual-rev-001" in governance_text
    if page.locator("#resource-rollback-id").count():
        page.fill("#resource-rollback-id", "visual-rev-001")
        page.click('[data-resource-rollback="dry-run"]')
        page.wait_for_timeout(200)
        governance_text = page.locator("#resource-governance-result").inner_text(timeout=5000)
        governance_rollback_previewed = "visual-rev-001" in governance_text and (
            "预演" in governance_text or "Dry" in governance_text or "尚未写入" in governance_text
        )
    if page.locator("[data-resource-disable]").count():
        page.once("dialog", lambda dialog: dialog.accept("visual smoke disable"))
        page.click("[data-resource-disable]")
        page.wait_for_timeout(500)
        governance_text = page.locator("#resource-governance-result").inner_text(timeout=5000)
        governance_disable_handled = (
            "visual-resource-request-001" in governance_text
            or "second delivery owner" in governance_text
            or "pending" in governance_text
        )
    if page.locator("[data-resource-governance-requests]").count():
        page.click("[data-resource-governance-requests]")
        page.wait_for_timeout(200)
        governance_text = page.locator("#resource-governance-result").inner_text(timeout=5000)
        governance_requests_loaded = "visual-resource-request-001" in governance_text
        resource_governance_sla_visible = (
            "SLA" in governance_text
            and ("overdue" in governance_text.lower() or "due" in governance_text.lower())
        ) or (
            "逾期" in governance_text
            and ("visual-resource-request-001" in governance_text or "交付资源治理" in governance_text)
        )
    if page.locator("[data-resource-governance-escalate-overdue]").count():
        page.once("dialog", lambda dialog: dialog.accept("visual smoke overdue escalation"))
        page.click("[data-resource-governance-escalate-overdue]")
        page.wait_for_timeout(300)
        governance_text = page.locator("#resource-governance-result").inner_text(timeout=5000)
        resource_governance_escalated = (
            "visual-resource-request-001" in governance_text
            and "escalated" in governance_text.lower()
        )
    resource_binding_added = False
    resource_binding_result = ""
    resource_option = page.evaluate(
        """() => Array.from(document.querySelectorAll('#object-resource-picker option'))
            .map((option) => option.value)
            .find((value) => value.startsWith('vision_models::')) || ''"""
    )
    if resource_option:
        page.select_option("#object-resource-picker", resource_option)
        page.click("[data-object-resource-add]")
        resource_id = resource_option.split("::", 1)[1]
        resource_binding_value = page.locator("#object-vision-models").input_value(timeout=5000)
        resource_binding_result = page.locator("#object-resource-picker-result").inner_text(timeout=5000)
        resource_binding_added = resource_id in resource_binding_value and any(
            marker in resource_binding_result for marker in ("Added", "加入", "已把")
        )
    body_text = page.locator("body").inner_text(timeout=5000)
    overflow = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
    )
    screenshot_path = output_dir / "askme-dashboard-projects-workspace.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "project_workspace",
        "screenshot": str(screenshot_path),
        "has_project_acceptance_snapshot": page.locator("[data-project-acceptance-snapshot]").count() > 0,
        "has_project_acceptance_summary_anchor": page.locator("#project-section-acceptance-summary").count() > 0,
        "has_project_positioning": "客户项目不是配置文件，是交付产品" in body_text,
        "has_workspace_nav": all(
            text in body_text
            for text in (
                "项目目录",
                "模板市场",
                "模板发布",
                "对象目录",
                "导入导出",
                "验收证据",
                "资源绑定",
                "事件归属",
                "多现场",
            )
        ),
        "has_template_governance": "模板发布治理" in body_text and "待复核发布" in body_text,
        "has_template_runtime_blueprint_binding": "运行蓝图绑定" in body_text
        and "blueprint" in body_text
        and "package" in body_text,
        "has_separate_artifact_inputs": all(
            text in body_text
            for text in ("项目交付包", "客户提案包", "验收证据包")
        ),
        "has_evidence_picker": "加入引用" in body_text and "先点击“查看现场证据”" in body_text,
        "has_delivery_resource_registry": (
            "交付资源目录" in body_text or "Delivery resource registry" in body_text
        )
        and ("登记资源" in body_text or "Register resource" in body_text),
        "has_resource_action_plan": "资源绑定行动计划" in body_text
        or "Resource binding action plan" in body_text,
        "has_resource_governance": (
            ("资源治理" in body_text and "回滚申请" in body_text and "审批队列" in body_text)
            or (
                "Resource governance" in body_text
                and "Request rollback" in body_text
                and "Approval queue" in body_text
            )
        ),
        "resource_governance_history_loaded": governance_history_loaded,
        "resource_governance_rollback_previewed": governance_rollback_previewed,
        "resource_governance_disable_handled": governance_disable_handled,
        "resource_governance_requests_loaded": governance_requests_loaded,
        "resource_governance_sla_visible": resource_governance_sla_visible,
        "resource_governance_escalated": resource_governance_escalated,
        "has_object_resource_picker": (
            ("绑定交付资源" in body_text and "加入绑定" in body_text)
            or ("Bind registered resource" in body_text and "Add binding" in body_text)
        ),
        "object_resource_binding_added": resource_binding_added,
        "object_resource_binding_result": resource_binding_result,
        "has_project_scope_copy": "solution provider" in body_text.lower()
        or "方案商" in body_text,
        "has_horizontal_overflow": int(overflow["scrollWidth"]) > int(overflow["clientWidth"]) + 2,
        "scroll_width": overflow["scrollWidth"],
        "client_width": overflow["clientWidth"],
    }


def _install_resource_governance_route_mocks(page: Any) -> None:
    """Mock resource governance actions so visual smoke never mutates real resources."""

    resource_catalog = {
        "summary": {
            "overall_status": "ready",
            "resource_count": 1,
            "used_resource_count": 1,
            "unregistered_resource_count": 0,
            "consumer_count": 1,
        },
        "resources": [
            {
                "resource_type": "vision_models",
                "resource_id": "visual-shared-detector",
                "display_name": "Visual shared detector",
                "status": "registered",
                "publish_status": "published",
                "source": "shared_registry",
                "version": "v1.0.0",
                "owner": "visual.qa",
                "consumer_count": 1,
            }
        ],
        "next_step": "Visual smoke resource catalog is mocked and does not write real files.",
    }
    history_payload = {
        "found": True,
        "revision_count": 1,
        "revisions": [
            {
                "revision_id": "visual-rev-001",
                "created_at": "2026-05-15T00:00:00Z",
                "operator_id": "visual.qa",
                "reason": "visual smoke governance fixture",
                "registry_sha256": "visual-sha256",
            }
        ],
    }
    disable_payload = {
        "accepted": True,
        "resource_type": "vision_models",
        "resource_id": "visual-shared-detector",
        "resource": {
            "resource_type": "vision_models",
            "resource_id": "visual-shared-detector",
            "publish_status": "disabled",
        },
    }
    rollback_payload = {
        "accepted": True,
        "dry_run": True,
        "revision_id": "visual-rev-001",
        "target_summary": {"resource_count": 1},
        "would_write": True,
    }
    governance_request_payload = {
        "accepted": True,
        "request": {
            "request_id": "visual-resource-request-001",
            "status": "pending",
            "action": "disable_resource",
            "operation": {
                "resource_type": "vision_models",
                "resource_id": "visual-shared-detector",
            },
            "requested_by": "visual.qa",
            "requested_at": 1778803200,
            "sla_target_s": 3600,
            "due_at": 1778806800,
            "review_sla": {
                "state": "overdue",
                "target_s": 3600,
                "requested_at": 1778803200,
                "due_at": 1778806800,
                "remaining_s": -7200,
                "overdue_s": 7200,
                "escalation_required": True,
                "escalation_policy": "delivery_owner_review_overdue",
                "message": "Review SLA is overdue; escalate to a delivery owner before applying customer-facing changes.",
            },
            "escalation_policy": "delivery_owner_review_overdue",
            "escalation_count": 0,
            "last_escalation": {},
            "escalations": [],
            "reason": "visual smoke disable",
        },
        "preview": {
            "accepted": True,
            "dry_run": True,
            "target_publish_status": "disabled",
            "impact": {
                "impact_type": "resource_disable",
                "analysis_status": "complete",
                "resource_type": "vision_models",
                "resource_id": "visual-shared-detector",
                "affected_consumer_count": 1,
                "affected_customer_project_count": 1,
                "affected_object_count": 1,
                "affected_template_count": 0,
                "affected_projects": [
                    {
                        "customer_id": "visual-customer",
                        "project_id": "visual-project",
                        "site_id": "visual-site",
                        "consumer_count": 1,
                        "object_count": 1,
                    }
                ],
                "affected_objects": [
                    {
                        "scope_type": "project",
                        "project_id": "visual-project",
                        "object_id": "visual-gate",
                        "display_name": "Visual gate",
                    }
                ],
                "affected_templates": [],
                "affected_consumers": [
                    {
                        "scope_type": "project",
                        "project_id": "visual-project",
                        "object_id": "visual-gate",
                        "display_name": "Visual gate",
                        "status": "linked",
                    }
                ],
                "message": "Affected customer projects, objects, and templates should be reviewed before approval.",
            },
        },
        "next_step": "A second delivery owner must review this request.",
    }
    governance_requests_payload = {
        "request_count": 1,
        "summary": {
            "pending_count": 1,
            "active_count": 0,
            "due_soon_count": 0,
            "overdue_count": 1,
            "approved_count": 0,
            "rejected_count": 0,
        },
        "requests": [
            {
                **governance_request_payload["request"],
                "preview": governance_request_payload["preview"],
            }
        ],
    }
    escalation_payload = {
        "accepted": True,
        "checked_count": 1,
        "escalated_count": 1,
        "skipped_count": 0,
        "escalations": [
            {
                "escalation_id": "visual-escalation-001",
                "status": "queued",
                "request_id": "visual-resource-request-001",
                "target": "vision_models/visual-shared-detector",
                "overdue_s": 7200,
                "delivery_group": "delivery_owners",
                "escalated_by": "visual.qa",
                "escalated_at": 1778810400,
                "notification": {
                    "channel": "delivery_owner_queue",
                    "status": "queued",
                    "message": "Delivery resource governance request is overdue: visual-resource-request-001.",
                },
            }
        ],
        "requests": [
            {
                **governance_request_payload["request"],
                "preview": governance_request_payload["preview"],
                "escalation_count": 1,
                "last_escalation": {
                    "escalation_id": "visual-escalation-001",
                    "status": "queued",
                    "request_id": "visual-resource-request-001",
                    "delivery_group": "delivery_owners",
                    "escalated_by": "visual.qa",
                    "escalated_at": 1778810400,
                    "notification": {
                        "message": "Delivery resource governance request is overdue: visual-resource-request-001.",
                    },
                },
            }
        ],
        "summary": governance_requests_payload["summary"],
        "next_step": "Escalated overdue resource governance requests to the delivery owner queue.",
    }

    def fulfill_json(route: Any, payload: dict[str, Any]) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload, ensure_ascii=False),
        )

    page.route(
        re.compile(r".*/api/field/customer-project-resource-catalog$"),
        lambda route: fulfill_json(route, resource_catalog),
    )
    page.route(
        re.compile(r".*/api/field/delivery-resource-registry/history.*"),
        lambda route: fulfill_json(route, history_payload),
    )
    page.route(
        re.compile(r".*/api/field/delivery-resource-governance-requests(?:\?.*)?$"),
        lambda route: fulfill_json(
            route,
            governance_request_payload
            if route.request.method.upper() == "POST"
            else governance_requests_payload,
        ),
    )
    page.route(
        re.compile(r".*/api/field/delivery-resource-governance-requests/escalate-overdue$"),
        lambda route: fulfill_json(route, escalation_payload),
    )
    page.route(
        re.compile(r".*/api/field/delivery-resource-registry/.*/disable$"),
        lambda route: fulfill_json(route, disable_payload),
    )
    page.route(
        re.compile(r".*/api/field/delivery-resource-registry/rollback$"),
        lambda route: fulfill_json(route, rollback_payload),
    )


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "visual-field-events.jsonl"
    if archive_path.exists():
        archive_path.unlink()
    incident_archive = archive_path.with_name("incident-alerts.jsonl")
    if incident_archive.exists():
        incident_archive.unlink()
    server_ctx = _start_server(archive_path)
    server = server_ctx["server"]
    thread = server_ctx["thread"]
    voice = server_ctx["voice"]
    base_url = str(server_ctx["base_url"])
    console_errors: list[str] = []
    page_errors: list[str] = []
    response_errors: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    try:
        _seed_field_event(base_url, output_dir)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(base_url=base_url)
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on(
                    "response",
                    lambda response: response_errors.append(
                        {"url": response.url, "status": response.status}
                    )
                    if response.status >= 400
                    else None,
                )
                checks.append(_check_viewport(page, name="desktop", width=1600, height=900, output_dir=output_dir))
                interactions.append(_exercise_conversation_wayfinding(page, output_dir=output_dir))
                interactions.append(_exercise_scenario_acceptance_page(page, output_dir=output_dir))
                interactions.append(_exercise_field_admission_decision(page, output_dir=output_dir))
                interactions.append(_exercise_interactions(page, output_dir=output_dir))
                interactions.append(_exercise_field_scenario_matrix(page, output_dir=output_dir))
                interactions.append(_exercise_project_workspace(page, output_dir=output_dir))
                interactions.append(_exercise_capability_readiness(page, output_dir=output_dir))
                interactions.append(_exercise_audit_delivery_dossier(page, output_dir=output_dir))
                checks.append(_check_viewport(page, name="mobile", width=390, height=844, output_dir=output_dir))
            finally:
                browser.close()
    except PlaywrightError as exc:
        return {
            "status": "failed",
            "reason": f"playwright_error:{exc.__class__.__name__}",
            "detail": str(exc),
            "base_url": base_url,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        voice.shutdown()
    failures = []
    for check in checks:
        if check["missing_required_text"]:
            failures.append(f"{check['name']}: missing {check['missing_required_text']}")
        if check["has_bad_text_marker"]:
            failures.append(f"{check['name']}: bad text marker")
        if check["has_horizontal_overflow"]:
            failures.append(f"{check['name']}: horizontal overflow")
    actionable_response_errors = [
        item
        for item in response_errors
        if not str(item["url"]).endswith("/favicon.ico")
        and not (str(item["url"]).endswith("/api/field/notification-preflight") and item["status"] == 409)
    ]
    if actionable_response_errors:
        failures.append(f"response_errors:{len(actionable_response_errors)}")
    if console_errors and actionable_response_errors:
        failures.append(f"console_errors:{len(console_errors)}")
    if page_errors:
        failures.append(f"page_errors:{len(page_errors)}")
    for interaction in interactions:
        if "has_submit_result" in interaction and not interaction["has_submit_result"]:
            failures.append(f"{interaction['name']}: missing submit result")
        if "has_event_context" in interaction and not interaction["has_event_context"]:
            failures.append(f"{interaction['name']}: missing event context")
        if "has_customer_language" in interaction and not interaction["has_customer_language"]:
            failures.append(f"{interaction['name']}: missing customer language")
        if "missing_scenarios" in interaction and interaction["missing_scenarios"]:
            failures.append(
                f"{interaction['name']}: missing scenarios {interaction['missing_scenarios']}"
            )
        if "missing_options" in interaction and interaction["missing_options"]:
            failures.append(
                f"{interaction['name']}: missing scenario options {interaction['missing_options']}"
            )
        if "has_notification_or_delivery" in interaction and not interaction["has_notification_or_delivery"]:
            failures.append(f"{interaction['name']}: missing notification or delivery copy")
        if "has_evidence_or_archive" in interaction and not interaction["has_evidence_or_archive"]:
            failures.append(f"{interaction['name']}: missing evidence or archive copy")
        if "has_context_selector" in interaction and not interaction["has_context_selector"]:
            failures.append(f"{interaction['name']}: missing conversation space context selector")
        if "has_space_answer" in interaction and not interaction["has_space_answer"]:
            failures.append(f"{interaction['name']}: missing park-space answer")
        if "has_evidence" in interaction and not interaction["has_evidence"]:
            failures.append(f"{interaction['name']}: missing conversation evidence")
        if "has_no_guide_policy" in interaction and not interaction["has_no_guide_policy"]:
            failures.append(f"{interaction['name']}: missing no-guide policy")
        if "has_no_chat_503" in interaction and not interaction["has_no_chat_503"]:
            failures.append(f"{interaction['name']}: chat endpoint unavailable in dashboard-only mode")
        if "has_acceptance_boundary" in interaction and not interaction["has_acceptance_boundary"]:
            failures.append(f"{interaction['name']}: missing acceptance boundary")
        if "has_production_boundary" in interaction and not interaction["has_production_boundary"]:
            failures.append(f"{interaction['name']}: missing production boundary")
        if "has_real_dependency_copy" in interaction and not interaction["has_real_dependency_copy"]:
            failures.append(f"{interaction['name']}: missing real dependency copy")
        if "has_device_entrypoint_copy" in interaction and not interaction["has_device_entrypoint_copy"]:
            failures.append(f"{interaction['name']}: missing device entrypoint copy")
        if "has_all_customer_scenarios" in interaction and not interaction["has_all_customer_scenarios"]:
            failures.append(f"{interaction['name']}: missing customer scenarios")
        if "has_admission_card" in interaction and not interaction["has_admission_card"]:
            failures.append(f"{interaction['name']}: missing admission decision card")
        if "has_admission_facts" in interaction and not interaction["has_admission_facts"]:
            failures.append(f"{interaction['name']}: missing admission evidence facts")
        if "has_block_reason_fact" in interaction and not interaction["has_block_reason_fact"]:
            failures.append(f"{interaction['name']}: missing blocked admission reason fact")
        if "has_next_step" in interaction and not interaction["has_next_step"]:
            failures.append(f"{interaction['name']}: missing admission next step")
        if "has_blocked_event_context" in interaction and not interaction["has_blocked_event_context"]:
            failures.append(f"{interaction['name']}: missing blocked event context")
        if "has_ingest_scope_card" in interaction and not interaction["has_ingest_scope_card"]:
            failures.append(f"{interaction['name']}: missing ingest scope card")
        if "has_ingest_scope_grid" in interaction and not interaction["has_ingest_scope_grid"]:
            failures.append(f"{interaction['name']}: missing ingest scope grid")
        if "has_ingest_scope_gate" in interaction and not interaction["has_ingest_scope_gate"]:
            failures.append(f"{interaction['name']}: missing ingest production gate")
        if "has_capability_page" in interaction and not interaction["has_capability_page"]:
            failures.append(f"{interaction['name']}: missing capability page")
        if (
            "has_customer_package_catalog" in interaction
            and not interaction["has_customer_package_catalog"]
        ):
            failures.append(f"{interaction['name']}: missing customer package catalog")
        if "has_inline_gate" in interaction and not interaction["has_inline_gate"]:
            failures.append(f"{interaction['name']}: missing inline readiness gate")
        if "has_release_claim_copy" in interaction and not interaction["has_release_claim_copy"]:
            failures.append(f"{interaction['name']}: missing release claim copy")
        if "has_release_summary" in interaction and not interaction["has_release_summary"]:
            failures.append(f"{interaction['name']}: missing release summary")
        if "has_recheck_button" in interaction and not interaction["has_recheck_button"]:
            failures.append(f"{interaction['name']}: missing recheck button")
        if "has_readiness_panel" in interaction and not interaction["has_readiness_panel"]:
            failures.append(f"{interaction['name']}: missing readiness panel")
        if "has_package_status" in interaction and not interaction["has_package_status"]:
            failures.append(f"{interaction['name']}: missing package status")
        if "has_missing_dependency_copy" in interaction and not interaction["has_missing_dependency_copy"]:
            failures.append(f"{interaction['name']}: missing missing-dependency copy")
        if "has_next_step_copy" in interaction and not interaction["has_next_step_copy"]:
            failures.append(f"{interaction['name']}: missing next-step copy")
        if "has_audit_dossier" in interaction and not interaction["has_audit_dossier"]:
            failures.append(f"{interaction['name']}: missing audit delivery dossier")
        if "has_allowed_uses" in interaction and not interaction["has_allowed_uses"]:
            failures.append(f"{interaction['name']}: missing audit allowed uses")
        if "has_blocked_uses" in interaction and not interaction["has_blocked_uses"]:
            failures.append(f"{interaction['name']}: missing audit blocked uses")
        if (
            "has_production_claim_boundary" in interaction
            and not interaction["has_production_claim_boundary"]
        ):
            failures.append(f"{interaction['name']}: missing production claim boundary")
        if "has_project_positioning" in interaction and not interaction["has_project_positioning"]:
            failures.append(f"{interaction['name']}: missing project positioning")
        if "has_workspace_nav" in interaction and not interaction["has_workspace_nav"]:
            failures.append(f"{interaction['name']}: missing workspace navigation")
        if (
            "has_project_acceptance_snapshot" in interaction
            and not interaction["has_project_acceptance_snapshot"]
        ):
            failures.append(f"{interaction['name']}: missing project acceptance snapshot")
        if (
            "has_project_acceptance_summary_anchor" in interaction
            and not interaction["has_project_acceptance_summary_anchor"]
        ):
            failures.append(f"{interaction['name']}: missing project acceptance summary anchor")
        if "has_template_governance" in interaction and not interaction["has_template_governance"]:
            failures.append(f"{interaction['name']}: missing template governance")
        if (
            "has_template_runtime_blueprint_binding" in interaction
            and not interaction["has_template_runtime_blueprint_binding"]
        ):
            failures.append(f"{interaction['name']}: missing template runtime blueprint binding")
        if (
            "has_separate_artifact_inputs" in interaction
            and not interaction["has_separate_artifact_inputs"]
        ):
            failures.append(f"{interaction['name']}: missing separate artifact inputs")
        if "has_evidence_picker" in interaction and not interaction["has_evidence_picker"]:
            failures.append(f"{interaction['name']}: missing evidence picker")
        if (
            "has_delivery_resource_registry" in interaction
            and not interaction["has_delivery_resource_registry"]
        ):
            failures.append(f"{interaction['name']}: missing delivery resource registry")
        if "has_resource_action_plan" in interaction and not interaction["has_resource_action_plan"]:
            failures.append(f"{interaction['name']}: missing resource action plan")
        if "has_resource_governance" in interaction and not interaction["has_resource_governance"]:
            failures.append(f"{interaction['name']}: missing resource governance")
        if (
            "resource_governance_history_loaded" in interaction
            and not interaction["resource_governance_history_loaded"]
        ):
            failures.append(f"{interaction['name']}: resource governance history did not load")
        if (
            "resource_governance_rollback_previewed" in interaction
            and not interaction["resource_governance_rollback_previewed"]
        ):
            failures.append(f"{interaction['name']}: resource rollback dry-run did not render")
        if (
            "resource_governance_disable_handled" in interaction
            and not interaction["resource_governance_disable_handled"]
        ):
            failures.append(f"{interaction['name']}: resource disable request did not render")
        if (
            "resource_governance_requests_loaded" in interaction
            and not interaction["resource_governance_requests_loaded"]
        ):
            failures.append(f"{interaction['name']}: resource governance queue did not render")
        if (
            "resource_governance_sla_visible" in interaction
            and not interaction["resource_governance_sla_visible"]
        ):
            failures.append(f"{interaction['name']}: resource governance SLA did not render")
        if (
            "resource_governance_escalated" in interaction
            and not interaction["resource_governance_escalated"]
        ):
            failures.append(f"{interaction['name']}: resource governance escalation did not render")
        if "has_object_resource_picker" in interaction and not interaction["has_object_resource_picker"]:
            failures.append(f"{interaction['name']}: missing object resource picker")
        if (
            "object_resource_binding_added" in interaction
            and not interaction["object_resource_binding_added"]
        ):
            failures.append(f"{interaction['name']}: object resource picker did not update binding field")
        if "has_project_scope_copy" in interaction and not interaction["has_project_scope_copy"]:
            failures.append(f"{interaction['name']}: missing project scope copy")
        if "has_horizontal_overflow" in interaction and interaction["has_horizontal_overflow"]:
            failures.append(f"{interaction['name']}: horizontal overflow")
    result = {
        "status": "passed" if not failures else "failed",
        "base_url": base_url,
        "checks": checks,
        "interactions": interactions,
        "console_errors": console_errors[:20],
        "page_errors": page_errors[:20],
        "response_errors": response_errors[:20],
        "failures": failures,
    }
    (output_dir / "dashboard-visual-smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="output/playwright",
        help="Directory for screenshots and JSON evidence.",
    )
    args = parser.parse_args()
    result = run(Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
