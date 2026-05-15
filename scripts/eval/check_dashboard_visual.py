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

from askme.health_server import build_health_snapshot, create_health_app
from askme.pipeline.field_operations import FieldOperationsService
from askme.skills.skill_manager import SkillManager
from askme.voice.tts import TTSEngine

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
    field_ops = FieldOperationsService(config={"archive_path": str(archive_path)})
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
        "现场事件闭环看板",
        "客户现在能看什么",
        "知识库",
        "现场事件",
        "语音音色",
        "交付检查",
    ]
    required = [
        "Customer acceptance view",
        "Field operations",
        "Customer projects",
        "Knowledge base",
        "Voice profiles",
        "Delivery checks",
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
        "has_customer_package_catalog": "Customer Enablement Packages" in body_before,
        "has_capability_page": "场景能力蓝图" in body_before,
        "has_inline_gate": "Enablement gate" in body_before,
        "has_release_summary": "Production claim" in body_before
        and "Production launch claims" in body_before,
        "has_release_claim_copy": "Release claim:" in body_before or "Release claim:" in body_text,
        "has_recheck_button": "Recheck enablement" in body_before,
        "has_readiness_panel": "Enablement Check" in body_text,
        "has_package_status": "package ready" in body_text or "package blocked" in body_text,
        "has_missing_dependency_copy": "Missing" in body_text,
        "has_next_step_copy": "Next step:" in body_text,
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
        "has_audit_dossier": "Customer Delivery Audit Dossier" in body_text,
        "has_allowed_uses": "Allowed use" in body_text,
        "has_blocked_uses": "Blocked claim" in body_text,
        "has_production_claim_boundary": "unattended production launch claim" in body_text,
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
        resource_binding_added = resource_id in resource_binding_value and "Added" in resource_binding_result
    body_text = page.locator("body").inner_text(timeout=5000)
    overflow = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
    )
    screenshot_path = output_dir / "askme-dashboard-projects-workspace.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return {
        "name": "project_workspace",
        "screenshot": str(screenshot_path),
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
        "has_separate_artifact_inputs": all(
            text in body_text
            for text in ("项目交付包", "客户提案包", "验收证据包")
        ),
        "has_evidence_picker": "加入引用" in body_text and "先点击“查看现场证据”" in body_text,
        "has_delivery_resource_registry": "Delivery resource registry" in body_text
        and "Register resource" in body_text,
        "has_resource_action_plan": "Resource binding action plan" in body_text,
        "has_resource_governance": "Resource governance" in body_text
        and "Request rollback" in body_text
        and "Approval queue" in body_text,
        "resource_governance_history_loaded": governance_history_loaded,
        "resource_governance_rollback_previewed": governance_rollback_previewed,
        "resource_governance_disable_handled": governance_disable_handled,
        "resource_governance_requests_loaded": governance_requests_loaded,
        "resource_governance_sla_visible": resource_governance_sla_visible,
        "resource_governance_escalated": resource_governance_escalated,
        "has_object_resource_picker": "Bind registered resource" in body_text
        and "Add binding" in body_text,
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
                interactions.append(_exercise_interactions(page, output_dir=output_dir))
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
        if "has_template_governance" in interaction and not interaction["has_template_governance"]:
            failures.append(f"{interaction['name']}: missing template governance")
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
