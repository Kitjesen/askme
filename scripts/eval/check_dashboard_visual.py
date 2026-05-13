"""Browser-level visual smoke check for the product dashboard."""

from __future__ import annotations

import argparse
import json
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
        if not interaction["has_submit_result"]:
            failures.append(f"{interaction['name']}: missing submit result")
        if not interaction["has_event_context"]:
            failures.append(f"{interaction['name']}: missing event context")
        if not interaction["has_customer_language"]:
            failures.append(f"{interaction['name']}: missing customer language")
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
