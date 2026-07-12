"""Run customer field-operation scenarios through the real HTTP API.

The runner is meant for product demos and pre-sales validation. It does not
pretend to use live hardware by default; it drives the same HTTP contracts that
camera/VMS, smoke sensors, robot diagnostics, Dashboard operators, and visitor
flows must use in a deployment.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.pipeline.field_operations import FieldOperationsService, sign_field_device_payload

from askme.health_server import build_health_snapshot, create_health_app

DEFAULT_OUTPUT_DIR = Path("artifacts/field_operations/live-demo")
DEFAULT_SITE_PROFILE = Path("deploy/site-profiles/park-demo.yaml")


class _DemoDispatcher:
    """Local dispatcher that records notification intent without external calls."""

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
        _ = message, severity, topic, payload
        self.last_delivery_report = [
            {"channel": "dingtalk", "status": "sent", "reason": "demo_local"},
            {"channel": "log", "status": "sent", "reason": ""},
        ]
        return ["dingtalk", "log"]


class _InProcessHttp:
    def __init__(self, output_dir: Path, site_profile: Path) -> None:
        service = FieldOperationsService(
            config=_local_service_config(output_dir, site_profile),
            alert_dispatcher_factory=_DemoDispatcher,
        )
        app = create_health_app(
            lambda: build_health_snapshot(
                app_name="askme",
                app_version="live-demo",
                model_name="demo-model",
                metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
                active_skills=[],
                voice_status={"enabled": True, "pipeline_ok": True},
            ),
            field_operations_handler=service,
        )
        self._client = TestClient(app)

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        response = self._client.get(path)
        return response.status_code, _response_json(response)

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = self._client.post(path, json=payload)
        return response.status_code, _response_json(response)


class _RemoteHttp:
    def __init__(self, server: str, timeout_s: float) -> None:
        self._server = server.rstrip("/")
        self._timeout_s = timeout_s

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        response = requests.get(f"{self._server}{path}", timeout=self._timeout_s)
        return response.status_code, _response_json(response)

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = requests.post(
            f"{self._server}{path}",
            json=payload,
            timeout=self._timeout_s,
        )
        return response.status_code, _response_json(response)


def run_live_demo(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    site_profile: Path = DEFAULT_SITE_PROFILE,
    server: str = "",
    timeout_s: float = 8.0,
    scenario_file: Path | None = None,
    refresh_scenario_timestamps: bool = False,
) -> dict[str, Any]:
    """Run the product demo through HTTP and write evidence artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    client: _InProcessHttp | _RemoteHttp
    mode = "remote_server" if server else "inprocess_http"
    client = _RemoteHttp(server, timeout_s) if server else _InProcessHttp(output_dir, site_profile)

    scenarios = _load_scenarios(scenario_file) if scenario_file else _demo_scenarios()
    scenario_results = [
        _run_http_scenario(
            client,
            scenario,
            refresh_timestamps=refresh_scenario_timestamps,
        )
        for scenario in scenarios
    ]
    events_status, events = client.get("/api/field/events?limit=50")
    readiness_status, readiness = client.get("/api/field/readiness")
    devices_status, devices = client.get("/api/field/devices")
    reports = _collect_reports(client, scenario_results)
    payload = {
        "status": _demo_status(scenario_results),
        "mode": mode,
        "server": server,
        "site_profile": str(site_profile),
        "scenario_source": str(scenario_file) if scenario_file else "built_in",
        "refresh_scenario_timestamps": refresh_scenario_timestamps,
        "output_dir": str(output_dir),
        "scenario_count": len(scenario_results),
        "accepted": sum(1 for item in scenario_results if item.get("accepted")),
        "failed": sum(1 for item in scenario_results if not item.get("accepted")),
        "scenarios": scenario_results,
        "events_status": events_status,
        "events": events,
        "readiness_status": readiness_status,
        "readiness": readiness,
        "devices_status": devices_status,
        "devices": devices,
        "reports": reports,
        "product_notes": [
            "这条验收会走真实 HTTP 接口，和摄像头、烟感、机器人诊断、现场操作员接入使用同一套事件入口。",
            "本地 in-process 模式证明软件链路已通，但不是硬件验收。现场验收要使用 --server 指向部署服务，并接入真实设备和外部服务。",
        ],
    }
    report_path = output_dir / "live-field-demo.json"
    guide_path = output_dir / "live-field-demo.md"
    html_report_path = output_dir / "live-field-demo.html"
    payload["report_path"] = str(report_path)
    payload["guide_path"] = str(guide_path)
    payload["html_report_path"] = str(html_report_path)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    guide_path.write_text(_render_markdown(payload), encoding="utf-8")
    html_report_path.write_text(_render_html(payload), encoding="utf-8")
    return payload


def _run_http_scenario(
    client: _InProcessHttp | _RemoteHttp,
    scenario: dict[str, Any],
    *,
    refresh_timestamps: bool = False,
) -> dict[str, Any]:
    payload = dict(scenario["payload"])
    if refresh_timestamps or scenario.get("refresh_timestamps"):
        payload["observed_at"] = time.time()
    if scenario.get("device_secret"):
        payload.setdefault("device_signature_timestamp", time.time())
        payload["device_signature"] = sign_field_device_payload(
            payload,
            secret=str(scenario["device_secret"]),
        )
    status_code, body = client.post(str(scenario["path"]), payload)
    event = body.get("event") if isinstance(body.get("event"), dict) else {}
    normalized = body.get("normalized") if isinstance(body.get("normalized"), dict) else {}
    return {
        "scenario_id": scenario["scenario_id"],
        "customer_scene": scenario["customer_scene"],
        "path": scenario["path"],
        "http_status": status_code,
        "accepted": status_code < 400 and body.get("accepted", True) is not False,
        "event_id": str(event.get("event_id") or ""),
        "normalized_scenario_id": normalized.get("scenario_id"),
        "incident_topic": event.get("incident_topic"),
        "priority": event.get("priority"),
        "severity": event.get("severity"),
        "notification_group": event.get("notification_group"),
        "voice_text": _voice_text(event),
        "runtime_status": _runtime_status(body),
        "delivery_report": event.get("delivery_report", []),
        "response_reason": body.get("reason", ""),
    }


def _collect_reports(
    client: _InProcessHttp | _RemoteHttp,
    scenario_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for item in scenario_results:
        event_id = item.get("event_id")
        if not event_id:
            continue
        status_code, body = client.get(f"/api/field/events/{event_id}/report")
        reports.append({
            "event_id": event_id,
            "status_code": status_code,
            "found": bool(body.get("found", status_code == 200)),
            "markdown_chars": len(str(body.get("markdown") or "")),
        })
    return reports


def _demo_scenarios() -> list[dict[str, Any]]:
    now = time.time()
    return [
        {
            "scenario_id": "fire_or_smoke",
            "customer_scene": "Fire and smoke sensor alert",
            "path": "/api/field/ingest",
            "device_secret": "smoke-secret",
            "payload": {
                "source": "sensor",
                "device_id": "smoke-warehouse-a",
                "observed_at": now,
                "sensor": {"temperature_c": 72, "smoke_level": 0.9},
                "zone_id": "warehouse-a",
                "location": "Warehouse A",
                "image_path": "artifacts/evidence/smoke.jpg",
            },
        },
        {
            "scenario_id": "illegal_parking",
            "customer_scene": "Vehicle illegally parked on a main road",
            "path": "/api/field/ingest",
            "device_secret": "camera-secret",
            "payload": {
                "source": "camera",
                "device_id": "camera-main-road-1",
                "observed_at": now,
                "zone_id": "main-road-1",
                "detections": [{"label": "vehicle", "confidence": 0.95}],
                "duration_s": 180,
                "plate_number": "DEMO-123",
                "image_path": "artifacts/evidence/car.jpg",
            },
        },
        {
            "scenario_id": "robot_abnormal_incident",
            "customer_scene": "Robot joint motor fault",
            "path": "/api/field/ingest",
            "device_secret": "robot-secret",
            "payload": {
                "source": "robot",
                "device_id": "robot-thunder-1",
                "observed_at": now,
                "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
                "zone_id": "warehouse-a",
                "location": "Warehouse A",
            },
        },
        {
            "scenario_id": "crowd_gathering",
            "customer_scene": "Crowd gathering near a main channel",
            "path": "/api/field/ingest",
            "device_secret": "camera-secret",
            "payload": {
                "source": "camera",
                "device_id": "camera-main-road-1",
                "observed_at": now,
                "zone_id": "main-road-1",
                "person_count": 8,
                "duration_min": 35,
                "detections": [{"label": "person", "confidence": 0.93, "count": 8}],
                "image_path": "artifacts/evidence/crowd.jpg",
            },
        },
        {
            "scenario_id": "wayfinding_help_point",
            "customer_scene": "Visitor asks for directions at a help point",
            "path": "/api/field/events",
            "payload": {
                "scenario_id": "wayfinding_help_point",
                "trigger_source": "visitor_help_point",
                "operator_id": "dashboard.operator",
                "service_point_id": "guide-point-01",
                "location": "North Gate Help Point",
                "requested_destination": "Service Center",
                "question": "Where is the service center?",
            },
        },
    ]


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    scenarios = raw.get("scenarios") if isinstance(raw, dict) else raw
    if not isinstance(scenarios, list):
        raise ValueError("scenario file must be a JSON array or an object with a scenarios array")
    loaded: list[dict[str, Any]] = []
    for index, item in enumerate(scenarios, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"scenario {index} must be an object")
        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"scenario {index} must contain a payload object")
        path_value = str(item.get("path") or "/api/field/ingest")
        if path_value not in {"/api/field/ingest", "/api/field/events"}:
            raise ValueError(f"scenario {index} uses unsupported path: {path_value}")
        loaded.append({
            "scenario_id": str(item.get("scenario_id") or f"scenario_{index}"),
            "customer_scene": str(item.get("customer_scene") or item.get("scenario_id") or f"Scenario {index}"),
            "path": path_value,
            "device_secret": item.get("device_secret"),
            "refresh_timestamps": bool(item.get("refresh_timestamps")),
            "payload": payload,
        })
    if not loaded:
        raise ValueError("scenario file must contain at least one scenario")
    return loaded


def _local_service_config(output_dir: Path, site_profile: Path) -> dict[str, Any]:
    return {
        "archive_path": str(output_dir / "field-events.jsonl"),
        "site_profile_path": str(site_profile),
        "dingtalk_webhooks": {
            "security": "http://demo.local/security",
            "cleaning": "http://demo.local/cleaning",
            "operations": "http://demo.local/operations",
        },
        "dingtalk_secrets": {
            "security": "demo-security",
            "cleaning": "demo-cleaning",
            "operations": "demo-operations",
        },
        "device_registry": {
            "smoke-warehouse-a": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "smoke-secret",
                "require_signature": True,
            },
            "camera-main-road-1": {
                "allowed_sources": ["camera"],
                "hmac_secret": "camera-secret",
                "require_signature": True,
            },
            "robot-thunder-1": {
                "allowed_sources": ["robot"],
                "hmac_secret": "robot-secret",
                "require_signature": True,
            },
        },
        "require_trusted_devices": True,
        "action_audit": {
            "enabled": True,
            "path": str(output_dir / "field-action-audit.jsonl"),
            "hmac_secret": "demo-action-audit",
            "swallow_errors": False,
        },
    }


def _demo_status(scenarios: list[dict[str, Any]]) -> str:
    return "passed" if scenarios and all(item.get("accepted") for item in scenarios) else "failed"


def _voice_text(event: dict[str, Any]) -> str:
    directive = event.get("voice_directive") if isinstance(event.get("voice_directive"), dict) else {}
    return str(directive.get("text") or event.get("voice_text") or "")


def _runtime_status(body: dict[str, Any]) -> str:
    delivery = body.get("runtime_delivery") if isinstance(body.get("runtime_delivery"), dict) else {}
    return str(delivery.get("status") or "")


def _response_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"raw": getattr(response, "text", "")}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Askme 现场场景验收报告",
        "",
        f"- 验收结果: {_status_label(str(payload.get('status') or ''))}",
        f"- 运行方式: {_mode_label(str(payload.get('mode') or ''))}",
        f"- 通过场景: {payload.get('accepted')}/{payload.get('scenario_count')}",
        f"- 交付状态: {payload.get('readiness', {}).get('status')}",
        "",
        "## 场景结果",
    ]
    for item in payload.get("scenarios", []):
        lines.extend([
            "",
            f"### {_display_scene(item)}",
            f"- 接口结果: HTTP {item.get('http_status')} / {'已接收' if item.get('accepted') else '未接收'}",
            f"- 事件编号: {item.get('event_id') or '-'}",
            f"- 通知对象: {_group_label(str(item.get('notification_group') or ''))}",
            f"- 播报话术: {item.get('voice_text') or '-'}",
        ])
    lines.extend([
        "",
        "## 产品说明",
    ])
    for note in payload.get("product_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _render_html(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "")
    mode = str(payload.get("mode") or "")
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    scenario_cards = "\n".join(_scenario_html(item) for item in payload.get("scenarios", []))
    note_rows = "\n".join(f"<li>{escape(str(note))}</li>" for note in payload.get("product_notes", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Askme 现场场景验收报告</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f4f8f6; color: #10251f; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ background: linear-gradient(135deg, #073d31, #12664f); color: white; border-radius: 20px; padding: 28px; box-shadow: 0 18px 50px rgba(8, 54, 43, .22); }}
    h1 {{ margin: 0 0 10px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 22px; }}
    .metric {{ background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.24); border-radius: 14px; padding: 14px; }}
    .metric span {{ display: block; opacity: .78; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 20px; }}
    .section {{ margin-top: 22px; background: white; border: 1px solid #dce8e3; border-radius: 18px; padding: 20px; }}
    .scenarios {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #dce8e3; border-radius: 14px; padding: 16px; background: #fbfefd; }}
    .ok {{ color: #0c7a4b; font-weight: 700; }}
    .warn {{ color: #a15c00; font-weight: 700; }}
    .label {{ color: #587168; font-size: 13px; margin-top: 10px; }}
    .voice {{ margin-top: 8px; padding: 10px 12px; background: #ecf7f2; border-radius: 10px; }}
    code {{ background: #eef4f1; border-radius: 6px; padding: 2px 5px; }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Askme 现场场景验收报告</h1>
    <p>这份报告用于演示、实验室验收和部署前自检。它说明哪些现场场景已经打通，哪些仍需要真实设备和外部服务补齐。</p>
    <div class="summary">
      <div class="metric"><span>验收结果</span><strong>{escape(_status_label(status))}</strong></div>
      <div class="metric"><span>通过场景</span><strong>{escape(str(payload.get('accepted')))} / {escape(str(payload.get('scenario_count')))}</strong></div>
      <div class="metric"><span>运行方式</span><strong>{escape(_mode_label(mode))}</strong></div>
      <div class="metric"><span>交付状态</span><strong>{escape(str(readiness.get('status') or '-'))}</strong></div>
    </div>
  </section>
  <section class="section">
    <h2>场景验收</h2>
    <div class="scenarios">{scenario_cards}</div>
  </section>
  <section class="section">
    <h2>真实接入说明</h2>
    <ul>{note_rows}</ul>
    <p>JSON 证据：<code>{escape(str(payload.get('report_path') or '-'))}</code></p>
  </section>
</main>
</body>
</html>
"""


def _scenario_html(item: dict[str, Any]) -> str:
    accepted = bool(item.get("accepted"))
    return f"""<article class="card">
  <h3>{escape(_display_scene(item))}</h3>
  <div class="{'ok' if accepted else 'warn'}">{'已接收并生成事件' if accepted else '未通过'}</div>
  <div class="label">接口结果</div>
  <div>HTTP {escape(str(item.get('http_status') or '-'))} · 事件 {escape(str(item.get('event_id') or '-'))}</div>
  <div class="label">通知对象</div>
  <div>{escape(_group_label(str(item.get('notification_group') or '')))}</div>
  <div class="label">现场播报</div>
  <div class="voice">{escape(str(item.get('voice_text') or '-'))}</div>
</article>"""


def _status_label(status: str) -> str:
    return "已通过" if status == "passed" else "未通过"


def _mode_label(mode: str) -> str:
    return "本地软件闭环" if mode == "inprocess_http" else "部署服务"


def _group_label(group: str) -> str:
    return {
        "security": "安保",
        "cleaning": "保洁",
        "operations": "运维",
        "none": "无需通知",
        "": "无需通知",
    }.get(group, group)


def _scene_label(scenario_id: str) -> str:
    return {
        "fire_or_smoke": "火灾/烟雾异常",
        "illegal_parking": "车辆违停",
        "robot_abnormal_incident": "机器人关节电机故障",
        "crowd_gathering": "人群聚集",
        "wayfinding_help_point": "路人指路",
    }.get(scenario_id, scenario_id)


def _display_scene(item: dict[str, Any]) -> str:
    scenario_id = str(item.get("scenario_id") or "")
    label = _scene_label(scenario_id)
    if label != scenario_id:
        return label
    return str(item.get("customer_scene") or scenario_id or "未命名场景")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE_PROFILE)
    parser.add_argument("--server", default="", help="Existing Askme runtime base URL")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=None,
        help="Replay customer/device scenario JSON instead of the built-in demo scenarios",
    )
    parser.add_argument(
        "--refresh-scenario-timestamps",
        action="store_true",
        help="Refresh observed_at in replayed scenarios for demo-only freshness gates",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args()

    payload = run_live_demo(
        output_dir=args.output_dir,
        site_profile=args.site_profile,
        server=args.server,
        timeout_s=args.timeout,
        scenario_file=args.scenario_file,
        refresh_scenario_timestamps=args.refresh_scenario_timestamps,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # noqa: T201
        return
    print(  # noqa: T201
        "live-field-demo: "
        f"{payload['status']} accepted={payload['accepted']}/{payload['scenario_count']} "
        f"mode={payload['mode']}"
    )
    print(f"report: {payload['report_path']}")  # noqa: T201
    print(f"guide: {payload['guide_path']}")  # noqa: T201
    print(f"html: {payload['html_report_path']}")  # noqa: T201


if __name__ == "__main__":
    main()
