"""Smoke-test FieldIncident -> runtime handoff -> signed callback archive roundtrip."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib import request

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient

from askme.cognition import WorldStateService
from askme.health_server import build_health_snapshot, create_health_app
from askme.pipeline.field_operations import FieldOperationsService
from askme.runtime.field_callbacks import (
    build_field_runtime_callback_sequence,
    field_event_id_from_runtime_result,
    post_field_runtime_callback_sequence,
)
from askme.runtime.handoff import RuntimeHandoffService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="", help="Existing askme HTTP base URL for live mode")
    parser.add_argument(
        "--start-local-server",
        action="store_true",
        help="Start a temporary uvicorn askme server, then run live HTTP smoke against it.",
    )
    parser.add_argument("--secret", default=os.getenv("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET", ""))
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--scenario-id", default="illegal_parking")
    parser.add_argument("--location", default="main road")
    parser.add_argument("--zone-name", default="main channel")
    parser.add_argument("--image-path", default="artifacts/evidence/car.jpg")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.secret:
        raise SystemExit("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET or --secret is required")
    event_body = {
        "scenario_id": args.scenario_id,
        "location": args.location,
        "zone_name": args.zone_name,
        "plate_number": "SMOKE-001",
        "image_path": args.image_path,
    }
    if args.start_local_server:
        result = _run_local_server_roundtrip(
            event_body=event_body,
            secret=args.secret,
            timeout_s=args.timeout_s,
        )
    elif args.base_url:
        result = _run_live_roundtrip(
            base_url=args.base_url,
            event_body=event_body,
            secret=args.secret,
            timeout_s=args.timeout_s,
        )
    else:
        result = _run_inprocess_roundtrip(event_body=event_body, secret=args.secret)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 2


def _run_inprocess_roundtrip(*, event_body: dict[str, Any], secret: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="askme-field-runtime-") as tmp:
        service = FieldOperationsService(
            config={"archive_path": str(Path(tmp) / "field-events.jsonl")}
        )
        runtime = RuntimeHandoffService(world_state=_runtime_world(), profile="shadow")
        client = TestClient(
            create_health_app(
                _health_snapshot,
                field_operations_handler=service,
                runtime_handler=runtime,
                field_runtime_callback_secret=secret,
            )
        )
        created = client.post("/api/field/events", json=event_body)
        created.raise_for_status()
        created_body = created.json()
        runtime_result = created_body["runtime_handoff_result"]
        event_id = field_event_id_from_runtime_result(runtime_result)
        payloads = build_field_runtime_callback_sequence(runtime_result, secret=secret)
        responses = [
            client.post(f"/api/field/events/{event_id}/runtime-delivery", json=payload)
            for payload in payloads
        ]
        archived = service.list_payload()["events"][0]
        return _roundtrip_summary(
            created_body=created_body,
            callback_payloads=payloads,
            callback_responses=[item.json() for item in responses],
            callback_status_codes=[item.status_code for item in responses],
            archived_event=archived,
            mode="inprocess",
        )


def _run_live_roundtrip(
    *,
    base_url: str,
    event_body: dict[str, Any],
    secret: str,
    timeout_s: float,
) -> dict[str, Any]:
    root = str(base_url or "").rstrip("/")
    created_body = _post_json(f"{root}/api/field/events", event_body, timeout_s=timeout_s)
    runtime_result = created_body.get("runtime_handoff_result")
    if not isinstance(runtime_result, dict):
        return {
            "ok": False,
            "mode": "live",
            "reason": "runtime_handoff_result_missing",
            "created": created_body,
        }
    event_id = field_event_id_from_runtime_result(runtime_result)
    payloads = build_field_runtime_callback_sequence(runtime_result, secret=secret)
    responses = post_field_runtime_callback_sequence(
        base_url=root,
        event_id=event_id,
        payloads=payloads,
        timeout_s=timeout_s,
    )
    return _roundtrip_summary(
        created_body=created_body,
        callback_payloads=payloads,
        callback_responses=responses,
        callback_status_codes=[200 if item.get("recorded") else 422 for item in responses],
        archived_event=(responses[-1].get("event") if responses else {}),
        mode="live",
    )


def _run_local_server_roundtrip(
    *,
    event_body: dict[str, Any],
    secret: str,
    timeout_s: float,
) -> dict[str, Any]:
    server_info = _start_local_smoke_server(secret=secret)
    try:
        result = _run_live_roundtrip(
            base_url=server_info["base_url"],
            event_body=event_body,
            secret=secret,
            timeout_s=timeout_s,
        )
        result["mode"] = "local_server"
        result["base_url"] = server_info["base_url"]
        return result
    finally:
        server_info["server"].should_exit = True
        server_info["thread"].join(timeout=5)
        server_info["tmpdir"].cleanup()


def _start_local_smoke_server(*, secret: str) -> dict[str, Any]:
    import socket
    import threading
    import time

    import uvicorn

    tmpdir = tempfile.TemporaryDirectory(prefix="askme-field-runtime-live-")
    archive_path = str(Path(tmpdir.name) / "field-events.jsonl")
    service = FieldOperationsService(config={"archive_path": archive_path})
    runtime = RuntimeHandoffService(world_state=_runtime_world(), profile="shadow")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
    app = create_health_app(
        _health_snapshot,
        field_operations_handler=service,
        runtime_handler=runtime,
        field_runtime_callback_secret=secret,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    deadline = time.time() + 5
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with request.urlopen(f"{base_url}/health", timeout=0.5) as response:  # noqa: S310 - local smoke server.
                response.read()
            return {"server": server, "thread": thread, "base_url": base_url, "tmpdir": tmpdir}
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    tmpdir.cleanup()
    raise RuntimeError(f"local field runtime smoke server did not start: {last_error}")


def _post_json(url: str, body: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310 - operator-provided askme URL.
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _roundtrip_summary(
    *,
    created_body: dict[str, Any],
    callback_payloads: list[dict[str, Any]],
    callback_responses: list[dict[str, Any]],
    callback_status_codes: list[int],
    archived_event: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    receipts = archived_event.get("runtime_delivery_receipts")
    if not isinstance(receipts, list):
        receipts = []
    delivery = archived_event.get("runtime_delivery")
    if not isinstance(delivery, dict):
        delivery = {}
    callback_statuses = [str(item.get("status") or "") for item in callback_payloads]
    ok = (
        bool(created_body.get("accepted"))
        and callback_status_codes == [200] * len(callback_payloads)
        and [str(item.get("status") or "") for item in receipts] == callback_statuses
        and str(delivery.get("status") or "") == (callback_statuses[-1] if callback_statuses else "")
    )
    return {
        "ok": ok,
        "mode": mode,
        "event_id": created_body.get("event", {}).get("event_id"),
        "runtime_statuses": callback_statuses,
        "callback_status_codes": callback_status_codes,
        "receipt_count": len(receipts),
        "final_runtime_delivery": delivery,
        "workflow": archived_event.get("incident_workflow", {}),
    }


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


def _health_snapshot() -> dict[str, Any]:
    return build_health_snapshot(
        app_name="askme",
        app_version="smoke",
        model_name="smoke",
        metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
        active_skills=[],
        voice_status={"enabled": True, "pipeline_ok": True},
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
