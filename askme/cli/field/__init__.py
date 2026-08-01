"""Field-operations CLI commands extracted from askme.cli."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from askme.cli.field_audit import _run_field_audit_anchor
from askme.cli.utils import (
    _cli_root_override,
    _field_action_audit_config,
    _field_signed_payload_text,
    _get_json,
    _load_field_ingest_events,
    _loopback_proxy_environment,
    _loopback_proxy_kwargs,
    _normalise_server_url,
    _post_json,
    _post_json_with_retries,
    _resolve_field_action_audit_hmac_secret,
    _resolve_field_device_signing_secret,
    _server_auth_headers,
    _single_device_id,
    _start_field_smoke_server,
    _start_local_webhook_collector,
    _write_field_smoke_events,
)


def _run_field_operations_eval(*, output: str) -> dict[str, Any]:
    from scripts.eval.evaluate_field_operations_scenarios import evaluate_scenarios, write_report

    payload = asyncio.run(evaluate_scenarios())
    report_path = write_report(payload, Path(output))
    payload["report_path"] = str(report_path)
    return payload


def _emit_field_operations_eval_payload(payload: dict[str, Any]) -> None:
    print(
        "field-operations: "
        f"{payload.get('status')} "
        f"{payload.get('passed', 0)}/{payload.get('scenario_count', 0)} scenarios "
        f"failed={payload.get('failed', 0)}"
    )
    print(f"report: {payload.get('report_path') or '-'}")
    product_demo = payload.get("product_demo")
    if not isinstance(product_demo, dict):
        return
    print(
        "product-demo: "
        f"{product_demo.get('suite_name') or 'field operations'} "
        f"ready={bool(product_demo.get('demo_ready'))} "
        f"real-integration-ready={bool(product_demo.get('real_integration_ready'))} "
        f"{product_demo.get('passed', 0)}/{product_demo.get('customer_scenario_count', 0)} scenes"
    )
    scenarios = product_demo.get("customer_scenarios")
    if isinstance(scenarios, list):
        print("customer-scenes:")
        for item in scenarios[:20]:
            if not isinstance(item, dict):
                continue
            actual = item.get("actual") if isinstance(item.get("actual"), dict) else {}
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            marker = "ok" if item.get("passed") else "fail"
            print(
                "  - "
                f"[{marker}] {item.get('customer_name') or item.get('name')}: "
                f"{item.get('expected_robot_action') or '-'} "
                f"notify={actual.get('notification_group') or '-'} "
                f"delivery={actual.get('delivery_status') or '-'} "
                f"event={evidence.get('event_id') or '-'}"
            )
    gaps = product_demo.get("blocked_on_real_integrations")
    if isinstance(gaps, list) and gaps:
        print("real-integration-gaps:")
        for gap in gaps[:10]:
            print(f"  - {gap}")


def _run_field_ingest_file(
    *,
    source: str,
    server: str,
    dry_run: bool,
    limit: int,
    device_secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    from askme.pipeline.field.field_ingest_adapters import normalize_field_ingest_payload
    from askme.pipeline.field.field_ingest_bridge import sign_field_ingest_payload

    events = _load_field_ingest_events(Path(source))
    if limit > 0:
        events = events[:limit]
    base_url = _normalise_server_url(server)
    results: list[dict[str, Any]] = []
    failures = 0
    signed_count = 0
    for index, event in enumerate(events, start=1):
        normalized = normalize_field_ingest_payload(event)
        normalized, signing = sign_field_ingest_payload(normalized, device_secrets or {})
        if signing.get("signed"):
            signed_count += 1
        item: dict[str, Any] = {
            "index": index,
            "normalized": normalized,
            "posted": False,
            "device_signing": signing,
        }
        if dry_run:
            item["status"] = "dry_run"
        else:
            try:
                response = _post_json(f"{base_url}/api/field/ingest", normalized)
            except Exception as exc:
                failures += 1
                item["status"] = "failed"
                item["error"] = str(exc)
            else:
                item["posted"] = True
                item["status"] = str(response.get("status") or "unknown")
                item["accepted"] = bool(response.get("accepted"))
                item["scenario_id"] = (
                    (response.get("normalized") or {}).get("scenario_id")
                    or normalized.get("scenario_id")
                    or ""
                )
                item["event_id"] = (response.get("event") or {}).get("event_id") or ""
        results.append(item)
    return {
        "status": "failed" if failures else "ok",
        "target": "field-ingest-file",
        "server": base_url,
        "dry_run": dry_run,
        "count": len(events),
        "failed": failures,
        "signed": signed_count,
        "results": results,
    }


def _emit_field_ingest_file_payload(payload: dict[str, Any]) -> None:
    print(
        "field-ingest-file: "
        f"{payload.get('status')} count={payload.get('count', 0)} "
        f"failed={payload.get('failed', 0)} "
        f"signed={payload.get('signed', 0)} "
        f"dry_run={payload.get('dry_run')}"
    )
    for item in payload.get("results", [])[:20]:
        print(
            f"- #{item.get('index')} {item.get('status')} "
            f"scenario={item.get('scenario_id') or item.get('normalized', {}).get('scenario_id') or '-'} "
            f"event={item.get('event_id') or '-'}"
        )


def _field_ingest_post_json(
    url: str,
    body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """POST a field-ingest payload with control auth and loopback proxy safety."""
    kwargs: dict[str, Any] = {
        "json": body,
        "timeout": timeout_s,
    }
    kwargs.update(_loopback_proxy_kwargs(url))
    headers = _server_auth_headers()
    if headers:
        kwargs["headers"] = headers
    http_requests = _cli_root_override("requests", requests)
    response = http_requests.post(url, **kwargs)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"status": "invalid_response"}


def _run_field_ingest_bridge(
    *,
    source: str,
    server: str,
    state_path: str | None,
    dry_run: bool,
    limit: int,
    timeout_s: float,
    device_secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    from askme.pipeline.field.field_ingest_bridge import run_field_ingest_bridge_once

    return run_field_ingest_bridge_once(
        source=source,
        server=server,
        state_path=state_path,
        dry_run=dry_run,
        limit=limit,
        timeout_s=timeout_s,
        device_secrets=device_secrets,
        post_func=_field_ingest_post_json,
    )


def _watch_field_ingest_bridge(
    *,
    source: str,
    server: str,
    state_path: str | None,
    interval_s: float,
    dry_run: bool,
    limit: int,
    timeout_s: float,
    device_secrets: dict[str, str] | None = None,
) -> None:
    from askme.pipeline.field.field_ingest_bridge import watch_field_ingest_bridge

    watch_field_ingest_bridge(
        source=source,
        server=server,
        state_path=state_path,
        interval_s=interval_s,
        dry_run=dry_run,
        limit=limit,
        timeout_s=timeout_s,
        device_secrets=device_secrets,
        post_func=_field_ingest_post_json,
    )


def _emit_field_ingest_bridge_payload(payload: dict[str, Any]) -> None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(
        "field-ingest-bridge: "
        f"{payload.get('status')} count={payload.get('count', 0)} "
        f"failed={payload.get('failed', 0)} "
        f"dry_run={payload.get('dry_run')} "
        f"state={payload.get('state_path') or '-'}"
    )
    if summary:
        scenarios = (
            summary.get("scenario_counts")
            if isinstance(summary.get("scenario_counts"), dict)
            else {}
        )
        sources = (
            summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {}
        )
        devices = (
            summary.get("device_counts") if isinstance(summary.get("device_counts"), dict) else {}
        )
        scenario_text = (
            ", ".join(f"{key}:{value}" for key, value in sorted(scenarios.items())) or "-"
        )
        source_text = ", ".join(f"{key}:{value}" for key, value in sorted(sources.items())) or "-"
        device_text = ", ".join(f"{key}:{value}" for key, value in sorted(devices.items())) or "-"
        print(
            "summary: "
            f"posted={summary.get('posted', 0)} "
            f"accepted={summary.get('accepted', 0)} "
            f"signed={summary.get('signed', 0)} "
            f"format={summary.get('source_format') or '-'} "
            f"scenarios={scenario_text} "
            f"sources={source_text} "
            f"devices={device_text}"
        )
    for item in payload.get("results", [])[:20]:
        print(
            f"- #{item.get('index')} {item.get('status')} "
            f"scenario={item.get('scenario_id') or item.get('normalized', {}).get('scenario_id') or '-'} "
            f"event={item.get('event_id') or '-'}"
        )


def _run_field_sign_device_payload(
    *,
    source: str,
    output: str,
    device_id: str,
    secret: str,
    secret_env: str,
    timestamp: float,
) -> dict[str, Any]:
    from askme.pipeline.field.field_device_signature import sign_field_device_payload

    resolved_secret = _resolve_field_device_signing_secret(secret=secret, secret_env=secret_env)
    if not resolved_secret:
        return {
            "status": "failed",
            "reason": "device_secret_missing",
            "source": source,
            "output": output,
            "count": 0,
            "message": "Provide --secret or --secret-env with a configured HMAC secret.",
        }

    source_path = Path(source)
    events = _load_field_ingest_events(source_path)
    signature_timestamp = float(timestamp or time.time())
    signed_events: list[dict[str, Any]] = []
    for event in events:
        signed = dict(event)
        if device_id:
            signed["device_id"] = device_id
        signed.pop("device_signature", None)
        signed.pop("signature", None)
        signed.pop("x_signature", None)
        signed["device_signature_alg"] = "hmac-sha256"
        signed["device_signature_timestamp"] = signature_timestamp
        signed["device_signature"] = sign_field_device_payload(signed, secret=resolved_secret)
        signed_events.append(signed)

    output_path = Path(output) if output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _field_signed_payload_text(
                signed_events, source_path=source_path, output_path=output_path
            ),
            encoding="utf-8",
            newline="\n",
        )
    return {
        "status": "signed",
        "source": str(source_path),
        "output": str(output_path or ""),
        "count": len(signed_events),
        "device_id": device_id or _single_device_id(signed_events),
        "signature_alg": "hmac-sha256",
        "signature_timestamp": signature_timestamp,
        "secret_source": f"env:{secret_env}" if secret_env else "argument",
        "signed_payload": signed_events[0]
        if len(signed_events) == 1 and output_path is None
        else None,
        "signed_payloads": signed_events
        if len(signed_events) > 1 and output_path is None
        else None,
    }


def _emit_field_sign_device_payload(payload: dict[str, Any]) -> None:
    print(
        "field-sign-device-payload: "
        f"{payload.get('status')} count={payload.get('count', 0)} "
        f"device={payload.get('device_id') or '-'} "
        f"alg={payload.get('signature_alg') or '-'} "
        f"output={payload.get('output') or 'stdout'}"
    )
    if payload.get("status") != "signed":
        print(f"reason: {payload.get('reason') or 'unknown'}")
        if payload.get("message"):
            print(f"message: {payload.get('message')}")
        return
    signed = payload.get("signed_payload")
    signed_many = payload.get("signed_payloads")
    if isinstance(signed, dict):
        print(json.dumps(signed, ensure_ascii=False, sort_keys=True, indent=2))
    elif isinstance(signed_many, list):
        for item in signed_many:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))


def _run_field_ingest_smoke(
    *,
    output_dir: str,
    server: str = "",
    audit_hmac_secret: str = "",
    require_device_signatures: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "device-events.jsonl"
    state_path = output / "device-events.state.json"
    archive_path = output / "field-events.jsonl"
    action_audit_path = output / "field-action-audit.jsonl"
    report_path = output / "field-ingest-smoke.json"
    _write_field_smoke_events(source)
    state_path.unlink(missing_ok=True)
    archive_path.unlink(missing_ok=True)
    action_audit_path.unlink(missing_ok=True)

    local_server = None
    base_url = server.strip()
    operator_action_payload: dict[str, Any] = {}
    if not base_url:
        field_config: dict[str, Any] = {
            "action_audit": _field_action_audit_config(
                action_audit_path,
                hmac_secret=audit_hmac_secret,
            )
        }
        if require_device_signatures:
            field_config.update(_field_ingest_smoke_trusted_device_config())
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config=field_config,
        )
        base_url = str(local_server["base_url"])

    device_secrets = _field_ingest_smoke_device_secrets() if require_device_signatures else None
    try:
        bridge_payload = _run_field_ingest_bridge(
            source=str(source),
            server=base_url,
            state_path=str(state_path),
            dry_run=False,
            limit=0,
            timeout_s=8.0,
            device_secrets=device_secrets,
        )
        read_headers = {"X-Askme-Operator-Id": "supervisor-1"}
        events_payload = _get_json(
            f"{_normalise_server_url(base_url)}/api/field/events?limit=20",
            headers=read_headers,
        )
        events = events_payload.get("events") if isinstance(events_payload, dict) else []
        first_event = next(
            (item for item in events if isinstance(item, dict) and item.get("event_id")), None
        )
        if first_event:
            operator_action_payload = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{first_event['event_id']}/acknowledge",
                {
                    "operator_id": "security-1",
                    "note": "field-smoke-suite acknowledges first incident for audit evidence",
                },
            )
            events_payload = _get_json(
                f"{_normalise_server_url(base_url)}/api/field/events?limit=20",
                headers=read_headers,
            )
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)

    events = events_payload.get("events") if isinstance(events_payload, dict) else []
    scenario_ids = {str(item.get("scenario_id") or "") for item in events if isinstance(item, dict)}
    required = {
        "illegal_parking",
        "fire_or_smoke",
        "robot_abnormal_incident",
        "trash_bin_full",
        "crowd_gathering",
    }
    bridge_summary = (
        bridge_payload.get("summary") if isinstance(bridge_payload.get("summary"), dict) else {}
    )
    passed = (
        bridge_payload.get("status") == "ok"
        and int(bridge_payload.get("count") or 0) == 8
        and int(bridge_summary.get("posted") or 0) == 8
        and int(bridge_summary.get("accepted") or 0) == 8
        and int(bridge_summary.get("events_created") or 0) == 8
        and (not require_device_signatures or int(bridge_summary.get("signed") or 0) == 8)
        and required.issubset(scenario_ids)
        and operator_action_payload.get("acknowledged") is True
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-ingest-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "source": str(source),
        "state_path": str(state_path),
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "require_device_signatures": require_device_signatures,
        "bridge": bridge_payload,
        "operator_action": operator_action_payload,
        "event_count": len(events) if isinstance(events, list) else 0,
        "expected_bridge_count": 8,
        "scenario_ids": sorted(scenario_ids),
        "required_scenario_ids": sorted(required),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_ingest_smoke_payload(payload: dict[str, Any]) -> None:
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), dict) else {}
    summary = bridge.get("summary") if isinstance(bridge.get("summary"), dict) else {}
    print(
        "field-ingest-smoke: "
        f"{payload.get('status')} "
        f"events={payload.get('event_count', 0)} "
        f"server={payload.get('server')}"
    )
    print(f"source: {payload.get('source')}")
    print(f"archive: {payload.get('archive_path')}")
    print(f"report: {payload.get('report_path')}")
    if summary:
        print(
            "bridge: "
            f"posted={summary.get('posted', 0)} "
            f"accepted={summary.get('accepted', 0)} "
            f"events_created={summary.get('events_created', 0)} "
            f"signed={summary.get('signed', 0)}"
        )
    print("scenarios: " + ", ".join(payload.get("scenario_ids", [])))


class _RecordingVoiceHandler:
    """Small voice handler used by field-voice-smoke without audio hardware."""

    def __init__(self) -> None:
        self.profiles: list[str] = []
        self.spoken: list[str] = []
        self.playback_started = False

    def set_voice_profile_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(body.get("profile_id") or "")
        self.profiles.append(profile_id)
        return {
            "updated": True,
            "active_profile": profile_id,
            "requested_profile": profile_id,
            "resolved_profile": profile_id,
            "profile": {"profile_id": profile_id},
        }

    def speak(self, text: str) -> None:
        self.spoken.append(str(text))

    def start_playback(self) -> None:
        self.playback_started = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "profiles": list(self.profiles),
            "spoken": list(self.spoken),
            "playback_started": self.playback_started,
        }


def _build_field_voice_smoke_handler(*, live_tts: bool) -> Any:
    if not live_tts:
        return _RecordingVoiceHandler()
    from askme.config import get_config
    from askme.providers import build_tts_provider

    return build_tts_provider(get_config())


def _field_smoke_run_id() -> str:
    import time

    return f"smoke-{int(time.time() * 1000)}"


def _make_field_voice_smoke_event_unique(payload: dict[str, Any]) -> None:
    """Avoid smoke-test events being deduped by earlier ingest-smoke events."""

    run_id_factory = _cli_root_override("_field_smoke_run_id", _field_smoke_run_id)
    run_id = run_id_factory()
    payload["smoke_run_id"] = run_id
    for key in ("location", "plate_number", "zone_id"):
        value = payload.get(key)
        if value:
            payload[key] = f"{value}-{run_id}"


def _field_voice_smoke_event(scenario: str) -> dict[str, Any]:
    import time

    if scenario == "joint_fault":
        return {
            "scenario_id": "robot_abnormal_incident",
            "source": "robot",
            "observed_at": time.time(),
            "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
            "fault_type": "joint_motor_fault",
            "joint_id": "hip-left",
            "fault_code": "MOTOR_OVER_CURRENT",
            "location": "A区东侧",
            "image_path": "artifacts/evidence/joint-fault.jpg",
        }
    if scenario == "illegal_parking":
        return {
            "scenario_id": "illegal_parking",
            "source": "camera",
            "observed_at": time.time(),
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "duration_s": 180,
            "image_path": "artifacts/evidence/car.jpg",
        }
    return {
        "scenario_id": "fire_or_smoke",
        "source": "sensor",
        "observed_at": time.time(),
        "location": "配电间门口",
        "temperature_c": 68,
        "smoke_level": 0.82,
        "image_path": "artifacts/evidence/smoke.jpg",
    }


def _run_field_voice_smoke(
    *,
    output_dir: str,
    server: str = "",
    scenario: str = "fire",
    live_tts: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-voice-events.jsonl"
    report_path = output / "field-voice-smoke.json"
    archive_path.unlink(missing_ok=True)

    local_server = None
    voice_handler: Any | None = None
    base_url = server.strip()
    if not base_url:
        voice_handler = _build_field_voice_smoke_handler(live_tts=live_tts)
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            voice_handler=voice_handler,
            voice_enabled=True,
        )
        base_url = str(local_server["base_url"])

    request_payload = _field_voice_smoke_event(scenario)
    _make_field_voice_smoke_event_unique(request_payload)
    response_payload: dict[str, Any] = {}
    status_code = 0
    try:
        endpoint = f"{_normalise_server_url(base_url)}/api/field/events"
        http_requests = _cli_root_override("requests", requests)
        response = http_requests.post(
            endpoint,
            json=request_payload,
            timeout=10,
            **_loopback_proxy_kwargs(endpoint),
        )
        status_code = response.status_code
        response_payload = response.json()
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)
        if live_tts and voice_handler is not None and hasattr(voice_handler, "shutdown"):
            voice_handler.shutdown()

    delivery = response_payload.get("voice_delivery") if isinstance(response_payload, dict) else {}
    event = response_payload.get("event") if isinstance(response_payload, dict) else {}
    directive = event.get("voice_directive") if isinstance(event, dict) else {}
    recorded = voice_handler.snapshot() if isinstance(voice_handler, _RecordingVoiceHandler) else {}
    passed = (
        status_code == 200
        and response_payload.get("accepted") is True
        and isinstance(delivery, dict)
        and delivery.get("status") == "queued"
        and isinstance(directive, dict)
        and bool(directive.get("resolved_profile"))
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-voice-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "live_tts": bool(live_tts),
        "scenario": scenario,
        "http_status": status_code,
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "request": request_payload,
        "response": response_payload,
        "voice_delivery": delivery if isinstance(delivery, dict) else {},
        "voice_directive": directive if isinstance(directive, dict) else {},
        "recorded_voice_handler": recorded,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_voice_smoke_payload(payload: dict[str, Any]) -> None:
    directive = (
        payload.get("voice_directive") if isinstance(payload.get("voice_directive"), dict) else {}
    )
    delivery = (
        payload.get("voice_delivery") if isinstance(payload.get("voice_delivery"), dict) else {}
    )
    print(
        "field-voice-smoke: "
        f"{payload.get('status')} "
        f"scenario={payload.get('scenario')} "
        f"delivery={delivery.get('status', '-')}"
    )
    print(f"server: {payload.get('server')}")
    print(
        f"voice: {directive.get('requested_profile', '-')} -> {directive.get('resolved_profile', '-')}"
    )
    print(f"report: {payload.get('report_path')}")


def _run_field_notification_smoke(
    *,
    output_dir: str,
    server: str = "",
    groups: str = "security,cleaning,operations",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-notification-events.jsonl"
    report_path = output / "field-notification-smoke.json"
    archive_path.unlink(missing_ok=True)

    group_names = [item.strip() for item in str(groups).split(",") if item.strip()]
    local_server = None
    collector = None
    base_url = server.strip()
    webhook_url = ""
    if not base_url:
        collector = _start_local_webhook_collector()
        webhook_url = str(collector["url"])
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config={
                "dingtalk_webhooks": {group: webhook_url for group in group_names},
            },
        )
        base_url = str(local_server["base_url"])

    results: list[dict[str, Any]] = []
    try:
        with _loopback_proxy_environment():
            for group in group_names:
                response = _post_json(
                    f"{_normalise_server_url(base_url)}/api/field/notification-test",
                    {
                        "notification_group": group,
                        "operator_id": "supervisor-1",
                        "message": f"Askme现场通知联调：{group}响应组。",
                    },
                )
                results.append(response)
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)
        if collector:
            collector["server"].shutdown()
            collector["thread"].join(timeout=5)

    collector_requests = list(collector["requests"]) if collector else []
    sent_groups = [
        str(item.get("notification_group") or "") for item in results if item.get("sent") is True
    ]
    passed = set(group_names).issubset(set(sent_groups))
    if collector is not None:
        passed = passed and len(collector_requests) >= len(group_names)
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-notification-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "local_webhook_collector": bool(collector),
        "external_services": bool(server.strip()),
        "groups": group_names,
        "sent_groups": sent_groups,
        "result_count": len(results),
        "collector_request_count": len(collector_requests),
        "collector_url": webhook_url,
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "results": results,
        "collector_requests": collector_requests[:10],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_notification_smoke_payload(payload: dict[str, Any]) -> None:
    print(
        "field-notification-smoke: "
        f"{payload.get('status')} "
        f"groups={','.join(payload.get('sent_groups') or [])} "
        f"collector_requests={payload.get('collector_request_count', 0)}"
    )
    print(f"server: {payload.get('server')}")
    print(f"report: {payload.get('report_path')}")


def _run_field_notification_preflight(
    *,
    server: str = "",
    groups: str = "security,cleaning,operations",
    require_secret: bool = True,
) -> dict[str, Any]:
    group_names = [item.strip() for item in str(groups).split(",") if item.strip()]
    if server.strip():
        return _get_json(f"{_normalise_server_url(server)}/api/field/notification-preflight")

    from askme.config import get_config
    from askme.pipeline.field.field_operations import FieldOperationsService

    cfg = get_config()
    field_cfg = dict(
        cfg.get("field_operations", {}) if isinstance(cfg.get("field_operations"), dict) else {}
    )
    service = FieldOperationsService(config=field_cfg)
    return service.notification_preflight_payload(
        groups=group_names,
        require_secret=require_secret,
    )


def _emit_field_notification_preflight_payload(payload: dict[str, Any]) -> None:
    print(f"field-notification-preflight: {payload.get('status')}")
    groups = payload.get("groups") if isinstance(payload.get("groups"), dict) else {}
    for group, result in groups.items():
        if not isinstance(result, dict):
            continue
        print(
            f"- {group}: "
            f"{'ready' if result.get('ready') else 'blocked'} "
            f"webhook={bool(result.get('webhook_configured'))} "
            f"secret={bool(result.get('secret_configured'))}"
        )
    for action in payload.get("next_actions", [])[:5]:
        print(f"next: {action}")


def _run_field_disposition_smoke(
    *,
    output_dir: str,
    server: str = "",
    audit_hmac_secret: str = "",
) -> dict[str, Any]:
    import time

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-events.jsonl"
    report_path = output / "field-disposition-smoke.json"

    local_server = None
    base_url = server.strip()
    if not base_url:
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config={
                "action_audit": _field_action_audit_config(
                    output / "field-action-audit.jsonl",
                    hmac_secret=audit_hmac_secret,
                )
            },
        )
        base_url = str(local_server["base_url"])

    unique_location = f"配电间门口-处置验收-{int(time.time() * 1000)}"
    event_request = {
        "scenario_id": "fire_or_smoke",
        "source": "sensor",
        "observed_at": time.time(),
        "location": unique_location,
        "temperature_c": 68,
        "smoke_level": 0.82,
        "image_path": "artifacts/evidence/smoke-disposition.jpg",
    }
    created: dict[str, Any] = {}
    acknowledged: dict[str, Any] = {}
    close_requested: dict[str, Any] = {}
    closed: dict[str, Any] = {}
    report: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    try:
        created = _post_json(f"{_normalise_server_url(base_url)}/api/field/events", event_request)
        event = created.get("event") if isinstance(created.get("event"), dict) else {}
        event_id = str(event.get("event_id") or "")
        if event_id:
            acknowledged = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/acknowledge",
                {"operator_id": "security-1", "note": "field disposition smoke acknowledged"},
            )
            close_requested = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/request-close",
                {"operator_id": "security-1", "note": "request supervisor close approval"},
            )
            closed = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/close",
                {
                    "operator_id": "supervisor-1",
                    "note": "现场已复核并完成处置",
                    "supervisor_approved": True,
                    "supervisor_id": "supervisor-1",
                },
            )
            report = _get_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/report"
            )
            integrity = _get_json(f"{_normalise_server_url(base_url)}/api/field/audit/integrity")
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)

    closed_event = closed.get("event") if isinstance(closed.get("event"), dict) else {}
    report_body = report.get("report") if isinstance(report.get("report"), dict) else {}
    timeline = report_body.get("timeline") if isinstance(report_body.get("timeline"), list) else []
    passed = (
        created.get("accepted") is True
        and acknowledged.get("acknowledged") is True
        and close_requested.get("requested") is True
        and closed_event.get("status") == "closed"
        and (closed_event.get("close_approval") or {}).get("approved") is True
        and len(timeline) >= 3
        and integrity.get("valid") is True
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-disposition-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "created": created,
        "acknowledged": acknowledged,
        "close_requested": close_requested,
        "closed": closed,
        "event_report": report,
        "action_audit_integrity": integrity,
        "timeline_count": len(timeline),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_disposition_smoke_payload(payload: dict[str, Any]) -> None:
    closed = payload.get("closed") if isinstance(payload.get("closed"), dict) else {}
    event = closed.get("event") if isinstance(closed.get("event"), dict) else {}
    print(
        "field-disposition-smoke: "
        f"{payload.get('status')} "
        f"event={event.get('event_id', '-')} "
        f"timeline={payload.get('timeline_count', 0)}"
    )
    print(f"server: {payload.get('server')}")
    print(f"report: {payload.get('report_path')}")


def _run_field_smoke_suite(
    *,
    output_dir: str,
    voice_scenario: str = "fire",
    groups: str = "security,cleaning,operations",
    live_tts: bool = False,
    audit_hmac_secret: str = "",
    audit_webhook_url: str = "",
    audit_webhook_retries: int = 3,
    include_audit_anchor: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_audit_hmac_secret = _resolve_field_action_audit_hmac_secret(audit_hmac_secret)
    scenario_report_path = output / "scenario-evaluation.json"
    suite_report_path = output / "field-smoke-suite.json"
    html_report_path = output / "field-smoke-suite.html"

    run_field_operations_eval = _cli_root_override(
        "_run_field_operations_eval",
        _run_field_operations_eval,
    )
    run_field_ingest_smoke = _cli_root_override("_run_field_ingest_smoke", _run_field_ingest_smoke)
    run_field_voice_smoke = _cli_root_override("_run_field_voice_smoke", _run_field_voice_smoke)
    run_field_notification_smoke = _cli_root_override(
        "_run_field_notification_smoke",
        _run_field_notification_smoke,
    )
    run_field_disposition_smoke = _cli_root_override(
        "_run_field_disposition_smoke",
        _run_field_disposition_smoke,
    )
    run_field_readiness = _cli_root_override("_run_field_readiness", _run_field_readiness)
    run_field_audit_anchor = _cli_root_override(
        "_run_field_audit_anchor",
        _run_field_audit_anchor,
    )

    scenario = run_field_operations_eval(output=str(scenario_report_path))
    ingest = run_field_ingest_smoke(
        output_dir=str(output),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    voice = run_field_voice_smoke(
        output_dir=str(output),
        scenario=voice_scenario,
        live_tts=live_tts,
    )
    notification = run_field_notification_smoke(
        output_dir=str(output),
        groups=groups,
    )
    disposition = run_field_disposition_smoke(
        output_dir=str(output),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    readiness = run_field_readiness(
        server="",
        archive_path=str(output / "field-events.jsonl"),
        scenario_report=str(scenario_report_path),
        smoke_report=str(output / "field-ingest-smoke.json"),
        voice_smoke_report=str(output / "field-voice-smoke.json"),
        notification_smoke_report=str(output / "field-notification-smoke.json"),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    audit_anchor = (
        run_field_audit_anchor(
            server="",
            archive_path=str(output / "field-events.jsonl"),
            audit_path=str(output / "field-action-audit.jsonl"),
            hmac_secret=resolved_audit_hmac_secret,
            output=str(output / "audit-checkpoint.json"),
            webhook_url=audit_webhook_url,
            webhook_retries=audit_webhook_retries,
            require_valid=True,
        )
        if include_audit_anchor
        else {"status": "skipped", "target": "field-audit-anchor"}
    )
    checks = {
        "scenario_eval": scenario.get("status") == "passed",
        "field_ingest_smoke": ingest.get("status") == "passed",
        "field_voice_smoke": voice.get("status") == "passed",
        "field_notification_smoke": notification.get("status") == "passed",
        "field_disposition_smoke": disposition.get("status") == "passed",
        "readiness_unblocked": not readiness.get("blockers"),
        "audit_checkpoint_created": audit_anchor.get("status") in {"anchored", "skipped"},
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-smoke-suite",
        "output_dir": str(output),
        "report_path": str(suite_report_path),
        "html_report_path": str(html_report_path),
        "customer_summary": _field_smoke_customer_summary(
            checks=checks,
            readiness=readiness,
            notification=notification,
            voice=voice,
        ),
        "checks": checks,
        "scenario_report": scenario,
        "ingest_smoke": ingest,
        "voice_smoke": voice,
        "notification_smoke": notification,
        "disposition_smoke": disposition,
        "readiness": readiness,
        "audit_anchor": audit_anchor,
    }
    suite_report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_report_path.write_text(_field_smoke_suite_html(payload), encoding="utf-8")
    return payload


def _emit_field_smoke_suite_payload(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    print(f"field-smoke-suite: {payload.get('status')}")  # noqa: T201
    for name, passed in checks.items():
        print(f"- {name}: {'passed' if passed else 'failed'}")  # noqa: T201
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    print(f"readiness: {readiness.get('status', '-')}")  # noqa: T201
    print(f"report: {payload.get('report_path')}")  # noqa: T201


def _field_smoke_customer_summary(
    *,
    checks: dict[str, bool],
    readiness: dict[str, Any],
    notification: dict[str, Any],
    voice: dict[str, Any],
) -> dict[str, Any]:
    warnings = [str(item) for item in readiness.get("warnings", []) if item]
    blockers = [str(item) for item in readiness.get("blockers", []) if item]
    return {
        "headline": "现场能力链路已通过本地实验室验证"
        if all(checks.values())
        else "现场能力链路仍有未通过项",
        "readiness_status": readiness.get("status", "unknown"),
        "passed_checks": [name for name, passed in checks.items() if passed],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "voice_verified": voice.get("status") == "passed",
        "voice_live_tts": bool(voice.get("live_tts")),
        "notification_verified": notification.get("status") == "passed",
        "notification_external_services": bool(notification.get("external_services")),
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": [str(item) for item in readiness.get("next_actions", []) if item],
    }


def _field_smoke_suite_html(payload: dict[str, Any]) -> str:
    summary = (
        payload.get("customer_summary") if isinstance(payload.get("customer_summary"), dict) else {}
    )
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    blockers = summary.get("blockers") if isinstance(summary.get("blockers"), list) else []
    actions = summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else []
    gates = readiness.get("gates") if isinstance(readiness.get("gates"), dict) else {}
    check_rows = "".join(
        f"<li><strong>{_html_escape(name)}</strong>: {'通过' if passed else '未通过'}</li>"
        for name, passed in checks.items()
    )
    gate_rows = "".join(
        f"<li><strong>{_html_escape(name)}</strong>: {'通过' if value else '未通过'}</li>"
        for name, value in gates.items()
    )
    blocker_rows = (
        "".join(f"<li>{_html_escape(item)}</li>" for item in blockers) or "<li>无阻塞项</li>"
    )
    warning_rows = (
        "".join(f"<li>{_html_escape(item)}</li>" for item in warnings) or "<li>无提醒项</li>"
    )
    action_rows = (
        "".join(f"<li>{_html_escape(item)}</li>" for item in actions) or "<li>无需额外动作</li>"
    )
    status = str(payload.get("status") or "unknown")
    readiness_status = str(summary.get("readiness_status") or readiness.get("status") or "unknown")
    headline = str(summary.get("headline") or "现场验收报告")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Askme 现场能力验收报告</title>
  <style>
    body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:32px;color:#16352c;background:#f6fbf8;}}
    h1,h2{{color:#0b5a3f;}}
    .card{{background:#fff;border:1px solid #dbe9e2;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 8px 24px rgba(15,50,35,.08);}}
    .status{{display:inline-block;border-radius:999px;padding:6px 12px;background:#e2f7ea;color:#0a6a44;font-weight:700;}}
    .warn{{background:#fff6d8;color:#815500;}}
    li{{margin:6px 0;}}
    code{{background:#eef7f2;padding:2px 5px;border-radius:5px;}}
  </style>
</head>
<body>
  <h1>Askme 现场能力验收报告</h1>
  <div class="card">
    <p class="status{" warn" if status != "passed" else ""}">Suite: {_html_escape(status)}</p>
    <p class="status{" warn" if readiness_status != "production_ready" else ""}">Readiness: {_html_escape(readiness_status)}</p>
    <h2>{_html_escape(headline)}</h2>
    <p>这份报告面向演示、实验室验收和部署前自检。它证明本地链路是否打通，同时明确哪些能力仍未接入真实设备或真实外部服务。</p>
  </div>
  <div class="card"><h2>验收检查</h2><ul>{check_rows}</ul></div>
  <div class="card"><h2>部署门禁</h2><ul>{gate_rows}</ul></div>
  <div class="card"><h2>阻塞项</h2><ul>{blocker_rows}</ul></div>
  <div class="card"><h2>提醒项</h2><ul>{warning_rows}</ul></div>
  <div class="card"><h2>下一步</h2><ul>{action_rows}</ul></div>
  <div class="card">
    <h2>原始证据</h2>
    <p>JSON 报告：<code>{_html_escape(str(payload.get("report_path") or "-"))}</code></p>
  </div>
</body>
</html>
"""


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _run_field_deployed_smoke(
    *,
    server: str,
    output_dir: str,
    voice_scenario: str = "fire",
    groups: str = "security,cleaning,operations",
    require_notification_ready: bool = True,
    require_device_signatures: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "field-deployed-smoke.json"
    base_url = _normalise_server_url(server)

    get_json = _cli_root_override("_get_json", _get_json)
    run_field_notification_preflight = _cli_root_override(
        "_run_field_notification_preflight",
        _run_field_notification_preflight,
    )
    run_field_ingest_smoke = _cli_root_override("_run_field_ingest_smoke", _run_field_ingest_smoke)
    run_field_voice_smoke = _cli_root_override("_run_field_voice_smoke", _run_field_voice_smoke)
    run_field_notification_smoke = _cli_root_override(
        "_run_field_notification_smoke",
        _run_field_notification_smoke,
    )

    health = get_json(f"{base_url}/health")
    notification_preflight = run_field_notification_preflight(
        server=base_url,
        groups=groups,
        require_secret=True,
    )
    notification_ready = notification_preflight.get("ready") is True
    ingest = run_field_ingest_smoke(
        output_dir=str(output),
        server=base_url,
        require_device_signatures=require_device_signatures,
    )
    voice = run_field_voice_smoke(
        output_dir=str(output),
        server=base_url,
        scenario=voice_scenario,
    )
    if require_notification_ready and not notification_ready:
        notification = {
            "status": "skipped",
            "target": "field-notification-smoke",
            "reason": "notification_preflight_blocked",
        }
    else:
        notification = run_field_notification_smoke(
            output_dir=str(output),
            server=base_url,
            groups=groups,
        )
    readiness = get_json(f"{base_url}/api/field/readiness")
    checks = {
        "health_reachable": health.get("status") in {"ok", "degraded"},
        "notification_preflight_ready": notification_ready or not require_notification_ready,
        "field_ingest_smoke": ingest.get("status") == "passed",
        "signed_device_ingest_smoke": (
            not require_device_signatures
            or (
                isinstance(ingest.get("bridge"), dict)
                and isinstance(ingest["bridge"].get("summary"), dict)
                and int(ingest["bridge"]["summary"].get("signed") or 0) >= 1
            )
        ),
        "field_voice_smoke": voice.get("status") == "passed",
        "field_notification_smoke": notification.get("status") == "passed"
        or (not require_notification_ready and notification.get("status") == "skipped"),
        "readiness_reachable": bool(readiness.get("status")),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "target": "field-deployed-smoke",
        "server": base_url,
        "output_dir": str(output),
        "report_path": str(report_path),
        "checks": checks,
        "health": health,
        "notification_preflight": notification_preflight,
        "ingest_smoke": ingest,
        "voice_smoke": voice,
        "notification_smoke": notification,
        "readiness": readiness,
        "require_notification_ready": bool(require_notification_ready),
        "require_device_signatures": bool(require_device_signatures),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_deployed_smoke_payload(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    print(f"field-deployed-smoke: {payload.get('status')} server={payload.get('server')}")  # noqa: T201
    for name, passed in checks.items():
        print(f"- {name}: {'passed' if passed else 'failed'}")  # noqa: T201
    print(f"report: {payload.get('report_path')}")  # noqa: T201


def _run_field_readiness(
    *,
    server: str,
    archive_path: str,
    scenario_report: str,
    smoke_report: str,
    voice_smoke_report: str,
    notification_smoke_report: str,
    site_profile: str = "",
    check_site_env: bool = False,
    audit_hmac_secret: str = "",
    review_path: str = "",
) -> dict[str, Any]:
    if server:
        return _get_json(f"{_normalise_server_url(server)}/api/field/readiness")
    from askme.pipeline.field.field_operations import FieldOperationsService

    action_audit_path = Path(archive_path).with_name("field-action-audit.jsonl")
    service = FieldOperationsService(
        config={
            "archive_path": archive_path,
            "scenario_report_path": scenario_report,
            "smoke_report_path": smoke_report,
            "voice_smoke_report_path": voice_smoke_report,
            "notification_smoke_report_path": notification_smoke_report,
            "site_profile_path": site_profile,
            "site_profile_check_env": check_site_env,
            "action_audit": _field_action_audit_config(
                action_audit_path,
                hmac_secret=audit_hmac_secret,
            ),
            "audit_review_path": review_path,
        }
    )
    return service.readiness_payload()


def _emit_field_readiness_payload(payload: dict[str, Any]) -> None:
    print(f"field-readiness: {payload.get('status')}")
    brief = payload.get("delivery_brief") if isinstance(payload.get("delivery_brief"), dict) else {}
    if brief:
        print(f"product-stage: {brief.get('stage_code') or '-'}")
        print(f"release-scope: {brief.get('release_scope') or '-'}")
    site_profile = (
        payload.get("site_profile") if isinstance(payload.get("site_profile"), dict) else {}
    )
    if site_profile:
        summary = (
            site_profile.get("summary") if isinstance(site_profile.get("summary"), dict) else {}
        )
        print(
            "site-profile: "
            f"configured={bool(site_profile.get('configured'))} "
            f"valid={bool(site_profile.get('valid'))} "
            f"site={summary.get('site_id') or '-'} "
            f"zones={summary.get('zone_count', 0)} "
            f"devices={summary.get('device_count', 0)}"
        )
    device_trust = (
        payload.get("device_trust") if isinstance(payload.get("device_trust"), dict) else {}
    )
    if device_trust:
        unsigned = device_trust.get("unsigned_device_ids")
        unsigned_ids = unsigned if isinstance(unsigned, list) else []
        unsigned_label = ",".join(str(item) for item in unsigned_ids[:5]) if unsigned_ids else "-"
        print(
            "device-trust: "
            f"registered={device_trust.get('registered_device_count', 0)} "
            f"signed={device_trust.get('signed_device_count', 0)} "
            f"unsigned={device_trust.get('unsigned_device_count', 0)} "
            f"all_ready={bool(device_trust.get('all_registered_devices_signature_ready'))} "
            f"unsigned_ids={unsigned_label}"
        )
    blockers = payload.get("blockers") or []
    warnings = payload.get("warnings") or []
    if blockers:
        print("blockers:")
        for item in blockers:
            print(f"- {item}")
    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"- {item}")
    actions = payload.get("next_actions") or []
    if actions:
        print("next actions:")
        for item in actions:
            print(f"- {item}")


def _run_field_device_trust(*, site_profile: str) -> dict[str, Any]:
    from askme.pipeline.field.customer_project_template_support import load_field_site_profile
    from askme.pipeline.field.customer_projects import build_site_profile_report

    path = Path(site_profile)
    try:
        profile = load_field_site_profile(path)
        report = build_site_profile_report(path, check_env=True)
    except Exception as exc:
        return {
            "status": "invalid_profile",
            "site_profile": str(path),
            "reason": str(exc),
            "devices": [],
            "summary": {
                "registered_device_count": 0,
                "signature_ready_count": 0,
                "missing_secret_count": 0,
            },
            "next_actions": ["修复站点配置文件后再检查设备信任状态。"],
        }

    devices = profile.get("devices") if isinstance(profile.get("devices"), dict) else {}
    rows: list[dict[str, Any]] = []
    for device_id, device in sorted(devices.items()):
        if not isinstance(device, dict):
            continue
        secret_env = str(device.get("secret_env") or "").strip()
        secret_configured = bool(secret_env and os.getenv(secret_env))
        rows.append(
            {
                "device_id": str(device_id),
                "name": str(device.get("name") or ""),
                "source": str(device.get("source") or ""),
                "zone_id": str(device.get("zone_id") or ""),
                "secret_env": secret_env,
                "secret_configured": secret_configured,
                "require_signature": True,
                "status": "ready" if secret_configured else "missing_secret",
                "signing_command": _field_device_signing_command(str(device_id), secret_env),
            }
        )
    missing = [row for row in rows if not row["secret_configured"]]
    valid = report.get("status") == "passed"
    if not valid:
        status = "invalid_profile"
    elif missing:
        status = "needs_secret"
    else:
        status = "ready"
    return {
        "status": status,
        "site_profile": str(path),
        "profile_valid": valid,
        "profile_errors": report.get("errors") or [],
        "profile_warnings": report.get("warnings") or [],
        "devices": rows,
        "summary": {
            "registered_device_count": len(rows),
            "signature_ready_count": len(rows) - len(missing),
            "missing_secret_count": len(missing),
            "missing_secret_envs": sorted(
                {str(row.get("secret_env") or "") for row in missing if row.get("secret_env")}
            ),
        },
        "next_actions": _field_device_trust_next_actions(status, missing),
    }


def _emit_field_device_trust_payload(payload: dict[str, Any], *, show_commands: bool) -> None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(
        "field-device-trust: "
        f"{payload.get('status')} "
        f"registered={summary.get('registered_device_count', 0)} "
        f"ready={summary.get('signature_ready_count', 0)} "
        f"missing={summary.get('missing_secret_count', 0)}"
    )
    if payload.get("reason"):
        print(f"reason: {payload.get('reason')}")
    warnings = (
        payload.get("profile_warnings") if isinstance(payload.get("profile_warnings"), list) else []
    )
    for warning in warnings[:8]:
        print(f"warning: {warning}")
    devices = payload.get("devices") if isinstance(payload.get("devices"), list) else []
    for item in devices:
        if not isinstance(item, dict):
            continue
        print(
            "  - "
            f"{item.get('device_id') or '-'} "
            f"source={item.get('source') or '-'} "
            f"zone={item.get('zone_id') or '-'} "
            f"secret_env={item.get('secret_env') or '-'} "
            f"status={item.get('status') or '-'}"
        )
        if show_commands and item.get("signing_command"):
            print(f"    sign: {item.get('signing_command')}")
    actions = payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []
    for action in actions:
        print(f"next: {action}")


def _field_device_signing_command(device_id: str, secret_env: str) -> str:
    secret_arg = f"--secret-env {secret_env}" if secret_env else "--secret <DEVICE_SECRET>"
    return (
        "python -m askme runtime field-sign-device-payload device-event.json "
        f"--device-id {device_id} {secret_arg} --output device-event.signed.json"
    )


def _field_device_trust_next_actions(status: str, missing: list[dict[str, Any]]) -> list[str]:
    if status == "invalid_profile":
        return ["修复站点配置文件中的设备、区域、响应组或阈值配置。"]
    if missing:
        envs = ", ".join(
            sorted({str(row.get("secret_env") or "") for row in missing if row.get("secret_env")})
        )
        if envs:
            return [f"在部署环境中配置设备 HMAC secret：{envs}。"]
        return ["为每个现场设备补充 secret_env，并在部署环境中配置对应 HMAC secret。"]
    return ["设备 HMAC secret 已就绪，可使用 field-sign-device-payload 生成可信事件。"]


def _run_field_site_env_template(*, site_profile: str, output: str = "") -> dict[str, Any]:
    from askme.pipeline.field.customer_project_profiles import (
        load_field_site_profile,
    )
    from askme.pipeline.field.customer_project_template_support import (
        site_profile_env_references,
    )
    from askme.pipeline.field.field_site_runtime_config import render_site_profile_env_template
    from askme.pipeline.field.field_site_validation import (
        validate_field_site_profile,
    )

    path = Path(site_profile)
    try:
        profile = load_field_site_profile(path)
        validation = validate_field_site_profile(profile, check_env=False)
        refs = site_profile_env_references(profile)
        template = render_site_profile_env_template(profile)
    except Exception as exc:
        return {
            "status": "invalid_profile",
            "site_profile": str(path),
            "reason": str(exc),
            "env_count": 0,
            "configured_count": 0,
            "missing_count": 0,
            "env_refs": [],
            "template": "",
            "output": "",
            "next_actions": ["修复站点配置文件后再生成环境变量模板。"],
        }

    output_path = str(output or "").strip()
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8", newline="\n")
    configured_count = sum(1 for ref in refs if ref.get("configured"))
    missing_count = len(refs) - configured_count
    return {
        "status": "ok" if validation.get("status") == "passed" else "invalid_profile",
        "site_profile": str(path),
        "profile_valid": validation.get("status") == "passed",
        "profile_errors": validation.get("errors") or [],
        "env_count": len(refs),
        "configured_count": configured_count,
        "missing_count": missing_count,
        "env_refs": refs,
        "template": "" if output_path else template,
        "output": output_path,
        "next_actions": _field_site_env_template_next_actions(
            profile_valid=validation.get("status") == "passed",
            missing_count=missing_count,
            output=output_path,
        ),
    }


def _field_site_env_template_next_actions(
    *,
    profile_valid: bool,
    missing_count: int,
    output: str,
) -> list[str]:
    if not profile_valid:
        return ["先修复站点配置文件中的必填项，再生成部署环境模板。"]
    actions = []
    if output:
        actions.append(f"将 {output} 交给现场交付人员填写真实密钥。")
    else:
        actions.append("将输出保存为部署环境的 .env 文件，并填写真实密钥。")
    if missing_count:
        actions.append("配置完成后运行 field-readiness --check-site-env 验证环境变量是否就绪。")
    else:
        actions.append("当前进程已读取到所有引用的环境变量，可继续做现场 readiness 检查。")
    return actions


def _emit_field_site_env_template_payload(payload: dict[str, Any]) -> None:
    print(
        "field-site-env-template: "
        f"{payload.get('status')} "
        f"envs={payload.get('env_count', 0)} "
        f"configured={payload.get('configured_count', 0)} "
        f"missing={payload.get('missing_count', 0)}"
    )
    if payload.get("reason"):
        print(f"reason: {payload.get('reason')}")
    if payload.get("output"):
        print(f"output: {payload.get('output')}")
    refs = payload.get("env_refs") if isinstance(payload.get("env_refs"), list) else []
    for item in refs[:20]:
        if not isinstance(item, dict):
            continue
        print(
            "  - "
            f"{item.get('env_name') or '-'} "
            f"category={item.get('category') or '-'} "
            f"configured={bool(item.get('configured'))} "
            f"ref={item.get('reference') or '-'}"
        )
    if not payload.get("output") and payload.get("template"):
        print("")
        print(payload.get("template"), end="")
    actions = payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []
    for action in actions:
        print(f"next: {action}")


def _run_field_live_demo(
    *,
    output_dir: str,
    site_profile: str,
    server: str = "",
    timeout_s: float = 8.0,
    scenario_file: str = "",
    refresh_scenario_timestamps: bool = False,
) -> dict[str, Any]:
    from scripts.demo.live_field_operations_demo import run_live_demo

    return run_live_demo(
        output_dir=Path(output_dir),
        site_profile=Path(site_profile),
        server=server,
        timeout_s=timeout_s,
        scenario_file=Path(scenario_file) if scenario_file else None,
        refresh_scenario_timestamps=refresh_scenario_timestamps,
    )


def _emit_field_live_demo_payload(payload: dict[str, Any]) -> None:
    print(
        "field-live-demo: "
        f"{payload.get('status')} "
        f"accepted={payload.get('accepted', 0)}/{payload.get('scenario_count', 0)} "
        f"mode={payload.get('mode') or '-'}"
    )
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    if readiness:
        print(f"readiness: {readiness.get('status') or '-'}")
    if payload.get("report_path"):
        print(f"report: {payload.get('report_path')}")
    if payload.get("guide_path"):
        print(f"guide: {payload.get('guide_path')}")
    if payload.get("html_report_path"):
        print(f"html: {payload.get('html_report_path')}")
    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    if scenarios:
        print("scenarios:")
        for item in scenarios[:10]:
            if not isinstance(item, dict):
                continue
            print(
                "- "
                f"{item.get('scenario_id') or '-'}: "
                f"http={item.get('http_status') or '-'} "
                f"accepted={bool(item.get('accepted'))} "
                f"event={item.get('event_id') or '-'}"
            )


def _field_ingest_smoke_device_secrets() -> dict[str, str]:
    return {
        "camera-main-road-1": "smoke-camera-main-road",
        "cam-main-road-01": "smoke-anpr-main-road",
        "camera-plaza-1": "smoke-camera-plaza",
        "smoke-power-room-1": "smoke-sensor-power-room",
        "robot-thunder-1": "smoke-robot-thunder",
        "bin-17": "smoke-bin-17",
    }


def _field_ingest_smoke_trusted_device_config() -> dict[str, Any]:
    return {
        "require_trusted_devices": True,
        "device_registry": {
            "camera-main-road-1": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-camera-main-road",
                "require_signature": True,
            },
            "cam-main-road-01": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-anpr-main-road",
                "require_signature": True,
            },
            "camera-plaza-1": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-camera-plaza",
                "require_signature": True,
            },
            "smoke-power-room-1": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "smoke-sensor-power-room",
                "require_signature": True,
            },
            "robot-thunder-1": {
                "allowed_sources": ["robot"],
                "hmac_secret": "smoke-robot-thunder",
                "require_signature": True,
            },
            "bin-17": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-bin-17",
                "require_signature": True,
            },
        },
    }
