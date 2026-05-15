"""Runtime bridge for forwarding field-device event files to field ingest.

Many camera, sensor, and robot diagnostic processes can already write JSON or
JSONL. The bridge keeps that integration thin: read new device events, normalize
them with the stable field-ingest adapter, then POST to ``/api/field/ingest``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from askme.pipeline.field_ingest_adapters import normalize_field_ingest_payload
from askme.pipeline.field_operations import sign_field_device_payload

PostFunc = Callable[[str, dict[str, Any], float], dict[str, Any]]


def run_field_ingest_bridge_once(
    *,
    source: str | Path,
    server: str,
    state_path: str | Path | None = None,
    dry_run: bool = False,
    limit: int = 0,
    timeout_s: float = 8.0,
    device_secrets: dict[str, str] | None = None,
    post_func: PostFunc | None = None,
) -> dict[str, Any]:
    """Process newly available events from ``source`` once."""
    source_path = Path(source)
    state_file = Path(state_path) if state_path else _default_state_path(source_path)
    state = _read_state(state_file)
    events, new_state = _read_new_events(source_path, state)
    if limit > 0:
        events = events[:limit]

    endpoint = _ingest_endpoint(server)
    post = post_func or _post_json
    results: list[dict[str, Any]] = []
    failures = 0
    for index, event in enumerate(events, start=1):
        normalized = normalize_field_ingest_payload(event)
        normalized, signing = _maybe_sign_device_payload(normalized, device_secrets or {})
        item: dict[str, Any] = {
            "index": index,
            "source": str(source_path),
            "normalized": normalized,
            "posted": False,
            "device_signing": signing,
        }
        if dry_run:
            item["status"] = "dry_run"
        else:
            try:
                response = post(endpoint, normalized, timeout_s)
            except Exception as exc:
                failures += 1
                item["status"] = "failed"
                item["error"] = str(exc)
            else:
                item["posted"] = True
                item["status"] = str(response.get("status") or "unknown")
                item["accepted"] = bool(response.get("accepted"))
                item["reason"] = str(response.get("reason") or "")
                item["scenario_id"] = (
                    (response.get("normalized") or {}).get("scenario_id")
                    or normalized.get("scenario_id")
                    or ""
                )
                item["event_id"] = (response.get("event") or {}).get("event_id") or ""
        results.append(item)

    _write_state(state_file, new_state)
    summary = _bridge_summary(results, new_state)
    return {
        "status": "failed" if failures else "ok",
        "target": "field-ingest-bridge",
        "source": str(source_path),
        "state_path": str(state_file),
        "server": endpoint.rsplit("/api/field/ingest", 1)[0],
        "dry_run": dry_run,
        "count": len(events),
        "failed": failures,
        "summary": summary,
        "results": results,
    }


def watch_field_ingest_bridge(
    *,
    source: str | Path,
    server: str,
    state_path: str | Path | None = None,
    interval_s: float = 1.0,
    dry_run: bool = False,
    limit: int = 0,
    timeout_s: float = 8.0,
    device_secrets: dict[str, str] | None = None,
) -> None:
    """Run a long-lived bridge loop until interrupted."""
    while True:
        payload = run_field_ingest_bridge_once(
            source=source,
            server=server,
            state_path=state_path,
            dry_run=dry_run,
            limit=limit,
            timeout_s=timeout_s,
            device_secrets=device_secrets,
        )
        if payload["count"]:
            print(_summary_line(payload), flush=True)
        time.sleep(max(0.1, interval_s))


def _read_new_events(path: Path, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _read_new_jsonl_events(path, state)
    return _read_changed_json_events(path, state)


def _read_new_jsonl_events(path: Path, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    offset = int(state.get("offset") or 0)
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read()
        new_offset = handle.tell()
    if not raw:
        return [], {**state, "offset": new_offset, "source": str(path)}
    text = raw.decode("utf-8-sig" if offset == 0 else "utf-8")
    events = _json_lines(text, path)
    return events, {**state, "offset": new_offset, "source": str(path), "format": "jsonl"}


def _read_changed_json_events(path: Path, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stat = path.stat()
    fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
    if state.get("fingerprint") == fingerprint:
        return [], {**state, "source": str(path)}
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    events = loaded if isinstance(loaded, list) else [loaded]
    result = _ensure_objects(events, path)
    return result, {
        **state,
        "fingerprint": fingerprint,
        "source": str(path),
        "format": "json",
    }


def _json_lines(text: str, path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    return _ensure_objects(events, path)


def _ensure_objects(events: list[Any], path: Path) -> list[dict[str, Any]]:
    result = [event for event in events if isinstance(event, dict)]
    if len(result) != len(events):
        raise ValueError(f"Field ingest source must contain JSON objects: {path}")
    return result


def _post_json(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    response = requests.post(url, json=body, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"status": "invalid_response"}


def _maybe_sign_device_payload(
    payload: dict[str, Any],
    device_secrets: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    device_id = _bridge_device_id(payload)
    source = str(payload.get("source") or "").strip()
    secret = ""
    if device_id:
        secret = str(device_secrets.get(device_id) or "")
    if not secret and source:
        secret = str(device_secrets.get(source) or "")
    if not secret:
        secret = str(device_secrets.get("*") or "")
    signing = {
        "requested": bool(device_secrets),
        "signed": False,
        "device_id": device_id,
        "source": source,
        "reason": "no_matching_secret" if device_secrets else "not_configured",
    }
    if not secret:
        return payload, signing
    if payload.get("device_signature"):
        signing["reason"] = "already_signed"
        signing["signed"] = True
        return payload, signing
    signed_payload = dict(payload)
    if device_id and not signed_payload.get("device_id"):
        signed_payload["device_id"] = device_id
    signed_payload.setdefault("device_signature_alg", "hmac-sha256")
    signed_payload.setdefault("device_signature_timestamp", time.time())
    signed_payload["device_signature"] = sign_field_device_payload(signed_payload, secret=secret)
    signing["reason"] = "signed"
    signing["signed"] = True
    return signed_payload, signing


def sign_field_ingest_payload(
    payload: dict[str, Any],
    device_secrets: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sign a normalized field-ingest payload with the same rules used by the bridge."""

    return _maybe_sign_device_payload(payload, device_secrets)


def _bridge_device_id(payload: dict[str, Any]) -> str:
    for key in ("device_id", "camera_id", "sensor_id", "robot_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    sensor = payload.get("sensor") if isinstance(payload.get("sensor"), dict) else {}
    robot = payload.get("robot") if isinstance(payload.get("robot"), dict) else {}
    for container in (sensor, robot):
        for key in ("device_id", "sensor_id", "robot_id", "hardware_id"):
            value = container.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return ""


def _bridge_summary(results: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    scenario_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    signed_count = 0
    posted_count = 0
    accepted_count = 0
    event_count = 0
    for item in results:
        signing = item.get("device_signing") if isinstance(item.get("device_signing"), dict) else {}
        if signing.get("signed"):
            signed_count += 1
        if item.get("posted"):
            posted_count += 1
        if item.get("accepted"):
            accepted_count += 1
        if item.get("event_id"):
            event_count += 1
        normalized = item.get("normalized") if isinstance(item.get("normalized"), dict) else {}
        scenario_id = str(item.get("scenario_id") or normalized.get("scenario_id") or "").strip()
        if scenario_id:
            scenario_counts[scenario_id] = scenario_counts.get(scenario_id, 0) + 1
        source = str(normalized.get("source") or "").strip()
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        device_id = _bridge_device_id(normalized)
        if device_id:
            device_counts[device_id] = device_counts.get(device_id, 0) + 1
    return {
        "processed": len(results),
        "posted": posted_count,
        "accepted": accepted_count,
        "failed": sum(1 for item in results if item.get("status") == "failed"),
        "signed": signed_count,
        "events_created": event_count,
        "scenario_counts": scenario_counts,
        "source_counts": source_counts,
        "device_counts": device_counts,
        "source_format": state.get("format") or "",
        "offset": state.get("offset"),
        "fingerprint": state.get("fingerprint") or "",
    }


def _ingest_endpoint(server: str) -> str:
    base = str(server or "http://127.0.0.1:8765").rstrip("/")
    if base.endswith("/api/field/ingest"):
        return base
    return f"{base}/api/field/ingest"


def _default_state_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".askme-state.json")


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_line(payload: dict[str, Any]) -> str:
    return (
        "field-ingest-bridge: "
        f"{payload.get('status')} count={payload.get('count')} "
        f"failed={payload.get('failed')} dry_run={payload.get('dry_run')}"
    )
