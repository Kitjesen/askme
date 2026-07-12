"""Field action audit CLI commands extracted from askme.cli."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from askme.cli.utils import (
    _acquire_field_audit_retry_lock,
    _append_field_audit_retry_queue,
    _cli_root_override,
    _get_json,
    _normalise_server_url,
    _post_json_with_retries,
    _resolve_field_action_audit_hmac_secret,
)


def _run_field_audit_integrity(
    *,
    server: str,
    archive_path: str,
    audit_path: str,
    hmac_secret: str = "",
) -> dict[str, Any]:
    if server:
        get_json = _cli_root_override("_get_json", _get_json)
        return get_json(f"{_normalise_server_url(server)}/api/field/audit/integrity")
    from askme.pipeline.field.field_operations import FieldOperationsService

    resolved_hmac_secret = _resolve_field_action_audit_hmac_secret(hmac_secret)
    action_audit: dict[str, Any] = {
        "enabled": True,
        "path": audit_path,
        "swallow_errors": False,
    }
    if resolved_hmac_secret:
        action_audit["hmac_secret"] = resolved_hmac_secret
    service = FieldOperationsService(
        config={
            "archive_path": archive_path,
            "action_audit": action_audit,
        }
    )
    return service.action_audit_integrity_payload()


def _emit_field_audit_integrity_payload(payload: dict[str, Any]) -> None:
    status = "valid" if payload.get("valid") else "invalid"
    print(f"field-audit-integrity: {status}")
    print(f"path: {payload.get('path') or '-'}")
    print(f"checked: {payload.get('checked_count', 0)} / expected: {payload.get('expected_count', '-')}")
    print(f"latest_hash: {payload.get('latest_hash') or '-'}")
    print(f"signed: {bool(payload.get('signed'))}")
    failures = payload.get("failures") or []
    if failures:
        print("failures:")
        for item in failures[:10]:
            line = item.get("line", 0)
            reason = item.get("reason") or "unknown"
            detail = item.get("detail")
            suffix = f" ({detail})" if detail else ""
            print(f"- line {line}: {reason}{suffix}")


def _run_field_audit_anchor(
    *,
    server: str,
    archive_path: str,
    audit_path: str,
    hmac_secret: str = "",
    output: str = "",
    webhook_url: str = "",
    webhook_retries: int = 3,
    retry_queue: str = "",
    require_valid: bool = True,
) -> dict[str, Any]:

    run_field_audit_integrity = _cli_root_override(
        "_run_field_audit_integrity",
        _run_field_audit_integrity,
    )
    post_json_with_retries = _cli_root_override(
        "_post_json_with_retries",
        _post_json_with_retries,
    )

    integrity = run_field_audit_integrity(
        server=server,
        archive_path=archive_path,
        audit_path=audit_path,
        hmac_secret=hmac_secret,
    )
    valid = integrity.get("enabled") is False or integrity.get("valid") is True
    checkpoint = {
        "path": integrity.get("path") or audit_path,
        "latest_hash": integrity.get("latest_hash") or "",
        "hash_alg": integrity.get("hash_alg") or "",
        "checked_count": integrity.get("checked_count", 0),
        "expected_count": integrity.get("expected_count", 0),
        "signed": bool(integrity.get("signed")),
        "signature_alg": integrity.get("signature_alg") or "",
    }
    payload: dict[str, Any] = {
        "status": "blocked" if require_valid and not valid else "anchored",
        "target": "field-audit-anchor",
        "generated_at": round(time.time(), 3),
        "source": "server" if server else "local",
        "server": _normalise_server_url(server) if server else "",
        "archive_path": archive_path,
        "audit_path": audit_path,
        "checkpoint": checkpoint,
        "integrity": integrity,
        "output_path": output,
        "webhook_url": webhook_url,
        "webhook_delivery": None,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if webhook_url and payload["status"] != "blocked":
        delivery = post_json_with_retries(
            webhook_url,
            payload,
            attempts=max(1, int(webhook_retries or 1)),
        )
        payload["webhook_delivery"] = delivery
        if delivery.get("status") != "sent":
            payload["status"] = "delivery_failed"
            if retry_queue:
                _append_field_audit_retry_queue(retry_queue, payload)
        if output:
            path = Path(output)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_audit_anchor_payload(payload: dict[str, Any]) -> None:
    checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
    print(f"field-audit-anchor: {payload.get('status')}")
    print(f"latest_hash: {checkpoint.get('latest_hash') or '-'}")
    print(f"checked: {checkpoint.get('checked_count', 0)} / expected: {checkpoint.get('expected_count', '-')}")
    print(f"signed: {bool(checkpoint.get('signed'))}")
    if payload.get("output_path"):
        print(f"output: {payload.get('output_path')}")
    if payload.get("webhook_url"):
        delivery = payload.get("webhook_delivery")
        print(f"webhook: {'sent' if delivery else 'not_sent'}")


def _run_field_audit_delivery_retry(
    *,
    queue: str,
    webhook_retries: int = 3,
    lock_timeout_s: float = 300.0,
) -> dict[str, Any]:
    path = Path(queue)
    if not path.exists():
        return {
            "status": "empty",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "attempted": 0,
            "sent": 0,
            "remaining": 0,
            "results": [],
        }
    lock = _acquire_field_audit_retry_lock(path, lock_timeout_s=lock_timeout_s)
    if not lock.get("acquired"):
        return {
            "status": "locked",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "lock": lock,
            "attempted": 0,
            "sent": 0,
            "remaining": None,
            "results": [],
        }
    remaining: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                remaining.append({"line": line_number, "raw": line, "error": str(exc)})
                results.append({"line": line_number, "status": "invalid_json", "error": str(exc)})
                continue
            webhook_url = str(record.get("webhook_url") or "")
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if not webhook_url or not payload:
                remaining.append(record)
                results.append({"line": line_number, "status": "invalid_record"})
                continue
            post_json_with_retries = _cli_root_override(
                "_post_json_with_retries",
                _post_json_with_retries,
            )
            delivery = post_json_with_retries(
                webhook_url,
                payload,
                attempts=max(1, int(webhook_retries or 1)),
            )
            results.append({"line": line_number, "webhook_url": webhook_url, "delivery": delivery})
            if delivery.get("status") != "sent":
                remaining.append(record)
        if remaining:
            path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining),
                encoding="utf-8",
            )
        else:
            path.unlink(missing_ok=True)
        sent = sum(1 for item in results if item.get("delivery", {}).get("status") == "sent")
        return {
            "status": "sent" if results and not remaining else "failed" if remaining else "empty",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "lock": lock,
            "attempted": len(results),
            "sent": sent,
            "remaining": len(remaining),
            "results": results,
        }
    finally:
        lock_path = lock.get("path")
        if lock_path:
            Path(str(lock_path)).unlink(missing_ok=True)


def _emit_field_audit_delivery_retry_payload(payload: dict[str, Any]) -> None:
    print(f"field-audit-retry-delivery: {payload.get('status')}")
    print(f"attempted: {payload.get('attempted', 0)}")
    print(f"sent: {payload.get('sent', 0)}")
    print(f"remaining: {payload.get('remaining', 0)}")
    print(f"queue: {payload.get('queue') or '-'}")
    lock = payload.get("lock") if isinstance(payload.get("lock"), dict) else {}
    if lock:
        print(f"lock: {lock.get('path') or '-'}")
        if lock.get("reason"):
            print(f"lock_reason: {lock.get('reason')}")


def _run_field_audit_retry_status(*, queue: str) -> dict[str, Any]:
    path = Path(queue)
    if not path.exists():
        return {
            "status": "empty",
            "target": "field-audit-retry-status",
            "queue": str(path),
            "pending": 0,
            "invalid": 0,
            "items": [],
        }
    items: list[dict[str, Any]] = []
    invalid = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid += 1
            items.append({"line": line_number, "status": "invalid_json", "error": str(exc)})
            continue
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
        items.append(
            {
                "line": line_number,
                "status": "pending",
                "webhook_url": record.get("webhook_url") or "",
                "queued_at": record.get("queued_at"),
                "latest_hash": checkpoint.get("latest_hash") or "",
                "checked_count": checkpoint.get("checked_count", 0),
            }
        )
    pending = sum(1 for item in items if item.get("status") == "pending")
    return {
        "status": "pending" if pending or invalid else "empty",
        "target": "field-audit-retry-status",
        "queue": str(path),
        "pending": pending,
        "invalid": invalid,
        "items": items,
    }


def _emit_field_audit_retry_status_payload(payload: dict[str, Any]) -> None:
    print(f"field-audit-retry-status: {payload.get('status')}")
    print(f"pending: {payload.get('pending', 0)}")
    print(f"invalid: {payload.get('invalid', 0)}")
    print(f"queue: {payload.get('queue') or '-'}")
    for item in (payload.get("items") or [])[:10]:
        print(f"- line {item.get('line')}: {item.get('status')} {item.get('latest_hash') or item.get('error') or ''}")
