"""Signed unified audit export packages and optional webhook delivery."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .query import AuditQueryService

PostJson = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


@dataclass(frozen=True)
class AuditExportConfig:
    output_dir: Path
    webhook_url: str = ""
    hmac_secret: str = ""
    signature_key_id: str = "local-unified-audit-export"
    retry_queue_path: Path | None = None
    timeout_s: float = 5.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> AuditExportConfig:
        cfg = config or {}
        audit_cfg = cfg.get("audit") if isinstance(cfg.get("audit"), dict) else {}
        export_cfg = audit_cfg.get("export") if isinstance(audit_cfg.get("export"), dict) else {}
        output_dir = Path(str(export_cfg.get("output_dir") or "artifacts/audit_exports"))
        retry_queue = str(export_cfg.get("retry_queue_path") or "").strip()
        return cls(
            output_dir=output_dir,
            webhook_url=str(export_cfg.get("webhook_url") or "").strip(),
            hmac_secret=str(export_cfg.get("hmac_secret") or "").strip(),
            signature_key_id=str(export_cfg.get("signature_key_id") or "local-unified-audit-export"),
            retry_queue_path=Path(retry_queue) if retry_queue else output_dir / "export-delivery-retry.jsonl",
            timeout_s=float(export_cfg.get("timeout_s") or 5.0),
        )


class AuditExportService:
    """Create immutable-ish local audit export packages for customer evidence."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        query_service: AuditQueryService | None = None,
        post_json: PostJson | None = None,
    ) -> None:
        self._config = config or {}
        self._export_config = AuditExportConfig.from_config(self._config)
        self._query_service = query_service or AuditQueryService(self._config)
        self._post_json = post_json or _post_json

    def create_export(
        self,
        *,
        actor_id: str,
        limit: int = 500,
        source: str = "",
        operator_id: str = "",
        action: str = "",
        outcome: str = "",
        q: str = "",
        deliver: bool = False,
        webhook_url: str = "",
    ) -> dict[str, Any]:
        payload = self._query_service.query(
            limit=limit,
            source=source,
            operator_id=operator_id,
            action=action,
            outcome=outcome,
            q=q,
        )
        export_id = self._export_id()
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        content = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        if content:
            content += "\n"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        output_dir = self._export_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / f"{export_id}.jsonl"
        manifest_path = output_dir / f"{export_id}.manifest.json"
        records_path.write_text(content, encoding="utf-8", newline="\n")
        manifest = {
            "export_id": export_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "actor_id": actor_id,
            "record_count": len(records),
            "filtered_total": payload.get("filtered_total", len(records)),
            "total": payload.get("total", len(records)),
            "filters": payload.get("filters", {}),
            "summary": payload.get("summary", {}),
            "sources": payload.get("sources", {}),
            "format": "jsonl",
            "records_path": str(records_path),
            "sha256": content_hash,
            "signature": self._signature(content_hash),
            "signature_alg": "hmac-sha256" if self._export_config.hmac_secret else "",
            "signature_key_id": self._export_config.signature_key_id if self._export_config.hmac_secret else "",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        result = {
            "ok": True,
            "export": {
                **manifest,
                "manifest_path": str(manifest_path),
            },
            "delivery": None,
        }
        target = str(webhook_url or self._export_config.webhook_url or "").strip()
        if deliver:
            result["delivery"] = self.deliver_export(manifest, target)
        return result

    def deliver_export(self, manifest: dict[str, Any], webhook_url: str) -> dict[str, Any]:
        if not webhook_url:
            return {"sent": False, "reason": "webhook_url_missing"}
        body = {
            "type": "askme.unified_audit_export",
            "manifest": manifest,
        }
        headers = {"Content-Type": "application/json"}
        if self._export_config.hmac_secret:
            headers["X-Askme-Audit-Signature"] = _sign_payload(
                body,
                secret=self._export_config.hmac_secret,
            )
            headers["X-Askme-Audit-Signature-Alg"] = "hmac-sha256"
            headers["X-Askme-Audit-Signature-Key-Id"] = self._export_config.signature_key_id
        try:
            return self._post_json(
                webhook_url,
                body,
                headers,
                max(0.5, float(self._export_config.timeout_s or 5.0)),
            )
        except Exception as exc:
            delivery = {
                "sent": False,
                "reason": "webhook_delivery_failed",
                "error": str(exc),
                "webhook_url": webhook_url,
            }
            self._queue_delivery_retry(webhook_url, body, headers, delivery)
            return delivery

    def retry_status(self, *, limit: int = 50) -> dict[str, Any]:
        """Return a customer-safe summary of pending export delivery retries."""
        queue = self._export_config.retry_queue_path
        if queue is None or not queue.exists():
            return {
                "status": "empty",
                "queue": str(queue) if queue is not None else "",
                "exists": False,
                "pending": 0,
                "invalid": 0,
                "items": [],
            }
        entries = self._read_retry_entries(queue)
        pending_items: list[dict[str, Any]] = []
        invalid_items: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("invalid"):
                invalid_items.append(
                    {
                        "line": entry.get("line"),
                        "status": "invalid",
                        "error": entry.get("error") or "invalid_retry_record",
                    }
                )
                continue
            record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
            pending_items.append(self._retry_item_summary(record, line=entry.get("line")))
        limited_items = (pending_items + invalid_items)[: max(0, int(limit or 0))]
        pending = len(pending_items)
        invalid = len(invalid_items)
        return {
            "status": "pending" if pending or invalid else "empty",
            "queue": str(queue),
            "exists": True,
            "pending": pending,
            "invalid": invalid,
            "items": limited_items,
        }

    def retry_queued_deliveries(self, *, limit: int = 50) -> dict[str, Any]:
        """Replay queued export deliveries and keep failed or invalid records queued."""
        queue = self._export_config.retry_queue_path
        if queue is None or not queue.exists():
            return {
                "status": "empty",
                "queue": str(queue) if queue is not None else "",
                "attempted": 0,
                "sent": 0,
                "failed": 0,
                "remaining": 0,
                "invalid": 0,
                "results": [],
            }
        lock = self._acquire_retry_lock(queue)
        if not lock.get("acquired"):
            return {
                "status": "locked",
                "queue": str(queue),
                "lock": lock,
                "attempted": 0,
                "sent": 0,
                "failed": 0,
                "remaining": None,
                "invalid": 0,
                "results": [],
            }
        attempted = 0
        sent = 0
        failed = 0
        invalid = 0
        results: list[dict[str, Any]] = []
        remaining_lines: list[str] = []
        max_attempts = max(0, int(limit or 0))
        try:
            for entry in self._read_retry_entries(queue):
                raw = str(entry.get("raw") or "")
                if entry.get("invalid"):
                    invalid += 1
                    remaining_lines.append(raw)
                    results.append(
                        {
                            "line": entry.get("line"),
                            "status": "invalid",
                            "error": entry.get("error") or "invalid_retry_record",
                        }
                    )
                    continue
                record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
                webhook_url = str(record.get("webhook_url") or "")
                body = record.get("body") if isinstance(record.get("body"), dict) else {}
                headers = record.get("headers") if isinstance(record.get("headers"), dict) else {}
                if not webhook_url or not body:
                    invalid += 1
                    remaining_lines.append(raw)
                    results.append({"line": entry.get("line"), "status": "invalid_record"})
                    continue
                if attempted >= max_attempts:
                    remaining_lines.append(raw)
                    continue
                attempted += 1
                delivery = self._retry_delivery(webhook_url, body, headers)
                summary = self._retry_item_summary(record, line=entry.get("line"))
                results.append({**summary, "delivery": delivery})
                if delivery.get("sent"):
                    sent += 1
                    continue
                failed += 1
                record["delivery"] = delivery
                record["last_retry_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                record["retry_count"] = int(record.get("retry_count") or 0) + 1
                remaining_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
            if remaining_lines:
                queue.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8", newline="\n")
            else:
                queue.unlink(missing_ok=True)
            status = "pending" if remaining_lines else "sent" if attempted else "empty"
            return {
                "status": status,
                "queue": str(queue),
                "lock": lock,
                "attempted": attempted,
                "sent": sent,
                "failed": failed,
                "remaining": len(remaining_lines),
                "invalid": invalid,
                "results": results,
            }
        finally:
            lock_path = lock.get("path")
            if lock_path:
                Path(str(lock_path)).unlink(missing_ok=True)

    def _queue_delivery_retry(
        self,
        webhook_url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        delivery: dict[str, Any],
    ) -> None:
        queue = self._export_config.retry_queue_path
        if queue is None:
            return
        try:
            queue.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "queued_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "webhook_url": webhook_url,
                "body": body,
                "headers": headers,
                "delivery": delivery,
            }
            with queue.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        except OSError:
            return

    def _retry_delivery(
        self,
        webhook_url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        try:
            return self._post_json(
                webhook_url,
                body,
                headers,
                max(0.5, float(self._export_config.timeout_s or 5.0)),
            )
        except Exception as exc:
            return {
                "sent": False,
                "reason": "webhook_delivery_failed",
                "error": str(exc),
                "webhook_url": webhook_url,
            }

    def _read_retry_entries(self, queue: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            lines = queue.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            return [{"line": 0, "raw": "", "invalid": True, "error": str(exc)}]
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                entries.append(
                    {
                        "line": line_number,
                        "raw": line,
                        "invalid": True,
                        "error": str(exc),
                    }
                )
                continue
            if not isinstance(record, dict):
                entries.append(
                    {
                        "line": line_number,
                        "raw": line,
                        "invalid": True,
                        "error": "retry_record_not_object",
                    }
                )
                continue
            entries.append({"line": line_number, "raw": line, "record": record, "invalid": False})
        return entries

    def _retry_item_summary(self, record: dict[str, Any], *, line: Any = None) -> dict[str, Any]:
        body = record.get("body") if isinstance(record.get("body"), dict) else {}
        manifest = body.get("manifest") if isinstance(body.get("manifest"), dict) else {}
        delivery = record.get("delivery") if isinstance(record.get("delivery"), dict) else {}
        return {
            "line": line,
            "status": "pending",
            "queued_at": record.get("queued_at") or "",
            "last_retry_at": record.get("last_retry_at") or "",
            "retry_count": int(record.get("retry_count") or 0),
            "webhook_url": record.get("webhook_url") or delivery.get("webhook_url") or "",
            "export_id": manifest.get("export_id") or "",
            "record_count": manifest.get("record_count", 0),
            "reason": delivery.get("reason") or "",
            "error": delivery.get("error") or "",
        }

    def _acquire_retry_lock(self, queue: Path) -> dict[str, Any]:
        lock_path = queue.with_suffix(queue.suffix + ".lock")
        now = time.time()
        payload = {
            "acquired": True,
            "path": str(lock_path),
            "pid": os.getpid(),
            "queue": str(queue),
            "acquired_at": round(now, 3),
            "expires_at": round(now + 300.0, 3),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(lock_path), flags)
        except FileExistsError:
            return {"acquired": False, "path": str(lock_path), "reason": "audit_export_retry_already_running"}
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return payload

    def _signature(self, content_hash: str) -> str:
        if not self._export_config.hmac_secret:
            return ""
        return hmac.new(
            self._export_config.hmac_secret.encode("utf-8"),
            content_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _export_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"audit-export-{timestamp}-{uuid4().hex[:8]}"


def _sign_payload(payload: dict[str, Any], *, secret: str) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "sent": 200 <= int(response.status) < 300,
                "status_code": int(response.status),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "response_preview": body[:300],
                "webhook_url": url,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "sent": False,
            "status_code": int(exc.code),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "response_preview": body[:300],
            "webhook_url": url,
        }
