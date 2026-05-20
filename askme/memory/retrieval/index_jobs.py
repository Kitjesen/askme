"""Durable job history for knowledge index rebuild operations."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from askme.config import project_root

_UTC = timezone(timedelta(0))


class KnowledgeIndexJobStore:
    """Small JSON-backed audit log for Knowledge Console index jobs."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        path: str | Path | None = None,
        enabled: bool | None = None,
        max_history: int | None = None,
        swallow_errors: bool | None = None,
    ) -> None:
        cfg = ((config or {}).get("memory") or {}).get("knowledge_index_jobs") or {}
        self.enabled = bool(cfg.get("enabled", True) if enabled is None else enabled)
        self.max_history = max(1, int(cfg.get("max_history", 100) if max_history is None else max_history))
        self.swallow_errors = bool(
            cfg.get("swallow_errors", True) if swallow_errors is None else swallow_errors,
        )
        raw_path = path or cfg.get("path") or "data/memory/catalog/index_jobs.json"
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        self.path = resolved
        self._lock = threading.RLock()
        self._jobs: list[dict[str, Any]] = []
        self._last_error = ""
        self._load()

    def health(self) -> dict[str, Any]:
        with self._lock:
            last = self._jobs[0] if self._jobs else {}
            return {
                "enabled": self.enabled,
                "path": str(self.path),
                "total": len(self._jobs),
                "last_job_id": last.get("job_id", ""),
                "last_status": last.get("status", ""),
                "last_error": self._last_error,
            }

    def list_jobs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), self.max_history))
        with self._lock:
            return [dict(job) for job in self._jobs[:safe_limit]]

    def record(self, job: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return dict(job)
        clean = self._clean_job(job)
        with self._lock:
            self._jobs = [clean, *self._jobs]
            self._jobs = self._jobs[: self.max_history]
            self._save()
        return dict(clean)

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            jobs = payload.get("jobs") if isinstance(payload, dict) else payload
            if isinstance(jobs, list):
                self._jobs = [
                    self._clean_job(job)
                    for job in jobs
                    if isinstance(job, dict) and str(job.get("job_id") or "").strip()
                ][: self.max_history]
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            if not self.swallow_errors:
                raise

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now(_UTC).isoformat(timespec="seconds"),
                "jobs": self._jobs,
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._last_error = ""
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            if not self.swallow_errors:
                raise

    @staticmethod
    def _clean_job(job: dict[str, Any]) -> dict[str, Any]:
        errors = job.get("errors")
        record_ids = job.get("record_ids")
        requested = job.get("requested_record_ids")
        return {
            "job_id": str(job.get("job_id") or "").strip(),
            "type": str(job.get("type") or "knowledge_rebuild_index"),
            "status": str(job.get("status") or "unknown"),
            "operator_id": str(job.get("operator_id") or ""),
            "started_at": str(job.get("started_at") or ""),
            "completed_at": str(job.get("completed_at") or ""),
            "duration_ms": _int_or_default(job.get("duration_ms"), 0),
            "requested_record_ids": [
                str(item)
                for item in requested
                if str(item or "").strip()
            ] if isinstance(requested, list) else [],
            "record_ids": [
                str(item)
                for item in record_ids
                if str(item or "").strip()
            ] if isinstance(record_ids, list) else [],
            "scanned": _int_or_default(job.get("scanned"), 0),
            "eligible": _int_or_default(job.get("eligible"), 0),
            "selected": _int_or_default(job.get("selected"), 0),
            "indexed": _int_or_default(job.get("indexed"), 0),
            "skipped": _int_or_default(job.get("skipped"), 0),
            "errors": [str(item) for item in errors] if isinstance(errors, list) else [],
            "backend": str(job.get("backend") or ""),
            "fallback_reason": str(job.get("fallback_reason") or ""),
            "include_ineligible": bool(job.get("include_ineligible", False)),
        }


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
