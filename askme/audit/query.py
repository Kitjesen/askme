"""Read-only unified audit timeline for product governance views."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.skills.audit import SkillAuditLog, default_skill_audit_path


@dataclass(frozen=True)
class AuditPaths:
    skill_audit: Path
    field_action_audit: Path | None = None
    field_event_archive: Path | None = None
    runtime_audit: Path | None = None


class AuditQueryService:
    """Aggregate append-only audit sources into one customer-facing timeline."""

    def __init__(self, config: dict[str, Any] | None = None, *, paths: AuditPaths | None = None) -> None:
        self._config = config or {}
        self._paths = paths or self._paths_from_config(self._config)

    def query(
        self,
        *,
        limit: int = 100,
        source: str = "",
        operator_id: str = "",
        action: str = "",
        outcome: str = "",
        q: str = "",
    ) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 100), 500))
        records = self._records()
        filtered = [
            record
            for record in records
            if _matches(record, source=source, operator_id=operator_id, action=action, outcome=outcome, q=q)
        ]
        filtered.sort(key=lambda item: item.get("sort_at") or 0.0, reverse=True)
        visible = [_public_record(item) for item in filtered[:safe_limit]]
        return {
            "records": visible,
            "count": len(visible),
            "total": len(records),
            "filtered_total": len(filtered),
            "limit": safe_limit,
            "filters": {
                "source": source,
                "operator_id": operator_id,
                "action": action,
                "outcome": outcome,
                "q": q,
            },
            "summary": _summary(filtered),
            "sources": {
                "skill_audit": str(self._paths.skill_audit),
                "field_action_audit": str(self._paths.field_action_audit or ""),
                "field_event_archive": str(self._paths.field_event_archive or ""),
                "runtime_audit": str(self._paths.runtime_audit or ""),
            },
        }

    def _records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        records.extend(self._skill_records())
        records.extend(self._field_records())
        records.extend(self._runtime_records())
        return records

    def _skill_records(self) -> list[dict[str, Any]]:
        log = SkillAuditLog(self._paths.skill_audit)
        return [_normalize_skill(item) for item in log.recent(limit=500)]

    def _field_records(self) -> list[dict[str, Any]]:
        if self._paths.field_action_audit and self._paths.field_action_audit.is_file():
            return [
                _normalize_field_action(item)
                for item in _read_jsonl(self._paths.field_action_audit, limit=1000)
            ]
        if self._paths.field_event_archive and self._paths.field_event_archive.is_file():
            return _field_records_from_archive(self._paths.field_event_archive)
        return []

    def _runtime_records(self) -> list[dict[str, Any]]:
        path = self._paths.runtime_audit
        if not path or not path.is_file():
            return []
        return [_normalize_runtime(item) for item in _read_jsonl(path, limit=1000)]

    @staticmethod
    def _paths_from_config(config: dict[str, Any]) -> AuditPaths:
        field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
        field_audit_cfg = field_cfg.get("action_audit") if isinstance(field_cfg.get("action_audit"), dict) else {}
        runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        handoff_cfg = runtime_cfg.get("handoff") if isinstance(runtime_cfg.get("handoff"), dict) else {}
        runtime_audit_cfg = handoff_cfg.get("audit") if isinstance(handoff_cfg.get("audit"), dict) else {}
        skill_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
        return AuditPaths(
            skill_audit=Path(str(skill_cfg.get("audit_path") or default_skill_audit_path())),
            field_action_audit=_optional_path(field_audit_cfg.get("path") or field_cfg.get("action_audit_path")),
            field_event_archive=_optional_path(field_cfg.get("archive_path")),
            runtime_audit=_optional_path(
                runtime_audit_cfg.get("path")
                or runtime_audit_cfg.get("jsonl_path")
                or handoff_cfg.get("audit_log_path")
            ),
        )


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-max(1, limit):]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _normalize_skill(item: dict[str, Any]) -> dict[str, Any]:
    timestamp = str(item.get("timestamp") or "")
    return {
        "source": "skill",
        "category": str(item.get("event_type") or "skill"),
        "action": str(item.get("action") or item.get("skill_name") or ""),
        "outcome": str(item.get("status") or ""),
        "operator_id": str(item.get("operator_id") or ""),
        "subject": str(item.get("skill_name") or ""),
        "reason": str(item.get("reason") or ""),
        "message": str(item.get("result_preview") or item.get("user_text_preview") or ""),
        "timestamp": timestamp,
        "sort_at": _sort_time(timestamp),
        "raw": item,
    }


def _normalize_field_action(item: dict[str, Any]) -> dict[str, Any]:
    audit = item.get("audit") if isinstance(item.get("audit"), dict) else {}
    at = audit.get("at") or item.get("created_at")
    return {
        "source": "field",
        "category": str(item.get("kind") or "field_event_action"),
        "action": str(audit.get("action") or ""),
        "outcome": str(audit.get("outcome") or ""),
        "operator_id": str(audit.get("operator_id") or ""),
        "subject": str(item.get("event_id") or ""),
        "reason": str(audit.get("reason") or ""),
        "message": str(audit.get("note") or audit.get("authorization_reason") or ""),
        "timestamp": _iso_from_any(at),
        "sort_at": _sort_time(at),
        "raw": item,
    }


def _field_records_from_archive(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in _read_jsonl(path, limit=500):
        audits = event.get("action_audit") if isinstance(event.get("action_audit"), list) else []
        for audit in audits:
            if isinstance(audit, dict):
                records.append(_normalize_field_action({
                    "kind": "field_event_action",
                    "event_id": event.get("event_id"),
                    "created_at": audit.get("at"),
                    "audit": audit,
                }))
    return records


def _normalize_runtime(item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    kind = str(item.get("kind") or "runtime")
    return {
        "source": "runtime",
        "category": kind,
        "action": str(action.get("action") or event.get("event_type") or kind),
        "outcome": str(action.get("outcome") or event.get("status") or item.get("state") or ""),
        "operator_id": str(action.get("operator_id") or event.get("operator_id") or ""),
        "subject": str(item.get("run_id") or event.get("run_id") or item.get("handoff_id") or ""),
        "reason": str(action.get("reason") or event.get("reason") or ""),
        "message": str(event.get("message") or action.get("note") or ""),
        "timestamp": _iso_from_any(item.get("created_at")),
        "sort_at": _sort_time(item.get("created_at")),
        "raw": item,
    }


def _matches(
    record: dict[str, Any],
    *,
    source: str,
    operator_id: str,
    action: str,
    outcome: str,
    q: str,
) -> bool:
    if source and record.get("source") != source:
        return False
    if operator_id and record.get("operator_id") != operator_id:
        return False
    if action and record.get("action") != action:
        return False
    if outcome and record.get("outcome") != outcome:
        return False
    if q:
        haystack = json.dumps(_public_record(record), ensure_ascii=False)
        return q.lower() in haystack.lower()
    return True


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(str(item.get("source") or "unknown") for item in records)
    by_outcome = Counter(str(item.get("outcome") or "unknown") for item in records)
    by_operator = Counter(str(item.get("operator_id") or "unknown") for item in records)
    return {
        "by_source": dict(sorted(by_source.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "top_operators": dict(by_operator.most_common(8)),
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record.get("source") or "",
        "category": record.get("category") or "",
        "action": record.get("action") or "",
        "outcome": record.get("outcome") or "",
        "operator_id": record.get("operator_id") or "",
        "subject": record.get("subject") or "",
        "reason": record.get("reason") or "",
        "message": record.get("message") or "",
        "timestamp": record.get("timestamp") or "",
    }


def _sort_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _iso_from_any(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    text = str(value or "").strip()
    return text
