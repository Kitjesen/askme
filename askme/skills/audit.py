"""Append-only audit records for skill execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_skill_audit_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "skill_audit.jsonl"


def _audit_text(value: Any, *, limit: int | None = None) -> str:
    text = str(value or "")
    return text[:limit] if limit else text


class SkillAuditLog:
    """Best-effort JSONL audit log for capability execution."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_skill_audit_path()

    def append(
        self,
        *,
        skill_name: str,
        status: str,
        event_type: str = "execution",
        user_text: str = "",
        source: str = "voice",
        safety_level: str = "",
        execution: str = "",
        operator_id: str = "",
        action: str = "",
        elapsed_ms: float | None = None,
        reason: str = "",
        result_preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": _audit_text(event_type),
            "skill_name": _audit_text(skill_name),
            "status": _audit_text(status),
            "source": _audit_text(source),
            "safety_level": _audit_text(safety_level),
            "execution": _audit_text(execution),
            "user_text_preview": _audit_text(user_text, limit=160),
        }
        if operator_id:
            payload["operator_id"] = _audit_text(operator_id)
        if action:
            payload["action"] = _audit_text(action)
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(float(elapsed_ms), 3)
        if reason:
            payload["reason"] = _audit_text(reason, limit=240)
        if result_preview:
            payload["result_preview"] = _audit_text(result_preview, limit=160)
        if metadata:
            payload["metadata"] = {
                _audit_text(key, limit=80): _audit_text(value, limit=240)
                for key, value in metadata.items()
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except (OSError, TypeError):
            return

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
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
