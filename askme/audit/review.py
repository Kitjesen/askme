"""Append-only review decisions for unified audit records."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

_CLEARING_DECISIONS = {"accepted", "resolved", "waived", "false_positive"}
_ACTIVE_DECISIONS = _CLEARING_DECISIONS | {"escalated", "rejected"}
_GENESIS_HASH = "GENESIS"


class AuditReviewService:
    """Persist supervisor review decisions without mutating audit sources."""

    def __init__(self, config: dict[str, Any] | None = None, *, path: str | Path | None = None) -> None:
        self._config = config or {}
        self._path = Path(str(path or _review_path_from_config(self._config)))

    @property
    def path(self) -> Path:
        return self._path

    def submit(
        self,
        *,
        record_id: str,
        reviewer_id: str,
        decision: str,
        note: str = "",
        created_at: float | None = None,
    ) -> dict[str, Any]:
        safe_record_id = str(record_id or "").strip()
        safe_reviewer_id = str(reviewer_id or "").strip()
        safe_decision = str(decision or "").strip().lower()
        if not safe_record_id:
            return {"ok": False, "reason": "record_id_required"}
        if not safe_reviewer_id:
            return {"ok": False, "reason": "reviewer_id_required"}
        if safe_decision not in _ACTIVE_DECISIONS:
            return {
                "ok": False,
                "reason": "invalid_review_decision",
                "allowed_decisions": sorted(_ACTIVE_DECISIONS),
            }

        previous = self._latest_checkpoint()
        sequence = int(previous.get("sequence") or 0) + 1
        record = {
            "kind": "audit_review_decision",
            "sequence": sequence,
            "record_id": safe_record_id,
            "reviewer_id": safe_reviewer_id,
            "decision": safe_decision,
            "note": str(note or "").strip(),
            "created_at": float(created_at if created_at is not None else time.time()),
            "previous_hash": str(previous.get("record_hash") or _GENESIS_HASH),
            "hash_alg": "sha256",
        }
        record["record_hash"] = _record_hash(record)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return {"ok": True, "record": _public_review(record), "path": str(self._path)}

    def latest(self) -> dict[str, dict[str, Any]]:
        decisions: dict[str, dict[str, Any]] = {}
        for record in _read_reviews(self._path):
            record_id = str(record.get("record_id") or "")
            if record_id:
                decisions[record_id] = _public_review(record)
        return decisions

    def list(self, *, limit: int = 100) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 100), 500))
        records = [_public_review(item) for item in _read_reviews(self._path)]
        visible = records[-safe_limit:]
        visible.reverse()
        return {
            "records": visible,
            "count": len(visible),
            "total": len(records),
            "path": str(self._path),
            "integrity": self.integrity(),
        }

    def integrity(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {
                "valid": True,
                "exists": False,
                "checked_count": 0,
                "failures": [],
                "path": str(self._path),
            }
        failures: list[dict[str, Any]] = []
        previous_hash = _GENESIS_HASH
        expected_sequence = 1
        checked = 0
        try:
            with self._path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        failures.append({
                            "line": line_number,
                            "reason": "invalid_json",
                            "detail": str(exc),
                        })
                        continue
                    if not isinstance(record, dict):
                        failures.append({"line": line_number, "reason": "invalid_record"})
                        continue
                    checked += 1
                    if int(record.get("sequence") or 0) != expected_sequence:
                        failures.append({"line": line_number, "reason": "sequence_mismatch"})
                    if str(record.get("previous_hash") or "") != previous_hash:
                        failures.append({"line": line_number, "reason": "previous_hash_mismatch"})
                    if str(record.get("record_hash") or "") != _record_hash(record):
                        failures.append({"line": line_number, "reason": "record_hash_mismatch"})
                    previous_hash = str(record.get("record_hash") or "")
                    expected_sequence += 1
        except OSError as exc:
            failures.append({"line": 0, "reason": "read_failed", "detail": str(exc)})
        return {
            "valid": not failures,
            "exists": True,
            "checked_count": checked,
            "failures": failures,
            "path": str(self._path),
            "latest_hash": previous_hash if checked else _GENESIS_HASH,
        }

    def _latest_checkpoint(self) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for record in _read_reviews(self._path):
            latest = record
        return latest


def decision_clears_review(decision: str) -> bool:
    return str(decision or "").strip().lower() in _CLEARING_DECISIONS


def _review_path_from_config(config: dict[str, Any]) -> Path:
    audit_cfg = config.get("audit") if isinstance(config.get("audit"), dict) else {}
    review_cfg = audit_cfg.get("review") if isinstance(audit_cfg.get("review"), dict) else {}
    raw = (
        review_cfg.get("path")
        or review_cfg.get("jsonl_path")
        or audit_cfg.get("review_path")
        or "artifacts/audit/reviews.jsonl"
    )
    return Path(str(raw))


def _read_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        return []
    return records


def _record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _public_review(record: dict[str, Any]) -> dict[str, Any]:
    decision = str(record.get("decision") or "")
    return {
        "record_id": str(record.get("record_id") or ""),
        "reviewer_id": str(record.get("reviewer_id") or ""),
        "decision": decision,
        "decision_label": _decision_label(decision),
        "clears_review": decision_clears_review(decision),
        "customer_effect": _customer_effect(decision),
        "next_step": _decision_next_step(decision),
        "note": str(record.get("note") or ""),
        "created_at": record.get("created_at"),
        "sequence": record.get("sequence"),
        "record_hash": str(record.get("record_hash") or ""),
        "previous_hash": str(record.get("previous_hash") or ""),
        "hash_alg": str(record.get("hash_alg") or ""),
    }


def _decision_label(decision: str) -> str:
    return {
        "accepted": "已复核通过",
        "resolved": "问题已处理",
        "waived": "复核豁免",
        "false_positive": "确认误报",
        "escalated": "升级处理中",
        "rejected": "复核拒绝",
    }.get(str(decision or "").strip().lower(), decision or "未记录")


def _customer_effect(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if decision_clears_review(normalized):
        return "该复核决定解除交付阻断，记录可进入客户验收审计包。"
    if normalized == "escalated":
        return "该记录仍需继续处理，暂不能进入客户验收审计包。"
    if normalized == "rejected":
        return "该记录复核未通过，继续阻断客户验收审计包。"
    return "该复核决定已记录，交付影响待确认。"


def _decision_next_step(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if decision_clears_review(normalized):
        return "重新生成审计包并随验收材料归档。"
    if normalized == "escalated":
        return "补齐处理结论后再次复核。"
    if normalized == "rejected":
        return "重新处理现场或运行问题后再提交复核。"
    return "由主管补充复核结论。"
