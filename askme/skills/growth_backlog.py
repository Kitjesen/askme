"""Online skill-growth backlog derived from real interaction evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit import SkillAuditLog

_DEFAULT_MIN_OCCURRENCES = 2
_MAX_EXAMPLES = 5


def default_skill_growth_state_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "skill_growth_backlog.json"


@dataclass(frozen=True)
class SkillGrowthCandidate:
    candidate_id: str
    summary: str
    suggested_skill_name: str
    suggested_voice_trigger: str
    evidence_count: int
    status: str = "candidate"
    priority: str = "P2"
    risk_level: str = "normal"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    current_skill_names: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)
    first_seen_at: str = ""
    last_seen_at: str = ""
    updated_by: str = ""
    updated_at: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "summary": self.summary,
            "suggested_skill_name": self.suggested_skill_name,
            "suggested_voice_trigger": self.suggested_voice_trigger,
            "evidence_count": self.evidence_count,
            "status": self.status,
            "priority": self.priority,
            "risk_level": self.risk_level,
            "reasons": list(self.reasons),
            "current_skill_names": list(self.current_skill_names),
            "examples": list(self.examples),
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
            "note": self.note,
        }


class SkillGrowthBacklog:
    """Build reviewable skill-growth candidates from audit evidence."""

    def __init__(
        self,
        audit_log: SkillAuditLog | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self._audit = audit_log or SkillAuditLog()
        self._state_path = Path(state_path) if state_path is not None else default_skill_growth_state_path()

    def payload(
        self,
        *,
        min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
        limit: int = 20,
    ) -> dict[str, Any]:
        candidates = self.list_candidates(min_occurrences=min_occurrences, limit=limit)
        return {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "summary": {
                "candidate_count": len(candidates),
                "open_count": sum(1 for item in candidates if item.status == "candidate"),
                "promoted_count": sum(1 for item in candidates if item.status == "promoted"),
                "dismissed_count": sum(1 for item in candidates if item.status == "dismissed"),
                "min_occurrences": max(1, int(min_occurrences)),
            },
            "policy": {
                "source": "skill_audit_log",
                "promotion_requires_generated_skill_review": True,
                "auto_create_or_enable_skills": False,
                "human_product_owner_required": True,
            },
        }

    def list_candidates(
        self,
        *,
        min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
        limit: int = 20,
    ) -> list[SkillGrowthCandidate]:
        min_occurrences = max(1, int(min_occurrences))
        state = self._load_state()
        buckets: dict[str, list[dict[str, Any]]] = {}
        for record in self._audit.recent(limit=500):
            if not _growth_signal(record):
                continue
            text = _clean_text(record.get("user_text_preview"))
            if not text:
                continue
            key = _bucket_key(text)
            buckets.setdefault(key, []).append(record)

        candidates: list[SkillGrowthCandidate] = []
        for key, records in buckets.items():
            if len(records) < min_occurrences:
                continue
            candidate_id = _candidate_id(key)
            overlay = state.get(candidate_id, {})
            examples = _unique(
                _clean_text(record.get("user_text_preview"))
                for record in records
                if _clean_text(record.get("user_text_preview"))
            )[:_MAX_EXAMPLES]
            reasons = _unique(
                _clean_text(record.get("reason") or record.get("status"))
                for record in records
                if _clean_text(record.get("reason") or record.get("status"))
            )
            skill_names = _unique(
                _clean_text(record.get("skill_name"))
                for record in records
                if _clean_text(record.get("skill_name"))
            )
            timestamps = sorted(
                _clean_text(record.get("timestamp"))
                for record in records
                if _clean_text(record.get("timestamp"))
            )
            summary = _summary_from_examples(examples)
            candidates.append(
                SkillGrowthCandidate(
                    candidate_id=candidate_id,
                    summary=summary,
                    suggested_skill_name=_suggest_skill_name(summary, candidate_id),
                    suggested_voice_trigger=examples[0] if examples else summary,
                    evidence_count=len(records),
                    status=_clean_text(overlay.get("status")) or "candidate",
                    priority=_priority(len(records), reasons),
                    risk_level=_risk_level(examples, reasons),
                    reasons=tuple(reasons),
                    current_skill_names=tuple(skill_names),
                    examples=tuple(examples),
                    first_seen_at=timestamps[0] if timestamps else "",
                    last_seen_at=timestamps[-1] if timestamps else "",
                    updated_by=_clean_text(overlay.get("updated_by")),
                    updated_at=_clean_text(overlay.get("updated_at")),
                    note=_clean_text(overlay.get("note")),
                )
            )
        return sorted(
            candidates,
            key=lambda item: (
                item.status != "candidate",
                _priority_rank(item.priority),
                -item.evidence_count,
                item.last_seen_at,
            ),
        )[: max(1, int(limit))]

    def get_candidate(
        self,
        candidate_id: str,
        *,
        min_occurrences: int = 1,
    ) -> SkillGrowthCandidate | None:
        target = _clean_text(candidate_id)
        if not target:
            return None
        for candidate in self.list_candidates(min_occurrences=min_occurrences, limit=500):
            if candidate.candidate_id == target:
                return candidate
        return None

    def mark(
        self,
        candidate_id: str,
        *,
        action: str,
        operator_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        candidate_id = _clean_text(candidate_id)
        if not candidate_id:
            return {"ok": False, "error": "candidate_id is required"}
        status = {
            "promote": "promoted",
            "dismiss": "dismissed",
            "reopen": "candidate",
            "observe": "candidate",
        }.get(_clean_text(action))
        if status is None:
            return {"ok": False, "error": f"unsupported action: {action}"}
        state = self._load_state()
        state[candidate_id] = {
            **state.get(candidate_id, {}),
            "status": status,
            "updated_by": _clean_text(operator_id),
            "updated_at": datetime.now(UTC).isoformat(),
            "note": _clean_text(note),
        }
        self._save_state(state)
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "status": status,
            "backlog": self.payload(),
        }

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self._state_path.is_file():
            return {}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        items = payload.get("candidates") if isinstance(payload, dict) else None
        return items if isinstance(items, dict) else {}

    def _save_state(self, state: dict[str, dict[str, Any]]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps({"candidates": state}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _growth_signal(record: dict[str, Any]) -> bool:
    event_type = _clean_text(record.get("event_type"))
    status = _clean_text(record.get("status"))
    reason = _clean_text(record.get("reason"))
    if event_type == "governance":
        return False
    if status in {"failed", "blocked"}:
        return reason not in {"estop_active", "cancel_token"}
    return reason in {"not_found", "disabled", "unsupported", "no_route", "no_skill"}


def _bucket_key(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.lower())
    normalized = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]]", "", normalized)
    return normalized[:80]


def _candidate_id(key: str) -> str:
    return "grow_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _summary_from_examples(examples: list[str]) -> str:
    if not examples:
        return "未命名增长候选"
    text = examples[0].strip()
    return text[:48] if len(text) > 48 else text


def _suggest_skill_name(summary: str, candidate_id: str) -> str:
    ascii_slug = re.sub(r"[^a-z0-9_]+", "_", summary.lower()).strip("_")
    ascii_slug = re.sub(r"_+", "_", ascii_slug)
    if ascii_slug and not ascii_slug[0].isdigit():
        return ascii_slug[:48]
    return f"skill_{candidate_id[-8:]}"


def _priority(count: int, reasons: list[str]) -> str:
    reason_set = set(reasons)
    if count >= 5 or reason_set & {"not_found", "no_skill", "no_route"}:
        return "P1"
    return "P2"


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def _risk_level(examples: list[str], reasons: list[str]) -> str:
    text = " ".join([*examples, *reasons])
    high_risk_tokens = ("急停", "停止", "移动", "导航", "带路", "开门", "抓取", "电机", "机械臂")
    return "dangerous" if any(token in text for token in high_risk_tokens) else "normal"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
