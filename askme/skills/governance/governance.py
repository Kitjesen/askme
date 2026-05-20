"""Governance records for generated skills.

Generated skills are product features, not temporary chat output.  This store
keeps their review state separate from the SKILL.md body so reviewers can
approve, reject, disable, and audit them without editing prompt files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.skills.core.skill_model import SkillDefinition

PENDING = "pending_approval"
APPROVED = "approved"
REJECTED = "rejected"
DISABLED = "disabled"
_ENABLED_STATES = {APPROVED}
_KNOWN_STATES = {PENDING, APPROVED, REJECTED, DISABLED}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SkillGovernanceRecord:
    skill_name: str
    status: str
    created_at: str
    updated_at: str
    created_by: str = "agent"
    reviewed_by: str = ""
    review_note: str = ""
    path: str = ""
    description: str = ""
    voice_trigger: str = ""
    safety_level: str = "normal"
    execution: str = "skill_executor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "reviewed_by": self.reviewed_by,
            "review_note": self.review_note,
            "path": self.path,
            "description": self.description,
            "voice_trigger": self.voice_trigger,
            "safety_level": self.safety_level,
            "execution": self.execution,
        }


class SkillGovernanceStore:
    """JSON-backed generated-skill review store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_generated_dir(cls, generated_skills_dir: str | Path) -> SkillGovernanceStore:
        return cls(Path(generated_skills_dir).parent / "skill_governance.json")

    def get(self, skill_name: str) -> SkillGovernanceRecord | None:
        raw = self._load().get(skill_name)
        if not isinstance(raw, dict):
            return None
        return self._record_from_dict(raw)

    def ensure_pending(
        self,
        skill: SkillDefinition,
        *,
        created_by: str = "agent",
    ) -> SkillGovernanceRecord:
        data = self._load()
        existing = data.get(skill.name) if isinstance(data.get(skill.name), dict) else {}
        now = _utc_now()
        status = str(existing.get("status") or PENDING)
        if status not in _KNOWN_STATES:
            status = PENDING
        payload = {
            **existing,
            "skill_name": skill.name,
            "status": status,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "created_by": existing.get("created_by") or created_by,
            "path": skill.path,
            "description": skill.description,
            "voice_trigger": skill.voice_trigger or "",
            "safety_level": skill.safety_level,
            "execution": skill.execution,
        }
        data[skill.name] = payload
        self._save(data)
        return self._record_from_dict(payload)

    def set_status(
        self,
        skill: SkillDefinition,
        *,
        status: str,
        reviewed_by: str,
        review_note: str = "",
    ) -> SkillGovernanceRecord:
        if status not in _KNOWN_STATES:
            raise ValueError(f"unknown skill governance status: {status}")
        data = self._load()
        existing = data.get(skill.name) if isinstance(data.get(skill.name), dict) else {}
        now = _utc_now()
        payload = {
            **existing,
            "skill_name": skill.name,
            "status": status,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "created_by": existing.get("created_by") or "agent",
            "reviewed_by": reviewed_by,
            "review_note": review_note,
            "path": skill.path,
            "description": skill.description,
            "voice_trigger": skill.voice_trigger or "",
            "safety_level": skill.safety_level,
            "execution": skill.execution,
        }
        data[skill.name] = payload
        self._save(data)
        return self._record_from_dict(payload)

    def list_records(self) -> list[SkillGovernanceRecord]:
        records: list[SkillGovernanceRecord] = []
        for raw in self._load().values():
            if isinstance(raw, dict):
                records.append(self._record_from_dict(raw))
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    @staticmethod
    def is_enabled_status(status: str) -> bool:
        return status in _ENABLED_STATES

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        records = payload.get("records") if isinstance(payload, dict) else None
        return records if isinstance(records, dict) else {}

    def _save(self, records: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": records}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _record_from_dict(payload: dict[str, Any]) -> SkillGovernanceRecord:
        now = _utc_now()
        return SkillGovernanceRecord(
            skill_name=str(payload.get("skill_name") or ""),
            status=str(payload.get("status") or PENDING),
            created_at=str(payload.get("created_at") or now),
            updated_at=str(payload.get("updated_at") or now),
            created_by=str(payload.get("created_by") or "agent"),
            reviewed_by=str(payload.get("reviewed_by") or ""),
            review_note=str(payload.get("review_note") or ""),
            path=str(payload.get("path") or ""),
            description=str(payload.get("description") or ""),
            voice_trigger=str(payload.get("voice_trigger") or ""),
            safety_level=str(payload.get("safety_level") or "normal"),
            execution=str(payload.get("execution") or "skill_executor"),
        )
