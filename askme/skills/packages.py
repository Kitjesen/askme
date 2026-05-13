"""Project-level skill packages for customer deployments.

Generated skills should not become global product behavior just because a
reviewer approved the prompt. A package records which approved skills are
enabled for a specific customer/site deployment so the same robot product can
grow online without turning every experiment into a production-wide ability.

Packages are release units: every package edit, skill assignment, rollout
change, and rollback creates an immutable history snapshot. This gives product
and delivery teams a customer-facing way to answer "what changed, who changed
it, and how do we roll back?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PACKAGE_ID = "default-demo"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_id(value: Any, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _clamp_percent(value: Any, *, default: int = 100) -> int:
    try:
        percent = int(value)
    except (TypeError, ValueError):
        percent = default
    return max(0, min(100, percent))


@dataclass(frozen=True)
class SkillPackageRecord:
    package_id: str
    display_name: str
    site_id: str = "demo"
    customer_name: str = ""
    description: str = ""
    enabled: bool = True
    skill_names: tuple[str, ...] = field(default_factory=tuple)
    release_version: int = 0
    release_channel: str = "draft"
    rollout_percent: int = 100
    last_published_at: str = ""
    last_rollback_at: str = ""
    rollback_of_version: int | None = None
    history_count: int = 0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    updated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "site_id": self.site_id,
            "customer_name": self.customer_name,
            "description": self.description,
            "enabled": self.enabled,
            "skill_names": list(self.skill_names),
            "release_version": self.release_version,
            "release_channel": self.release_channel,
            "rollout_percent": self.rollout_percent,
            "last_published_at": self.last_published_at,
            "last_rollback_at": self.last_rollback_at,
            "rollback_of_version": self.rollback_of_version,
            "history_count": self.history_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


class SkillPackageStore:
    """JSON-backed project/customer ability package store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_generated_dir(cls, generated_skills_dir: str | Path) -> SkillPackageStore:
        return cls(Path(generated_skills_dir).parent / "skill_packages.json")

    def payload(self) -> dict[str, Any]:
        records = self.list_packages()
        assigned = {
            skill_name
            for record in records
            if record.enabled and record.rollout_percent > 0
            for skill_name in record.skill_names
        }
        return {
            "packages": [record.to_dict() for record in records],
            "summary": {
                "package_count": len(records),
                "enabled_package_count": sum(1 for record in records if record.enabled),
                "assigned_skill_count": len(assigned),
                "gray_release_count": sum(
                    1 for record in records if 0 < record.rollout_percent < 100
                ),
                "rollback_ready_count": sum(1 for record in records if record.history_count > 1),
            },
            "policy": {
                "approved_generated_skills_require_package": True,
                "default_package_id": DEFAULT_PACKAGE_ID,
                "customer_scoped_enablement": True,
                "package_release_versioning": True,
                "gray_release_supported": True,
                "rollback_supported": True,
                "rollout_percent_zero_disables_package_skills": True,
            },
        }

    def list_packages(self) -> list[SkillPackageRecord]:
        document = self._with_default_document(self._load_document())
        data = document["packages"]
        return sorted(
            (
                self._record_from_dict(
                    item,
                    history_count=len(document["history"].get(str(item.get("package_id")), [])),
                )
                for item in data.values()
                if isinstance(item, dict)
            ),
            key=lambda item: (item.site_id, item.package_id),
        )

    def get(self, package_id: str) -> SkillPackageRecord:
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        payload = document["packages"].get(package_id, self._default_package())
        return self._record_from_dict(
            payload,
            history_count=len(document["history"].get(package_id, [])),
        )

    def history(self, package_id: str, *, limit: int = 20) -> dict[str, Any]:
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        records = list(document["history"].get(package_id, []))
        records = records[-max(1, int(limit)):]
        return {
            "package_id": package_id,
            "records": records,
            "count": len(records),
            "current": self._record_from_dict(
                document["packages"].get(package_id, self._default_package()),
                history_count=len(document["history"].get(package_id, [])),
            ).to_dict(),
        }

    def upsert_package(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        display_name: str = "",
        site_id: str = "demo",
        customer_name: str = "",
        description: str = "",
        enabled: bool = True,
        release_channel: str = "",
        rollout_percent: int | None = None,
        operator_id: str = "",
    ) -> SkillPackageRecord:
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        existing = document["packages"].get(package_id) if isinstance(document["packages"].get(package_id), dict) else {}
        payload = {
            **existing,
            "package_id": package_id,
            "display_name": display_name or existing.get("display_name") or _default_display_name(package_id),
            "site_id": site_id or existing.get("site_id") or "demo",
            "customer_name": customer_name or existing.get("customer_name") or "",
            "description": description or existing.get("description") or "",
            "enabled": enabled,
            "skill_names": self._list(existing.get("skill_names")),
            "release_channel": _clean_id(release_channel, default=str(existing.get("release_channel") or "draft")),
            "rollout_percent": _clamp_percent(
                rollout_percent if rollout_percent is not None else existing.get("rollout_percent"),
                default=100,
            ),
            "created_at": existing.get("created_at") or _utc_now(),
        }
        return self._save_package_update(
            document,
            package_id,
            payload,
            action="upsert_package",
            operator_id=operator_id,
        )

    def update_release(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        release_channel: str = "pilot",
        rollout_percent: int = 100,
        operator_id: str = "",
        note: str = "",
    ) -> SkillPackageRecord:
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        existing = document["packages"].get(package_id)
        if not isinstance(existing, dict):
            existing = self._default_package(package_id=package_id)
        payload = {
            **existing,
            "release_channel": _clean_id(release_channel, default="pilot"),
            "rollout_percent": _clamp_percent(rollout_percent),
            "last_published_at": _utc_now(),
        }
        return self._save_package_update(
            document,
            package_id,
            payload,
            action="release_package",
            operator_id=operator_id,
            note=note,
        )

    def rollback_package(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        target_version: int | None = None,
        operator_id: str = "",
        note: str = "",
    ) -> SkillPackageRecord:
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        history = list(document["history"].get(package_id, []))
        if not history:
            raise ValueError(f"no package history for {package_id}")
        current = document["packages"].get(package_id, {})
        current_version = int(current.get("release_version") or 0)
        target = self._rollback_target(history, target_version, current_version)
        if target is None:
            raise ValueError(f"rollback target not found for {package_id}")
        snapshot = target.get("package")
        if not isinstance(snapshot, dict):
            raise ValueError(f"rollback snapshot is invalid for {package_id}")
        now = _utc_now()
        payload = {
            **snapshot,
            "package_id": package_id,
            "created_at": current.get("created_at") or snapshot.get("created_at") or now,
            "last_rollback_at": now,
            "rollback_of_version": int(snapshot.get("release_version") or 0),
        }
        return self._save_package_update(
            document,
            package_id,
            payload,
            action="rollback_package",
            operator_id=operator_id,
            note=note,
        )

    def enabled_package_ids_for_skill(self, skill_name: str) -> list[str]:
        clean = _clean_id(skill_name)
        if not clean:
            return []
        return [
            record.package_id
            for record in self.list_packages()
            if record.enabled and record.rollout_percent > 0 and clean in record.skill_names
        ]

    def is_skill_enabled(self, skill_name: str) -> bool:
        return bool(self.enabled_package_ids_for_skill(skill_name))

    def assign_skill(
        self,
        skill_name: str,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        operator_id: str = "",
        site_id: str = "demo",
        display_name: str = "",
        customer_name: str = "",
        description: str = "",
    ) -> SkillPackageRecord:
        clean_skill = _clean_id(skill_name)
        if not clean_skill:
            raise ValueError("skill_name is required")
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        existing = document["packages"].get(package_id) if isinstance(document["packages"].get(package_id), dict) else {}
        skills = _unique([*self._list(existing.get("skill_names")), clean_skill])
        payload = {
            **existing,
            "package_id": package_id,
            "display_name": display_name or existing.get("display_name") or _default_display_name(package_id),
            "site_id": site_id or existing.get("site_id") or "demo",
            "customer_name": customer_name or existing.get("customer_name") or "",
            "description": description or existing.get("description") or "",
            "enabled": bool(existing.get("enabled", True)),
            "skill_names": skills,
            "release_channel": existing.get("release_channel") or "draft",
            "rollout_percent": _clamp_percent(existing.get("rollout_percent"), default=100),
            "created_at": existing.get("created_at") or _utc_now(),
        }
        return self._save_package_update(
            document,
            package_id,
            payload,
            action="assign_skill",
            operator_id=operator_id,
        )

    def unassign_skill(
        self,
        skill_name: str,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        operator_id: str = "",
    ) -> SkillPackageRecord:
        clean_skill = _clean_id(skill_name)
        package_id = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
        document = self._with_default_document(self._load_document())
        existing = document["packages"].get(package_id) if isinstance(document["packages"].get(package_id), dict) else self._default_package()
        skills = [item for item in self._list(existing.get("skill_names")) if item != clean_skill]
        payload = {
            **existing,
            "package_id": package_id,
            "skill_names": skills,
        }
        return self._save_package_update(
            document,
            package_id,
            payload,
            action="unassign_skill",
            operator_id=operator_id,
        )

    def _load_document(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {"packages": {}, "history": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"packages": {}, "history": {}}
        if not isinstance(payload, dict):
            return {"packages": {}, "history": {}}
        packages = payload.get("packages")
        history = payload.get("history")
        return {
            "packages": packages if isinstance(packages, dict) else {},
            "history": history if isinstance(history, dict) else {},
        }

    def _save_document(self, document: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _with_default_document(self, document: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        packages = document.setdefault("packages", {})
        document.setdefault("history", {})
        if DEFAULT_PACKAGE_ID not in packages:
            packages[DEFAULT_PACKAGE_ID] = self._default_package()
        return document

    def _save_package_update(
        self,
        document: dict[str, dict[str, Any]],
        package_id: str,
        payload: dict[str, Any],
        *,
        action: str,
        operator_id: str = "",
        note: str = "",
    ) -> SkillPackageRecord:
        packages = document.setdefault("packages", {})
        history = document.setdefault("history", {})
        existing = packages.get(package_id) if isinstance(packages.get(package_id), dict) else {}
        now = _utc_now()
        release_version = int(existing.get("release_version") or 0) + 1
        payload = {
            **payload,
            "package_id": package_id,
            "release_version": release_version,
            "rollout_percent": _clamp_percent(payload.get("rollout_percent"), default=100),
            "release_channel": _clean_id(payload.get("release_channel"), default="draft"),
            "created_at": payload.get("created_at") or existing.get("created_at") or now,
            "updated_at": now,
            "updated_by": operator_id,
        }
        packages[package_id] = payload
        history.setdefault(package_id, [])
        history[package_id].append(
            {
                "version": release_version,
                "action": action,
                "operator_id": operator_id,
                "timestamp": now,
                "note": note,
                "changed_fields": _changed_fields(existing, payload),
                "package": dict(payload),
            }
        )
        self._save_document(document)
        return self._record_from_dict(payload, history_count=len(history[package_id]))

    @staticmethod
    def _rollback_target(
        history: list[dict[str, Any]],
        target_version: int | None,
        current_version: int,
    ) -> dict[str, Any] | None:
        if target_version is not None:
            for item in reversed(history):
                if int(item.get("version") or 0) == int(target_version):
                    return item
            return None
        for item in reversed(history):
            version = int(item.get("version") or 0)
            if version < current_version:
                return item
        return None

    @staticmethod
    def _default_package(*, package_id: str = DEFAULT_PACKAGE_ID) -> dict[str, Any]:
        now = _utc_now()
        return {
            "package_id": package_id,
            "display_name": _default_display_name(package_id),
            "site_id": "demo",
            "customer_name": "",
            "description": "Default package for local demo and lab validation.",
            "enabled": True,
            "skill_names": [],
            "release_version": 0,
            "release_channel": "draft",
            "rollout_percent": 100,
            "last_published_at": "",
            "last_rollback_at": "",
            "rollback_of_version": None,
            "created_at": now,
            "updated_at": now,
            "updated_by": "",
        }

    @staticmethod
    def _record_from_dict(payload: dict[str, Any], *, history_count: int = 0) -> SkillPackageRecord:
        now = _utc_now()
        rollback_raw = payload.get("rollback_of_version")
        rollback_of_version: int | None
        try:
            rollback_of_version = int(rollback_raw) if rollback_raw not in (None, "") else None
        except (TypeError, ValueError):
            rollback_of_version = None
        return SkillPackageRecord(
            package_id=str(payload.get("package_id") or DEFAULT_PACKAGE_ID),
            display_name=str(payload.get("display_name") or _default_display_name(payload.get("package_id"))),
            site_id=str(payload.get("site_id") or "demo"),
            customer_name=str(payload.get("customer_name") or ""),
            description=str(payload.get("description") or ""),
            enabled=bool(payload.get("enabled", True)),
            skill_names=tuple(_unique(SkillPackageStore._list(payload.get("skill_names")))),
            release_version=int(payload.get("release_version") or 0),
            release_channel=str(payload.get("release_channel") or "draft"),
            rollout_percent=_clamp_percent(payload.get("rollout_percent"), default=100),
            last_published_at=str(payload.get("last_published_at") or ""),
            last_rollback_at=str(payload.get("last_rollback_at") or ""),
            rollback_of_version=rollback_of_version,
            history_count=history_count,
            created_at=str(payload.get("created_at") or now),
            updated_at=str(payload.get("updated_at") or now),
            updated_by=str(payload.get("updated_by") or ""),
        )

    @staticmethod
    def _list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _default_display_name(package_id: Any) -> str:
    clean = _clean_id(package_id, default=DEFAULT_PACKAGE_ID)
    return "Demo customer ability package" if clean == DEFAULT_PACKAGE_ID else clean


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    fields = sorted(set(before) | set(after))
    hidden = {"updated_at", "updated_by", "release_version"}
    changes: list[dict[str, Any]] = []
    for field_name in fields:
        if field_name in hidden:
            continue
        before_value = before.get(field_name)
        after_value = after.get(field_name)
        if before_value == after_value:
            continue
        changes.append(
            {
                "field": field_name,
                "before": before_value,
                "after": after_value,
            }
        )
    return changes
