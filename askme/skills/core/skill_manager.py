"""
Skill manager for askme.

Discovers SKILL.md files from three locations, parses YAML frontmatter,
manages enable/disable state, and builds prompts with template variables.

Skill search locations (later overrides earlier):
  1. Built-in:  askme/skills/builtin/<name>/SKILL.md
  2. User:      ~/.askme/skills/<name>/SKILL.md
  3. Project:   <cwd>/skills/<name>/SKILL.md
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from askme.config import project_root
from askme.skills.contracts import contracts_builtin as _contracts_builtin  # noqa: F401
from askme.skills.contracts.contracts import (
    SkillContract,
    build_skills_openapi,
    registered_skill_contracts,
)
from askme.skills.core.skill_model import SkillDefinition
from askme.skills.core.validation import validate_generated_skill
from askme.skills.governance.audit import SkillAuditLog
from askme.skills.governance.governance import (
    APPROVED,
    DISABLED,
    PENDING,
    REJECTED,
    SkillGovernanceStore,
)
from askme.skills.governance.packages import DEFAULT_PACKAGE_ID, SkillPackageStore

logger = logging.getLogger(__name__)

# Default data dir: askme project root / data
_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # askme/skills/
_PROJECT_ROOT = project_root()
_DATA_DIR = _PROJECT_ROOT / "data"
_SETTINGS_FILE = _DATA_DIR / "skills_settings.json"
_GENERATED_SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [{tags}]
depends: []
conflicts: []
safety_level: {safety_level}
voice_trigger: {voice_trigger}
confirm_before_execute: {confirm_before_execute}
---

## Tools

{tools_section}

## Prompt

{prompt}
"""


class SkillManager:
    """Discover, load, and manage SKILL.md-based skills."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        self._project_dir = Path(project_dir) if project_dir else Path.cwd()
        self._skills: dict[str, SkillDefinition] = {}
        self._disabled: set[str] = set()

    # ── Public API ──────────────────────────────────────────────

    def load(self) -> None:
        """Discover and load all skills from all locations."""
        self._skills.clear()
        self._load_settings()

        locations: list[tuple[Path, str]] = [
            (_PACKAGE_DIR / "builtin", "builtin"),
            (Path.home() / ".askme" / "skills", "user"),
            (self._project_dir / "skills", "project"),
            (self.generated_skills_dir, "generated"),  # LLM-created skills
        ]

        for directory, source in locations:
            self._discover_from(directory, source)

        # Apply persisted enabled/disabled state
        for name in self._disabled:
            if name in self._skills:
                self._skills[name].enabled = False
        self._apply_generated_skill_governance()

        logger.info("Loaded %d skills", len(self._skills))

    def hot_reload(self, router: Any | None = None) -> int:
        """Reload all skills from disk and optionally update the router's voice triggers.

        Args:
            router: An IntentRouter instance with update_voice_triggers().
                    If provided, the router's trigger map is refreshed immediately.

        Returns:
            Number of skills loaded after reload.
        """
        self.load()
        if router is not None:
            router.update_voice_triggers(self.get_voice_triggers())
            logger.info("Hot-reload: router voice triggers updated (%d triggers)",
                        len(self.get_voice_triggers()))
        return len(self._skills)

    @property
    def generated_skills_dir(self) -> Path:
        """Directory where LLM-created skills are stored."""
        return _DATA_DIR / "skills"

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all(self) -> list[SkillDefinition]:
        """Get all skills (including disabled ones)."""
        return list(self._skills.values())

    def get_enabled(self) -> list[SkillDefinition]:
        """Get only enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

    def get_contract(self, name: str) -> SkillContract | None:
        """Return the authoritative contract for a loaded skill."""
        skill = self._skills.get(name)
        if skill is None:
            return None

        explicit = registered_skill_contracts().get(name)
        if explicit is None:
            return skill.to_contract()

        return explicit.with_fallbacks(
            description=skill.description,
            version=skill.version,
            safety_level=skill.safety_level,
            execution=skill.execution,
            tags=skill.tags,
            confirm_before_execute=skill.confirm_before_execute,
        )

    def get_contracts(self) -> list[SkillContract]:
        """Return contracts for all loaded skills."""
        contracts: list[SkillContract] = []
        for skill in self.get_all():
            contract = self.get_contract(skill.name)
            if contract is not None:
                contracts.append(contract)
        return contracts

    def get_contract_catalog(self) -> list[dict[str, Any]]:
        """Return a capability-friendly catalog generated from contracts."""
        entries: list[dict[str, Any]] = []
        for skill in self.get_all():
            contract = self.get_contract(skill.name)
            base = contract.summary() if contract is not None else {
                "name": skill.name,
                "description": skill.description,
                "contract_source": "unavailable",
            }
            base.update({
                "enabled": skill.enabled,
                "trigger": skill.trigger,
                "voice_trigger": skill.voice_trigger,
                "legacy_source": skill.source,
            })
            entries.append(base)
        return entries

    def get_capability_center(self) -> dict[str, Any]:
        """Return customer-facing grouped capabilities for product UI."""
        from askme.skills.catalog.capability_center import build_capability_center

        return build_capability_center(
            self.get_all(),
            voice_triggers=self.get_voice_triggers(),
        )

    def get_generated_skill_governance(self) -> dict[str, Any]:
        """Return review state for generated skills."""
        store = self._governance_store()
        records = {record.skill_name: record for record in store.list_records()}
        items: list[dict[str, Any]] = []
        for skill in sorted(
            (item for item in self.get_all() if item.source == "generated"),
            key=lambda item: item.name,
        ):
            record = records.get(skill.name) or store.ensure_pending(skill)
            payload = record.to_dict()
            payload.update(
                {
                    "enabled": skill.enabled,
                    "tags": list(skill.tags),
                    "installed": True,
                    "package_ids": self._package_store().enabled_package_ids_for_skill(skill.name),
                    "validation": validate_generated_skill(
                        skill,
                        all_skills=self.get_all(),
                    ),
                }
            )
            items.append(payload)
        known_names = {item["skill_name"] for item in items}
        for record in records.values():
            if record.skill_name not in known_names:
                payload = record.to_dict()
                payload.update({"enabled": False, "installed": False, "tags": [], "validation": {}})
                items.append(payload)
        return {
            "records": items,
            "summary": {
                "total": len(items),
                "pending_approval": sum(1 for item in items if item["status"] == PENDING),
                "approved": sum(1 for item in items if item["status"] == APPROVED),
                "rejected": sum(1 for item in items if item["status"] == REJECTED),
                "disabled": sum(1 for item in items if item["status"] == DISABLED),
            },
            "policy": {
                "generated_skills_default_state": PENDING,
                "auto_enable_generated_skills": False,
                "approval_required": True,
                "approved_generated_skills_require_package": True,
            },
        }

    def get_skill_packages(self) -> dict[str, Any]:
        """Return customer/site ability packages for generated skills."""
        payload = self._package_store().payload()
        known_skills = {skill.name for skill in self.get_all()}
        for package in payload.get("packages", []):
            if not isinstance(package, dict):
                continue
            skill_names = [
                name for name in package.get("skill_names", [])
                if isinstance(name, str)
            ]
            package["missing_skill_names"] = [
                name for name in skill_names if name not in known_skills
            ]
            package["active_skill_names"] = [
                name for name in skill_names
                if self._skills.get(name) is not None and self._skills[name].enabled
            ]
        return payload

    def review_generated_skill(
        self,
        name: str,
        *,
        action: str,
        operator_id: str,
        note: str = "",
        router: Any | None = None,
    ) -> dict[str, Any]:
        """Approve, reject, disable, or return a generated skill to review."""
        self.load()
        skill = self._skills.get(name)
        if skill is None or skill.source != "generated":
            return {"ok": False, "error": "generated skill not found", "skill_name": name}

        action_to_status = {
            "approve": APPROVED,
            "enable": APPROVED,
            "reject": REJECTED,
            "disable": DISABLED,
            "request_review": PENDING,
            "pending": PENDING,
        }
        status = action_to_status.get(action)
        if status is None:
            return {"ok": False, "error": f"unsupported action: {action}", "skill_name": name}
        validation = validate_generated_skill(skill, all_skills=self.get_all())
        if status == APPROVED and not validation.get("ok"):
            SkillAuditLog().append(
                skill_name=name,
                status="validation_failed",
                event_type="governance",
                source="skill_governance",
                safety_level=skill.safety_level,
                execution=skill.execution,
                operator_id=operator_id,
                action=action,
                reason="generated skill validation failed",
                metadata={
                    "validation_ok": validation.get("ok"),
                    "issue_count": len(validation.get("issues", []))
                    if isinstance(validation.get("issues"), list) else 0,
                },
            )
            return {
                "ok": False,
                "error": "generated skill validation failed",
                "skill_name": name,
                "validation": validation,
            }

        record = self._governance_store().set_status(
            skill,
            status=status,
            reviewed_by=operator_id,
            review_note=note,
        )
        if status == APPROVED:
            self._package_store().assign_skill(
                skill.name,
                package_id=DEFAULT_PACKAGE_ID,
                operator_id=operator_id,
            )
        SkillAuditLog().append(
            skill_name=skill.name,
            status=status,
            event_type="governance",
            source="skill_governance",
            safety_level=skill.safety_level,
            execution=skill.execution,
            operator_id=operator_id,
            action=action,
            reason=note,
            result_preview=f"generated skill {status}",
            metadata={
                "enabled_after_review": self._generated_skill_enabled_by_governance(skill),
                "package_ids": ",".join(
                    self._package_store().enabled_package_ids_for_skill(skill.name)
                ),
                "validation_ok": validation.get("ok"),
            },
        )
        self.set_enabled(skill.name, self._generated_skill_enabled_by_governance(skill))
        self.hot_reload(router)
        reviewed = self._skills.get(skill.name)
        payload = record.to_dict()
        payload.update(
            {
                "ok": True,
                "enabled": bool(reviewed.enabled) if reviewed is not None else False,
                "validation": validation,
                "voice_triggers": [
                    phrase for phrase, target in self.get_voice_triggers().items()
                    if target == skill.name
                ],
            }
        )
        return payload

    def update_skill_package(
        self,
        *,
        skill_name: str,
        package_id: str = DEFAULT_PACKAGE_ID,
        action: str = "assign",
        operator_id: str = "",
    ) -> dict[str, Any]:
        """Assign or remove a generated skill from a customer/site package."""
        self.load()
        skill = self._skills.get(skill_name)
        if skill is None or skill.source != "generated":
            return {"ok": False, "error": "generated skill not found", "skill_name": skill_name}
        record = self._governance_store().get(skill.name)
        if record is None or record.status != APPROVED:
            return {
                "ok": False,
                "error": "generated skill must be approved before package enablement",
                "skill_name": skill.name,
            }
        if action == "assign":
            package = self._package_store().assign_skill(
                skill.name,
                package_id=package_id,
                operator_id=operator_id,
            )
        elif action == "unassign":
            package = self._package_store().unassign_skill(
                skill.name,
                package_id=package_id,
                operator_id=operator_id,
            )
        else:
            return {"ok": False, "error": f"unsupported package action: {action}"}
        enabled = self._generated_skill_enabled_by_governance(skill)
        self.set_enabled(skill.name, enabled)
        self.hot_reload()
        SkillAuditLog().append(
            skill_name=skill.name,
            status="package_assigned" if action == "assign" else "package_unassigned",
            event_type="governance",
            source="skill_package",
            safety_level=skill.safety_level,
            execution=skill.execution,
            operator_id=operator_id,
            action=action,
            result_preview=f"{action} {skill.name} in {package.package_id}",
            metadata={"package_id": package.package_id, "enabled_after_package_update": enabled},
        )
        return {
            "ok": True,
            "skill_name": skill.name,
            "enabled": enabled,
            "package": package.to_dict(),
            "packages": self.get_skill_packages(),
        }

    def upsert_skill_package(
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
    ) -> dict[str, Any]:
        """Create or update a customer/site ability package."""
        package = self._package_store().upsert_package(
            package_id=package_id,
            display_name=display_name,
            site_id=site_id,
            customer_name=customer_name,
            description=description,
            enabled=enabled,
            release_channel=release_channel,
            rollout_percent=rollout_percent,
            operator_id=operator_id,
        )
        self.load()
        SkillAuditLog().append(
            skill_name="skill_package",
            status="package_updated",
            event_type="governance",
            source="skill_package",
            operator_id=operator_id,
            action="upsert_package",
            result_preview=f"updated package {package.package_id}",
            metadata={
                "package_id": package.package_id,
                "enabled": package.enabled,
                "release_version": package.release_version,
                "release_channel": package.release_channel,
                "rollout_percent": package.rollout_percent,
            },
        )
        return {
            "ok": True,
            "package": package.to_dict(),
            "packages": self.get_skill_packages(),
        }

    def release_skill_package(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        release_channel: str = "pilot",
        rollout_percent: int = 100,
        operator_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Publish or gray-release a customer/site ability package."""
        package = self._package_store().update_release(
            package_id=package_id,
            release_channel=release_channel,
            rollout_percent=rollout_percent,
            operator_id=operator_id,
            note=note,
        )
        self.load()
        SkillAuditLog().append(
            skill_name="skill_package",
            status="package_released",
            event_type="governance",
            source="skill_package",
            operator_id=operator_id,
            action="release_package",
            reason=note,
            result_preview=f"released package {package.package_id} v{package.release_version}",
            metadata={
                "package_id": package.package_id,
                "release_version": package.release_version,
                "release_channel": package.release_channel,
                "rollout_percent": package.rollout_percent,
            },
        )
        return {
            "ok": True,
            "package": package.to_dict(),
            "packages": self.get_skill_packages(),
        }

    def rollback_skill_package(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        target_version: int | None = None,
        operator_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Rollback a customer/site ability package to a previous snapshot."""
        try:
            package = self._package_store().rollback_package(
                package_id=package_id,
                target_version=target_version,
                operator_id=operator_id,
                note=note,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "package_id": package_id}
        self.load()
        SkillAuditLog().append(
            skill_name="skill_package",
            status="package_rolled_back",
            event_type="governance",
            source="skill_package",
            operator_id=operator_id,
            action="rollback_package",
            reason=note,
            result_preview=f"rolled back package {package.package_id} to v{package.rollback_of_version}",
            metadata={
                "package_id": package.package_id,
                "release_version": package.release_version,
                "rollback_of_version": package.rollback_of_version,
            },
        )
        return {
            "ok": True,
            "package": package.to_dict(),
            "history": self._package_store().history(package.package_id),
            "packages": self.get_skill_packages(),
        }

    def get_skill_package_history(
        self,
        *,
        package_id: str = DEFAULT_PACKAGE_ID,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return version snapshots for one customer/site ability package."""
        return self._package_store().history(package_id, limit=limit)

    def create_generated_skill_draft(
        self,
        *,
        name: str,
        description: str,
        prompt: str,
        voice_trigger: str = "",
        tools_section: str = "",
        tags: str | list[str] = "generated",
        safety_level: str = "normal",
        confirm_before_execute: bool = False,
        operator_id: str = "",
        source: str = "skill_growth",
        router: Any | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Create a generated SKILL.md draft that still needs review."""
        skill_name = _sanitize_skill_name(name)
        if not skill_name:
            return {"ok": False, "error": "invalid skill name"}
        if not description.strip():
            return {"ok": False, "error": "description is required", "skill_name": skill_name}
        if len(prompt.strip()) < 10:
            return {"ok": False, "error": "prompt is too short", "skill_name": skill_name}

        self.load()
        existing = self._skills.get(skill_name)
        if existing is not None and existing.source != "generated":
            return {
                "ok": False,
                "error": "skill name conflicts with existing non-generated skill",
                "skill_name": skill_name,
                "existing_source": existing.source,
            }

        skill_dir = self.generated_skills_dir / skill_name
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists() and not overwrite:
            return {
                "ok": False,
                "error": "generated skill already exists",
                "skill_name": skill_name,
                "path": str(skill_file),
            }

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(
                _GENERATED_SKILL_TEMPLATE.format(
                    name=skill_name,
                    description=description.replace("\n", " "),
                    tags=_format_tags(tags),
                    safety_level=safety_level or "normal",
                    voice_trigger=voice_trigger or "",
                    confirm_before_execute="true" if confirm_before_execute else "false",
                    tools_section=tools_section or "",
                    prompt=prompt.strip(),
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            return {"ok": False, "error": f"failed to write skill file: {exc}", "skill_name": skill_name}

        loaded_count = self.hot_reload(router)
        skill = self._skills.get(skill_name)
        record = None
        validation: dict[str, Any] = {}
        if skill is not None:
            record = self._governance_store().ensure_pending(skill)
            self.set_enabled(skill.name, False)
            validation = validate_generated_skill(skill, all_skills=self.get_all())

        SkillAuditLog().append(
            skill_name=skill_name,
            status="draft_created",
            event_type="governance",
            source=source,
            safety_level=safety_level,
            execution=skill.execution if skill is not None else "",
            operator_id=operator_id,
            action="create_generated_skill_draft",
            result_preview=f"generated skill draft created: {skill_name}",
            metadata={
                "path": str(skill_file),
                "validation_ok": validation.get("ok", False),
                "loaded_count": loaded_count,
            },
        )
        return {
            "ok": True,
            "skill_name": skill_name,
            "path": str(skill_file),
            "loaded_count": loaded_count,
            "status": record.status if record is not None else PENDING,
            "enabled": False,
            "validation": validation,
        }

    def openapi_document(self) -> dict[str, Any]:
        """Generate an OpenAPI document from the loaded contracts."""
        return build_skills_openapi(self.get_contracts())

    def get_agent_shell_skills(self) -> set[str]:
        """Return names of enabled skills with execution='agent_shell'."""
        return {s.name for s in self._skills.values()
                if s.enabled and s.execution == "agent_shell"}

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a skill. Returns False if skill not found."""
        skill = self._skills.get(name)
        if skill is None:
            return False
        skill.enabled = enabled
        if enabled:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)
        self._save_settings()
        return True

    def build_prompt(self, name: str, context: dict[str, str] | None = None) -> str | None:
        """Build a prompt for a skill with template variable substitution.

        Returns None if skill not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            return None
        return skill.build_prompt(context)

    def get_voice_triggers(self) -> dict[str, str]:
        """Return a mapping of voice_trigger phrase -> skill name for enabled skills.

        The ``voice_trigger`` field supports comma-separated phrases, e.g.
        ``voice_trigger: 现在几点,星期几,今天几号`` will register three
        separate trigger phrases pointing to the same skill.
        """
        triggers: dict[str, str] = {}
        for skill in self.get_enabled():
            if skill.voice_trigger:
                for phrase in skill.voice_trigger.split(","):
                    phrase = phrase.strip()
                    if phrase:
                        triggers[phrase] = skill.name
        return triggers

    def check_dependencies(self, name: str) -> tuple[bool, list[str]]:
        """Check if a skill's dependencies are satisfied.

        Returns:
            (ok, missing_list)
        """
        skill = self._skills.get(name)
        if skill is None or not skill.depends:
            return (True, [])
        missing = [dep for dep in skill.depends if dep not in self._skills]
        return (len(missing) == 0, missing)

    def check_conflicts(self, name: str) -> tuple[bool, list[str]]:
        """Check if a skill conflicts with any active (enabled) skill.

        Returns:
            (ok, conflicting_list)
        """
        skill = self._skills.get(name)
        if skill is None or not skill.conflicts:
            return (True, [])
        active_conflicts = [
            c for c in skill.conflicts
            if c in self._skills and self._skills[c].enabled
        ]
        return (len(active_conflicts) == 0, active_conflicts)

    def get_skill_catalog(self) -> str:
        """Generate a comma-separated catalog of enabled skill names."""
        enabled = self.get_enabled()
        if not enabled:
            return "none"
        return ", ".join(s.name for s in enabled)

    # ── Discovery & Parsing ─────────────────────────────────────

    def _discover_from(self, directory: Path, source: str) -> None:
        """Scan a directory for <name>/SKILL.md subdirectories."""
        if not directory.is_dir():
            return
        try:
            for entry in sorted(directory.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if not skill_file.is_file():
                    continue
                skill = self._parse_skill_md(skill_file, source)
                if skill is not None:
                    self._skills[skill.name] = skill
        except OSError as exc:
            logger.warning("Failed to scan %s: %s", directory, exc)

    def _parse_skill_md(self, file_path: Path, source: str) -> SkillDefinition | None:
        """Parse a SKILL.md file into a SkillDefinition."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return None

        # --- YAML frontmatter ---
        fm_match = re.match(r"^---\r?\n(.*?)\r?\n---", content, re.DOTALL)
        if fm_match is None:
            return None
        meta = self._parse_yaml(fm_match.group(1))

        # --- ## Prompt section ---
        prompt_match = re.search(
            r"^## Prompt[^\S\r\n]*\r?\n(.*?)(?=^## |\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        )
        prompt = prompt_match.group(1).strip() if prompt_match else ""

        # --- ## Tools section ---
        tools_match = re.search(
            r"^## Tools[^\S\r\n]*\r?\n(.*?)(?=^## |\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        )
        tools_section = tools_match.group(1).strip() if tools_match else ""

        # Determine name: explicit in meta, or directory name
        name = meta.get("name", file_path.parent.name)

        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            version=meta.get("version", "1.0.0"),
            trigger=meta.get("trigger", "manual"),
            model=meta.get("model", ""),
            timeout=int(meta.get("timeout", 30)),
            tags=self._ensure_list(meta.get("tags", [])),
            depends=self._ensure_list(meta.get("depends", [])),
            conflicts=self._ensure_list(meta.get("conflicts", [])),
            safety_level=meta.get("safety_level", "normal"),
            voice_trigger=meta.get("voice_trigger"),
            required_prompt=meta.get("required_prompt", ""),
            confirm_before_execute=bool(meta.get("confirm_before_execute", False)),
            required_slots=self._parse_slot_specs(meta.get("required_slots", [])),
            schedule=meta.get("schedule"),
            prompt_template=prompt,
            tools_section=tools_section,
            execution=meta.get("execution", "skill_executor"),
            source=source,
            path=str(file_path),
            enabled=bool(meta.get("enabled", True)),
        )

    @staticmethod
    def _parse_slot_specs(specs_raw: Any) -> list:
        """Convert raw YAML list into SlotSpec objects."""
        from askme.skills.core.skill_model import SlotSpec
        if not isinstance(specs_raw, list):
            return []
        result = []
        for item in specs_raw:
            if isinstance(item, dict):
                result.append(SlotSpec(
                    name=str(item.get("name", "slot")),
                    type=str(item.get("type", "text")),
                    prompt=str(item.get("prompt", "")),
                    optional=bool(item.get("optional", False)),
                    default=str(item.get("default", "")),
                ))
        return result

    @staticmethod
    def _parse_yaml(yaml_text: str) -> dict[str, Any]:
        """Parse YAML frontmatter using PyYAML."""
        try:
            result = yaml.safe_load(yaml_text)
            return result if isinstance(result, dict) else {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _ensure_list(value: Any) -> list[str]:
        """Ensure a value is a list of strings."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return []

    def _governance_store(self) -> SkillGovernanceStore:
        return SkillGovernanceStore.for_generated_dir(self.generated_skills_dir)

    def _package_store(self) -> SkillPackageStore:
        return SkillPackageStore.for_generated_dir(self.generated_skills_dir)

    def _apply_generated_skill_governance(self) -> None:
        store = self._governance_store()
        changed = False
        for skill in self._skills.values():
            if skill.source != "generated":
                continue
            record = store.ensure_pending(skill)
            should_enable = (
                SkillGovernanceStore.is_enabled_status(record.status)
                and self._package_store().is_skill_enabled(skill.name)
            )
            if not should_enable:
                skill.enabled = False
                if skill.name not in self._disabled:
                    self._disabled.add(skill.name)
                    changed = True
            elif skill.name in self._disabled:
                skill.enabled = True
                self._disabled.discard(skill.name)
                changed = True
        if changed:
            self._save_settings()

    def _generated_skill_enabled_by_governance(self, skill: SkillDefinition) -> bool:
        record = self._governance_store().get(skill.name)
        return (
            record is not None
            and SkillGovernanceStore.is_enabled_status(record.status)
            and self._package_store().is_skill_enabled(skill.name)
        )

    # ── Settings Persistence ────────────────────────────────────

    def _load_settings(self) -> None:
        """Load disabled-skills list from data/skills_settings.json."""
        try:
            if _SETTINGS_FILE.is_file():
                data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                self._disabled = set(data.get("disabled", []))
        except (OSError, json.JSONDecodeError):
            self._disabled = set()

    def _save_settings(self) -> None:
        """Persist disabled-skills list to data/skills_settings.json."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = {"disabled": sorted(self._disabled)}
            _SETTINGS_FILE.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save skill settings: %s", exc)


def _sanitize_skill_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", str(value or "").lower().strip())


def _format_tags(value: str | list[str]) -> str:
    raw = value if isinstance(value, list) else str(value or "").split(",")
    tags = [
        re.sub(r"[^a-zA-Z0-9_-]", "_", str(item).strip())
        for item in raw
        if str(item).strip()
    ]
    if "generated" not in tags:
        tags.insert(0, "generated")
    return ",".join(tags)
