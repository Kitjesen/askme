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
import os
import re
from pathlib import Path
from typing import Any

from .skill_model import SkillDefinition

logger = logging.getLogger(__name__)

# Default data dir: askme project root / data
_PACKAGE_DIR = Path(__file__).resolve().parent            # askme/skills/
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent                # askme repo root
_DATA_DIR = _PROJECT_ROOT / "data"
_SETTINGS_FILE = _DATA_DIR / "skills_settings.json"


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
        ]

        for directory, source in locations:
            self._discover_from(directory, source)

        # Apply persisted enabled/disabled state
        for name in self._disabled:
            if name in self._skills:
                self._skills[name].enabled = False

        logger.info("Loaded %d skills", len(self._skills))

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all(self) -> list[SkillDefinition]:
        """Get all skills (including disabled ones)."""
        return list(self._skills.values())

    def get_enabled(self) -> list[SkillDefinition]:
        """Get only enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

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
        """Return a mapping of voice_trigger phrase -> skill name for enabled skills."""
        triggers: dict[str, str] = {}
        for skill in self.get_enabled():
            if skill.voice_trigger:
                triggers[skill.voice_trigger] = skill.name
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
        return ", ".join(s.name for s in enabled) + ", none"

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
        prompt_match = re.search(r"## Prompt\s*\r?\n(.*?)(?=\r?\n## |\Z)", content, re.DOTALL)
        prompt = prompt_match.group(1).strip() if prompt_match else ""

        # --- ## Tools section ---
        tools_match = re.search(r"## Tools\s*\r?\n(.*?)(?=\r?\n## |\Z)", content, re.DOTALL)
        tools_section = tools_match.group(1).strip() if tools_match else ""

        # Determine name: explicit in meta, or directory name
        name = meta.get("name", file_path.parent.name)

        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            version=meta.get("version", "1.0.0"),
            trigger=meta.get("trigger", "manual"),
            model=meta.get("model", "deepseek-chat"),
            timeout=int(meta.get("timeout", 30)),
            tags=self._ensure_list(meta.get("tags", [])),
            depends=self._ensure_list(meta.get("depends", [])),
            conflicts=self._ensure_list(meta.get("conflicts", [])),
            safety_level=meta.get("safety_level", "normal"),
            voice_trigger=meta.get("voice_trigger"),
            schedule=meta.get("schedule"),
            prompt_template=prompt,
            tools_section=tools_section,
            source=source,
            path=str(file_path),
            enabled=True,
        )

    @staticmethod
    def _parse_yaml(yaml_text: str) -> dict[str, Any]:
        """Lightweight YAML frontmatter parser (handles simple key: value pairs)."""
        result: dict[str, Any] = {}
        for line in yaml_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\w[\w_-]*):\s*(.+)$", line)
            if m is None:
                continue
            key = m.group(1)
            val: Any = m.group(2).strip()
            # Strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Parse inline list: [a, b, c]
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                val = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
            # Parse booleans
            elif val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            # Parse None
            elif val.lower() in ("null", "none", "~"):
                val = None
            result[key] = val
        return result

    @staticmethod
    def _ensure_list(value: Any) -> list[str]:
        """Ensure a value is a list of strings."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return []

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
