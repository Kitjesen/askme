"""Tests for SkillManager — skill discovery, state management, triggers, and prompts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from askme.skills.skill_manager import SkillManager
from askme.skills.skill_model import SkillDefinition

# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_skill(
    base_dir: Path,
    name: str,
    *,
    description: str = "A test skill",
    enabled: bool = True,
    tags: str = "",
    depends: str = "",
    conflicts: str = "",
    voice_trigger: str = "",
    trigger: str = "manual",
    execution: str = "skill_executor",
    prompt: str = "Do {{action}} now.",
    safety_level: str = "normal",
) -> Path:
    """Write a minimal SKILL.md under base_dir/<name>/SKILL.md."""
    skill_dir = base_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"enabled: {'true' if enabled else 'false'}",
        f"trigger: {trigger}",
        f"execution: {execution}",
        f"safety_level: {safety_level}",
    ]
    if tags:
        lines.append(f"tags: [{tags}]")
    if depends:
        lines.append(f"depends: [{depends}]")
    if conflicts:
        lines.append(f"conflicts: [{conflicts}]")
    if voice_trigger:
        lines.append(f"voice_trigger: {voice_trigger}")
    lines += [
        "---",
        "",
        "## Prompt",
        prompt,
    ]
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("\n".join(lines), encoding="utf-8")
    return skill_file


def _make_manager(tmp_path: Path, monkeypatch) -> SkillManager:
    """Create a SkillManager scoped to tmp_path with isolated settings."""
    settings_file = tmp_path / "settings" / "skills_settings.json"
    data_dir = tmp_path / "settings"
    monkeypatch.setattr("askme.skills.skill_manager._SETTINGS_FILE", settings_file)
    monkeypatch.setattr("askme.skills.skill_manager._DATA_DIR", data_dir)
    return SkillManager(project_dir=tmp_path)


def _load_with_skills(
    tmp_path: Path, monkeypatch, names: list[str] | None = None
) -> SkillManager:
    """Create manager, write skills under <tmp_path>/skills/, load, return."""
    mgr = _make_manager(tmp_path, monkeypatch)
    skills_dir = tmp_path / "skills"
    for n in (names or []):
        _write_skill(skills_dir, n)
    mgr.load()
    return mgr


# ── TestLoad ─────────────────────────────────────────────────────────────────

class TestLoad:
    def test_empty_project_dir_loads_zero_project_skills(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        # No project skills, but builtins may exist — just verify no crash
        assert isinstance(mgr.get_all(), list)

    def test_single_skill_discovered(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        assert mgr.get("patrol") is not None

    def test_multiple_skills_all_loaded(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        for name in ("nav", "arm", "report"):
            _write_skill(skills_dir, name)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        assert mgr.get("nav") is not None
        assert mgr.get("arm") is not None
        assert mgr.get("report") is not None

    def test_disabled_in_frontmatter_starts_disabled(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "inactive", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        skill = mgr.get("inactive")
        assert skill is not None
        assert skill.enabled is False

    def test_missing_skill_md_directory_ignored(self, tmp_path, monkeypatch):
        # Create a non-directory file at skills/<name> — should not crash
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "not_a_dir.txt").write_text("hello")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()  # should not raise

    def test_skill_without_frontmatter_skipped(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills" / "bad_skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("No frontmatter here.")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        assert mgr.get("bad_skill") is None

    def test_reload_clears_old_skills(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "alpha")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        # Remove the skill file and reload
        import shutil
        shutil.rmtree(skills_dir / "alpha")
        mgr.load()
        assert mgr.get("alpha") is None


# ── TestGetMethods ────────────────────────────────────────────────────────────

class TestGetMethods:
    def test_get_returns_none_for_missing(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        assert mgr.get("nonexistent") is None

    def test_get_returns_skill_definition(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "navigate", description="Navigate to a location")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        skill = mgr.get("navigate")
        assert isinstance(skill, SkillDefinition)
        assert skill.name == "navigate"
        assert skill.description == "Navigate to a location"

    def test_get_all_includes_disabled(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "active", enabled=True)
        _write_skill(skills_dir, "disabled_skill", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        names = {s.name for s in mgr.get_all()}
        assert "active" in names
        assert "disabled_skill" in names

    def test_get_enabled_excludes_disabled(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "active", enabled=True)
        _write_skill(skills_dir, "off", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        enabled_names = {s.name for s in mgr.get_enabled()}
        assert "active" in enabled_names
        assert "off" not in enabled_names

    def test_get_all_returns_list(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, ["a", "b"])
        assert isinstance(mgr.get_all(), list)

    def test_get_skill_catalog_no_enabled_returns_none_string(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "only", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        # Only disabled skills — catalog should return "none"
        # (builtins might be loaded, so only check if no project skills are enabled)
        # For isolated test: empty project dir
        mgr2 = _make_manager(tmp_path, monkeypatch)
        mgr2._skills = {}  # manually clear
        catalog = mgr2.get_skill_catalog()
        assert catalog == "none"

    def test_get_skill_catalog_lists_enabled_names(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol")
        _write_skill(skills_dir, "report")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        catalog = mgr.get_skill_catalog()
        # Both names should appear somewhere
        assert "patrol" in catalog
        assert "report" in catalog


# ── TestSetEnabled ────────────────────────────────────────────────────────────

class TestSetEnabled:
    def test_disable_existing_skill(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "target", enabled=True)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.set_enabled("target", False)
        assert result is True
        assert mgr.get("target").enabled is False

    def test_enable_existing_skill(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "target", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.set_enabled("target", True)
        assert result is True
        assert mgr.get("target").enabled is True

    def test_returns_false_for_unknown_skill(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        assert mgr.set_enabled("ghost", False) is False

    def test_disabled_set_updated_on_disable(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        mgr.set_enabled("patrol", False)
        assert "patrol" in mgr._disabled

    def test_disabled_set_cleared_on_enable(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol", enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        mgr.set_enabled("patrol", True)
        assert "patrol" not in mgr._disabled

    def test_settings_persisted_after_disable(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "nav")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        mgr.set_enabled("nav", False)
        settings_file = tmp_path / "settings" / "skills_settings.json"
        data = json.loads(settings_file.read_text())
        assert "nav" in data["disabled"]


# ── TestVoiceTriggers ─────────────────────────────────────────────────────────

class TestVoiceTriggers:
    def test_single_trigger_phrase(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "clock", voice_trigger="现在几点")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert "现在几点" in triggers
        assert triggers["现在几点"] == "clock"

    def test_multi_phrase_comma_split(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "clock", voice_trigger="现在几点,星期几,今天几号")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert "现在几点" in triggers
        assert "星期几" in triggers
        assert "今天几号" in triggers
        assert triggers["星期几"] == "clock"

    def test_whitespace_around_phrases_stripped(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "clock", voice_trigger="  现在几点 ,  星期几  ")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert "现在几点" in triggers
        assert "星期几" in triggers

    def test_disabled_skill_not_in_triggers(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        unique_trigger = "zzz_unique_trigger_phrase_xyz_disabled"
        _write_skill(skills_dir, "clock", voice_trigger=unique_trigger, enabled=False)
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert unique_trigger not in triggers

    def test_no_voice_trigger_not_included(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "nav")  # no voice_trigger
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert "nav" not in triggers.values()

    def test_empty_phrases_after_split_ignored(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "test", voice_trigger="phrase1,,phrase2,")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        triggers = mgr.get_voice_triggers()
        assert "phrase1" in triggers
        assert "phrase2" in triggers
        assert "" not in triggers


# ── TestDependencies ──────────────────────────────────────────────────────────

class TestDependencies:
    def test_no_depends_returns_ok(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "simple")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, missing = mgr.check_dependencies("simple")
        assert ok is True
        assert missing == []

    def test_unknown_skill_returns_ok(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        ok, missing = mgr.check_dependencies("ghost")
        assert ok is True
        assert missing == []

    def test_satisfied_dependency(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "base")
        _write_skill(skills_dir, "derived", depends="base")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, missing = mgr.check_dependencies("derived")
        assert ok is True
        assert missing == []

    def test_missing_dependency(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "derived", depends="missing_dep")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, missing = mgr.check_dependencies("derived")
        assert ok is False
        assert "missing_dep" in missing


# ── TestConflicts ─────────────────────────────────────────────────────────────

class TestConflicts:
    def test_no_conflicts_returns_ok(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "peaceful")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, conflicting = mgr.check_conflicts("peaceful")
        assert ok is True
        assert conflicting == []

    def test_unknown_skill_returns_ok(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        ok, conflicting = mgr.check_conflicts("ghost")
        assert ok is True
        assert conflicting == []

    def test_conflict_with_enabled_skill(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "alpha")
        _write_skill(skills_dir, "beta", conflicts="alpha")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, conflicting = mgr.check_conflicts("beta")
        assert ok is False
        assert "alpha" in conflicting

    def test_conflict_with_disabled_skill_not_reported(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "alpha", enabled=False)
        _write_skill(skills_dir, "beta", conflicts="alpha")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        ok, conflicting = mgr.check_conflicts("beta")
        assert ok is True
        assert "alpha" not in conflicting


# ── TestBuildPrompt ───────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_returns_none_for_unknown_skill(self, tmp_path, monkeypatch):
        mgr = _load_with_skills(tmp_path, monkeypatch, [])
        assert mgr.build_prompt("ghost") is None

    def test_substitutes_template_variable(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "action", prompt="Execute {{task}} immediately.")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.build_prompt("action", {"task": "patrol"})
        assert "patrol" in result
        assert "{{task}}" not in result

    def test_unresolved_placeholders_removed(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "action", prompt="Go to {{location}} now.")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.build_prompt("action")  # no context
        assert "{{location}}" not in result
        assert "Go to" in result

    def test_multiple_substitutions(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(
            skills_dir, "multi",
            prompt="{{verb}} to {{place}} at {{time}}."
        )
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.build_prompt(
            "multi", {"verb": "Navigate", "place": "warehouse", "time": "14:00"}
        )
        assert "Navigate" in result
        assert "warehouse" in result
        assert "14:00" in result

    def test_empty_context_returns_prompt_without_placeholders(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "simple", prompt="Stand by.")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        result = mgr.build_prompt("simple", {})
        assert result == "Stand by."


# ── TestSettingsPersistence ───────────────────────────────────────────────────

class TestSettingsPersistence:
    def test_disabled_skills_survive_reload(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        mgr.set_enabled("patrol", False)

        # Second manager shares the same settings file
        mgr2 = _make_manager(tmp_path, monkeypatch)
        mgr2.load()
        assert mgr2.get("patrol").enabled is False

    def test_re_enabled_skill_not_in_saved_disabled(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "patrol")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()
        mgr.set_enabled("patrol", False)
        mgr.set_enabled("patrol", True)

        settings_file = tmp_path / "settings" / "skills_settings.json"
        data = json.loads(settings_file.read_text())
        assert "patrol" not in data["disabled"]

    def test_corrupt_settings_file_falls_back_gracefully(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings" / "skills_settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("not valid json", encoding="utf-8")
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.load()  # should not raise
        assert mgr._disabled == set()


# ── TestHotReload ─────────────────────────────────────────────────────────────

class TestHotReload:
    def test_hot_reload_returns_count(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "a")
        _write_skill(skills_dir, "b")
        mgr = _make_manager(tmp_path, monkeypatch)
        count = mgr.hot_reload()
        assert count >= 2  # at least project skills, may include builtins

    def test_hot_reload_with_router_calls_update(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "clock", voice_trigger="几点了")
        mgr = _make_manager(tmp_path, monkeypatch)
        router = MagicMock()
        mgr.hot_reload(router=router)
        router.update_voice_triggers.assert_called_once()
        triggers_arg = router.update_voice_triggers.call_args[0][0]
        assert "几点了" in triggers_arg

    def test_hot_reload_without_router_no_crash(self, tmp_path, monkeypatch):
        mgr = _make_manager(tmp_path, monkeypatch)
        mgr.hot_reload(router=None)  # should not raise


# ── TestEnsureList ────────────────────────────────────────────────────────────

class TestEnsureList:
    def test_list_passthrough(self):
        assert SkillManager._ensure_list(["a", "b"]) == ["a", "b"]

    def test_string_splits_on_comma(self):
        result = SkillManager._ensure_list("alpha, beta, gamma")
        assert result == ["alpha", "beta", "gamma"]

    def test_none_returns_empty(self):
        assert SkillManager._ensure_list(None) == []

    def test_non_string_non_list_returns_empty(self):
        assert SkillManager._ensure_list(42) == []


# ── TestParseYaml ─────────────────────────────────────────────────────────────

class TestParseYaml:
    def test_valid_yaml_parsed(self):
        result = SkillManager._parse_yaml("name: patrol\nenabled: true")
        assert result["name"] == "patrol"
        assert result["enabled"] is True

    def test_invalid_yaml_returns_empty_dict(self):
        result = SkillManager._parse_yaml(": : : invalid")
        assert result == {}

    def test_non_dict_yaml_returns_empty_dict(self):
        result = SkillManager._parse_yaml("- item1\n- item2")
        assert result == {}
