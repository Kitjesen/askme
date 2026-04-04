"""Tests for CreateSkillTool — execute routing, name sanitization, file creation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from askme.tools.skill_tools import CreateSkillTool, register_skill_tools


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool(tmp_path: Path) -> CreateSkillTool:
    """Create a CreateSkillTool with mocked SkillManager and IntentRouter."""
    tool = CreateSkillTool()
    mgr = MagicMock()
    mgr.generated_skills_dir = tmp_path / "skills" / "generated"
    mgr.generated_skills_dir.mkdir(parents=True)
    mgr.hot_reload.return_value = 5  # 5 skills after reload
    mgr.get_voice_triggers.return_value = {"巡逻吧": "my_skill"}
    router = MagicMock()
    tool.set_context(mgr, router)
    return tool


# ── No context ────────────────────────────────────────────────────────────────

class TestNoContext:
    def test_execute_without_context_returns_error(self):
        tool = CreateSkillTool()
        result = tool.execute(name="test", description="desc", prompt="do something")
        assert "[Error]" in result


# ── Name sanitization ─────────────────────────────────────────────────────────

class TestNameSanitization:
    def test_uppercase_lowercased(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="MySkill", description="d", prompt="p")
        assert "[Error]" not in result
        # skill dir name should be sanitized
        assert (tmp_path / "skills" / "generated" / "myskill").exists()

    def test_spaces_replaced_with_underscore(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="my skill name", description="d", prompt="p")
        assert "[Error]" not in result
        assert (tmp_path / "skills" / "generated" / "my_skill_name").exists()

    def test_special_chars_replaced(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="my-skill!", description="d", prompt="p")
        assert "[Error]" not in result

    def test_empty_name_returns_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="", description="d", prompt="p")
        assert "[Error]" in result

    def test_only_special_chars_sanitized_to_underscores(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="!!!@@@", description="d", prompt="p")
        # The name becomes "______" which is valid
        assert "[Error]" not in result


# ── File creation ─────────────────────────────────────────────────────────────

class TestFileCreation:
    def test_skill_md_created(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="my_skill", description="does stuff", prompt="behave like agent")
        skill_file = tmp_path / "skills" / "generated" / "my_skill" / "SKILL.md"
        assert skill_file.exists()

    def test_skill_md_contains_name(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="patrol_skill", description="patrols", prompt="go patrol")
        content = (tmp_path / "skills" / "generated" / "patrol_skill" / "SKILL.md").read_text()
        assert "patrol_skill" in content

    def test_skill_md_contains_prompt(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="skill_a", description="d", prompt="my custom prompt text")
        content = (tmp_path / "skills" / "generated" / "skill_a" / "SKILL.md").read_text()
        assert "my custom prompt text" in content

    def test_skill_md_contains_voice_trigger(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="skill_b", description="d", prompt="p", voice_trigger="开始巡逻,执行任务")
        content = (tmp_path / "skills" / "generated" / "skill_b" / "SKILL.md").read_text()
        assert "开始巡逻" in content

    def test_skill_md_contains_tools_section(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="skill_c", description="d", prompt="p", tools_section="web_search")
        content = (tmp_path / "skills" / "generated" / "skill_c" / "SKILL.md").read_text()
        assert "web_search" in content

    def test_tags_in_skill_md(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="skill_d", description="d", prompt="p", tags="robot,sensor")
        content = (tmp_path / "skills" / "generated" / "skill_d" / "SKILL.md").read_text()
        assert "robot" in content


# ── Hot reload and result ─────────────────────────────────────────────────────

class TestHotReload:
    def test_hot_reload_called(self, tmp_path):
        tool = _make_tool(tmp_path)
        tool.execute(name="skill_e", description="d", prompt="p")
        tool._mgr.hot_reload.assert_called_once()

    def test_result_contains_skill_count(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="skill_f", description="d", prompt="p")
        assert "5" in result  # mock returns 5 skills

    def test_result_contains_voice_triggers(self, tmp_path):
        tool = _make_tool(tmp_path)
        # Set up trigger mapping to match the created skill name
        tool._mgr.get_voice_triggers.return_value = {"执行巡逻": "skill_g"}
        result = tool.execute(name="skill_g", description="d", prompt="p")
        assert "执行巡逻" in result

    def test_result_contains_file_path(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = tool.execute(name="skill_h", description="d", prompt="p")
        assert "SKILL.md" in result


# ── register_skill_tools ──────────────────────────────────────────────────────

class TestRegisterSkillTools:
    def test_registers_create_skill_tool(self, tmp_path):
        registry = MagicMock()
        mgr = MagicMock()
        router = MagicMock()
        register_skill_tools(registry, mgr, router)
        registry.register.assert_called_once()
        registered = registry.register.call_args[0][0]
        assert isinstance(registered, CreateSkillTool)


# ── Tool metadata ─────────────────────────────────────────────────────────────

class TestToolMetadata:
    def test_name(self):
        assert CreateSkillTool.name == "create_skill"

    def test_agent_allowed(self):
        assert CreateSkillTool.agent_allowed is True

    def test_required_params(self):
        required = CreateSkillTool.parameters["required"]
        assert "name" in required
        assert "description" in required
        assert "prompt" in required
