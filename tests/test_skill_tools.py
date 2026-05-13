"""Tests for generated skill creation tooling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from askme.skills.audit import SkillAuditLog
from askme.skills.skill_manager import SkillManager
from askme.tools.skill_tools import CreateAgentProfileTool, CreateSkillTool, register_skill_tools


def _make_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CreateSkillTool:
    import askme.skills.skill_manager as skill_manager_module

    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        tmp_path / "skills_settings.json",
    )
    monkeypatch.setattr(
        skill_manager_module,
        "SkillAuditLog",
        lambda: SkillAuditLog(tmp_path / "skill-audit.jsonl"),
    )
    tool = CreateSkillTool()
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = MagicMock()
    tool.set_context(manager, router)
    return tool


def test_execute_without_context_returns_error() -> None:
    tool = CreateSkillTool()

    result = tool.execute(name="test", description="desc", prompt="do something")

    assert "[Error]" in result


def test_skill_name_is_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _make_tool(tmp_path, monkeypatch)

    result = tool.execute(name="My Skill!", description="d", prompt="prompt text")

    assert "[Error]" not in result
    assert (tmp_path / "skills" / "my_skill_").exists()


def test_empty_name_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _make_tool(tmp_path, monkeypatch)

    result = tool.execute(name="", description="d", prompt="prompt text")

    assert "[Error]" in result


def test_skill_md_created_with_prompt_trigger_and_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(tmp_path, monkeypatch)

    tool.execute(
        name="patrol_skill",
        description="patrols",
        prompt="my custom prompt text",
        voice_trigger="start patrol",
        tools_section="web_search",
        tags="robot,sensor",
    )

    content = (
        tmp_path / "skills" / "patrol_skill" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "patrol_skill" in content
    assert "my custom prompt text" in content
    assert "start patrol" in content
    assert "web_search" in content
    assert "robot,sensor" in content


def test_hot_reload_called_and_result_explains_review_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(tmp_path, monkeypatch)

    result = tool.execute(name="skill_f", description="d", prompt="prompt text")

    assert tool._router.update_voice_triggers.called
    assert "待审批" in result
    assert "审批通过后" in result
    assert "SKILL.md" in result


def test_register_skill_tools_registers_create_skill_tool() -> None:
    registry = MagicMock()
    manager = MagicMock()
    router = MagicMock()

    register_skill_tools(registry, manager, router)

    assert registry.register.call_count == 2
    registered = [call.args[0] for call in registry.register.call_args_list]
    assert any(isinstance(tool, CreateSkillTool) for tool in registered)
    assert any(isinstance(tool, CreateAgentProfileTool) for tool in registered)


def test_tool_metadata() -> None:
    assert CreateSkillTool.name == "create_skill"
    assert CreateSkillTool.agent_allowed is True
    required = CreateSkillTool.parameters["required"]
    assert {"name", "description", "prompt"}.issubset(required)


def test_create_agent_profile_tool_metadata() -> None:
    assert CreateAgentProfileTool.name == "create_agent_profile"
    assert CreateAgentProfileTool.agent_allowed is True
    required = CreateAgentProfileTool.parameters["required"]
    assert {"name", "description", "instructions"}.issubset(required)


def test_create_agent_profile_tool_rejects_unknown_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tool = CreateAgentProfileTool(known_tools={"read_file", "create_agent_profile"})

    result = tool.execute(
        name="unsafe profile",
        description="Attempts to bypass the agent tool boundary.",
        instructions="This profile should be rejected before it is written.",
        tools="unknown_tool",
    )

    assert "[Error] unknown tools requested" in result
    assert not (tmp_path / ".askme" / "agents" / "unsafe_profile.md").exists()


def test_register_skill_tools_passes_registry_allowlist_to_agent_profile_tool() -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.registered = []

        def get_agent_allowed_names(self):
            return {"read_file", "robot_api"}

        def register(self, tool) -> None:
            self.registered.append(tool)

    registry = FakeRegistry()
    manager = MagicMock()
    router = MagicMock()

    register_skill_tools(registry, manager, router)

    profile_tool = next(tool for tool in registry.registered if isinstance(tool, CreateAgentProfileTool))
    assert {"read_file", "robot_api", "create_skill", "create_agent_profile"}.issubset(
        profile_tool._known_tools
    )
