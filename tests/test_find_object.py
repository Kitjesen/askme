"""Tests for the find_object skill chain: vision tools + skill loading + routing."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from askme.tools.vision_tool import LookAroundTool, FindTargetTool, register_vision_tools
from askme.tools.tool_registry import ToolRegistry
from askme.skills.skill_manager import SkillManager


# ---------- Vision Tools ----------

class TestLookAroundTool:
    def test_no_vision_returns_message(self):
        tool = LookAroundTool()
        result = tool.execute()
        assert "视觉不可用" in result

    def test_with_vision_unavailable(self):
        tool = LookAroundTool()
        vision = MagicMock()
        vision.available = False
        tool.set_vision(vision)
        result = tool.execute()
        assert "视觉不可用" in result

    def test_metadata(self):
        tool = LookAroundTool()
        assert tool.name == "look_around"
        assert tool.safety_level == "normal"


class TestFindTargetTool:
    def test_no_target_returns_error(self):
        tool = FindTargetTool()
        vision = MagicMock()
        vision.available = True
        tool.set_vision(vision)
        result = tool.execute(target="")
        assert "请指定" in result

    def test_no_vision_returns_message(self):
        tool = FindTargetTool()
        result = tool.execute(target="bottle")
        assert "视觉不可用" in result

    def test_not_found(self):
        tool = FindTargetTool()
        vision = MagicMock()
        vision.available = True
        tool.set_vision(vision)
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.is_running.return_value = False
            with patch("asyncio.run", return_value=None):
                result = tool.execute(target="bottle")
        assert "未找到" in result

    def test_found_returns_json(self):
        tool = FindTargetTool()
        vision = MagicMock()
        vision.available = True
        tool.set_vision(vision)
        mock_result = {
            "class": "bottle",
            "confidence": 0.87,
            "center": {"x": 320, "y": 240},
            "bbox": {"x": 300, "y": 200, "w": 40, "h": 80},
        }
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.is_running.return_value = False
            with patch("asyncio.run", return_value=mock_result):
                result = tool.execute(target="bottle")
        data = json.loads(result)
        assert data["found"] is True
        assert data["confidence"] == 0.87

    def test_metadata(self):
        tool = FindTargetTool()
        assert tool.name == "find_target"
        assert "target" in tool.parameters["properties"]


# ---------- Registration ----------

class TestRegistration:
    def test_register_vision_tools(self):
        registry = ToolRegistry()
        vision = MagicMock()
        register_vision_tools(registry, vision)
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert "look_around" in names
        assert "find_target" in names


# ---------- Skill Loading ----------

class TestFindObjectSkill:
    def test_skill_loaded(self):
        mgr = SkillManager()
        mgr.load()
        skills = {s.name: s for s in mgr.get_all()}
        assert "find_object" in skills

    def test_voice_triggers(self):
        mgr = SkillManager()
        mgr.load()
        triggers = mgr.get_voice_triggers()
        # At least some find_object triggers should be present
        find_triggers = [phrase for phrase, skill in triggers.items() if skill == "find_object"]
        assert len(find_triggers) >= 5
        # Check key triggers exist
        trigger_set = set(find_triggers)
        assert "帮我找" in trigger_set or "找一下" in trigger_set

    def test_skill_has_tools(self):
        mgr = SkillManager()
        mgr.load()
        skills = {s.name: s for s in mgr.get_all()}
        skill = skills["find_object"]
        assert "look_around" in skill.tools_section
        assert "find_target" in skill.tools_section
        assert "robot_api" in skill.tools_section


# ---------- Agent Shell Whitelist ----------

class TestAgentShellWhitelist:
    def test_vision_tools_in_whitelist(self):
        from askme.agent_shell.thunder_agent_shell import _AGENT_ALLOWED_TOOLS
        assert "look_around" in _AGENT_ALLOWED_TOOLS
        assert "find_target" in _AGENT_ALLOWED_TOOLS


# ---------- Brain Pipeline Routing ----------

class TestBrainPipelineRouting:
    def test_find_object_in_agent_shell_skills(self):
        """find_object should be routed to agent shell like agent_task."""
        # Read the source to verify the routing set
        from pathlib import Path
        src = Path("askme/pipeline/brain_pipeline.py").read_text(encoding="utf-8")
        assert '"find_object"' in src
        assert "_AGENT_SHELL_SKILLS" in src
