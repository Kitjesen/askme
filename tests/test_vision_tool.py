"""Tests for LookAroundTool, FindTargetTool, and register_vision_tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from askme.tools.vision_tool import FindTargetTool, LookAroundTool, register_vision_tools

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_vision(*, available: bool = True) -> MagicMock:
    v = MagicMock()
    v.available = available
    v.describe_scene = AsyncMock(return_value="走廊里有一张桌子。")
    v.describe_scene_with_question = AsyncMock(return_value="没有发现方便面。")
    v.find_object = AsyncMock(return_value=None)
    return v


# ── LookAroundTool ────────────────────────────────────────────────────────────

class TestLookAroundTool:
    def test_name(self):
        assert LookAroundTool.name == "look_around"

    def test_agent_allowed(self):
        assert LookAroundTool.agent_allowed is True

    def test_vision_unavailable_returns_message(self):
        tool = LookAroundTool()
        tool.set_vision(_make_vision(available=False))
        result = tool.execute()
        assert "视觉不可用" in result

    def test_no_vision_returns_unavailable(self):
        tool = LookAroundTool()
        result = tool.execute()
        assert "视觉不可用" in result

    def test_describe_scene_called(self):
        tool = LookAroundTool()
        vision = _make_vision()
        tool.set_vision(vision)
        result = tool.execute()
        assert "走廊" in result

    def test_empty_description_returns_fallback(self):
        tool = LookAroundTool()
        vision = _make_vision()
        vision.describe_scene = AsyncMock(return_value="")
        tool.set_vision(vision)
        result = tool.execute()
        assert "未检测到" in result

    def test_question_uses_describe_with_question(self):
        tool = LookAroundTool()
        vision = _make_vision()
        tool.set_vision(vision)
        result = tool.execute(question="有没有方便面")
        vision.describe_scene_with_question.assert_called_once_with("有没有方便面")
        assert "没有发现方便面" in result

    def test_question_empty_response_returns_fallback(self):
        tool = LookAroundTool()
        vision = _make_vision()
        vision.describe_scene_with_question = AsyncMock(return_value="")
        tool.set_vision(vision)
        result = tool.execute(question="有没有方便面")
        assert "无法回答" in result

    def test_exception_returns_error_message(self):
        tool = LookAroundTool()
        vision = _make_vision()
        vision.describe_scene = AsyncMock(side_effect=OSError("camera error"))
        tool.set_vision(vision)
        result = tool.execute()
        assert "视觉错误" in result


# ── FindTargetTool ────────────────────────────────────────────────────────────

class TestFindTargetTool:
    def test_name(self):
        assert FindTargetTool.name == "find_target"

    def test_agent_allowed(self):
        assert FindTargetTool.agent_allowed is True

    def test_empty_target_returns_error(self):
        tool = FindTargetTool()
        result = tool.execute(target="")
        assert "[错误]" in result

    def test_vision_unavailable_returns_message(self):
        tool = FindTargetTool()
        tool.set_vision(_make_vision(available=False))
        result = tool.execute(target="bottle")
        assert "视觉不可用" in result

    def test_not_found_returns_message(self):
        tool = FindTargetTool()
        vision = _make_vision()
        vision.find_object = AsyncMock(return_value=None)
        tool.set_vision(vision)
        result = tool.execute(target="cup")
        assert "未找到" in result
        assert "cup" in result

    def test_found_returns_json_with_found_true(self):
        tool = FindTargetTool()
        vision = _make_vision()
        vision.find_object = AsyncMock(return_value={
            "class_id": "bottle",
            "confidence": 0.92,
            "center": [320, 240],
            "bbox": [100, 100, 200, 200],
        })
        tool.set_vision(vision)
        result = tool.execute(target="bottle")
        data = json.loads(result)
        assert data["found"] is True
        assert data["object"] == "bottle"
        assert data["confidence"] == 0.92

    def test_found_with_distance(self):
        tool = FindTargetTool()
        vision = _make_vision()
        vision.find_object = AsyncMock(return_value={
            "class_id": "person",
            "confidence": 0.85,
            "distance_m": 2.5,
        })
        tool.set_vision(vision)
        result = tool.execute(target="person")
        data = json.loads(result)
        assert data["distance_m"] == 2.5

    def test_find_object_called_with_target(self):
        tool = FindTargetTool()
        vision = _make_vision()
        tool.set_vision(vision)
        tool.execute(target="chair")
        vision.find_object.assert_called_once_with("chair")

    def test_exception_returns_error(self):
        tool = FindTargetTool()
        vision = _make_vision()
        vision.find_object = AsyncMock(side_effect=OSError("cam error"))
        tool.set_vision(vision)
        result = tool.execute(target="bottle")
        assert "视觉错误" in result


# ── register_vision_tools ─────────────────────────────────────────────────────

class TestRegisterVisionTools:
    def test_registers_two_tools(self):
        registry = MagicMock()
        vision = _make_vision()
        register_vision_tools(registry, vision)
        assert registry.register.call_count == 2

    def test_registers_look_around_and_find_target(self):
        registry = MagicMock()
        vision = _make_vision()
        register_vision_tools(registry, vision)
        registered_names = {
            call[0][0].name for call in registry.register.call_args_list
        }
        assert "look_around" in registered_names
        assert "find_target" in registered_names
