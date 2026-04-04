"""Tests for ScanAroundTool — detection formatting, VLM question, rotation request."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.tools.scan_tool import ScanAroundTool, register_scan_tools


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool(*, detections=None, vlm_answer="场景描述") -> ScanAroundTool:
    tool = ScanAroundTool()
    vision = MagicMock()
    vision._read_daemon_detections.return_value = detections or []
    vision.describe_scene_with_question = AsyncMock(return_value=vlm_answer)
    tool.set_vision(vision)
    return tool


# ── No vision ─────────────────────────────────────────────────────────────────

class TestNoVision:
    def test_no_vision_still_returns_string(self):
        tool = ScanAroundTool()
        result = tool.execute()
        assert isinstance(result, str)

    def test_no_vision_shows_no_objects(self):
        tool = ScanAroundTool()
        result = tool.execute()
        assert "无" in result or "扫描完成" in result


# ── Detections ────────────────────────────────────────────────────────────────

class TestDetections:
    def test_empty_detections_shows_no_objects(self):
        tool = _make_tool(detections=[])
        result = tool.execute()
        assert "无 COCO 物体" in result

    def test_single_detection_no_distance(self):
        tool = _make_tool(detections=[{"class_id": "person"}])
        result = tool.execute()
        assert "person" in result

    def test_detection_with_distance(self):
        tool = _make_tool(detections=[{"class_id": "chair", "distance_m": 2.5}])
        result = tool.execute()
        assert "chair(2.5m)" in result

    def test_multiple_detections_joined(self):
        tool = _make_tool(detections=[
            {"class_id": "person"},
            {"class_id": "bottle"},
        ])
        result = tool.execute()
        assert "person" in result
        assert "bottle" in result

    def test_result_starts_with_scan_completion(self):
        tool = _make_tool(detections=[])
        result = tool.execute()
        assert result.startswith("扫描完成")


# ── VLM question ──────────────────────────────────────────────────────────────

class TestVlmQuestion:
    def test_question_triggers_vlm(self):
        tool = _make_tool(vlm_answer="没有发现方便面")
        result = tool.execute(question="有没有方便面")
        assert "没有发现方便面" in result
        tool._vision.describe_scene_with_question.assert_called_once_with("有没有方便面")

    def test_no_question_skips_vlm(self):
        tool = _make_tool()
        tool.execute(question="")
        tool._vision.describe_scene_with_question.assert_not_called()

    def test_empty_vlm_answer_not_appended(self):
        tool = _make_tool(vlm_answer="")
        result = tool.execute(question="什么都没有吗")
        assert "VLM回答" not in result

    def test_vlm_exception_shows_error(self):
        tool = ScanAroundTool()
        vision = MagicMock()
        vision._read_daemon_detections.return_value = []
        vision.describe_scene_with_question = AsyncMock(side_effect=OSError("vlm fail"))
        tool.set_vision(vision)
        result = tool.execute(question="有没有人")
        assert "VLM 错误" in result


# ── Full 360° scan ────────────────────────────────────────────────────────────

class TestFullScan:
    def test_full_false_no_rotation_request(self):
        tool = _make_tool()
        with patch("askme.tools.scan_tool.ScanAroundTool._request_rotation_scan") as mock:
            tool.execute(full=False)
        mock.assert_not_called()

    def test_full_true_calls_rotation_request(self):
        tool = _make_tool()
        with patch.object(tool, "_request_rotation_scan", return_value="360°扫描已请求") as mock:
            result = tool.execute(full=True)
        mock.assert_called_once()
        assert "360°扫描已请求" in result

    def test_rotation_unavailable_shows_message(self):
        tool = _make_tool()
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"error": "service unreachable"}):
            msg = tool._request_rotation_scan()
        assert "不可用" in msg

    def test_rotation_success_message(self):
        tool = _make_tool()
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"status": "accepted"}):
            msg = tool._request_rotation_scan()
        assert "360°旋转扫描已请求" in msg


# ── register_scan_tools ───────────────────────────────────────────────────────

class TestRegisterScanTools:
    def test_registers_tool(self):
        registry = MagicMock()
        register_scan_tools(registry)
        registry.register.assert_called_once()
        tool = registry.register.call_args[0][0]
        assert isinstance(tool, ScanAroundTool)

    def test_registers_with_vision(self):
        registry = MagicMock()
        vision = MagicMock()
        register_scan_tools(registry, vision=vision)
        tool = registry.register.call_args[0][0]
        assert tool._vision is vision


# ── Tool metadata ─────────────────────────────────────────────────────────────

class TestMetadata:
    def test_name(self):
        assert ScanAroundTool.name == "scan_around"

    def test_agent_allowed(self):
        assert ScanAroundTool.agent_allowed is True

    def test_safety_level(self):
        assert ScanAroundTool.safety_level == "normal"
