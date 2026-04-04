"""Tests for MoveRobotTool — execute routing, _go_to, _dispatch_control, _call_runtime_api."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from askme.tools.move_tool import MoveRobotTool, _call_runtime_api, register_move_tools


# ── _call_runtime_api ─────────────────────────────────────────────────────────

class TestCallRuntimeApi:
    def test_unknown_service_returns_error(self):
        result = _call_runtime_api("unknown_svc", "GET", "/path")
        assert "error" in result
        assert "unknown service" in result["error"]

    def test_url_error_returns_friendly_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = _call_runtime_api("control", "GET", "/path")
        assert "error" in result
        assert "服务不可达" in result["error"]

    def test_generic_exception_returns_error(self):
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = _call_runtime_api("nav", "GET", "/path")
        assert "error" in result

    def test_env_url_override(self, monkeypatch):
        monkeypatch.setenv("DOG_NAV_SERVICE_URL", "http://myhost:9999")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = _call_runtime_api("nav", "POST", "/api/test", {"key": "val"})
        call_args = mock_open.call_args[0][0]
        assert "myhost:9999" in call_args.full_url


# ── MoveRobotTool.execute routing ─────────────────────────────────────────────

class TestMoveRobotToolExecute:
    def setup_method(self):
        self.tool = MoveRobotTool()

    def test_unknown_action_returns_error(self):
        result = self.tool.execute(action="fly")
        assert "[错误]" in result
        assert "fly" in result

    def test_empty_action_returns_error(self):
        result = self.tool.execute(action="")
        assert "[错误]" in result

    def test_go_to_calls_go_to(self):
        with patch.object(self.tool, "_go_to", return_value="nav ok") as mock:
            result = self.tool.execute(action="go_to", target="厨房")
        mock.assert_called_once_with("厨房")
        assert result == "nav ok"

    def test_rotate_calls_dispatch_control(self):
        with patch.object(self.tool, "_dispatch_control", return_value="rotated") as mock:
            result = self.tool.execute(action="rotate", angle=90)
        mock.assert_called_once_with("rotate", {"angle_deg": 90})

    def test_forward_calls_dispatch_control(self):
        with patch.object(self.tool, "_dispatch_control", return_value="moved") as mock:
            result = self.tool.execute(action="forward", distance=1.5)
        mock.assert_called_once_with("walk_forward", {"distance_m": 1.5})

    def test_stop_calls_dispatch_control(self):
        with patch.object(self.tool, "_dispatch_control", return_value="stopped") as mock:
            result = self.tool.execute(action="stop")
        mock.assert_called_once_with("stop")


# ── _go_to ────────────────────────────────────────────────────────────────────

class TestGoTo:
    def setup_method(self):
        self.tool = MoveRobotTool()

    def test_empty_target_returns_error(self):
        result = self.tool._go_to("")
        assert "[错误]" in result

    def test_service_unavailable_returns_friendly_message(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"error": "服务不可达 (nav:5090): refused"}):
            result = self.tool._go_to("厨房")
        assert "[导航不可用]" in result
        assert "厨房" in result

    def test_other_error_returns_nav_error(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"error": "timeout"}):
            result = self.tool._go_to("走廊")
        assert "[导航错误]" in result

    def test_success_returns_task_id(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"task_id": "abc123", "status": "accepted"}):
            result = self.tool._go_to("仓库")
        assert "仓库" in result
        assert "abc123" in result

    def test_success_with_id_field(self):
        # Some APIs return 'id' instead of 'task_id'
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"id": "xyz789"}):
            result = self.tool._go_to("大厅")
        assert "xyz789" in result


# ── _dispatch_control ─────────────────────────────────────────────────────────

class TestDispatchControl:
    def setup_method(self):
        self.tool = MoveRobotTool()

    def test_service_unavailable_returns_friendly_message(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"error": "服务不可达 (control:5080): refused"}):
            result = self.tool._dispatch_control("rotate", {"angle_deg": 90})
        assert "[控制不可用]" in result

    def test_other_error_returns_control_error(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"error": "permission denied"}):
            result = self.tool._dispatch_control("stop")
        assert "[控制错误]" in result

    def test_success_returns_capability_name(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"status": "ok"}):
            result = self.tool._dispatch_control("rotate")
        assert "rotate" in result

    def test_no_params_defaults_to_empty_dict(self):
        with patch("askme.tools.move_tool._call_runtime_api",
                   return_value={"status": "ok"}) as mock:
            self.tool._dispatch_control("stop")
        call_body = mock.call_args[0][3]
        assert call_body["parameters"] == {}


# ── Tool metadata ─────────────────────────────────────────────────────────────

class TestToolMetadata:
    def test_name(self):
        assert MoveRobotTool.name == "move_robot"

    def test_description_not_empty(self):
        assert MoveRobotTool.description != ""

    def test_parameters_schema_has_action(self):
        params = MoveRobotTool.parameters
        assert "action" in params["properties"]
        assert "action" in params["required"]

    def test_agent_allowed(self):
        assert MoveRobotTool.agent_allowed is True


# ── register_move_tools ───────────────────────────────────────────────────────

class TestRegisterMoveTools:
    def test_registers_tool(self):
        registry = MagicMock()
        register_move_tools(registry)
        registry.register.assert_called_once()
        registered = registry.register.call_args[0][0]
        assert isinstance(registered, MoveRobotTool)
