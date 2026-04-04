"""Tests for robot_tools — arm controller integration, metadata, _run_coro."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.tools.robot_tools import (
    RobotEmergencyStopTool,
    RobotGetStateTool,
    RobotGrabTool,
    RobotHomeTool,
    RobotMoveTool,
    RobotReleaseTool,
    _ROBOT_TOOL_CLASSES,
    _run_coro,
    register_robot_tools,
)


# ── _run_coro helper ──────────────────────────────────────────────────────────

class TestRunCoro:
    def test_runs_coroutine_no_loop(self):
        async def simple():
            return 42

        result = _run_coro(simple())
        assert result == 42

    async def test_runs_coroutine_inside_loop(self):
        async def simple():
            return "hello"

        # When called from inside an async context, uses ThreadPoolExecutor
        result = _run_coro(simple())
        assert result == "hello"


# ── RobotMoveTool ─────────────────────────────────────────────────────────────

class TestRobotMoveTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.arm.execute = AsyncMock(return_value={"status": "ok"})
        self.tool = RobotMoveTool(self.arm)

    def test_name(self):
        assert RobotMoveTool.name == "robot_move"

    def test_safety_level_dangerous(self):
        assert RobotMoveTool.safety_level == "dangerous"

    def test_execute_returns_json_string(self):
        result = self.tool.execute(x=100.0, y=200.0, z=300.0)
        data = json.loads(result)
        assert data["status"] == "ok"

    def test_execute_calls_arm_move(self):
        self.tool.execute(x=10.0, y=20.0, z=30.0)
        self.arm.execute.assert_called_once_with("move", params={"x": 10.0, "y": 20.0, "z": 30.0})

    def test_parameters_have_required_xyz(self):
        required = RobotMoveTool.parameters["required"]
        assert "x" in required and "y" in required and "z" in required


# ── RobotGrabTool ─────────────────────────────────────────────────────────────

class TestRobotGrabTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.arm.execute = AsyncMock(return_value={"grabbed": True})
        self.tool = RobotGrabTool(self.arm)

    def test_name(self):
        assert RobotGrabTool.name == "robot_grab"

    def test_execute_calls_grab(self):
        self.tool.execute()
        self.arm.execute.assert_called_once_with("grab")

    def test_execute_returns_json(self):
        result = self.tool.execute()
        data = json.loads(result)
        assert data["grabbed"] is True


# ── RobotReleaseTool ──────────────────────────────────────────────────────────

class TestRobotReleaseTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.arm.execute = AsyncMock(return_value={"released": True})
        self.tool = RobotReleaseTool(self.arm)

    def test_name(self):
        assert RobotReleaseTool.name == "robot_release"

    def test_execute_calls_release(self):
        self.tool.execute()
        self.arm.execute.assert_called_once_with("release")


# ── RobotHomeTool ─────────────────────────────────────────────────────────────

class TestRobotHomeTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.arm.execute = AsyncMock(return_value={"position": "home"})
        self.tool = RobotHomeTool(self.arm)

    def test_name(self):
        assert RobotHomeTool.name == "robot_home"

    def test_execute_calls_home(self):
        self.tool.execute()
        self.arm.execute.assert_called_once_with("home")

    def test_execute_returns_json(self):
        result = self.tool.execute()
        data = json.loads(result)
        assert data["position"] == "home"


# ── RobotGetStateTool ─────────────────────────────────────────────────────────

class TestRobotGetStateTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.arm.get_state.return_value = {"joints": [0, 0, 0], "position": {"x": 0, "y": 0, "z": 0}}
        self.tool = RobotGetStateTool(self.arm)

    def test_name(self):
        assert RobotGetStateTool.name == "robot_get_state"

    def test_safety_level_normal(self):
        assert RobotGetStateTool.safety_level == "normal"

    def test_execute_calls_get_state(self):
        self.tool.execute()
        self.arm.get_state.assert_called_once()

    def test_execute_returns_json(self):
        result = self.tool.execute()
        data = json.loads(result)
        assert "joints" in data


# ── RobotEmergencyStopTool ────────────────────────────────────────────────────

class TestRobotEmergencyStopTool:
    def setup_method(self):
        self.arm = MagicMock()
        self.tool = RobotEmergencyStopTool(self.arm)

    def test_name(self):
        assert RobotEmergencyStopTool.name == "robot_emergency_stop"

    def test_safety_level_critical(self):
        assert RobotEmergencyStopTool.safety_level == "critical"

    def test_execute_calls_emergency_stop(self):
        self.tool.execute()
        self.arm.emergency_stop.assert_called_once()

    def test_execute_returns_confirmation_json(self):
        result = self.tool.execute()
        data = json.loads(result)
        assert data["status"] == "emergency_stop_activated"


# ── register_robot_tools ──────────────────────────────────────────────────────

class TestRegisterRobotTools:
    def test_registers_all_tool_classes(self):
        registry = MagicMock()
        arm = MagicMock()
        register_robot_tools(registry, arm)
        assert registry.register.call_count == len(_ROBOT_TOOL_CLASSES)

    def test_each_registered_with_arm_controller(self):
        registry = MagicMock()
        arm = MagicMock()
        register_robot_tools(registry, arm)
        for call in registry.register.call_args_list:
            tool = call[0][0]
            assert tool._arm is arm
