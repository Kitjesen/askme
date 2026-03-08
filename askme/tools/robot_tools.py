"""
Robot control tools for askme.

These tools wrap an ArmController instance and expose robot operations
as LLM-callable tools. Only registered when robot mode is enabled.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any

from .tool_registry import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from ..robot.arm_controller import ArmController


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from synchronous code safely.

    Works both when an event loop is already running (e.g. inside
    BrainPipeline) and when there is no loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Event loop is running — offload to a thread
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


class RobotMoveTool(BaseTool):
    """Move the robot arm to a target position (x, y, z)."""

    name = "robot_move"
    description = "移动机械臂到指定位置 (x, y, z)，单位为毫米"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "X 坐标 (mm)"},
            "y": {"type": "number", "description": "Y 坐标 (mm)"},
            "z": {"type": "number", "description": "Z 坐标 (mm)"},
        },
        "required": ["x", "y", "z"],
    }
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, *, x: float = 0, y: float = 0, z: float = 0, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("move", params={"x": x, "y": y, "z": z}))
        return json.dumps(result, ensure_ascii=False)


class RobotGrabTool(BaseTool):
    """Close the gripper to grab an object."""

    name = "robot_grab"
    description = "关闭夹爪抓取物体"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("grab"))
        return json.dumps(result, ensure_ascii=False)


class RobotReleaseTool(BaseTool):
    """Open the gripper to release an object."""

    name = "robot_release"
    description = "打开夹爪释放物体"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("release"))
        return json.dumps(result, ensure_ascii=False)


class RobotHomeTool(BaseTool):
    """Move the robot arm to its home (rest) position."""

    name = "robot_home"
    description = "将机械臂移动到初始位置"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("home"))
        return json.dumps(result, ensure_ascii=False)


class RobotGetStateTool(BaseTool):
    """Get the current state of the robot (joint angles, position)."""

    name = "robot_get_state"
    description = "获取机械臂当前状态（关节角度、位置等）"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        state = self._arm.get_state()
        return json.dumps(state, ensure_ascii=False)


class RobotEmergencyStopTool(BaseTool):
    """Immediately stop all robot motion (emergency stop)."""

    name = "robot_emergency_stop"
    description = "紧急停止机械臂所有运动"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def __init__(self, arm_controller: ArmController) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        self._arm.emergency_stop()
        return '{"status": "emergency_stop_activated", "message": "机械臂已紧急停止"}'


# ── Convenience registration ────────────────────────────────────

_ROBOT_TOOL_CLASSES: list[type[BaseTool]] = [
    RobotMoveTool,
    RobotGrabTool,
    RobotReleaseTool,
    RobotHomeTool,
    RobotGetStateTool,
    RobotEmergencyStopTool,
]


def register_robot_tools(registry: ToolRegistry, arm_controller: ArmController) -> None:
    """Instantiate and register all robot tools, injecting the arm controller."""
    for tool_cls in _ROBOT_TOOL_CLASSES:
        registry.register(tool_cls(arm_controller))  # type: ignore[call-arg]
