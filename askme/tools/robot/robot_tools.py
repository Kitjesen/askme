"""Robot arm control tools.

These tools depend on ``ArmControlPort``. Concrete arm implementations and
transport details stay behind the provider layer.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any

from askme.tools.core.tool_registry import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from askme.ports import ArmControlPort


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from synchronous code safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


class RobotMoveTool(BaseTool):
    """Move the robot arm to a target position."""

    name = "robot_move"
    description = "Move the mechanical arm to an x/y/z position in millimetres."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "X coordinate in millimetres"},
            "y": {"type": "number", "description": "Y coordinate in millimetres"},
            "z": {"type": "number", "description": "Z coordinate in millimetres"},
        },
        "required": ["x", "y", "z"],
    }
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, *, x: float = 0, y: float = 0, z: float = 0, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("move", params={"x": x, "y": y, "z": z}))
        return json.dumps(result, ensure_ascii=False)


class RobotGrabTool(BaseTool):
    """Close the gripper to grab an object."""

    name = "robot_grab"
    description = "Close the mechanical arm gripper."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("grab"))
        return json.dumps(result, ensure_ascii=False)


class RobotReleaseTool(BaseTool):
    """Open the gripper to release an object."""

    name = "robot_release"
    description = "Open the mechanical arm gripper."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("release"))
        return json.dumps(result, ensure_ascii=False)


class RobotHomeTool(BaseTool):
    """Move the robot arm to its home position."""

    name = "robot_home"
    description = "Move the mechanical arm to its home position."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        result = _run_coro(self._arm.execute("home"))
        return json.dumps(result, ensure_ascii=False)


class RobotGetStateTool(BaseTool):
    """Get the current state of the robot arm."""

    name = "robot_get_state"
    description = "Get mechanical arm state, including connection and joint data."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        state = self._arm.get_state()
        return json.dumps(state, ensure_ascii=False)


class RobotEmergencyStopTool(BaseTool):
    """Immediately stop all robot arm motion."""

    name = "robot_emergency_stop"
    description = "Immediately stop all mechanical arm motion."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def __init__(self, arm_controller: ArmControlPort) -> None:
        self._arm = arm_controller

    def execute(self, **kwargs: Any) -> str:
        self._arm.emergency_stop()
        return json.dumps(
            {
                "status": "emergency_stop_activated",
                "message": "Mechanical arm emergency stop activated.",
            },
            ensure_ascii=False,
        )


_ROBOT_TOOL_CLASSES: list[type[BaseTool]] = [
    RobotMoveTool,
    RobotGrabTool,
    RobotReleaseTool,
    RobotHomeTool,
    RobotGetStateTool,
    RobotEmergencyStopTool,
]


def register_robot_tools(registry: ToolRegistry, arm_controller: ArmControlPort) -> None:
    """Instantiate and register all robot-arm tools."""
    for tool_cls in _ROBOT_TOOL_CLASSES:
        registry.register(tool_cls(arm_controller))  # type: ignore[call-arg]
