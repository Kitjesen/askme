"""Robot control and robot service tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "MoveRobotTool": ("askme.tools.robot.move_tool", "MoveRobotTool"),
    "RobotApiTool": ("askme.tools.robot.robot_api_tool", "RobotApiTool"),
    "RobotEmergencyStopTool": (
        "askme.tools.robot.robot_tools",
        "RobotEmergencyStopTool",
    ),
    "RobotGetStateTool": ("askme.tools.robot.robot_tools", "RobotGetStateTool"),
    "RobotGrabTool": ("askme.tools.robot.robot_tools", "RobotGrabTool"),
    "RobotHomeTool": ("askme.tools.robot.robot_tools", "RobotHomeTool"),
    "RobotMoveTool": ("askme.tools.robot.robot_tools", "RobotMoveTool"),
    "RobotReleaseTool": ("askme.tools.robot.robot_tools", "RobotReleaseTool"),
    "register_move_tools": ("askme.tools.robot.move_tool", "register_move_tools"),
    "register_robot_tools": ("askme.tools.robot.robot_tools", "register_robot_tools"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve robot tools on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
