"""Robot control provider adapters."""

from __future__ import annotations

from typing import Any

from askme.ports import RobotControlPort
from askme.robot.dog.control_client import DogControlClient


def build_robot_control(config: dict[str, Any] | None = None) -> RobotControlPort:
    """Build the configured robot-control implementation."""
    return DogControlClient(config)


__all__ = ["build_robot_control"]
