"""Mechanical arm provider adapters."""

from __future__ import annotations

from typing import Any

from askme.ports import ArmControlPort
from askme.robot.arm.arm_controller import ArmController


def build_arm_control(config: dict[str, Any] | None = None) -> ArmControlPort:
    """Build the configured mechanical-arm implementation."""
    return ArmController(config)


def get_arm_safety_defaults() -> dict[str, Any]:
    """Return the default mechanical-arm safety configuration."""

    from askme.robot.arm.safety import _DEFAULT_CONFIG

    return dict(_DEFAULT_CONFIG)


__all__ = ["build_arm_control", "get_arm_safety_defaults"]
