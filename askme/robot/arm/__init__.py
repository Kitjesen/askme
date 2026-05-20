"""Standalone arm controller and local arm safety modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ArmController": ("askme.robot.arm.arm_controller", "ArmController"),
    "PolicyRunner": ("askme.robot.arm.policy_runner", "PolicyRunner"),
    "SafetyChecker": ("askme.robot.arm.safety", "SafetyChecker"),
    "SerialBridge": ("askme.robot.arm.serial_bridge", "SerialBridge"),
    "get_command": ("askme.robot.arm.direct_commands", "get_command"),
    "list_commands": ("askme.robot.arm.direct_commands", "list_commands"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve arm controller contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
