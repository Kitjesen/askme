"""Robot package.

Owner subpackages:
- ``arm``: standalone arm controller, serial bridge, safety and policies.
- ``dog``: Thunder dog runtime HTTP clients, direct commands and health.
- ``indicators``: LED state and customer-visible status indicators.
- ``telemetry``: pulse bus and pubsub adapters.

Historical imports such as ``askme.robot.arm_controller`` remain available for
compatibility. New code should import from the owner subpackage.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.robot.arm_controller": "askme.robot.arm.arm_controller",
    "askme.robot.control_client": "askme.robot.dog.control_client",
    "askme.robot.direct_commands": "askme.robot.arm.direct_commands",
    "askme.robot.led_controller": "askme.robot.indicators.led_controller",
    "askme.robot.mock_pulse": "askme.robot.telemetry.mock_pulse",
    "askme.robot.ota_bridge": "askme.robot.telemetry.ota_bridge",
    "askme.robot.policy_runner": "askme.robot.arm.policy_runner",
    "askme.robot.pubsub": "askme.robot.telemetry.pubsub",
    "askme.robot.pulse": "askme.robot.telemetry.pulse",
    "askme.robot.runtime_health": "askme.robot.dog.runtime_health",
    "askme.robot.safety": "askme.robot.arm.safety",
    "askme.robot.safety_client": "askme.robot.dog.safety_client",
    "askme.robot.serial_bridge": "askme.robot.arm.serial_bridge",
    "askme.robot.state_led_bridge": "askme.robot.indicators.state_led_bridge",
}

_LAZY_EXPORTS = {
    "ArmController": ("askme.robot.arm.arm_controller", "ArmController"),
    "PolicyRunner": ("askme.robot.arm.policy_runner", "PolicyRunner"),
    "SafetyChecker": ("askme.robot.arm.safety", "SafetyChecker"),
    "SerialBridge": ("askme.robot.arm.serial_bridge", "SerialBridge"),
}

__all__ = sorted(_LAZY_EXPORTS)


install_legacy_aliases(__name__, _LEGACY_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    legacy_module = _LEGACY_MODULE_ALIASES.get(f"{__name__}.{name}")
    if legacy_module:
        value = import_module(legacy_module)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
