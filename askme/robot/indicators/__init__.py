"""LED and customer-visible robot indicator modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "HttpLedController": ("askme.robot.indicators.led_controller", "HttpLedController"),
    "LedController": ("askme.robot.indicators.led_controller", "LedController"),
    "LedStateKind": ("askme.robot.indicators.led_controller", "LedStateKind"),
    "NullLedController": ("askme.robot.indicators.led_controller", "NullLedController"),
    "StateLedBridge": ("askme.robot.indicators.state_led_bridge", "StateLedBridge"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve robot indicator contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
