"""Status LED provider adapters."""

from __future__ import annotations

from typing import Any

from askme.ports import LedBridgePort, LedControllerPort, SafetyPort
from askme.robot.indicators.led_controller import HttpLedController, NullLedController
from askme.robot.indicators.state_led_bridge import StateLedBridge


def build_led_controller(config: dict[str, Any] | None = None) -> LedControllerPort:
    """Build the configured LED controller implementation."""
    cfg = config if isinstance(config, dict) else {}
    base_url = str(cfg.get("base_url") or "").strip()
    if base_url:
        return HttpLedController(base_url)
    return NullLedController()


def build_status_led(
    config: dict[str, Any] | None = None,
    *,
    audio: Any = None,
    dispatcher: Any = None,
    safety: SafetyPort | None = None,
) -> tuple[LedControllerPort, LedBridgePort]:
    """Build the LED controller and state bridge as one adapter stack."""
    controller = build_led_controller(config)
    bridge = StateLedBridge(
        audio=audio,
        dispatcher=dispatcher,
        safety=safety,
        led=controller,
    )
    return controller, bridge


__all__ = ["StateLedBridge", "build_led_controller", "build_status_led"]
