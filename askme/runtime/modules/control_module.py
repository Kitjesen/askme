"""ControlModule - wraps RobotControlPort as a declarative module.

Canonical wiring::

    dog_control = build_robot_control(cfg.get("runtime", {}).get("dog_control", {}))
"""

from __future__ import annotations

import logging
from typing import Any

from askme.ports import RobotControlPort
from askme.providers import build_robot_control
from askme.runtime.core.module import Module, ModuleRegistry, Out

logger = logging.getLogger(__name__)


class ControlModule(Module):
    """Provides the robot-control port to the runtime."""

    name = "control"
    depends_on = ("pulse",)
    provides = ("dog_control",)

    control: Out[RobotControlPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        control_cfg = cfg.get("runtime", {}).get("dog_control", {})
        self.client = build_robot_control(control_cfg)
        tools_mod = registry.get("tools")
        bind_robot_control = getattr(tools_mod, "bind_robot_control_client", None)
        if callable(bind_robot_control):
            bind_robot_control(self.client)
        logger.info("ControlModule: built (configured=%s)", self.client.is_configured())

    # -- typed accessors ------------------------------------------------
    @property
    def control_client(self) -> RobotControlPort:
        """The robot-control port instance."""
        return self.client

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "configured": self.client.is_configured()}
