"""ControlModule — wraps DogControlClient as a declarative module.

Mirrors the control client creation from ``assembly.py`` line 491::

    dog_control = DogControlClient(cfg.get("runtime", {}).get("dog_control", {}))
"""

from __future__ import annotations

import logging
from typing import Any

from askme.robot.control_client import DogControlClient
from askme.runtime.module import Module, ModuleRegistry, Out

logger = logging.getLogger(__name__)


class ControlModule(Module):
    """Provides the DogControlClient to the runtime."""

    name = "control"
    depends_on = ("pulse",)
    provides = ("dog_control",)

    control: Out[DogControlClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        control_cfg = cfg.get("runtime", {}).get("dog_control", {})
        self.client = DogControlClient(control_cfg)
        logger.info("ControlModule: built (configured=%s)", bool(self.client._base_url))

    def health(self) -> dict[str, Any]:
        configured = bool(self.client._base_url)
        return {"status": "ok", "configured": configured}
