"""SafetyModule - wraps SafetyPort as a declarative module.

Canonical wiring::

    dog_safety = build_safety(
        cfg.get("runtime", {}).get("dog_safety", {}),
        pulse=pulse,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from askme.ports import SafetyPort
from askme.providers import build_safety
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
from askme.schemas.messages import EstopState

logger = logging.getLogger(__name__)


class SafetyModule(Module):
    """Provides the robot-safety port to the runtime."""

    name = "safety"
    depends_on = ("pulse",)
    provides = ("dog_safety",)

    # In port: auto-wired to PulseModule (which has Out[EstopState])
    estop: In[EstopState]

    safety_client: Out[SafetyPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # In[EstopState] auto-wired to PulseModule by _auto_wire()
        pulse_mod = self.estop
        pulse_bus = getattr(pulse_mod, "bus", None) if pulse_mod else None

        safety_cfg = cfg.get("runtime", {}).get("dog_safety", {})
        self.client = build_safety(safety_cfg, pulse=pulse_bus)
        logger.info(
            "SafetyModule: built (configured=%s, pulse=%s)",
            self.client.is_configured(),
            pulse_mod is not None,
        )

    # -- typed accessors ------------------------------------------------
    @property
    def safety_client(self) -> SafetyPort:
        """The robot-safety port instance."""
        return self.client

    def health(self) -> dict[str, Any]:
        configured = self.client.is_configured()
        estop_active = self.client.is_estop_active() if configured else False
        return {
            "status": "ok",
            "configured": configured,
            "estop_active": estop_active,
        }
