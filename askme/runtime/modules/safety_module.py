"""SafetyModule — wraps DogSafetyClient as a declarative module.

Mirrors the safety wiring from ``assembly.py`` lines 487-490::

    dog_safety = DogSafetyClient(
        cfg.get("runtime", {}).get("dog_safety", {}),
        pulse=pulse,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.robot.safety_client import DogSafetyClient
from askme.schemas.messages import EstopState

logger = logging.getLogger(__name__)


class SafetyModule(Module):
    """Provides the DogSafetyClient to the runtime."""

    name = "safety"
    depends_on = ("pulse",)
    provides = ("dog_safety",)

    # In port: auto-wired to PulseModule (which has Out[EstopState])
    estop: In[EstopState]

    safety_client: Out[DogSafetyClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # In[EstopState] auto-wired to PulseModule by _auto_wire()
        pulse_mod = getattr(self, "estop", None)  # None if not wired or standalone
        pulse_bus = getattr(pulse_mod, "bus", None) if pulse_mod else None

        safety_cfg = cfg.get("runtime", {}).get("dog_safety", {})
        self.client = DogSafetyClient(safety_cfg, pulse=pulse_bus)
        logger.info(
            "SafetyModule: built (configured=%s, pulse=%s)",
            self.client.is_configured(),
            pulse_mod is not None,
        )

    # -- typed accessors ------------------------------------------------
    @property
    def safety_client(self) -> DogSafetyClient:
        """The DogSafetyClient instance."""
        return self.client

    def health(self) -> dict[str, Any]:
        configured = self.client.is_configured()
        estop_active = self.client.is_estop_active() if configured else False
        return {
            "status": "ok",
            "configured": configured,
            "estop_active": estop_active,
        }
