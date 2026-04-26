"""PulseModule — wraps the Pulse DDS bus as a declarative module.

Canonical wiring::

    pulse = Pulse(cfg.get("pulse", {}))
"""

from __future__ import annotations

import logging
from typing import Any

from askme.robot.pulse import Pulse
from askme.runtime.module import Module, ModuleRegistry, Out
from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)

logger = logging.getLogger(__name__)


class PulseModule(Module):
    """Provides the Pulse DDS data bus to the runtime."""

    name = "pulse"
    provides = ("telemetry", "dds")

    detections: Out[DetectionFrame]
    estop: Out[EstopState]
    joints: Out[JointStateSnapshot]
    imu: Out[ImuSnapshot]
    cms_state: Out[CmsState]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        pulse_cfg = cfg.get("pulse", {})
        self._bus = Pulse(pulse_cfg)
        logger.info("PulseModule: built (enabled=%s)", self._bus.available)

    # -- typed accessors ------------------------------------------------
    @property
    def bus(self) -> Pulse:
        """The Pulse DDS data bus instance."""
        return self._bus

    async def start(self) -> None:
        await self._bus.start()

    async def stop(self) -> None:
        await self._bus.stop()

    def health(self) -> dict[str, Any]:
        return self._bus.health()
