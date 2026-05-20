"""PulseModule wraps the configured telemetry bus as a declarative module.

Canonical wiring::

    bus = build_bus(cfg.get("pulse", {}))
"""

from __future__ import annotations

import logging
from typing import Any

from askme.interfaces.bus import BusBackend
from askme.providers import build_bus
from askme.runtime.core.module import Module, ModuleRegistry, Out
from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)

logger = logging.getLogger(__name__)


class PulseModule(Module):
    """Provides the telemetry data bus to the runtime."""

    name = "pulse"
    provides = ("telemetry", "dds")

    detections: Out[DetectionFrame]
    estop: Out[EstopState]
    joints: Out[JointStateSnapshot]
    imu: Out[ImuSnapshot]
    cms_state: Out[CmsState]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        pulse_cfg = cfg.get("pulse", {})
        self._bus = build_bus(pulse_cfg)
        logger.info(
            "PulseModule: built (backend=%s, available=%s)",
            pulse_cfg.get("backend", "pulse"),
            getattr(self._bus, "available", True),
        )

    @property
    def bus(self) -> BusBackend:
        """The telemetry data bus instance."""

        return self._bus

    async def start(self) -> None:
        await self._bus.start()

    async def stop(self) -> None:
        await self._bus.stop()

    def health(self) -> dict[str, Any]:
        return self._bus.health()
