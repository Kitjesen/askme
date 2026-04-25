"""PerceptionModule — wraps VisionBridge + ChangeDetector as a declarative module.

Canonical wiring::

    vision = VisionBridge()
    change_detector = ChangeDetector(config=cfg, pulse=pulse)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.perception.vision_bridge import VisionBridge
from askme.schemas.messages import DetectionFrame

logger = logging.getLogger(__name__)


class PerceptionModule(Module):
    """Provides VisionBridge and ChangeDetector to the runtime."""

    name = "perception"
    depends_on = ("pulse",)
    provides = ("vision", "change_monitor")

    # In port: auto-wired to PulseModule (which has Out[DetectionFrame])
    detections: In[DetectionFrame]

    vision: Out[VisionBridge]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.vision_bridge = VisionBridge()

        self.change_detector = None
        # In[DetectionFrame] auto-wired to PulseModule by _auto_wire()
        pulse_mod = self.detections
        pulse_bus = getattr(pulse_mod, "bus", None) if pulse_mod else None

        try:
            from askme.perception.change_detector import ChangeDetector
            self.change_detector = ChangeDetector(config=cfg, pulse=pulse_bus)
        except Exception as exc:
            logger.debug("ChangeDetector not available: %s", exc)

        logger.info(
            "PerceptionModule: built (change_detector=%s, pulse=%s)",
            "enabled" if self.change_detector else "disabled",
            pulse_mod is not None,
        )

    # -- typed accessors ------------------------------------------------
    @property
    def vision(self) -> VisionBridge:  # type: ignore[override]
        """The VisionBridge instance."""
        return self.vision_bridge

    async def start(self) -> None:
        if self.change_detector is not None:
            import asyncio
            self._cd_task = asyncio.create_task(
                self.change_detector.run(asyncio.Event()),
                name="askme-change-detector",
            )

    async def stop(self) -> None:
        task = getattr(self, "_cd_task", None)
        if task is not None and not task.done():
            task.cancel()
            import asyncio
            await asyncio.gather(task, return_exceptions=True)

    def health(self) -> dict[str, Any]:
        cd_active = (
            self.change_detector.is_active
            if self.change_detector is not None
            else False
        )
        return {"status": "ok", "change_detector_active": cd_active}
