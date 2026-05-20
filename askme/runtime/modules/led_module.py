"""LEDModule - wraps the configured status LED provider as a runtime module.

Canonical wiring::

    led_controller, led_bridge = build_status_led(...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
from askme.ports import AudioFrontendPort, SafetyPort
from askme.providers import build_status_led
from askme.runtime.core.module import In, Module, ModuleRegistry

logger = logging.getLogger(__name__)


class LEDModule(Module):
    """Provides status LED bridge and controller to the runtime."""

    name = "led"
    depends_on = ("voice", "skill", "safety")
    provides = ("indicators",)

    voice_in: In[AudioFrontendPort]
    skill_in: In[SkillDispatcher]
    safety_in: In[SafetyPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        led_cfg = cfg.get("led", {})

        voice_mod = self.voice_in
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        skill_mod = self.skill_in
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        safety_mod = self.safety_in
        dog_safety = getattr(safety_mod, "client", None) if safety_mod else None

        self.led_controller, self.led_bridge = build_status_led(
            led_cfg,
            audio=audio,
            dispatcher=dispatcher,
            safety=dog_safety,
        )

        logger.info(
            "LEDModule: built (controller=%s)",
            type(self.led_controller).__name__,
        )

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self.led_bridge.run(), name="askme-led-bridge"
        )

    async def stop(self) -> None:
        task = getattr(self, "_task", None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
