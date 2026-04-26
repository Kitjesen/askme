"""LEDModule — wraps StateLedBridge + LedController as a declarative module.

Canonical wiring::

    led_controller = HttpLedController(led_base_url) if led_base_url else NullLedController()
    led_bridge = StateLedBridge(audio=audio, dispatcher=dispatcher, safety=..., led=led_controller)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.robot.safety_client import DogSafetyClient
from askme.runtime.module import In, Module, ModuleRegistry

try:
    from askme.voice.audio_agent import AudioAgent
except ModuleNotFoundError:
    AudioAgent = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class LEDModule(Module):
    """Provides StateLedBridge and LedController to the runtime."""

    name = "led"
    depends_on = ("voice", "skill", "safety")
    provides = ("indicators",)

    voice_in: In[AudioAgent]
    skill_in: In[SkillDispatcher]
    safety_in: In[DogSafetyClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.robot.led_controller import HttpLedController, NullLedController
        from askme.robot.state_led_bridge import StateLedBridge

        led_cfg = cfg.get("led", {})
        led_base_url = led_cfg.get("base_url", "").strip()

        self.led_controller = (
            HttpLedController(led_base_url)
            if led_base_url
            else NullLedController()
        )

        voice_mod = self.voice_in
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        skill_mod = self.skill_in
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        safety_mod = self.safety_in
        dog_safety = getattr(safety_mod, "client", None) if safety_mod else None

        self.led_bridge = StateLedBridge(
            audio=audio,
            dispatcher=dispatcher,
            safety=dog_safety,
            led=self.led_controller,
        )

        logger.info(
            "LEDModule: built (controller=%s)",
            f"http({led_base_url})" if led_base_url else "null",
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
