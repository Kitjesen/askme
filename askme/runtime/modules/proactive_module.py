"""ProactiveModule — wraps ProactiveAgent as a declarative module.

Mirrors the proactive wiring from ``assembly.py`` lines 577-588::

    proactive = ProactiveAgent(
        vision=vision, audio=audio, episodic=episodic, llm=llm, config=cfg,
    )
    proactive.set_solve_callback(
        lambda anomaly_text: pipeline.execute_skill("solve_problem", anomaly_text)
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.pipeline.proactive_agent import ProactiveAgent
from askme.runtime.module import Module, ModuleRegistry

logger = logging.getLogger(__name__)


class ProactiveModule(Module):
    """Provides the ProactiveAgent to the runtime."""

    name = "proactive"
    depends_on = ("pipeline", "memory")
    provides = ("supervision",)

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        llm_mod = registry.get("llm")
        llm = getattr(llm_mod, "client", None) if llm_mod else None

        mem_mod = registry.get("memory")
        episodic = getattr(mem_mod, "episodic", None) if mem_mod else None

        perception_mod = registry.get("perception")
        vision = getattr(perception_mod, "vision_bridge", None) if perception_mod else None

        voice_mod = registry.get("voice")
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        self.agent = ProactiveAgent(
            vision=vision,
            audio=audio,
            episodic=episodic,
            llm=llm,
            config=cfg,
        )

        if pipeline is not None:
            self.agent.set_solve_callback(
                lambda anomaly_text: pipeline.execute_skill("solve_problem", anomaly_text)
            )

        logger.info(
            "ProactiveModule: built (enabled=%s)",
            self.agent._enabled,
        )

    async def start(self) -> None:
        if self.agent._enabled:
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(
                self.agent.run(self._stop_event),
                name="askme-proactive",
            )

    async def stop(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        task = getattr(self, "_task", None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "enabled": self.agent._enabled,
        }
