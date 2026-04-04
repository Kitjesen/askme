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

from askme.llm.client import LLMClient
from askme.perception.vision_bridge import VisionBridge
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.proactive_agent import ProactiveAgent
from askme.runtime.module import In, Module, ModuleRegistry
from askme.schemas.messages import MemoryContext
try:
    from askme.voice.audio_agent import AudioAgent
except ModuleNotFoundError:
    AudioAgent = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class ProactiveModule(Module):
    """Provides the ProactiveAgent to the runtime."""

    name = "proactive"
    depends_on = ("llm", "memory", "pipeline")
    provides = ("supervision",)

    llm_in: In[LLMClient]
    memory_in: In[MemoryContext]
    perception_in: In[VisionBridge]
    voice_in: In[AudioAgent]
    pipeline_in: In[BrainPipeline]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # Get dependencies via In[T] ports (auto-wired; getattr for standalone test compat)
        llm_mod = getattr(self, "llm_in", None)
        llm = getattr(llm_mod, "client", None) if llm_mod else None

        mem_mod = getattr(self, "memory_in", None)
        episodic = getattr(mem_mod, "episodic", None) if mem_mod else None

        perception_mod = getattr(self, "perception_in", None)
        vision = getattr(perception_mod, "vision_bridge", None) if perception_mod else None

        voice_mod = getattr(self, "voice_in", None)
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        pipeline_mod = getattr(self, "pipeline_in", None)
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
