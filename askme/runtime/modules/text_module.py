"""TextModule -wraps TextLoop + CommandHandler as a declarative module.

Canonical wiring::

    commands = CommandHandler(conversation=conversation, skill_manager=skill_manager)
    text_loop = TextLoop(router=router, pipeline=pipeline, ...)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.llm.core.client import LLMClient
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
from askme.ports import AudioFrontendPort
from askme.runtime.core.module import In, Module, ModuleRegistry
from askme.runtime.modules.voice_stack import (
    build_runtime_voice_stack,
    runtime_voice_stack_from_module,
)
from askme.schemas.messages import MemoryContext

logger = logging.getLogger(__name__)


class TextModule(Module):
    """Provides TextLoop and CommandHandler to the runtime."""

    name = "text"
    depends_on = ("llm", "memory", "skill", "pipeline", "cognition")
    provides = ("text_io",)

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    memory_in: In[MemoryContext]
    skill_in: In[SkillDispatcher]
    pipeline_in: In[BrainPipeline]
    voice_in: In[AudioFrontendPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.pipeline.channels.commands import CommandHandler
        from askme.pipeline.channels.text_loop import TextLoop
        from askme.telemetry.ota_bridge import OTABridgeMetrics

        llm_mod = self.llm_in
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else OTABridgeMetrics()

        mem_mod = self.memory_in
        conversation = getattr(mem_mod, "conversation", None) if mem_mod else None

        skill_mod = self.skill_in
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        pipeline_mod = self.pipeline_in
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        cognition_handler = registry.get("cognition")

        # Reuse voice module's audio if available, else create text-only AudioAgent
        voice_mod = self.voice_in
        if voice_mod is not None:
            voice_stack = runtime_voice_stack_from_module(voice_mod)
        else:
            voice_stack = build_runtime_voice_stack(
                cfg,
                voice_mode=False,
                metrics=ota_metrics,
                skill_manager=skill_manager,
            )
        audio = voice_stack.audio

        # Wire audio into pipeline if not already done by VoiceModule
        if pipeline is not None and getattr(pipeline, "_audio", None) is None:
            pipeline.set_audio(audio)

        # CommandHandler
        self._commands = CommandHandler(
            conversation=conversation,
            skill_manager=skill_manager,
        )

        # TextLoop
        self._text_loop = TextLoop(
            router=voice_stack.router,
            pipeline=pipeline,
            commands=self._commands,
            conversation=conversation,
            skill_manager=skill_manager,
            audio=audio,
            voice_runtime_bridge=voice_stack.voice_gateway,
            dispatcher=dispatcher,
            cognition_handler=cognition_handler,
        )

        logger.info("TextModule: built")

    # -- typed accessors ------------------------------------------------
    @property
    def text_loop(self) -> Any:
        """The TextLoop instance."""
        return self._text_loop

    @property
    def commands(self) -> Any:
        """The CommandHandler instance."""
        return self._commands

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
