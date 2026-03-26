"""TextModule — wraps TextLoop + CommandHandler as a declarative module.

Mirrors the text loop creation from ``assembly.py`` lines 548-561::

    commands = CommandHandler(conversation=conversation, skill_manager=skill_manager)
    text_loop = TextLoop(router=router, pipeline=pipeline, ...)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.llm.client import LLMClient
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.runtime.module import In, Module, ModuleRegistry
from askme.schemas.messages import MemoryContext
from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)


class TextModule(Module):
    """Provides TextLoop and CommandHandler to the runtime."""

    name = "text"
    depends_on = ("llm", "memory", "skill", "pipeline")
    provides = ("text_io",)

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    memory_in: In[MemoryContext]
    skill_in: In[SkillDispatcher]
    pipeline_in: In[BrainPipeline]
    voice_in: In[AudioAgent]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.llm.intent_router import IntentRouter
        from askme.pipeline.commands import CommandHandler
        from askme.pipeline.text_loop import TextLoop
        from askme.robot.ota_bridge import OTABridgeMetrics
        from askme.voice.audio_agent import AudioAgent
        from askme.voice.runtime_bridge import VoiceRuntimeBridge

        llm_mod = getattr(self, "llm_in", None)
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else OTABridgeMetrics()

        mem_mod = getattr(self, "memory_in", None)
        conversation = getattr(mem_mod, "conversation", None) if mem_mod else None

        skill_mod = getattr(self, "skill_in", None)
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        pipeline_mod = getattr(self, "pipeline_in", None)
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        # Reuse voice module's audio if available, else create text-only AudioAgent
        voice_mod = getattr(self, "voice_in", None)
        if voice_mod is not None:
            audio = getattr(voice_mod, "audio", None)
            router = getattr(voice_mod, "router", None)
            voice_runtime_bridge = getattr(voice_mod, "voice_runtime_bridge", None)
        else:
            audio = AudioAgent(cfg, voice_mode=False, metrics=ota_metrics)
            voice_triggers = skill_manager.get_voice_triggers() if skill_manager else {}
            router = IntentRouter(voice_triggers=voice_triggers)
            voice_runtime_bridge = VoiceRuntimeBridge(
                cfg.get("runtime", {}).get("voice_bridge", {})
            )

        # Wire audio into pipeline if not already done by VoiceModule
        if pipeline is not None and getattr(pipeline, "_audio", None) is None:
            pipeline._audio = audio

        # CommandHandler
        self._commands = CommandHandler(
            conversation=conversation,
            skill_manager=skill_manager,
        )

        # TextLoop
        self._text_loop = TextLoop(
            router=router,
            pipeline=pipeline,
            commands=self._commands,
            conversation=conversation,
            skill_manager=skill_manager,
            audio=audio,
            voice_runtime_bridge=voice_runtime_bridge,
            dispatcher=dispatcher,
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
