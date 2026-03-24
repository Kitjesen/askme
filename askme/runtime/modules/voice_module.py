"""VoiceModule — wraps AudioAgent + IntentRouter + VoiceLoop + AddressDetector.

Mirrors the voice wiring from ``assembly.py`` lines 466-575::

    router = IntentRouter(...)
    audio = AudioAgent(cfg, ...)
    voice_loop = VoiceLoop(router=router, pipeline=pipeline, ...)
    address_detector = AddressDetector(...)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import Module, ModuleRegistry

logger = logging.getLogger(__name__)


class VoiceModule(Module):
    """Provides AudioAgent, IntentRouter, VoiceLoop, and AddressDetector."""

    name = "voice"
    depends_on = ("pipeline", "skill")
    provides = ("voice", "tts", "asr")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.llm.intent_router import IntentRouter
        from askme.pipeline.voice_loop import VoiceLoop
        from askme.robot.ota_bridge import OTABridgeMetrics
        from askme.tools.builtin_tools import SpeakProgressTool
        from askme.tools.voice_tools import register_voice_tools
        from askme.voice.address_detector import AddressDetector
        from askme.voice.audio_agent import AudioAgent
        from askme.voice.audio_router import AudioRouter
        from askme.voice.runtime_bridge import VoiceRuntimeBridge

        llm_mod = registry.get("llm")
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else OTABridgeMetrics()

        tools_mod = registry.get("tools")
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        skill_mod = registry.get("skill")
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        executor_mod = registry.get("executor")
        agent_shell = getattr(executor_mod, "shell", None) if executor_mod else None

        # AudioRouter
        self._audio_router = AudioRouter()

        # AudioAgent
        self._audio = AudioAgent(
            cfg,
            voice_mode=True,
            metrics=ota_metrics,
            audio_router=self._audio_router,
        )

        # Register voice tools
        if tools is not None:
            register_voice_tools(tools, self._audio)
            tools.register(SpeakProgressTool(self._audio))

        # Cross-link: pipeline, agent_shell, and dispatcher need the audio agent
        # for TTS playback and voice state queries.  VoiceModule owns the AudioAgent
        # so it is responsible for injecting it into objects built by earlier modules.
        if pipeline is not None:
            pipeline._audio = self._audio
        if agent_shell is not None:
            agent_shell._audio = self._audio
        if dispatcher is not None:
            dispatcher._audio = self._audio

        # VoiceRuntimeBridge
        self._voice_runtime_bridge = VoiceRuntimeBridge(
            cfg.get("runtime", {}).get("voice_bridge", {})
        )

        # IntentRouter
        voice_triggers = skill_manager.get_voice_triggers() if skill_manager else {}
        self._router = IntentRouter(voice_triggers=voice_triggers)

        # VoiceLoop
        self._voice_loop = VoiceLoop(
            router=self._router,
            pipeline=pipeline,
            audio=self._audio,
            voice_runtime_bridge=self._voice_runtime_bridge,
            dispatcher=dispatcher,
            audio_router=self._audio_router,
        )

        # AddressDetector
        self._address_detector = AddressDetector(
            cfg.get("voice", {}).get("address_detection", {})
        )
        self._voice_loop.set_address_detector(self._address_detector)

        logger.info("VoiceModule: built")

    # -- typed accessors ------------------------------------------------
    @property
    def audio(self) -> Any:
        """The AudioAgent instance."""
        return self._audio

    @property
    def voice_loop(self) -> Any:
        """The VoiceLoop instance."""
        return self._voice_loop

    @property
    def router(self) -> Any:
        """The IntentRouter instance."""
        return self._router

    @property
    def voice_runtime_bridge(self) -> Any:
        """The VoiceRuntimeBridge instance."""
        return self._voice_runtime_bridge

    @property
    def address_detector(self) -> Any:
        """The AddressDetector instance."""
        return self._address_detector

    @property
    def audio_router(self) -> Any:
        """The AudioRouter instance."""
        return self._audio_router

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "voice_mode": True,
        }
