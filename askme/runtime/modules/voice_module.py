"""VoiceModule — wraps AudioAgent + IntentRouter + VoiceLoop + AddressDetector.

Canonical wiring::

    router = IntentRouter(...)
    audio = AudioAgent(cfg, ...)
    voice_loop = VoiceLoop(router=router, pipeline=pipeline, ...)
    address_detector = AddressDetector(...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
from askme.llm.client import LLMClient
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.tools.tool_registry import ToolRegistry

try:
    from askme.voice.audio_agent import AudioAgent
except ModuleNotFoundError:
    AudioAgent = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class VoiceModule(Module):
    """Provides AudioAgent, IntentRouter, VoiceLoop, and AddressDetector."""

    name = "voice"
    depends_on = ("llm", "tools", "skill", "pipeline")
    provides = ("voice", "tts", "asr")

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    skill_in: In[SkillDispatcher]
    pipeline_in: In[BrainPipeline]
    executor_in: In[ThunderAgentShell]

    # Out port
    audio_out: Out[AudioAgent]

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

        llm_mod = self.llm_in
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else OTABridgeMetrics()

        tools_mod = self.tool_registry_in
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        skill_mod = self.skill_in
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        pipeline_mod = self.pipeline_in
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        executor_mod = self.executor_in
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
            pipeline.set_audio(self._audio)
        if agent_shell is not None:
            agent_shell.set_audio(self._audio)
        if dispatcher is not None:
            dispatcher.set_audio(self._audio)

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

        self._task: asyncio.Task[None] | None = None
        logger.info("VoiceModule: built")

    async def start(self) -> None:
        """Open mic persistently, then start the VoiceLoop."""
        self._audio._mic.start()  # mic stays open across listen/speak cycles
        self._task = asyncio.create_task(self._voice_loop.run(), name="voice-loop")
        logger.info("VoiceModule: voice loop started (mic persistent)")

    async def stop(self) -> None:
        """Cancel the voice loop task and close mic."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._audio._mic.stop()
        self._audio.shutdown()
        logger.info("VoiceModule: stopped")

    # -- typed accessors ------------------------------------------------
    @property
    def audio_out(self) -> AudioAgent:
        """The AudioAgent instance (Out port)."""
        return self._audio

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
