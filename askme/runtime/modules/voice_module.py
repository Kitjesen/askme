"""VoiceModule - wires the voice gateway, interaction router, and audio ports.

Canonical wiring::

    router = IntentRouter(...)
    voice_stack = build_audio_frontend(cfg, ...)
    voice_loop = VoiceLoop(router=router, pipeline=pipeline, ...)
    address_detector = AddressDetector(...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.agent_shell import AgentShell
from askme.llm.core.client import LLMClient
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
from askme.ports import AudioFrontendPort, AudioRouterPort
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
from askme.runtime.modules.voice_stack import build_runtime_voice_stack
from askme.tools.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class VoiceModule(Module):
    """Provides audio frontend, voice gateway, router, loop, and gates."""

    name = "voice"
    depends_on = ("llm", "tools", "skill", "pipeline")
    provides = ("voice", "tts", "asr")

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    skill_in: In[SkillDispatcher]
    pipeline_in: In[BrainPipeline]
    executor_in: In[AgentShell]

    # Out port
    audio_out: Out[AudioFrontendPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.pipeline.channels.voice_loop import VoiceLoop
        from askme.robot_interaction import (
            RobotInteractionService,
        )
        from askme.telemetry.ota_bridge import OTABridgeMetrics
        from askme.tools.core.builtin_tools import SpeakProgressTool
        from askme.tools.voice.voice_tools import register_voice_tools

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

        voice_stack = build_runtime_voice_stack(
            cfg,
            voice_mode=True,
            metrics=ota_metrics,
            skill_manager=skill_manager,
        )
        self._audio = voice_stack.audio
        self._audio_router = voice_stack.audio_router
        self._asr_provider = voice_stack.asr_provider
        self._tts_provider = voice_stack.tts_provider
        self._voice_runtime_bridge = voice_stack.voice_runtime_bridge
        self._voice_gateway = voice_stack.voice_gateway
        self._router = voice_stack.router

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

        self._interaction_service = RobotInteractionService(self._router)

        # VoiceLoop
        self._voice_loop = VoiceLoop(
            router=self._router,
            pipeline=pipeline,
            audio=self._audio,
            voice_runtime_bridge=self._voice_gateway,
            dispatcher=dispatcher,
            audio_router=self._audio_router,
        )

        # Runtime voice stack owns gate construction; VoiceModule injects into VoiceLoop.
        self._address_detector = voice_stack.address_detector
        self._voice_loop.set_address_detector(self._address_detector)
        self._interaction_gate = voice_stack.interaction_gate
        self._voice_loop.set_interaction_gate(self._interaction_gate)
        self._interaction_perception_provider = _build_interaction_perception_provider(
            registry
        )
        self._voice_loop.set_interaction_perception_provider(
            self._interaction_perception_provider
        )

        self._task: asyncio.Task[None] | None = None
        logger.info("VoiceModule: built")

    async def start(self) -> None:
        """Open mic persistently, then start the VoiceLoop."""
        self._audio.start_input()  # mic stays open across listen/speak cycles
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
        self._audio.stop_input()
        self._audio.shutdown()
        logger.info("VoiceModule: stopped")

    # -- typed accessors ------------------------------------------------
    @property
    def audio_out(self) -> AudioFrontendPort:
        """The audio frontend instance (Out port)."""
        return self._audio

    @property
    def audio(self) -> AudioFrontendPort:
        """The audio frontend instance."""
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
    def voice_gateway(self) -> Any:
        """The VoiceGatewayService instance."""
        return self._voice_gateway

    @property
    def interaction_service(self) -> Any:
        """The RobotInteractionService instance."""
        return self._interaction_service

    @property
    def address_detector(self) -> Any:
        """The AddressDetector instance."""
        return self._address_detector

    @property
    def interaction_gate(self) -> Any:
        """The InteractionGate instance."""
        return self._interaction_gate

    @property
    def audio_router(self) -> AudioRouterPort:
        """The audio router instance."""
        return self._audio_router

    @property
    def asr_provider(self) -> Any:
        """The ASR provider behind the audio frontend."""
        return self._asr_provider

    @property
    def tts_provider(self) -> Any:
        """The TTS provider behind the audio frontend."""
        return self._tts_provider

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "voice_mode": True,
            "interaction_gate": {
                "enabled": bool(getattr(self._interaction_gate, "enabled", False)),
                "min_asr_confidence": getattr(
                    self._interaction_gate,
                    "min_asr_confidence",
                    None,
                ),
                "max_perception_age_s": getattr(
                    self._interaction_gate,
                    "max_perception_age_s",
                    None,
                ),
                "max_interaction_distance_m": getattr(
                    self._interaction_gate,
                    "max_interaction_distance_m",
                    None,
                ),
                **self._voice_loop.interaction_status_snapshot(),
            },
        }


def _build_interaction_perception_provider(registry: ModuleRegistry) -> Any:
    def _provider() -> dict[str, Any] | None:
        perception_mod = registry.get("perception")
        snapshot = getattr(perception_mod, "interaction_snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            if isinstance(payload, dict):
                return payload

        cognition_mod = registry.get("cognition")
        world_state = getattr(cognition_mod, "world_state", None)
        scene_fact = (
            world_state.get_fact("scene.objects", include_stale=True)
            if hasattr(world_state, "get_fact")
            else None
        )
        if scene_fact is not None:
            objects = scene_fact.value if isinstance(scene_fact.value, list) else []
            return {
                "source": scene_fact.source or "cognition_world_state",
                "observed_at": scene_fact.observed_at,
                "reason": "stale" if scene_fact.is_stale() else "fresh",
                "objects": [dict(item) for item in objects if isinstance(item, dict)],
            }

        world_snapshot = getattr(world_state, "snapshot", None)
        if callable(world_snapshot):
            payload = world_snapshot()
            if isinstance(payload, dict):
                scene = payload.get("scene")
                return {
                    "source": "cognition_world_state",
                    "observed_at": scene.get("observed_at") if isinstance(scene, dict) else None,
                    "reason": (
                        "snapshot_scene_observed_at"
                        if isinstance(scene, dict) and scene.get("observed_at") is not None
                        else "no_scene_fact_timestamp"
                    ),
                    "objects": scene.get("objects", []) if isinstance(scene, dict) else [],
                }

        return None

    return _provider
