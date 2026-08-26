"""VoiceModule - wires the voice gateway, interaction router, and audio ports.

Canonical wiring::

    router = IntentRouter(...)
    voice_stack = build_audio_frontend(cfg, ...)
    voice_loop = VoiceLoop(router=router, pipeline=pipeline, ...)
    address_detector = AddressDetector(...)
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, cast

from askme.agent_shell import AgentShell
from askme.llm.core.client import LLMClient
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.core.persona import persona_from_brain_config
from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
from askme.ports import (
    AudioFrontendPort,
    AudioRouterPort,
    PlaybackTarget,
    SpeechActor,
    SpeechDelivery,
    SpeechPlaybackPort,
    SpeechPlaybackRequest,
    SpeechPriority,
)
from askme.providers import build_speech_playback
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
from askme.runtime.modules.voice_stack import build_runtime_voice_stack
from askme.runtime.task.voice_lifecycle import (
    VoiceTaskLifecycleService,
    VoiceTaskOperatorContext,
)
from askme.runtime.voice_control import VoiceControlStateStore, deep_merge
from askme.tools.core.tool_registry import ToolRegistry
from askme.voice.output.phrase_prime import (
    PhrasePrimeEntry,
    configured_feedback_phrases,
    prime_phrase_cache,
    resolve_phrase_prime_entries,
)

logger = logging.getLogger(__name__)

_BACKGROUND_TASK_STOP_TIMEOUT_SECONDS = 1.0


class _LLMControlModule(Protocol):
    """Control-plane contract required for runtime LLM switching."""

    client: LLMClient
    llm_client: Any

    def replace_config(self, brain_cfg: dict[str, Any]) -> LLMClient: ...

    def prepare_client(self, brain_cfg: dict[str, Any]) -> LLMClient: ...

    async def validate_client(
        self,
        client: LLMClient,
        *,
        timeout_s: float = 10.0,
        model: str | None = None,
        purpose: str = "assistant_response",
    ) -> None: ...

    def commit_client(
        self,
        next_client: LLMClient,
        *,
        warmup_model: str | None = None,
    ) -> None: ...


class VoiceModule(Module):
    """Provides audio frontend, voice gateway, router, loop, and gates."""

    name = "voice"
    depends_on = ("llm", "tools", "skill", "mission", "pipeline", "runtime_handoff")
    provides = ("voice", "tts", "asr")

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    skill_in: In[SkillDispatcher]
    pipeline_in: In[BrainPipeline]
    executor_in: In[AgentShell]

    # Out port
    audio_out: Out[AudioFrontendPort]  # type: ignore[no-redef]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.pipeline.channels.voice_loop import VoiceLoop
        from askme.robot_interaction import (
            RobotInteractionService,
        )
        from askme.telemetry.ota_bridge import OTABridgeMetrics
        from askme.tools.core.builtin_tools import SpeakProgressTool
        from askme.tools.voice.voice_tools import register_voice_tools

        self._registry = registry
        self._base_cfg = deepcopy(cfg)
        self._state_store = VoiceControlStateStore(cfg)
        self._control_state = self._state_store.load()
        self._component_switch_lock = asyncio.Lock()
        self._runtime_switch_state_lock = threading.RLock()
        effective_cfg = self._effective_startup_config(cfg)

        llm_mod = self.llm_in
        self._voice_cfg = dict(effective_cfg.get("voice", {}) or {})
        self._input_retry_seconds = max(
            0.1,
            float(self._voice_cfg.get("input_retry_seconds", 5.0)),
        )
        self._ensure_runtime_switch_state()
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else OTABridgeMetrics()

        tools_mod = self.tool_registry_in
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        skill_mod = self.skill_in
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        dispatcher = getattr(skill_mod, "skill_dispatcher", None) if skill_mod else None

        pipeline_mod = self.pipeline_in
        pipeline_candidate = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        if pipeline_candidate is None:
            raise RuntimeError("VoiceModule requires an available brain pipeline")
        pipeline = cast(BrainPipeline, pipeline_candidate)

        executor_mod = self.executor_in
        agent_shell = getattr(executor_mod, "shell", None) if executor_mod else None

        voice_stack = build_runtime_voice_stack(
            effective_cfg,
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
        self._speech_playback = build_speech_playback(
            effective_cfg,
            audio=self._audio,
        )
        runtime_switch_setter = getattr(self._audio, "set_runtime_switch_callback", None)
        if callable(runtime_switch_setter):
            runtime_switch_setter(self._handle_runtime_switch_outcome)

        # Register voice tools
        if tools is not None:
            register_voice_tools(tools, self._audio)
            tools.register(SpeakProgressTool(self._audio))

        # Cross-link: pipeline, agent_shell, and dispatcher need the audio agent
        # for TTS playback and voice state queries.  VoiceModule owns the AudioAgent
        # so it is responsible for injecting it into objects built by earlier modules.
        if pipeline is not None:
            pipeline.set_audio(self._audio)
            set_barge_in_callback = getattr(
                self._audio, "set_barge_in_callback", None
            )
            cancel_current_turn = getattr(pipeline, "cancel_current_turn", None)
            if callable(set_barge_in_callback) and callable(cancel_current_turn):
                def _cancel_voice_turn() -> None:
                    cancel_current_turn(owner="voice")

                set_barge_in_callback(_cancel_voice_turn)
        if agent_shell is not None:
            agent_shell.set_audio(self._audio)
        if dispatcher is not None:
            dispatcher.set_audio(self._audio)

        interaction_gate_cfg = self._voice_cfg.get("interaction_gate", {})
        if not isinstance(interaction_gate_cfg, dict):
            raise ValueError("voice.interaction_gate must be a mapping")
        anonymous_encounter_idle_seconds = interaction_gate_cfg.get(
            "anonymous_encounter_idle_seconds", 25.0
        )
        self._interaction_service = RobotInteractionService(self._router)
        self._voice_task_lifecycle = _build_voice_task_lifecycle(effective_cfg, registry)
        voice_task_operator_provider = _build_voice_task_operator_provider(
            effective_cfg,
            self._audio,
            self._voice_task_lifecycle,
        )

        # VoiceLoop
        self._voice_loop = VoiceLoop(
            router=self._router,
            pipeline=pipeline,
            audio=self._audio,
            voice_runtime_bridge=self._voice_gateway,
            dispatcher=dispatcher,
            audio_router=self._audio_router,
            voice_task_lifecycle=self._voice_task_lifecycle,
            voice_task_operator_provider=voice_task_operator_provider,
            anonymous_encounter_idle_seconds=anonymous_encounter_idle_seconds,
        )

        # Runtime voice stack owns gate construction; VoiceModule injects into VoiceLoop.
        self._address_detector = voice_stack.address_detector
        self._voice_loop.set_address_detector(self._address_detector)
        self._interaction_gate = voice_stack.interaction_gate
        self._voice_loop.set_interaction_gate(self._interaction_gate)
        self._interaction_perception_provider = _build_interaction_perception_provider(registry)
        self._voice_loop.set_interaction_perception_provider(self._interaction_perception_provider)
        self._mission_context_provider = _build_mission_context_provider(effective_cfg, registry)
        self._voice_loop.set_mission_context_provider(self._mission_context_provider)

        self._task: asyncio.Task[None] | None = None
        self._phrase_prime_task: asyncio.Task[None] | None = None
        self._phrase_prime_stop = threading.Event()
        self._apply_persisted_llm_and_prompt()
        logger.info("VoiceModule: built")

    def _effective_startup_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        effective = deepcopy(cfg)
        asr_state = self._control_state.get("asr", {})
        if isinstance(asr_state, dict) and asr_state:
            effective["voice"] = self._resolve_asr_voice_config(asr_state, source=effective)
        tts_state = self._control_state.get("tts", {})
        if isinstance(tts_state, dict) and tts_state:
            voice_cfg = effective.get("voice", {})
            voice_cfg["tts"] = self._resolve_tts_config(tts_state, source=effective)
        return effective

    def _apply_persisted_llm_and_prompt(self) -> None:
        llm_state = self._control_state.get("llm", {})
        if isinstance(llm_state, dict) and llm_state:
            try:
                brain_cfg = self._resolve_llm_config(llm_state)
                llm_mod = self._require_llm_control_module(
                    required_methods=("replace_config",),
                    required_attributes=("llm_client",),
                )
                llm_mod.replace_config(brain_cfg)
                self._publish_llm(llm_mod.llm_client, brain_cfg)
            except Exception as exc:
                logger.warning("VoiceModule: persisted LLM selection ignored: %s", exc)
        prompt_state = self._control_state.get("prompt", {})
        if isinstance(prompt_state, dict) and prompt_state:
            try:
                self._apply_prompt_settings(prompt_state)
            except Exception as exc:
                logger.warning("VoiceModule: persisted prompt selection ignored: %s", exc)

    async def start(self) -> None:
        """Start a voice task that recovers when microphone hardware is absent."""
        speech_playback = getattr(self, "_speech_playback", None)
        if speech_playback is not None:
            await speech_playback.start()
        self._task = asyncio.create_task(
            self._run_with_input_recovery(),
            name="voice-loop",
        )
        self._start_phrase_prime_task()
        logger.info("VoiceModule: voice task started")

    async def _run_with_input_recovery(self) -> None:
        retry_seconds = max(0.0, float(getattr(self, "_input_retry_seconds", 5.0)))
        while True:
            try:
                self._audio.start_input()
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "VoiceModule: microphone unavailable; retrying in %.1fs: %s",
                    retry_seconds,
                    exc,
                )
                try:
                    self._audio.stop_input()
                except Exception:
                    logger.debug(
                        "VoiceModule: failed to clean up input after open error",
                        exc_info=True,
                    )
                await asyncio.sleep(retry_seconds)

        logger.info("VoiceModule: voice loop started (mic persistent)")
        await self._voice_loop.run()

    async def stop(self) -> None:
        """Cancel the voice loop task and close mic."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        speech_playback = getattr(self, "_speech_playback", None)
        if speech_playback is not None:
            await speech_playback.shutdown()
        phrase_prime_stop = getattr(self, "_phrase_prime_stop", None)
        if phrase_prime_stop is not None:
            phrase_prime_stop.set()
        phrase_prime_task = getattr(self, "_phrase_prime_task", None)
        await self._harvest_background_task(phrase_prime_task, "phrase cache prime")
        self._audio.stop_input()
        self._audio.shutdown()
        logger.info("VoiceModule: stopped")

    async def _harvest_background_task(
        self,
        task: asyncio.Task[None] | None,
        label: str,
    ) -> None:
        if task is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=_BACKGROUND_TASK_STOP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "VoiceModule: %s did not stop within %.1fs; detaching",
                label,
                _BACKGROUND_TASK_STOP_TIMEOUT_SECONDS,
            )
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

    def _start_phrase_prime_task(self) -> None:
        current_task = getattr(self, "_phrase_prime_task", None)
        if current_task is not None and not current_task.done():
            return
        voice_cfg = getattr(self, "_voice_cfg", {})
        tts_cfg = dict(voice_cfg.get("tts", {}) or {})
        if not bool(tts_cfg.get("phrase_cache_enabled", True)):
            return
        if not bool(tts_cfg.get("phrase_prime_enabled", True)):
            return
        configured = tts_cfg.get("phrase_prime_list", ())
        policy = getattr(getattr(self, "_router", None), "_policy", None)
        quick_replies = getattr(policy, "quick_replies", {})
        entries = resolve_phrase_prime_entries(
            configured,
            quick_replies=quick_replies,
            feedback_phrases=configured_feedback_phrases(voice_cfg),
        )
        if not entries:
            return
        self._phrase_prime_stop = threading.Event()
        self._phrase_prime_task = asyncio.create_task(
            self._prime_phrase_cache_background(tts_cfg, entries),
            name="voice-phrase-cache-prime",
        )

    async def _prime_phrase_cache_background(
        self,
        tts_cfg: dict[str, Any],
        entries: list[PhrasePrimeEntry],
    ) -> None:
        try:
            results = await asyncio.to_thread(
                prime_phrase_cache,
                tts_cfg,
                entries,
                stop_event=self._phrase_prime_stop,
            )
        except Exception as exc:
            logger.warning("VoiceModule: phrase cache prime failed: %s", exc)
            return
        created = sum(bool(result.get("created")) for result in results)
        cached = sum(bool(result.get("cached")) for result in results)
        logger.info(
            "VoiceModule: phrase cache prime finished (%d cached, %d created)",
            cached,
            created,
        )

    # -- typed accessors ------------------------------------------------
    @property  # type: ignore[no-redef]
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
    def voice_task_lifecycle(self) -> VoiceTaskLifecycleService | None:
        """External task lifecycle used by the voice session, when configured."""
        return self._voice_task_lifecycle

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

        if self._audio_router is None:
            raise RuntimeError("VoiceModule audio router is not available")
        return self._audio_router

    @property
    def asr_provider(self) -> Any:
        """The ASR provider behind the audio frontend."""
        return getattr(self._audio, "_asr_mgr", self._asr_provider)

    @property
    def tts_provider(self) -> Any:
        """The TTS provider behind the audio frontend."""
        return getattr(self._audio, "tts", self._tts_provider)

    @property
    def speech_playback(self) -> SpeechPlaybackPort:
        """Shared playback job port used by HTTP, MCP, and internal callers."""
        return self._speech_playback

    async def speak_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue literal text for the local robot; never invoke the LLM."""
        semantics = str(payload.get("semantics") or "verbatim").strip().lower()
        if semantics != "verbatim":
            raise ValueError("speak_payload only accepts semantics=verbatim")
        job = await self._speech_playback.submit(
            self._speech_request(payload, delivery=SpeechDelivery.PLAYBACK)
        )
        return job.to_payload()

    async def synthesize_speech_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Synthesize literal text without releasing it to the speaker."""
        job = await self._speech_playback.submit(
            self._speech_request(payload, delivery=SpeechDelivery.SYNTHESIZE_ONLY)
        )
        return job.to_payload()

    async def speech_playback_audio_payload(
        self,
        playback_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del payload
        artifact = await self._speech_playback.artifact_file(playback_id)
        return {
            "path": str(artifact.path),
            "filename": artifact.filename,
            "media_type": artifact.media_type,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
        }

    async def speech_playback_status_payload(
        self,
        playback_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del payload
        return (await self._speech_playback.status(playback_id)).to_payload()

    async def cancel_speech_playback_payload(
        self,
        playback_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        job = await self._speech_playback.cancel(
            playback_id,
            reason=str(payload.get("reason") or "operator_cancelled"),
            actor=self._speech_actor(payload),
        )
        return job.to_payload()

    def _speech_request(
        self,
        payload: dict[str, Any],
        *,
        delivery: SpeechDelivery,
    ) -> SpeechPlaybackRequest:
        return SpeechPlaybackRequest(
            text=str(payload.get("text") or ""),
            target=PlaybackTarget(
                robot_id=str(payload.get("robot_id") or ""),
                device_id=str(payload.get("device_id") or ""),
                site_id=str(payload.get("site_id") or ""),
            ),
            actor=self._speech_actor(payload),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            delivery=delivery,
            priority=SpeechPriority(str(payload.get("priority") or "normal")),
            queue_policy=str(payload.get("queue_policy") or "enqueue"),
            voice_profile_id=str(payload.get("voice_profile_id") or ""),
            speed=_optional_float(payload.get("speed")),
            pitch=_optional_float(payload.get("pitch")),
            volume=_optional_float(payload.get("volume")),
            ttl_s=float(payload.get("ttl_s", 60.0)),
        )

    @staticmethod
    def _speech_actor(payload: dict[str, Any]) -> SpeechActor:
        auth = payload.get("operator_auth")
        operator = auth.get("operator") if isinstance(auth, dict) else {}
        operator = operator if isinstance(operator, dict) else {}
        roles = operator.get("roles") if isinstance(operator.get("roles"), list) else []
        return SpeechActor(
            operator_id=str(
                operator.get("operator_id")
                or payload.get("operator_id")
                or "unknown.operator"
            ).strip(),
            roles=frozenset(str(role).strip() for role in roles if str(role).strip()),
            surface=str(payload.get("surface") or "fastapi"),
        )

    def set_tts_activation_callback(self, callback: Any | None) -> None:
        """Connect the runtime warm-session owner to TTS activation events."""

        setter = getattr(self._audio, "set_tts_activation_callback", None)
        if callable(setter):
            setter(callback)

    def voice_profiles_payload(self) -> dict[str, Any]:
        return self.tts_provider.voice_profiles_payload()

    def set_voice_profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.tts_provider.set_voice_profile_payload(payload)

    async def system_control_payload(self) -> dict[str, Any]:
        """Return the complete non-secret state used by the management console."""

        audio_status = _status_snapshot(self._audio)
        llm_mod = self._registry.get("llm")
        llm_client = getattr(llm_mod, "client", None)
        llm_status = (
            llm_client.provider_status()
            if llm_client is not None and hasattr(llm_client, "provider_status")
            else {}
        )
        pipeline = self._pipeline()
        prompt = pipeline.prompt_settings() if pipeline is not None else {}
        memory_mod = self._registry.get("memory")
        memory_payload: dict[str, Any] = {}
        memory_health = getattr(memory_mod, "health_payload", None)
        if callable(memory_health):
            result = memory_health({})
            memory_payload = await result if asyncio.iscoroutine(result) else result
        memory_payload = memory_payload if isinstance(memory_payload, dict) else {}
        gate_status = self._voice_loop.interaction_status_snapshot()
        interaction_policy = {
            "enabled": bool(getattr(self._interaction_gate, "enabled", False)),
            "mode": "strict_public_site"
            if (
                bool(getattr(self._interaction_gate, "silent_on_ambiguous", False))
                and not bool(
                    getattr(
                        self._interaction_gate,
                        "allow_unaddressed_public_help",
                        True,
                    )
                )
                and not bool(
                    getattr(
                        self._interaction_gate,
                        "allow_unaddressed_robot_tasks",
                        True,
                    )
                )
            )
            else "permissive",
            "silent_on_ambiguous": bool(
                getattr(self._interaction_gate, "silent_on_ambiguous", False)
            ),
            "allow_unaddressed_public_help": bool(
                getattr(
                    self._interaction_gate,
                    "allow_unaddressed_public_help",
                    True,
                )
            ),
            "allow_unaddressed_robot_tasks": bool(
                getattr(
                    self._interaction_gate,
                    "allow_unaddressed_robot_tasks",
                    True,
                )
            ),
            "wake_timeout_s": audio_status.get("wake_timeout_s"),
        }
        interaction_status = {
            **(audio_status.get("interaction", {}) or {}),
            **gate_status,
            "policy": interaction_policy,
            "wake_source": audio_status.get("last_turn_wake_source", "none"),
            "wake_timeout_remaining_s": audio_status.get(
                "wake_timeout_remaining_s",
                0.0,
            ),
        }
        issues = self._runtime_issues(
            audio_status,
            memory_payload,
            prompt,
            interaction_status,
        )
        return {
            "status": "ready" if not issues else "degraded",
            "runtime": {
                "llm": llm_status,
                "asr": audio_status.get("asr", {}),
                "tts": audio_status.get("tts", {}),
                "playback": self._speech_playback.snapshot(),
                "switches": self._runtime_switches_snapshot(),
                "kws": {
                    "enabled": audio_status.get("wake_word_enabled", False),
                    "keyword": "小算",
                },
                "vad": {
                    "state": audio_status.get("input", {}).get("vad_state", "idle"),
                    "threshold": self._voice_cfg.get("vad", {}).get("threshold"),
                },
                "audio": {
                    "input_ready": audio_status.get("input_ready", False),
                    "output_ready": audio_status.get("output_ready", False),
                    "input": audio_status.get("input", {}),
                    "media": audio_status.get("media", {}),
                    "pending_updates": audio_status.get("pending_runtime_updates", {}),
                },
                "interaction": interaction_status,
                "latency": audio_status.get("voice_turn", {}).get("latency_summary", {}),
            },
            "catalog": self._provider_catalog(memory_payload),
            "prompt": {
                **prompt,
                "persona": deepcopy(self._base_cfg.get("brain", {}).get("persona", {})),
            },
            "memory": memory_payload,
            "issues": issues,
            "resolved_issues": [
                {
                    "id": "prompt_system_preserved",
                    "label": "直连模型保留 system prompt",
                },
                {
                    "id": "volcengine_idle_preconnect_disabled",
                    "label": "火山 ASR 仅在检测到语音后连接",
                },
                {
                    "id": "local_onnx_cpu",
                    "label": "本地 ONNX 组件固定 CPU provider",
                },
                {
                    "id": "reflection_json_repair",
                    "label": "记忆反思支持 JSON 修复与文本降级",
                },
                {
                    "id": "ambient_speech_admission",
                    "label": "旁人语音在记忆和大模型前静默门控",
                },
                {
                    "id": "wake_window_not_authority",
                    "label": "追问收音窗口不再等同于关键词授权",
                },
            ],
            "persistence": {
                "path": str(self._state_store.path),
                "restored": bool(self._control_state),
                "secrets_persisted": False,
            },
        }

    async def switch_system_component_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        switch_lock = getattr(self, "_component_switch_lock", None)
        if switch_lock is None:
            switch_lock = asyncio.Lock()
            self._component_switch_lock = switch_lock

        async with switch_lock:
            component = str(payload.get("component") or "").strip().lower()
            if component == "llm":
                result = await self._switch_llm(payload)
            elif component == "asr":
                result = await self._switch_asr(payload)
            elif component == "tts":
                result = await self._switch_tts(payload)
            else:
                raise ValueError("component must be one of: llm, asr, tts")
            if result.get("updated") and (
                component not in {"asr", "tts"} or result.get("state") == "active"
            ):
                self._persist_control_state()
            return result

    def update_prompt_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._apply_prompt_settings(payload)
        persona = payload.get("persona")
        persisted = {
            "system_prompt": settings.get("system_prompt", ""),
            "user_prefix": settings.get("user_prefix", ""),
            "relay_compat_mode": bool(settings.get("relay_compat_mode", False)),
        }
        if isinstance(persona, dict):
            persisted["persona"] = deepcopy(persona)
        self._control_state["prompt"] = persisted
        self._persist_control_state()
        return {
            "updated": True,
            "component": "prompt",
            "state": "active",
            "runtime": settings,
        }

    async def _switch_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        brain_cfg = self._resolve_llm_config(payload)
        required_methods = ["prepare_client", "commit_client"]
        if bool(payload.get("validate", True)):
            required_methods.append("validate_client")
        llm_mod = self._require_llm_control_module(
            required_methods=tuple(required_methods),
            required_attributes=("client", "llm_client"),
        )

        previous_client = llm_mod.client
        previous_payload = dict(self._control_state.get("llm", {}) or {})
        previous_brain_cfg = self._resolve_llm_config(previous_payload)
        previous_warmup_model = str(
            previous_brain_cfg.get("health_model")
            or previous_brain_cfg.get("voice_model")
            or previous_client.model
        ).strip()
        candidate = llm_mod.prepare_client(brain_cfg)
        candidate_warmup_model = str(
            brain_cfg.get("health_model") or brain_cfg.get("voice_model") or candidate.model
        ).strip()
        try:
            if candidate.provider_name == "litellm" and candidate_warmup_model != "health-probe":
                raise ValueError("LiteLLM switch requires health_model='health-probe'")
            if bool(payload.get("validate", True)):
                validation_timeout = min(15.0, float(brain_cfg.get("timeout", 10.0)))
                await llm_mod.validate_client(
                    candidate,
                    timeout_s=validation_timeout,
                )
                if candidate.provider_name == "litellm":
                    await llm_mod.validate_client(
                        candidate,
                        timeout_s=validation_timeout,
                        model=candidate_warmup_model,
                        purpose="health_probe",
                    )
        except Exception:
            await self._retire_llm_client(candidate, label="failed validation")
            raise

        llm_mod.commit_client(
            candidate,
            warmup_model=candidate_warmup_model,
        )
        try:
            self._publish_llm(llm_mod.llm_client, brain_cfg)
        except Exception:
            logger.exception("LLM consumer publication failed; rolling back provider switch")
            try:
                llm_mod.commit_client(
                    previous_client,
                    warmup_model=previous_warmup_model,
                )
            except Exception:
                logger.exception("LLM module rollback failed")
            try:
                self._publish_llm(llm_mod.llm_client, previous_brain_cfg)
            except Exception:
                logger.exception("LLM consumer rollback failed")
            await self._defer_llm_client_retirement(llm_mod, candidate, label="failed candidate")
            raise

        await self._defer_llm_client_retirement(llm_mod, previous_client, label="previous provider")
        self._control_state["llm"] = {
            "provider": candidate.provider_name,
            "model": candidate.model,
            "voice_model": str(brain_cfg.get("voice_model") or candidate.model),
            "health_model": candidate_warmup_model,
            "fallback_models": list(brain_cfg.get("fallback_models", [])),
        }
        return {
            "updated": True,
            "component": "llm",
            "state": "active",
            "runtime": candidate.provider_status(),
        }

    def _require_llm_control_module(
        self,
        *,
        required_methods: tuple[str, ...],
        required_attributes: tuple[str, ...] = (),
    ) -> _LLMControlModule:
        module = self._registry.get("llm")
        if module is None:
            raise RuntimeError("LLM module is not available")

        missing_methods: list[str] = []
        for name in required_methods:
            try:
                method = getattr(module, name)
            except Exception:
                method = None
            if not callable(method):
                missing_methods.append(name)

        missing_attributes: list[str] = []
        for name in required_attributes:
            try:
                value = getattr(module, name)
            except Exception:
                value = None
            if value is None:
                missing_attributes.append(name)

        if missing_methods or missing_attributes:
            missing = ", ".join((*missing_methods, *missing_attributes))
            raise RuntimeError(f"LLM module control contract is unavailable: {missing}")
        return cast(_LLMControlModule, module)

    async def _switch_asr(self, payload: dict[str, Any]) -> dict[str, Any]:
        voice_cfg = self._resolve_asr_voice_config(payload)
        desired = self._asr_selection(voice_cfg)
        effective = self._current_asr_selection()
        reconfigure = getattr(self._audio, "reconfigure_asr", None)
        if not callable(reconfigure):
            raise RuntimeError("audio frontend does not support ASR hot switching")
        try:
            result = await asyncio.to_thread(reconfigure, voice_cfg)
        except Exception as exc:
            self._record_runtime_switch_failure("asr", desired, str(exc))
            raise
        if str(result.get("state") or "active") == "pending":
            self._record_runtime_switch_pending("asr", desired)
            return self._runtime_switch_result(
                result,
                desired=desired,
                effective=effective,
                pending=desired,
            )

        self._voice_cfg = voice_cfg
        self._control_state["asr"] = desired
        self._record_runtime_switch_active("asr", desired)
        return self._runtime_switch_result(
            result,
            desired=desired,
            effective=desired,
        )

    async def _switch_tts(self, payload: dict[str, Any]) -> dict[str, Any]:
        tts_cfg = self._resolve_tts_config(payload)
        desired = self._tts_selection(tts_cfg)
        effective = self._current_tts_selection()
        reconfigure = getattr(self._audio, "reconfigure_tts", None)
        if not callable(reconfigure):
            raise RuntimeError("audio frontend does not support TTS hot switching")
        try:
            result = await asyncio.to_thread(reconfigure, tts_cfg)
        except Exception as exc:
            self._record_runtime_switch_failure("tts", desired, str(exc))
            raise
        if str(result.get("state") or "active") == "pending":
            self._record_runtime_switch_pending("tts", desired)
            return self._runtime_switch_result(
                result,
                desired=desired,
                effective=effective,
                pending=desired,
            )

        self._voice_cfg["tts"] = tts_cfg
        self._control_state["tts"] = desired
        self._record_runtime_switch_active("tts", desired)
        return self._runtime_switch_result(
            result,
            desired=desired,
            effective=desired,
        )

    def _current_asr_selection(self) -> dict[str, str]:
        persisted = self._control_state.get("asr", {})
        if isinstance(persisted, dict) and persisted.get("provider"):
            provider = str(persisted.get("provider") or "local")
            return {
                "provider": provider,
                "model": str(persisted.get("model") or ("local" if provider == "local" else "")),
            }
        return self._asr_selection(self._voice_cfg)

    @staticmethod
    def _asr_selection(voice_cfg: dict[str, Any]) -> dict[str, str]:
        cloud_cfg = voice_cfg.get("cloud_asr", {})
        if not isinstance(cloud_cfg, dict) or not cloud_cfg.get("enabled"):
            return {"provider": "local", "model": "local"}
        return {
            "provider": str(cloud_cfg.get("provider") or ""),
            "model": str(cloud_cfg.get("model") or ""),
        }

    def _current_tts_selection(self) -> dict[str, str]:
        persisted = self._control_state.get("tts", {})
        if isinstance(persisted, dict) and persisted.get("backend"):
            return {
                "backend": str(persisted.get("backend") or ""),
                "model": str(persisted.get("model") or ""),
                "voice_id": str(persisted.get("voice_id") or ""),
            }
        return self._tts_selection(self._voice_cfg.get("tts", {}))

    @staticmethod
    def _tts_selection(tts_cfg: dict[str, Any]) -> dict[str, str]:
        backend = str(tts_cfg.get("backend") or "")
        if backend == "volcengine":
            model = str(tts_cfg.get("volcengine_tts_resource_id") or "")
            voice_id = str(tts_cfg.get("volcengine_tts_speaker") or "")
        elif backend == "edge":
            model = str(tts_cfg.get("voice") or "")
            voice_id = model
        elif backend == "local":
            model = Path(str(tts_cfg.get("model_dir") or "")).name
            voice_id = str(tts_cfg.get("sid") or "")
        else:
            model = str(tts_cfg.get("minimax_tts_model") or "")
            voice_id = str(tts_cfg.get("minimax_voice_id") or "")
        return {
            "backend": backend,
            "model": model,
            "voice_id": voice_id,
        }

    def _runtime_switch_lock(self) -> Any:
        lock = getattr(self, "_runtime_switch_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._runtime_switch_state_lock = lock
        return lock

    def _ensure_runtime_switch_state(self) -> dict[str, dict[str, Any]]:
        lock = self._runtime_switch_lock()
        with lock:
            current = getattr(self, "_runtime_switch_state", None)
            if isinstance(current, dict) and {"asr", "tts"} <= current.keys():
                return current
            asr = self._current_asr_selection()
            tts = self._current_tts_selection()
            current = {
                "asr": {
                    "state": "active",
                    "desired": dict(asr),
                    "effective": dict(asr),
                    "pending": None,
                    "failed": None,
                },
                "tts": {
                    "state": "active",
                    "desired": dict(tts),
                    "effective": dict(tts),
                    "pending": None,
                    "failed": None,
                },
            }
            self._runtime_switch_state = current
            return current

    def _record_runtime_switch_pending(
        self,
        component: str,
        desired: dict[str, str],
    ) -> None:
        with self._runtime_switch_lock():
            state = self._ensure_runtime_switch_state()[component]
            state.update(
                {
                    "state": "pending",
                    "desired": dict(desired),
                    "pending": dict(desired),
                    "failed": None,
                }
            )

    def _record_runtime_switch_active(
        self,
        component: str,
        effective: dict[str, str],
    ) -> None:
        with self._runtime_switch_lock():
            state = self._ensure_runtime_switch_state()[component]
            state.update(
                {
                    "state": "active",
                    "desired": dict(effective),
                    "effective": dict(effective),
                    "pending": None,
                    "failed": None,
                }
            )

    def _record_runtime_switch_failure(
        self,
        component: str,
        desired: dict[str, str],
        reason: str,
    ) -> None:
        failure = {"reason": str(reason or "runtime activation failed")}
        with self._runtime_switch_lock():
            state = self._ensure_runtime_switch_state()[component]
            state.update(
                {
                    "state": "failed",
                    "desired": dict(desired),
                    "pending": None,
                    "failed": failure,
                }
            )

    def _runtime_switches_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._runtime_switch_lock():
            return deepcopy(self._ensure_runtime_switch_state())

    def _handle_runtime_switch_outcome(self, outcome: dict[str, Any]) -> None:
        component = str(outcome.get("component") or "").strip().lower()
        if component not in {"asr", "tts"}:
            logger.warning("Ignoring runtime switch outcome for unknown component: %s", component)
            return
        config = outcome.get("config")
        if not isinstance(config, dict):
            logger.warning("Ignoring %s runtime switch outcome without config", component)
            return
        desired = self._asr_selection(config) if component == "asr" else self._tts_selection(config)
        outcome_state = str(outcome.get("state") or "").strip().lower()

        if outcome_state == "failed":
            with self._runtime_switch_lock():
                pending = self._ensure_runtime_switch_state()[component].get("pending")
                if pending is not None and pending != desired:
                    logger.info("Ignoring stale %s runtime switch failure", component)
                    return
                self._record_runtime_switch_failure(
                    component,
                    desired,
                    str(outcome.get("reason") or "runtime activation failed"),
                )
            return
        if outcome_state != "active":
            logger.warning(
                "Ignoring %s runtime switch outcome with state %r",
                component,
                outcome_state,
            )
            return

        with self._runtime_switch_lock():
            current = self._ensure_runtime_switch_state()[component]
            pending = current.get("pending")
            if component == "asr":
                self._voice_cfg = dict(config)
            else:
                self._voice_cfg["tts"] = dict(config)
            self._control_state[component] = dict(desired)
            if pending is not None and pending != desired:
                current["effective"] = dict(desired)
                current["failed"] = None
            else:
                self._record_runtime_switch_active(component, desired)
            self._persist_control_state()

    @staticmethod
    def _runtime_switch_result(
        result: dict[str, Any],
        *,
        desired: dict[str, str],
        effective: dict[str, str],
        pending: dict[str, str] | None = None,
        failed: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = dict(result)
        activation = payload.get("effective")
        if activation is not None and not isinstance(activation, dict):
            payload["activation"] = activation
        payload.update(
            {
                "desired": dict(desired),
                "effective": dict(effective),
                "pending": dict(pending) if pending is not None else None,
                "failed": dict(failed) if failed is not None else None,
            }
        )
        return payload

    async def _defer_llm_client_retirement(
        self,
        llm_module: Any,
        client: Any,
        *,
        label: str,
    ) -> None:
        retire = getattr(llm_module, "retire_client", None)
        if callable(retire):
            try:
                result = retire(client)
                if inspect.isawaitable(result):
                    await result
                return
            except Exception:
                logger.warning(
                    "VoiceModule: failed to defer %s LLM retirement",
                    label,
                    exc_info=True,
                )
                return
        await self._retire_llm_client(client, label=label)

    async def _retire_llm_client(self, client: Any, *, label: str) -> None:
        close = getattr(client, "aclose", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=2.0)
        except Exception:
            logger.warning("VoiceModule: failed to retire %s LLM client", label, exc_info=True)

    def _publish_llm(self, client: Any, brain_cfg: dict[str, Any]) -> None:
        pipeline = self._pipeline()
        if pipeline is not None:
            pipeline.replace_llm(
                client,
                voice_model=str(brain_cfg.get("voice_model") or getattr(client, "model", "")),
            )
        memory_mod = self._registry.get("memory")
        replace_llm = getattr(memory_mod, "replace_llm", None)
        if callable(replace_llm):
            replace_llm(client)

    def _pipeline(self) -> Any | None:
        pipeline_mod = self._registry.get("pipeline")
        return getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

    def _resolve_llm_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = deepcopy(self._base_cfg.get("brain", {}))
        provider = (
            str(
                payload.get("provider")
                or self._control_state.get("llm", {}).get("provider")
                or base.get("provider")
                or ""
            )
            .strip()
            .lower()
        )
        presets = self._llm_presets()
        preset = presets.get(provider)
        if preset is None:
            raise ValueError(f"LLM provider is not configured: {provider}")
        resolved = deep_merge(base, preset)
        resolved["provider"] = provider
        resolved["model"] = str(
            payload.get("model") or preset.get("model") or base.get("model") or ""
        )
        resolved["voice_model"] = str(payload.get("voice_model") or resolved["model"])
        if isinstance(payload.get("fallback_models"), list):
            resolved["fallback_models"] = [
                str(item).strip() for item in payload["fallback_models"] if str(item).strip()
            ]
        return resolved

    def _llm_presets(self) -> dict[str, dict[str, Any]]:
        brain = self._base_cfg.get("brain", {})
        current_provider = str(brain.get("provider") or "").strip().lower()
        presets: dict[str, dict[str, Any]] = {
            current_provider: {
                "api_key": brain.get("api_key", ""),
                "base_url": brain.get("base_url", ""),
                "model": brain.get("model", ""),
                "fallback_models": brain.get("fallback_models", []),
            }
        }
        configured = brain.get("provider_presets", {})
        if isinstance(configured, dict):
            for name, preset in configured.items():
                if isinstance(preset, dict):
                    presets[str(name).strip().lower()] = deepcopy(preset)
        return {key: value for key, value in presets.items() if key}

    def _resolve_asr_voice_config(
        self,
        payload: dict[str, Any],
        *,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = source or self._base_cfg
        voice_cfg = deepcopy(root.get("voice", {}))
        provider = (
            str(
                payload.get("provider") or voice_cfg.get("cloud_asr", {}).get("provider") or "local"
            )
            .strip()
            .lower()
        )
        cloud_cfg = deepcopy(voice_cfg.get("cloud_asr", {}))
        if provider == "local":
            cloud_cfg["enabled"] = False
            cloud_cfg["preconnect"] = False
        else:
            current_provider = str(cloud_cfg.get("provider") or "").strip().lower()
            if provider != current_provider:
                presets = voice_cfg.get("asr_provider_presets", {})
                preset = presets.get(provider) if isinstance(presets, dict) else None
                if not isinstance(preset, dict):
                    raise ValueError(f"ASR provider is not configured: {provider}")
                cloud_cfg = deep_merge(cloud_cfg, preset)
            cloud_cfg["provider"] = provider
            cloud_cfg["enabled"] = True
            cloud_cfg["model"] = str(payload.get("model") or cloud_cfg.get("model") or "")
            if provider in {"volcengine", "doubao", "seed_asr", "volcengine_seed_asr"}:
                cloud_cfg["preconnect"] = bool(payload.get("preconnect", False))
        voice_cfg["cloud_asr"] = cloud_cfg
        return voice_cfg

    def _resolve_tts_config(
        self,
        payload: dict[str, Any],
        *,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = source or self._base_cfg
        voice_cfg = root.get("voice", {})
        tts_cfg = deepcopy(voice_cfg.get("tts", {}))
        backend = str(payload.get("backend") or tts_cfg.get("backend") or "local").strip().lower()
        if backend == "volc":
            backend = "volcengine"
        if backend not in {"local", "edge", "minimax", "volcengine"}:
            raise ValueError("TTS backend must be one of: local, edge, minimax, volcengine")
        tts_cfg["backend"] = backend
        if payload.get("model"):
            model = str(payload["model"])
            if backend == "volcengine":
                # Volcengine's V3 route selects the TTS product/model through
                # X-Api-Resource-Id, not an undocumented req_params.model.
                tts_cfg["volcengine_tts_resource_id"] = model
                tts_cfg["volcengine_tts_model"] = model
            else:
                tts_cfg["minimax_tts_model"] = model
        if payload.get("voice_id"):
            voice_key = {
                "edge": "voice",
                "volcengine": "volcengine_tts_speaker",
            }.get(backend, "minimax_voice_id")
            tts_cfg[voice_key] = str(payload["voice_id"])
        return tts_cfg

    def _apply_prompt_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        pipeline = self._pipeline()
        if pipeline is None:
            raise RuntimeError("pipeline is not available")
        current = pipeline.prompt_settings()
        brain_cfg = deepcopy(self._base_cfg.get("brain", {}))
        persona_patch = payload.get("persona")
        if isinstance(persona_patch, dict):
            brain_cfg["persona"] = deep_merge(
                brain_cfg.get("persona", {}),
                persona_patch,
            )
        persona = persona_from_brain_config(brain_cfg)
        regenerate = bool(persona_patch) or bool(payload.get("regenerate_persona"))
        base_prompt = (
            persona.build_system_prompt()
            if regenerate
            else str(payload.get("system_prompt", current.get("system_prompt", ""))).strip()
        )
        prompt_seed = persona.build_prompt_seed() if regenerate else current.get("prompt_seed", [])
        user_prefix = (
            persona.build_user_prefix()
            if regenerate
            else str(payload.get("user_prefix", current.get("user_prefix", "")))
        )
        relay_mode = bool(payload.get("relay_compat_mode", current.get("relay_compat_mode", False)))
        return pipeline.update_prompt(
            base_prompt=base_prompt,
            prompt_seed=prompt_seed,
            user_prefix=user_prefix,
            relay_compat_mode=relay_mode,
        )

    def _provider_catalog(self, memory: dict[str, Any]) -> dict[str, Any]:
        llm_entries = []
        for provider, preset in self._llm_presets().items():
            models = [str(preset.get("model") or "")]
            models.extend(str(item) for item in preset.get("fallback_models", []) if str(item))
            llm_entries.append(
                {
                    "provider": provider,
                    "models": list(dict.fromkeys(item for item in models if item)),
                    "credential_ready": bool(preset.get("api_key")),
                }
            )
        base_tts = self._base_cfg.get("voice", {}).get("tts", {})
        base_asr = self._base_cfg.get("voice", {}).get("cloud_asr", {})
        local_model = Path(str(base_tts.get("model_dir") or ""))
        return {
            "llm": llm_entries,
            "asr": [
                {"provider": "local", "models": ["sherpa-onnx"], "credential_ready": True},
                {
                    "provider": str(base_asr.get("provider") or "volcengine"),
                    "models": [str(base_asr.get("model") or "bigmodel")],
                    "credential_ready": bool(
                        base_asr.get("api_key")
                        or (base_asr.get("app_id") and base_asr.get("access_token"))
                    ),
                },
            ],
            "tts": [
                {
                    "backend": "minimax",
                    "models": [str(base_tts.get("minimax_tts_model") or "")],
                    "credential_ready": bool(base_tts.get("minimax_api_key")),
                },
                {
                    "backend": "volcengine",
                    "models": [str(base_tts.get("volcengine_tts_resource_id") or "")],
                    "credential_ready": bool(
                        (
                            base_tts.get("volcengine_tts_api_key")
                            or (
                                base_tts.get("volcengine_tts_app_id")
                                and base_tts.get("volcengine_tts_access_key")
                            )
                        )
                        and base_tts.get("volcengine_tts_resource_id")
                        and base_tts.get("volcengine_tts_speaker")
                    ),
                },
                {
                    "backend": "edge",
                    "models": [str(base_tts.get("voice") or "")],
                    "credential_ready": importlib.util.find_spec("edge_tts") is not None,
                },
                {
                    "backend": "local",
                    "models": [local_model.name],
                    "credential_ready": local_model.is_dir(),
                },
            ],
            "memory": memory.get("backend_dependencies", {}),
        }

    @staticmethod
    def _runtime_issues(
        audio: dict[str, Any],
        memory: dict[str, Any],
        prompt: dict[str, Any],
        interaction: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if not audio.get("input_ready"):
            issues.append(
                {"id": "audio_input", "severity": "critical", "label": "麦克风输入未就绪"}
            )
        if not audio.get("output_ready"):
            issues.append({"id": "audio_output", "severity": "critical", "label": "语音输出未就绪"})
        if memory and not memory.get("ready"):
            issues.append(
                {"id": "memory_not_ready", "severity": "high", "label": "记忆检索尚未就绪"}
            )
        elif memory.get("status") == "degraded":
            issues.append(
                {"id": "memory_degraded", "severity": "medium", "label": "记忆正在使用降级后端"}
            )
        asr_error = str(audio.get("asr", {}).get("cloud", {}).get("last_error") or "")
        if asr_error and "45000081" not in asr_error:
            issues.append(
                {"id": "asr_provider_error", "severity": "medium", "label": asr_error[:120]}
            )
        pending = audio.get("pending_runtime_updates", {})
        if any(bool(value) for value in pending.values()):
            issues.append(
                {
                    "id": "runtime_switch_pending",
                    "severity": "info",
                    "label": "模型切换将在当前语音轮次结束后生效",
                }
            )
        if prompt.get("relay_compat_mode"):
            issues.append(
                {
                    "id": "relay_compat_prompt",
                    "severity": "info",
                    "label": "Prompt 正在使用旧中继兼容模式",
                }
            )
        policy = (interaction or {}).get("policy", {})
        if policy and policy.get("mode") != "strict_public_site":
            issues.append(
                {
                    "id": "ambient_admission_permissive",
                    "severity": "medium",
                    "label": "对话准入仍允许未称呼小算的模糊现场语音",
                }
            )
        return issues

    def _persist_control_state(self) -> None:
        saved = self._state_store.save(self._control_state)
        self._control_state = {
            key: value for key, value in saved.items() if key in {"llm", "asr", "tts", "prompt"}
        }

    def health(self) -> dict[str, Any]:
        audio_status = _status_snapshot(self._audio)
        bridge_status = _status_snapshot(self._voice_gateway)
        readiness = _voice_product_readiness(
            audio_status,
            bridge_status,
            self._voice_cfg,
        )
        return {
            "status": "ok" if readiness["ready"] else "degraded",
            "voice_mode": True,
            "product_readiness": readiness,
            "audio": audio_status,
            "runtime_bridge": bridge_status,
            "interaction_gate": {
                "enabled": bool(getattr(self._interaction_gate, "enabled", False)),
                "silent_on_ambiguous": bool(
                    getattr(self._interaction_gate, "silent_on_ambiguous", False)
                ),
                "allow_unaddressed_public_help": bool(
                    getattr(
                        self._interaction_gate,
                        "allow_unaddressed_public_help",
                        True,
                    )
                ),
                "allow_unaddressed_robot_tasks": bool(
                    getattr(
                        self._interaction_gate,
                        "allow_unaddressed_robot_tasks",
                        True,
                    )
                ),
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
            if world_state is not None and hasattr(world_state, "get_fact")
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


def _build_voice_task_lifecycle(
    cfg: dict[str, Any],
    registry: ModuleRegistry,
) -> VoiceTaskLifecycleService | None:
    runtime_module = registry.get("runtime_handoff")
    if runtime_module is None or not bool(getattr(runtime_module, "enabled", True)):
        return None
    handoff_service = getattr(runtime_module, "runtime_handoff_service", None)
    supervisor = getattr(runtime_module, "external_task_supervisor", None)
    if handoff_service is None or supervisor is None:
        return None

    handoff_cfg = cfg.get("runtime_handoff", {}) if isinstance(cfg, dict) else {}
    if not isinstance(handoff_cfg, dict):
        handoff_cfg = {}
    lifecycle_cfg = handoff_cfg.get("voice_task", {})
    if not isinstance(lifecycle_cfg, dict):
        lifecycle_cfg = {}
    if not bool(lifecycle_cfg.get("enabled", True)):
        return None
    run_service = getattr(handoff_service, "run_service", None)
    if not bool(getattr(run_service, "durable_store_ready", False)):
        raise ValueError(
            "runtime_handoff.voice_task requires a durable TaskRun store with "
            "swallow_errors=false"
        )
    operator_cfg = lifecycle_cfg.get("operator", {})
    if not isinstance(operator_cfg, dict):
        operator_cfg = {}
    roles_cfg = operator_cfg.get("roles", [])
    if isinstance(roles_cfg, str):
        roles_cfg = [roles_cfg]
    if not isinstance(roles_cfg, (list, tuple, set)):
        roles_cfg = []
    roles = tuple(
        dict.fromkeys(str(role).strip().lower() for role in roles_cfg if str(role).strip())
    )
    permissions_cfg = operator_cfg.get("permissions", [])
    if isinstance(permissions_cfg, str):
        permissions_cfg = [permissions_cfg]
    if not isinstance(permissions_cfg, (list, tuple, set)):
        permissions_cfg = []
    permissions = tuple(
        dict.fromkeys(
            str(permission).strip().lower()
            for permission in permissions_cfg
            if str(permission).strip()
        )
    )
    operator_context = None
    session_scope = str(operator_cfg.get("session_scope") or "per_turn").strip().lower()
    if operator_cfg and session_scope == "single_operator":
        operator_context = VoiceTaskOperatorContext(
            operator_id=str(operator_cfg.get("operator_id") or "").strip(),
            roles=roles,
            authenticated=operator_cfg.get("authenticated") is True,
            source=str(operator_cfg.get("source") or "").strip(),
            person_id=str(operator_cfg.get("person_id") or "").strip(),
            permissions=permissions,
        )
    mission_module = registry.get("mission")
    mission_service = getattr(mission_module, "mission_service", None)
    return VoiceTaskLifecycleService(
        handoff_service=handoff_service,
        supervisor=supervisor,
        mission_service=mission_service,
        operator_context=operator_context,
        approval_ttl_s=float(lifecycle_cfg.get("approval_ttl_seconds", 60.0)),
        clarification_ttl_s=float(lifecycle_cfg.get("clarification_ttl_seconds", 45.0)),
        delivery_ttl_s=float(lifecycle_cfg.get("delivery_ttl_seconds", 120.0)),
        delivery_retry_delay_s=float(lifecycle_cfg.get("delivery_retry_delay_seconds", 0.25)),
        max_delivery_attempts=int(lifecycle_cfg.get("max_delivery_attempts", 3)),
    )


def _build_voice_task_operator_provider(
    cfg: dict[str, Any],
    audio: AudioFrontendPort,
    lifecycle: VoiceTaskLifecycleService | None,
) -> Callable[[str, str], VoiceTaskOperatorContext | None] | None:
    """Resolve identity per captured turn; static identity needs explicit session scope."""

    if lifecycle is None:
        return None
    handoff_cfg = cfg.get("runtime_handoff", {}) if isinstance(cfg, dict) else {}
    if not isinstance(handoff_cfg, dict):
        handoff_cfg = {}
    lifecycle_cfg = handoff_cfg.get("voice_task", {})
    if not isinstance(lifecycle_cfg, dict):
        lifecycle_cfg = {}
    operator_cfg = lifecycle_cfg.get("operator", {})
    if not isinstance(operator_cfg, dict):
        operator_cfg = {}
    static_context = lifecycle.default_operator_context
    single_operator_session = (
        str(operator_cfg.get("session_scope") or "").strip().lower()
        == "single_operator"
    )
    resolver = getattr(audio, "voice_task_operator_context_for_turn", None)

    def _provider(session_id: str, turn_id: str) -> VoiceTaskOperatorContext | None:
        if callable(resolver):
            payload = resolver(session_id, turn_id)
            context = VoiceTaskOperatorContext.from_mapping(payload)
            if context is not None:
                return context
        if single_operator_session:
            return static_context
        return None

    return _provider


def _build_mission_context_provider(
    cfg: dict[str, Any],
    registry: ModuleRegistry,
) -> Any:
    voice_cfg = cfg.get("voice", {}) if isinstance(cfg, dict) else {}
    gate_cfg = voice_cfg.get("interaction_gate", {}) if isinstance(voice_cfg, dict) else {}
    default_mode = str(gate_cfg.get("default_mission_mode", "idle"))
    actor_role = str(gate_cfg.get("default_actor_role", "operator"))

    def _provider() -> dict[str, Any]:
        safety_mod = registry.get("safety")
        safety_health = getattr(safety_mod, "health", None)
        if callable(safety_health):
            snapshot = safety_health()
            if isinstance(snapshot, dict) and snapshot.get("estop_active") is True:
                return {
                    "mission_mode": "emergency",
                    "actor_role": actor_role,
                    "source": "safety",
                    "runtime_state": "estop_active",
                }

        runtime_mod = registry.get("runtime_handoff")
        service = getattr(runtime_mod, "runtime_handoff_service", None)
        run_service = getattr(service, "run_service", None)
        active_run = getattr(run_service, "active_run", None)
        run = active_run() if callable(active_run) else None
        if run is None:
            return {
                "mission_mode": default_mode,
                "actor_role": actor_role,
                "source": "config",
                "runtime_state": "",
            }

        runtime_state = str(getattr(run, "current_state", "") or "").strip().lower()
        if runtime_state == "paused":
            mission_mode = "paused"
        elif runtime_state in {
            "created",
            "submitted",
            "validating",
            "preflight",
            "queued",
            "executing",
            "cancel_requested",
        }:
            mission_mode = "mission_active"
        else:
            mission_mode = default_mode
        return {
            "mission_mode": mission_mode,
            "actor_role": actor_role,
            "source": "runtime_handoff",
            "runtime_state": runtime_state,
        }

    return _provider


def _voice_product_readiness(
    audio_status: dict[str, Any],
    bridge_status: dict[str, Any],
    voice_cfg: dict[str, Any],
) -> dict[str, Any]:
    readiness_cfg = voice_cfg.get("product_readiness", {})
    if not isinstance(readiness_cfg, dict):
        readiness_cfg = {}
    require_wake_word = bool(readiness_cfg.get("require_wake_word", True))
    require_runtime_bridge = bool(readiness_cfg.get("require_runtime_bridge", False))
    blockers: list[str] = []

    if audio_status.get("input_ready") is False:
        blockers.append("voice_input_not_ready")
    if audio_status.get("output_ready") is False:
        blockers.append("voice_output_not_ready")
    if audio_status.get("pipeline_ok") is False:
        blockers.append("voice_pipeline_not_ready")
    if require_wake_word and audio_status.get("wake_word_enabled") is not True:
        blockers.append("wake_word_not_ready")
    if require_runtime_bridge and (
        bridge_status.get("enabled") is not True or bridge_status.get("circuit_open") is True
    ):
        blockers.append("runtime_bridge_not_ready")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "degraded_mode": (
            "kws_unavailable_safety_only"
            if audio_status.get("kws_unavailable_safety_only") is True
            else None
        ),
        "requirements": {
            "wake_word": require_wake_word,
            "runtime_bridge": require_runtime_bridge,
        },
    }


def _status_snapshot(component: Any) -> dict[str, Any]:
    snapshot = getattr(component, "status_snapshot", None)
    if not callable(snapshot):
        return {}
    try:
        payload = snapshot()
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
        }
    return dict(payload) if isinstance(payload, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
