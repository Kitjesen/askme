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
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from askme.agent_shell import AgentShell
from askme.llm.core.client import LLMClient
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.core.persona import persona_from_brain_config
from askme.pipeline.skills.skill_dispatcher import SkillDispatcher
from askme.ports import AudioFrontendPort, AudioRouterPort
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
from askme.runtime.modules.voice_stack import build_runtime_voice_stack
from askme.runtime.voice_control import VoiceControlStateStore, deep_merge
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

        self._registry = registry
        self._base_cfg = deepcopy(cfg)
        self._state_store = VoiceControlStateStore(cfg)
        self._control_state = self._state_store.load()
        effective_cfg = self._effective_startup_config(cfg)

        llm_mod = self.llm_in
        self._voice_cfg = dict(effective_cfg.get("voice", {}) or {})
        self._input_retry_seconds = max(
            0.1,
            float(self._voice_cfg.get("input_retry_seconds", 5.0)),
        )
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
        self._mission_context_provider = _build_mission_context_provider(effective_cfg, registry)
        self._voice_loop.set_mission_context_provider(self._mission_context_provider)

        self._task: asyncio.Task[None] | None = None
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
                llm_mod = self._registry.get("llm")
                client = llm_mod.replace_config(brain_cfg)
                self._publish_llm(client, brain_cfg)
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
        self._task = asyncio.create_task(
            self._run_with_input_recovery(),
            name="voice-loop",
        )
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
        return getattr(self._audio, "_asr_mgr", self._asr_provider)

    @property
    def tts_provider(self) -> Any:
        """The TTS provider behind the audio frontend."""
        return getattr(self._audio, "tts", self._tts_provider)

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
            "mode": "strict_public_site" if (
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
            ) else "permissive",
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
        component = str(payload.get("component") or "").strip().lower()
        if component == "llm":
            result = await self._switch_llm(payload)
        elif component == "asr":
            result = await self._switch_asr(payload)
        elif component == "tts":
            result = await self._switch_tts(payload)
        else:
            raise ValueError("component must be one of: llm, asr, tts")
        if result.get("updated"):
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
        llm_mod = self._registry.get("llm")
        if llm_mod is None:
            raise RuntimeError("LLM module is not available")
        candidate = llm_mod.prepare_client(brain_cfg)
        if bool(payload.get("validate", True)):
            await llm_mod.validate_client(
                candidate,
                timeout_s=min(15.0, float(brain_cfg.get("timeout", 10.0))),
            )
        llm_mod.commit_client(candidate)
        self._publish_llm(candidate, brain_cfg)
        self._control_state["llm"] = {
            "provider": candidate.provider_name,
            "model": candidate.model,
            "voice_model": str(brain_cfg.get("voice_model") or candidate.model),
            "fallback_models": list(brain_cfg.get("fallback_models", [])),
        }
        return {
            "updated": True,
            "component": "llm",
            "state": "active",
            "runtime": candidate.provider_status(),
        }

    async def _switch_asr(self, payload: dict[str, Any]) -> dict[str, Any]:
        voice_cfg = self._resolve_asr_voice_config(payload)
        reconfigure = getattr(self._audio, "reconfigure_asr", None)
        if not callable(reconfigure):
            raise RuntimeError("audio frontend does not support ASR hot switching")
        result = await asyncio.to_thread(reconfigure, voice_cfg)
        cloud_cfg = voice_cfg.get("cloud_asr", {})
        provider = "local" if not cloud_cfg.get("enabled") else str(cloud_cfg.get("provider") or "")
        self._voice_cfg = voice_cfg
        self._control_state["asr"] = {
            "provider": provider,
            "model": str(cloud_cfg.get("model") or "local"),
        }
        return dict(result)

    async def _switch_tts(self, payload: dict[str, Any]) -> dict[str, Any]:
        tts_cfg = self._resolve_tts_config(payload)
        reconfigure = getattr(self._audio, "reconfigure_tts", None)
        if not callable(reconfigure):
            raise RuntimeError("audio frontend does not support TTS hot switching")
        result = await asyncio.to_thread(reconfigure, tts_cfg)
        self._voice_cfg["tts"] = tts_cfg
        self._control_state["tts"] = {
            "backend": str(tts_cfg.get("backend") or ""),
            "model": str(tts_cfg.get("minimax_tts_model") or ""),
            "voice_id": str(tts_cfg.get("minimax_voice_id") or ""),
        }
        return dict(result)

    def _publish_llm(self, client: Any, brain_cfg: dict[str, Any]) -> None:
        pipeline = self._pipeline()
        if pipeline is not None:
            pipeline.replace_llm(
                client,
                voice_model=str(brain_cfg.get("voice_model") or client.model),
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
        provider = str(
            payload.get("provider")
            or self._control_state.get("llm", {}).get("provider")
            or base.get("provider")
            or ""
        ).strip().lower()
        presets = self._llm_presets()
        preset = presets.get(provider)
        if preset is None:
            raise ValueError(f"LLM provider is not configured: {provider}")
        resolved = deep_merge(base, preset)
        resolved["provider"] = provider
        resolved["model"] = str(payload.get("model") or preset.get("model") or base.get("model") or "")
        resolved["voice_model"] = str(
            payload.get("voice_model") or resolved["model"]
        )
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
        if brain.get("minimax_api_key"):
            presets["minimax"] = {
                "api_key": brain.get("minimax_api_key", ""),
                "base_url": brain.get("minimax_base_url", "https://api.minimax.chat/v1"),
                "model": brain.get("minimax_model", "MiniMax-M2.5-highspeed"),
                "fallback_models": [],
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
        provider = str(payload.get("provider") or voice_cfg.get("cloud_asr", {}).get("provider") or "local").strip().lower()
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
        if backend not in {"local", "edge", "minimax"}:
            raise ValueError("TTS backend must be one of: local, edge, minimax")
        tts_cfg["backend"] = backend
        if payload.get("model"):
            tts_cfg["minimax_tts_model"] = str(payload["model"])
        if payload.get("voice_id"):
            tts_cfg["minimax_voice_id"] = str(payload["voice_id"])
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
        relay_mode = bool(
            payload.get("relay_compat_mode", current.get("relay_compat_mode", False))
        )
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
                {"backend": "minimax", "models": [str(base_tts.get("minimax_tts_model") or "")], "credential_ready": bool(base_tts.get("minimax_api_key"))},
                {"backend": "edge", "models": [str(base_tts.get("voice") or "")], "credential_ready": importlib.util.find_spec("edge_tts") is not None},
                {"backend": "local", "models": [local_model.name], "credential_ready": local_model.is_dir()},
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
            issues.append({"id": "audio_input", "severity": "critical", "label": "麦克风输入未就绪"})
        if not audio.get("output_ready"):
            issues.append({"id": "audio_output", "severity": "critical", "label": "语音输出未就绪"})
        if memory and not memory.get("ready"):
            issues.append({"id": "memory_not_ready", "severity": "high", "label": "记忆检索尚未就绪"})
        elif memory.get("status") == "degraded":
            issues.append({"id": "memory_degraded", "severity": "medium", "label": "记忆正在使用降级后端"})
        asr_error = str(audio.get("asr", {}).get("cloud", {}).get("last_error") or "")
        if asr_error and "45000081" not in asr_error:
            issues.append({"id": "asr_provider_error", "severity": "medium", "label": asr_error[:120]})
        pending = audio.get("pending_runtime_updates", {})
        if any(bool(value) for value in pending.values()):
            issues.append({"id": "runtime_switch_pending", "severity": "info", "label": "模型切换将在当前语音轮次结束后生效"})
        if prompt.get("relay_compat_mode"):
            issues.append({"id": "relay_compat_prompt", "severity": "info", "label": "Prompt 正在使用旧中继兼容模式"})
        policy = (interaction or {}).get("policy", {})
        if policy and policy.get("mode") != "strict_public_site":
            issues.append({
                "id": "ambient_admission_permissive",
                "severity": "medium",
                "label": "对话准入仍允许未称呼小算的模糊现场语音",
            })
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
        bridge_status.get("enabled") is not True
        or bridge_status.get("circuit_open") is True
    ):
        blockers.append("runtime_bridge_not_ready")

    return {
        "ready": not blockers,
        "blockers": blockers,
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
