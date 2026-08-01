"""Adapt built runtime apps to the stable MCP context contract.

MCP tools consume :class:`askme.mcp.context.AppContext` fields. Runtime modules
own richer in-process objects. This adapter is the narrow bridge between those
two surfaces: it reads already-built runtime modules and populates MCP-facing
fields without constructing providers or importing hardware implementations.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from askme.mcp.context import AppContext
from askme.runtime.core.profiles import MCP_PROFILE


def app_context_from_runtime_app(
    runtime_app: Any,
    *,
    config: dict[str, Any] | None = None,
    runtime_profile: dict[str, Any] | None = None,
) -> AppContext:
    """Return an MCP ``AppContext`` backed by an already-built runtime app."""

    ctx = AppContext(
        config=dict(config or {}),
        runtime_app=runtime_app,
        runtime_profile=dict(runtime_profile or MCP_PROFILE.snapshot()),
    )
    bind_runtime_app_to_context(ctx, runtime_app)
    return ctx


def bind_runtime_app_to_context(ctx: AppContext, runtime_app: Any) -> AppContext:
    """Populate *ctx* fields from runtime modules and return *ctx*."""

    ctx.runtime_app = runtime_app

    llm_mod = _module(runtime_app, "llm")
    memory_mod = _module(runtime_app, "memory")
    tools_mod = _module(runtime_app, "tools")
    skill_mod = _module(runtime_app, "skill")
    perception_mod = _module(runtime_app, "perception")
    voice_mod = _module(runtime_app, "voice")
    control_mod = _module(runtime_app, "control")

    ctx.llm_client = _pick(llm_mod, "llm_client")

    ctx.conversation = _pick(memory_mod, "conversation")
    ctx.memory_bridge = _pick(memory_mod, "memory_bridge")
    ctx.session_memory = _pick(memory_mod, "session_memory")
    ctx.episodic_memory = _pick(memory_mod, "episodic_memory", "episodic")

    ctx.tool_registry = _pick(tools_mod, "registry", "tool_registry")
    ctx.navigation_client = _pick(tools_mod, "navigation_client")
    ctx.temporal_memory_client = _pick(tools_mod, "temporal_memory_client")

    ctx.skill_manager = _pick(skill_mod, "skill_manager")
    ctx.skill_executor = _pick(skill_mod, "skill_executor")

    ctx.vision_bridge = _pick(perception_mod, "vision_bridge", "vision")
    ctx.scene_intelligence = _pick(perception_mod, "scene_intelligence")

    ctx.robot_control_client = _pick(control_mod, "control_client", "client", "control")
    ctx.arm_controller = _arm_controller_from_module(control_mod)
    ctx.robot_enabled = ctx.arm_controller is not None

    _bind_voice(ctx, voice_mod)
    return ctx


@dataclass
class AudioFrontendVoiceIO:
    """Blocking MCP voice I/O wrapper around a runtime audio frontend."""

    audio: Any

    def listen_once(self) -> str | None:
        listen_once = getattr(self.audio, "listen_once", None)
        if callable(listen_once):
            result = listen_once()
            return str(result) if result is not None else None

        listen_loop = getattr(self.audio, "listen_loop", None)
        if callable(listen_loop):
            result = listen_loop()
            return str(result) if result is not None else None

        return None

    def speak_and_wait(self, text: str) -> None:
        direct = getattr(self.audio, "speak_and_wait", None)
        if callable(direct) and not inspect.iscoroutinefunction(direct):
            _run_sync_or_awaitable(direct(text))
            return

        speak = getattr(self.audio, "speak", None)
        start = getattr(self.audio, "start_playback", None)
        wait = getattr(self.audio, "wait_speaking_done", None)
        stop = getattr(self.audio, "stop_playback", None)

        if callable(speak):
            speak(text)
        if callable(start):
            start()
        try:
            if callable(wait):
                wait()
        finally:
            if callable(stop):
                stop()

    def shutdown(self) -> None:
        shutdown = getattr(self.audio, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _module(runtime_app: Any, name: str) -> Any | None:
    modules = getattr(runtime_app, "modules", None)
    if isinstance(modules, dict):
        return modules.get(name)
    get = getattr(runtime_app, "get", None)
    if callable(get):
        return get(name)
    return getattr(runtime_app, name, None)


def _pick(obj: Any | None, *names: str) -> Any | None:
    if obj is None:
        return None
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    return None


def _arm_controller_from_module(control_mod: Any | None) -> Any | None:
    candidate = _pick(
        control_mod,
        "arm_controller",
        "arm",
        "controller",
        "control_client",
        "client",
    )
    if candidate is None:
        return None
    required = ("execute", "get_state", "emergency_stop")
    if all(callable(getattr(candidate, name, None)) for name in required):
        return candidate
    return None


def _bind_voice(ctx: AppContext, voice_mod: Any | None) -> None:
    voice_io = _pick(voice_mod, "voice_io")
    audio = _pick(voice_mod, "audio")
    if voice_io is None and audio is not None:
        voice_io = AudioFrontendVoiceIO(audio)

    ctx.voice_io = voice_io
    ctx.tts_engine = (
        _pick(voice_mod, "tts_engine", "tts_provider") or _pick(audio, "tts")
        if audio is not None
        else _pick(voice_mod, "tts_engine", "tts_provider")
    )
    ctx.asr_engine = _pick(voice_mod, "asr_engine", "asr_provider")
    ctx.vad_engine = _pick(voice_mod, "vad_engine", "vad_provider")
    ctx.voice_enabled = voice_io is not None or ctx.tts_engine is not None


def _run_sync_or_awaitable(result: Any) -> None:
    if not inspect.isawaitable(result):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(result)
        return
    raise RuntimeError("async audio speak_and_wait cannot be driven from a synchronous MCP call")


__all__ = [
    "AudioFrontendVoiceIO",
    "app_context_from_runtime_app",
    "bind_runtime_app_to_context",
]
