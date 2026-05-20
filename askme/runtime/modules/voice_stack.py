"""Runtime-local voice stack assembly shared by text and voice modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from askme.ports import AudioFrontendPort, AudioRouterPort
from askme.providers import build_audio_frontend, build_voice_runtime_bridge
from askme.voice_gateway import VoiceGatewayService


@dataclass(frozen=True)
class RuntimeVoiceStack:
    """Runtime-owned voice services built from provider ports."""

    audio: AudioFrontendPort
    audio_router: AudioRouterPort | None
    asr_provider: Any | None
    tts_provider: Any | None
    router: Any
    address_detector: Any
    interaction_gate: Any
    voice_runtime_bridge: Any
    voice_gateway: Any


def build_runtime_voice_stack(
    cfg: dict[str, Any],
    *,
    voice_mode: bool,
    metrics: Any | None,
    skill_manager: Any | None,
) -> RuntimeVoiceStack:
    """Build audio frontend, intent router, and voice gateway for runtime modules."""

    from askme.robot_interaction import (
        AddressDetector,
        IntentRouter,
        InteractionGate,
    )

    provider_stack = build_audio_frontend(
        cfg,
        voice_mode=voice_mode,
        metrics=metrics,
    )
    voice_triggers = skill_manager.get_voice_triggers() if skill_manager else {}
    router = IntentRouter(voice_triggers=voice_triggers)
    address_detector = AddressDetector(cfg.get("voice", {}).get("address_detection", {}))
    interaction_gate = InteractionGate(cfg.get("voice", {}).get("interaction_gate", {}))
    runtime_bridge = build_voice_runtime_bridge(
        cfg.get("runtime", {}).get("voice_bridge", {})
    )
    voice_gateway = VoiceGatewayService(runtime_bridge)
    return RuntimeVoiceStack(
        audio=provider_stack.audio,
        audio_router=provider_stack.audio_router,
        asr_provider=provider_stack.asr,
        tts_provider=provider_stack.tts,
        router=router,
        address_detector=address_detector,
        interaction_gate=interaction_gate,
        voice_runtime_bridge=runtime_bridge,
        voice_gateway=voice_gateway,
    )


def runtime_voice_stack_from_module(voice_mod: Any) -> RuntimeVoiceStack:
    """Adapt an already-built VoiceModule to the shared stack shape."""

    voice_gateway = getattr(voice_mod, "voice_gateway", None)
    voice_runtime_bridge = getattr(voice_mod, "voice_runtime_bridge", None)
    if voice_gateway is None and voice_runtime_bridge is not None:
        voice_gateway = VoiceGatewayService(voice_runtime_bridge)
    return RuntimeVoiceStack(
        audio=getattr(voice_mod, "audio", None),
        audio_router=getattr(voice_mod, "audio_router", None),
        asr_provider=getattr(voice_mod, "asr_provider", None),
        tts_provider=getattr(voice_mod, "tts_provider", None),
        router=getattr(voice_mod, "router", None),
        address_detector=getattr(voice_mod, "address_detector", None),
        interaction_gate=getattr(voice_mod, "interaction_gate", None),
        voice_runtime_bridge=voice_runtime_bridge,
        voice_gateway=voice_gateway,
    )
