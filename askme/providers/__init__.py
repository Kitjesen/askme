"""Provider adapter layer for cloud, local, and in-house services."""

from __future__ import annotations

from importlib import import_module

from askme.providers.arm_control import build_arm_control, get_arm_safety_defaults
from askme.providers.led import StateLedBridge, build_led_controller, build_status_led
from askme.providers.perception import (
    PerceptionStack,
    analyze_image_base64,
    build_perception,
    build_scene_intelligence,
    capture_snapshot_payload,
    read_depth_info,
)
from askme.providers.register_defaults import register_default_provider_backends
from askme.providers.robot_control import build_robot_control
from askme.providers.safety import build_safety
from askme.providers.spatial import build_navigation, build_temporal_memory
from askme.providers.telemetry import build_bus
from askme.providers.voice import (
    VoiceProviderStack,
    build_audio_frontend,
    build_edge_voice_io,
    build_speech_playback,
    build_tts_provider,
    build_voice_runtime_bridge,
    resolve_voice_profile_id,
)

__all__ = [
    "PerceptionStack",
    "StateLedBridge",
    "VoiceProviderStack",
    "analyze_image_base64",
    "build_arm_control",
    "build_audio_frontend",
    "build_bus",
    "build_edge_voice_io",
    "build_speech_playback",
    "build_navigation",
    "build_tts_provider",
    "build_voice_runtime_bridge",
    "build_scene_intelligence",
    "get_arm_safety_defaults",
    "build_led_controller",
    "build_perception",
    "build_robot_control",
    "build_safety",
    "build_status_led",
    "build_temporal_memory",
    "capture_snapshot_payload",
    "resolve_voice_profile_id",
    "read_depth_info",
    "register_default_provider_backends",
]


def __getattr__(name: str):
    if name == "voice_runtime":
        module = import_module("askme.providers.voice_runtime")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
