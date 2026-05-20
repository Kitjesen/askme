"""Voice sub-package -- ASR, VAD, KWS, TTS engines and AudioAgent.

Hardware-dependent modules are loaded lazily so importing ``askme.voice``
does not initialize optional audio/ASR bindings such as ``sherpa_onnx``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.voice.address_detector": "askme.robot_interaction.address_detector",
    "askme.voice.asr": "askme.voice.input.asr",
    "askme.voice.asr_manager": "askme.voice.input.asr_manager",
    "askme.voice.audio_agent": "askme.voice.orchestration.audio_agent",
    "askme.voice.audio_devices": "askme.voice.diagnostics.audio_devices",
    "askme.voice.audio_filter": "askme.voice.input.audio_filter",
    "askme.voice.audio_processor": "askme.voice.input.audio_processor",
    "askme.voice.audio_router": "askme.voice.output.audio_router",
    "askme.voice.cloud_asr": "askme.voice.input.cloud_asr",
    "askme.voice.generated_contracts": "askme.voice.core.generated_contracts",
    "askme.voice.health_check": "askme.voice.diagnostics.health_check",
    "askme.voice.interaction_gate": "askme.robot_interaction.interaction_gate",
    "askme.voice.kws": "askme.voice.input.kws",
    "askme.voice.media_contracts": "askme.voice.core.media_contracts",
    "askme.voice.mic_calibration": "askme.voice.diagnostics.mic_calibration",
    "askme.voice.mic_input": "askme.voice.input.mic_input",
    "askme.voice.minimax_hybrid": "askme.voice.diagnostics.minimax_hybrid",
    "askme.voice.noise_reduction": "askme.voice.input.noise_reduction",
    "askme.voice.online_smoke": "askme.voice.diagnostics.online_smoke",
    "askme.voice.perception_context": "askme.robot_interaction.perception_context",
    "askme.voice.punctuation": "askme.voice.core.punctuation",
    "askme.voice.runtime_bridge": "askme.voice_gateway.runtime_bridge",
    "askme.voice.s100p_readiness_bundle": "askme.voice.diagnostics.s100p_readiness_bundle",
    "askme.voice.stream_splitter": "askme.voice.core.stream_splitter",
    "askme.voice.sunrise_audio_doctor": "askme.voice.diagnostics.sunrise_audio_doctor",
    "askme.voice.sunrise_readiness": "askme.voice.diagnostics.sunrise_readiness",
    "askme.voice.tts": "askme.voice.output.tts",
    "askme.voice.turn_trace": "askme.voice.core.turn_trace",
    "askme.voice.vad": "askme.voice.input.vad",
    "askme.voice.vad_controller": "askme.voice.input.vad_controller",
    "askme.voice.voice_profiles": "askme.voice.output.voice_profiles",
}


install_legacy_aliases(__name__, _LEGACY_MODULE_ALIASES)

_LAZY_EXPORTS = {
    "ASREngine": ("askme.voice.input.asr", "ASREngine"),
    "AudioAgent": ("askme.voice.orchestration.audio_agent", "AudioAgent"),
    "KWSEngine": ("askme.voice.input.kws", "KWSEngine"),
    "InterruptDecision": ("askme.voice.core.media_contracts", "InterruptDecision"),
    "InterruptRequest": ("askme.voice.core.media_contracts", "InterruptRequest"),
    "VoiceMediaFrame": ("askme.voice.core.media_contracts", "VoiceMediaFrame"),
    "VoiceMediaStatus": ("askme.voice.core.media_contracts", "VoiceMediaStatus"),
    "VoiceTurnEvent": ("askme.voice.core.media_contracts", "VoiceTurnEvent"),
    "VoiceTurnEventType": ("askme.voice.core.media_contracts", "VoiceTurnEventType"),
    "StreamSplitter": ("askme.voice.core.stream_splitter", "StreamSplitter"),
    "TTSEngine": ("askme.voice.output.tts", "TTSEngine"),
    "VADEngine": ("askme.voice.input.vad", "VADEngine"),
}

_OPTIONAL_DEPENDENCY_FALLBACKS = {
    "ASREngine": frozenset({"sherpa_onnx"}),
    "AudioAgent": frozenset({"sounddevice"}),
    "KWSEngine": frozenset({"sherpa_onnx"}),
}

__all__ = [
    "ASREngine",
    "AudioAgent",
    "InterruptDecision",
    "InterruptRequest",
    "KWSEngine",
    "StreamSplitter",
    "TTSEngine",
    "VADEngine",
    "VoiceMediaFrame",
    "VoiceMediaStatus",
    "VoiceTurnEvent",
    "VoiceTurnEventType",
]


def __getattr__(name: str) -> Any:
    """Resolve public voice classes on first access."""
    legacy_module = _LEGACY_MODULE_ALIASES.get(f"{__name__}.{name}")
    if legacy_module:
        value = import_module(legacy_module)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or exc.name not in _OPTIONAL_DEPENDENCY_FALLBACKS.get(
            name,
            frozenset(),
        ):
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
