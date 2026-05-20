"""Voice input package.

Public imports:

    from askme.voice.input import ASREngine, MicInput, VADController
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ASREngine": ("askme.voice.input.asr", "ASREngine"),
    "ASRManager": ("askme.voice.input.asr_manager", "ASRManager"),
    "ASRResult": ("askme.voice.input.asr_manager", "ASRResult"),
    "AddressDetector": ("askme.robot_interaction.address_detector", "AddressDetector"),
    "AudioFilter": ("askme.voice.input.audio_filter", "AudioFilter"),
    "AudioProcessor": ("askme.voice.input.audio_processor", "AudioProcessor"),
    "CloudASR": ("askme.voice.input.cloud_asr", "CloudASR"),
    "KWSEngine": ("askme.voice.input.kws", "KWSEngine"),
    "MicInput": ("askme.voice.input.mic_input", "MicInput"),
    "NoiseGateCalibrator": ("askme.voice.input.noise_reduction", "NoiseGateCalibrator"),
    "SpectralSubtractor": ("askme.voice.input.noise_reduction", "SpectralSubtractor"),
    "VADEngine": ("askme.voice.input.vad", "VADEngine"),
    "VADController": ("askme.voice.input.vad_controller", "VADController"),
    "VADEvent": ("askme.voice.input.vad_controller", "VADEvent"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
