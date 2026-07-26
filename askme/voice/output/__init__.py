"""Voice output package.

Public imports:

    from askme.voice.output import TTSEngine, AudioRouter
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AudioErrorKind": ("askme.voice.output.audio_router", "AudioErrorKind"),
    "AudioRouter": ("askme.voice.output.audio_router", "AudioRouter"),
    "TTSEngine": ("askme.voice.output.tts", "TTSEngine"),
    "VoiceProfile": ("askme.voice.output.voice_profiles", "VoiceProfile"),
    "build_voice_profiles": ("askme.voice.output.voice_profiles", "build_voice_profiles"),
    "resolve_voice_profile_id": ("askme.voice.output.voice_profiles", "resolve_voice_profile_id"),
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
