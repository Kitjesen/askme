"""Voice sub-package -- ASR, VAD, KWS, TTS engines and AudioAgent.

Hardware-dependent modules are loaded lazily so importing ``askme.voice``
does not initialize optional audio/ASR bindings such as ``sherpa_onnx``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ASREngine": ("askme.voice.asr", "ASREngine"),
    "AudioAgent": ("askme.voice.audio_agent", "AudioAgent"),
    "KWSEngine": ("askme.voice.kws", "KWSEngine"),
    "InterruptDecision": ("askme.voice.media_contracts", "InterruptDecision"),
    "InterruptRequest": ("askme.voice.media_contracts", "InterruptRequest"),
    "VoiceMediaFrame": ("askme.voice.media_contracts", "VoiceMediaFrame"),
    "VoiceMediaStatus": ("askme.voice.media_contracts", "VoiceMediaStatus"),
    "VoiceTurnEvent": ("askme.voice.media_contracts", "VoiceTurnEvent"),
    "VoiceTurnEventType": ("askme.voice.media_contracts", "VoiceTurnEventType"),
    "StreamSplitter": ("askme.voice.stream_splitter", "StreamSplitter"),
    "TTSEngine": ("askme.voice.tts", "TTSEngine"),
    "VADEngine": ("askme.voice.vad", "VADEngine"),
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
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
