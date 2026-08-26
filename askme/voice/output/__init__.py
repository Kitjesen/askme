"""Voice synthesis and cached audio output package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CachedPhrase": ("askme.voice.output.phrase_cache", "CachedPhrase"),
    "PhraseAudioCache": (
        "askme.voice.output.phrase_cache",
        "PhraseAudioCache",
    ),
    "TTSEngine": ("askme.voice.output.tts", "TTSEngine"),
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
