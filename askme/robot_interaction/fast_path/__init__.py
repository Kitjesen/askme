"""Deterministic interaction fast-path policies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "FastVoiceIntent": (
        "askme.robot_interaction.fast_path.voice_intents",
        "FastVoiceIntent",
    ),
    "FastVoiceIntentKind": (
        "askme.robot_interaction.fast_path.voice_intents",
        "FastVoiceIntentKind",
    ),
    "default_cached_phrases": (
        "askme.robot_interaction.fast_path.voice_intents",
        "default_cached_phrases",
    ),
    "match_fast_voice_intent": (
        "askme.robot_interaction.fast_path.voice_intents",
        "match_fast_voice_intent",
    ),
    "normalize_fast_voice_text": (
        "askme.robot_interaction.fast_path.voice_intents",
        "normalize_fast_voice_text",
    ),
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
