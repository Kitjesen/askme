"""Voice-control tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "MuteMicTool": ("askme.tools.voice.voice_tools", "MuteMicTool"),
    "StopSpeakingTool": ("askme.tools.voice.voice_tools", "StopSpeakingTool"),
    "UnmuteMicTool": ("askme.tools.voice.voice_tools", "UnmuteMicTool"),
    "register_voice_tools": ("askme.tools.voice.voice_tools", "register_voice_tools"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve voice-control tools on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
