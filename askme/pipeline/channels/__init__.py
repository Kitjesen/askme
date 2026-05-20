"""Text and voice channel loop entrypoints.

Use this package for product-facing channel orchestration imports:

    from askme.pipeline.channels import TextLoop, VoiceLoop
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CommandHandler": ("askme.pipeline.channels.commands", "CommandHandler"),
    "TextLoop": ("askme.pipeline.channels.text_loop", "TextLoop"),
    "VoiceLoop": ("askme.pipeline.channels.voice_loop", "VoiceLoop"),
    "record_external_turn": (
        "askme.pipeline.channels.external_turns",
        "record_external_turn",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if name != "VoiceLoop" or exc.name == module_name:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
