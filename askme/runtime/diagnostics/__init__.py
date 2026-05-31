"""Runtime-level diagnostic smoke checks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "print_dialogue_burst_summary": (
        "askme.runtime.diagnostics.dialogue_smoke",
        "print_dialogue_burst_summary",
    ),
    "print_dialogue_smoke_summary": (
        "askme.runtime.diagnostics.dialogue_smoke",
        "print_dialogue_smoke_summary",
    ),
    "run_dialogue_burst_sync": (
        "askme.runtime.diagnostics.dialogue_smoke",
        "run_dialogue_burst_sync",
    ),
    "run_dialogue_smoke_sync": (
        "askme.runtime.diagnostics.dialogue_smoke",
        "run_dialogue_smoke_sync",
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
