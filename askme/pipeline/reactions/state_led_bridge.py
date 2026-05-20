"""Compatibility facade for the status LED state bridge.

Runtime composition should use ``providers.build_status_led``. This module keeps
historical pipeline imports working without importing robot implementations
directly from pipeline code.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.providers import StateLedBridge as StateLedBridge

__all__ = ["StateLedBridge"]


def __getattr__(name: str) -> Any:
    if name != "StateLedBridge":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("askme.providers"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
