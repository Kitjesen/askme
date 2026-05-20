"""Product LLM package.

Root imports stay backward compatible. Real implementation lives under
``core/``, provider adapters under ``providers/``, and runtime policy under
``policy/``. Cross-domain exports are resolved lazily so importing
``askme.llm`` does not eagerly initialize memory or interaction modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ConversationManager": ("askme.memory.core.conversation", "ConversationManager"),
    "IntentRouter": ("askme.robot_interaction.intent_router", "IntentRouter"),
    "LLMClient": ("askme.llm.core.client", "LLMClient"),
    "LLMConfig": ("askme.llm.core.config", "LLMConfig"),
    "LLMGateway": ("askme.llm.core.gateway", "LLMGateway"),
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
