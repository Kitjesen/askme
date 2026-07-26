"""Core LLM gateway primitives exposed without eager import cycles."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.llm.core.client import LLMClient
    from askme.llm.core.config import LLMConfig
    from askme.llm.core.contracts import (
        LLMCallContext,
        LLMDeadlineExceeded,
        LLMNoSemanticResponse,
        LLMProvider,
        LLMRequest,
        Message,
        ToolSpec,
    )
    from askme.llm.core.factory import (
        available_llm_providers,
        create_llm_provider,
        resolve_provider_name,
    )
    from askme.llm.core.gateway import LLMGateway

_LAZY_EXPORTS = {
    "LLMCallContext": ("askme.llm.core.contracts", "LLMCallContext"),
    "LLMDeadlineExceeded": ("askme.llm.core.contracts", "LLMDeadlineExceeded"),
    "LLMNoSemanticResponse": (
        "askme.llm.core.contracts",
        "LLMNoSemanticResponse",
    ),
    "LLMClient": ("askme.llm.core.client", "LLMClient"),
    "LLMConfig": ("askme.llm.core.config", "LLMConfig"),
    "LLMGateway": ("askme.llm.core.gateway", "LLMGateway"),
    "LLMProvider": ("askme.llm.core.contracts", "LLMProvider"),
    "LLMRequest": ("askme.llm.core.contracts", "LLMRequest"),
    "Message": ("askme.llm.core.contracts", "Message"),
    "ToolSpec": ("askme.llm.core.contracts", "ToolSpec"),
    "available_llm_providers": (
        "askme.llm.core.factory",
        "available_llm_providers",
    ),
    "create_llm_provider": ("askme.llm.core.factory", "create_llm_provider"),
    "resolve_provider_name": ("askme.llm.core.factory", "resolve_provider_name"),
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
