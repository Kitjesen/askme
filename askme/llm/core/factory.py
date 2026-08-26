"""Factory for LLM gateway dependencies."""

from __future__ import annotations

from collections.abc import Callable

from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMProvider
from askme.llm.providers.domestic import (
    DashScopeProvider,
    DeepSeekProvider,
    DoubaoProvider,
    MiniMaxProvider,
    ZhipuProvider,
)
from askme.llm.providers.fake import FakeLLMProvider
from askme.llm.providers.litellm import LiteLLMProxyProvider
from askme.llm.providers.openai_compatible import OpenAICompatibleProvider
from askme.llm.providers.profiles import normalize_provider_name
from askme.llm.streaming.retry import default_backoff

_PROVIDER_CLASSES = {
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "litellm": LiteLLMProxyProvider,
    "minimax": MiniMaxProvider,
    "doubao": DoubaoProvider,
    "dashscope": DashScopeProvider,
    "deepseek": DeepSeekProvider,
    "zhipu": ZhipuProvider,
}


def create_llm_provider(
    config: LLMConfig,
    *,
    backoff_func: Callable[[int], float] = default_backoff,
) -> LLMProvider:
    """Create the provider transport for the configured LLM backend."""

    provider_name = resolve_provider_name(config)
    if provider_name == "fake":
        text = str(config.provider_options.get("response_text", "ok"))
        return FakeLLMProvider(response_text=text)

    cls = _PROVIDER_CLASSES.get(provider_name)
    if cls is None:
        available = ", ".join(available_llm_providers())
        raise KeyError(f"Unknown LLM provider {provider_name!r}. Available: {available}")
    return cls(config, backoff_func=backoff_func)


def resolve_provider_name(config: LLMConfig) -> str:
    explicit = normalize_provider_name(config.provider)
    if not explicit:
        raise ValueError("LLMConfig.provider is required; implicit provider inference is disabled")
    return explicit


def available_llm_providers() -> list[str]:
    return sorted([*_PROVIDER_CLASSES.keys(), "fake"])
