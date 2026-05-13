"""Domestic OpenAI-compatible provider adapters.

These classes are intentionally thin.  MiniMax, Doubao Ark, DashScope/Qwen,
DeepSeek, and Zhipu all expose OpenAI-compatible chat APIs in common
deployments, but their endpoints and keys must remain configuration-owned.
The class name gives the product/runtime a stable provider identity without
hardcoding customer credentials or deployment URLs.
"""

from __future__ import annotations

from askme.llm.providers.openai_compatible import OpenAICompatibleProvider


class MiniMaxProvider(OpenAICompatibleProvider):
    provider_name = "minimax"


class DoubaoProvider(OpenAICompatibleProvider):
    provider_name = "doubao"


class DashScopeProvider(OpenAICompatibleProvider):
    provider_name = "dashscope"


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_name = "deepseek"


class ZhipuProvider(OpenAICompatibleProvider):
    provider_name = "zhipu"
