"""Compatibility import for the product LLM client."""

from askme.llm.core.client import LLMClient, _backoff

__all__ = ["LLMClient", "_backoff"]
