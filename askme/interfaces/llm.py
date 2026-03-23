"""LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from askme.runtime.registry import BackendRegistry


class LLMBackend(ABC):
    """Abstract LLM interface — chat, stream, tool calling."""

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Single-turn chat completion. Returns response text."""

    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        """Streaming chat completion. Yields tokens."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model identifier."""

    def supports_tools(self) -> bool:
        """Whether this backend supports tool/function calling."""
        return False

    def supports_vision(self) -> bool:
        """Whether this backend supports image inputs."""
        return False


llm_registry = BackendRegistry("llm", LLMBackend, default="minimax")
