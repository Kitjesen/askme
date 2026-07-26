"""LLM backend interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from askme.interfaces.core.registry import BackendRegistry

if TYPE_CHECKING:
    from askme.llm.core.contracts import LLMCallContext


class LLMBackend(ABC):
    """Abstract LLM interface — chat, stream, tool calling.

    Matches the public API of :class:`askme.llm.core.client.LLMClient`.
    """

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        context: LLMCallContext | None = None,
    ) -> str:
        """Single-turn chat completion. Returns response text."""

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
        cancel_token: asyncio.Event | None = None,
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        """Streaming chat completion. Yields ``ChatCompletionChunk``."""

    @property
    def model_name(self) -> str:
        """Current model identifier."""
        return getattr(self, "model", "")

    def supports_tools(self) -> bool:
        """Whether this backend supports tool/function calling."""
        return False

    def supports_vision(self) -> bool:
        """Whether this backend supports image inputs."""
        return False


llm_registry = BackendRegistry("llm", LLMBackend, default="minimax")
