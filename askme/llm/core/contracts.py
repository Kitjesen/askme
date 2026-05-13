"""Contracts for product-grade LLM access.

Business code should depend on these contracts instead of constructing a
provider client directly.  The concrete provider can be MiniMax, another
OpenAI-compatible endpoint, or a fake test provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]
ToolSpec = dict[str, Any]


@dataclass(frozen=True)
class LLMRequest:
    """Normalized request metadata before it reaches a model provider."""

    messages: Sequence[Message]
    model: str
    temperature: float
    stream: bool
    max_tokens: int = 0
    tools: list[ToolSpec] | None = None
    tool_choice: str | None = None
    thinking: bool = False
    cancel_token: asyncio.Event | None = field(default=None, compare=False)


@dataclass(frozen=True)
class LLMCallContext:
    """Audit metadata for an LLM call.

    This is intentionally small.  It is safe to log and useful for tracing
    latency, model choice, task purpose, and whether a response was grounded.
    """

    purpose: str = "general"
    operator_id: str | None = None
    session_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


class LLMProvider(Protocol):
    """Provider-level OpenAI-compatible transport."""

    @property
    def raw_client(self) -> Any:
        """Underlying SDK client for low-level integrations."""

    @property
    def minimax_client(self) -> Any | None:
        """Optional secondary MiniMax client."""

    def client_for_model(self, model: str) -> Any:
        """Return the SDK client that should serve this model."""

    async def stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncIterator[Any]:
        """Create a streaming completion with transport-level retry."""

    async def completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Create a non-streaming completion with transport-level retry."""
