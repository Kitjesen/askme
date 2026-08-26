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
    """Internal correlation and policy context for one LLM call.

    The complete object is not safe to serialize or log because it may contain
    session, operator, or evidence identifiers. Providers must project an
    explicit allowlist before sending context across a process boundary.
    """

    trace_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    call_id: str | None = None
    purpose: str = "general"
    channel: str = "text"
    request_class: str = "text"
    latency_budget_ms: int | None = None
    privacy_class: str = "conversation"
    allow_cache: bool = False
    operator_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


class LLMDeadlineExceeded(TimeoutError):
    """Raised when an LLM call exhausts its product latency budget."""

    def __init__(self, *, phase: str, context: LLMCallContext) -> None:
        self.phase = str(phase)
        self.trace_id = context.trace_id
        self.turn_id = context.turn_id
        self.request_class = context.request_class
        self.latency_budget_ms = context.latency_budget_ms
        super().__init__(
            "LLM deadline exceeded "
            f"during {self.phase} (request_class={self.request_class}, "
            f"budget_ms={self.latency_budget_ms})"
        )


class LLMNoSemanticResponse(RuntimeError):
    """Raised when a completed stream never produces text or a tool call."""

    def __init__(
        self,
        *,
        model_alias: str,
        context: LLMCallContext | None = None,
    ) -> None:
        self.model_alias = str(model_alias)
        self.trace_id = context.trace_id if context is not None else None
        self.turn_id = context.turn_id if context is not None else None
        self.request_class = context.request_class if context is not None else None
        super().__init__(f"LLM stream ended without semantic payload (model={self.model_alias})")


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

    def stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_token: asyncio.Event | None = None,
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        """Create a streaming completion with transport-level retry."""

    async def completion_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        context: LLMCallContext | None = None,
    ) -> Any:
        """Create a non-streaming completion with transport-level retry."""
