"""Deterministic fake LLM provider for offline product tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from askme.llm.core.contracts import LLMCallContext


class FakeLLMProvider:
    """Small provider that mimics OpenAI chat completion objects.

    Use this for CI, demos without keys, and contract tests.  It never touches
    the network and still exercises gateway policy, tools passthrough, and
    streaming orchestration.
    """

    def __init__(self, *, response_text: str = "ok") -> None:
        self.response_text = response_text
        self.raw_client = SimpleNamespace(name="fake-llm")
        self.minimax_client = None
        self.calls: list[dict[str, Any]] = []

    def client_for_model(self, model: str) -> Any:
        return self.raw_client

    async def stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_token: asyncio.Event | None = None,
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        _ = context
        self.calls.append(dict(kwargs))
        if cancel_token is not None and cancel_token.is_set():
            return
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=self.response_text, tool_calls=None),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )

    async def completion_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        context: LLMCallContext | None = None,
    ) -> Any:
        _ = context
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response_text, tool_calls=None),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )
