"""
LLM client wrapper for any OpenAI-compatible API (Claude Relay, DeepSeek, etc.).

Features:
  - Configurable timeout on all API calls
  - Automatic retry with exponential backoff for transient errors
  - Model fallback chain (e.g. Opus -> Sonnet -> Haiku)

Usage::

    from askme.brain import LLMClient

    client = LLMClient()                       # reads config automatically
    stream = client.chat_stream(messages)       # async generator
    response = await client.chat(messages)      # full response string
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, AsyncIterator, Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from askme.config import get_config
from askme.ota_bridge import OTABridgeMetrics

logger = logging.getLogger(__name__)

# Retryable HTTP status codes (transient server errors)
_RETRYABLE_STATUS = {500, 502, 503, 504, 529}


class LLMClient:
    """Async wrapper around ``AsyncOpenAI`` with timeout, retry, and fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        metrics: OTABridgeMetrics | None = None,
    ) -> None:
        cfg = get_config().get("brain", {})

        self.api_key: str = api_key or cfg.get("api_key", "")
        self.base_url: str = base_url or cfg.get("base_url", "https://cursor.scihub.edu.kg/api/v1")
        self.model: str = model or cfg.get("model", "claude-opus-4-6")
        self.max_tokens: int = cfg.get("max_tokens", 0)
        self.temperature: float = cfg.get("temperature", 0.7)

        # Retry / resilience config
        self._timeout: float = cfg.get("timeout", 30.0)
        self._max_retries: int = cfg.get("max_retries", 2)
        self._fallback_models: list[str] = cfg.get("fallback_models", [])
        self._metrics = metrics

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Return an async streaming iterator of ``ChatCompletionChunk``.

        Retries on transient errors and falls back to alternate models.
        """
        started_at = time.perf_counter()
        success = False
        last_model_name = model or self.model

        try:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "stream": True,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if tools:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            models_to_try = self._model_chain(model)

            for model_name in models_to_try:
                last_model_name = model_name
                kwargs["model"] = model_name
                streaming_started = False
                try:
                    async for chunk in self._stream_with_retry(kwargs):
                        streaming_started = True
                        yield chunk
                    success = True
                    return
                except (APITimeoutError, APIConnectionError) as exc:
                    if streaming_started:
                        raise
                    logger.warning("[LLM] %s failed (%s), trying next model", model_name, exc)
                    continue
                except APIStatusError as exc:
                    if streaming_started:
                        raise
                    if exc.status_code in _RETRYABLE_STATUS:
                        logger.warning("[LLM] %s returned %d, trying next model", model_name, exc.status_code)
                        continue
                    raise

            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        finally:
            if self._metrics is not None:
                self._metrics.record_llm_call(
                    time.perf_counter() - started_at,
                    success=success,
                    model=last_model_name,
                    mode="stream",
                )

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming convenience: return the full assistant reply as a string.

        Retries on transient errors and falls back to alternate models.
        """
        response = await self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Return the raw non-streaming completion object with retry/fallback."""
        started_at = time.perf_counter()
        success = False
        last_model_name = model or self.model

        try:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "stream": False,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if tools:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            models_to_try = self._model_chain(model)

            for model_name in models_to_try:
                last_model_name = model_name
                kwargs["model"] = model_name
                try:
                    result = await self._completion_with_retry(kwargs)
                    success = True
                    return result
                except (APITimeoutError, APIConnectionError) as exc:
                    logger.warning("[LLM] %s failed (%s), trying next model", model_name, exc)
                    continue
                except APIStatusError as exc:
                    if exc.status_code in _RETRYABLE_STATUS:
                        logger.warning("[LLM] %s returned %d, trying next model", model_name, exc.status_code)
                        continue
                    raise

            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        finally:
            if self._metrics is not None:
                self._metrics.record_llm_call(
                    time.perf_counter() - started_at,
                    success=success,
                    model=last_model_name,
                    mode="completion",
                )

    # ------------------------------------------------------------------
    # Expose underlying client for advanced usage
    # ------------------------------------------------------------------

    @property
    def raw_client(self) -> AsyncOpenAI:
        """Direct access to the ``AsyncOpenAI`` instance."""
        return self._client

    # ------------------------------------------------------------------
    # Internal: retry logic
    # ------------------------------------------------------------------

    def _model_chain(self, override: str | None = None) -> list[str]:
        """Build ordered list of models to try: override/primary -> fallbacks."""
        primary = override or self.model
        chain = [primary]
        for fb in self._fallback_models:
            if fb != primary:
                chain.append(fb)
        return chain

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Retry streaming call up to ``_max_retries`` times with backoff.

        Only the connection phase is retried. Once chunks start flowing,
        mid-stream errors propagate to the caller because retrying would replay
        already-spoken audio.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = _backoff(attempt)
                    logger.warning(
                        "[LLM] Retry %d/%d after %.1fs (%s)",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                continue
            except APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                    last_exc = exc
                    wait = _backoff(attempt)
                    logger.warning(
                        "[LLM] Retry %d/%d after %.1fs (HTTP %d)",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc.status_code,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

            async for chunk in response:
                yield chunk
            return

        if last_exc:
            raise last_exc

    async def _completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Retry non-streaming call up to ``_max_retries`` times with backoff."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = _backoff(attempt)
                    logger.warning(
                        "[LLM] Retry %d/%d after %.1fs (%s)",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                continue
            except APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                    last_exc = exc
                    wait = _backoff(attempt)
                    logger.warning(
                        "[LLM] Retry %d/%d after %.1fs (HTTP %d)",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc.status_code,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        if last_exc:
            raise last_exc
        return None


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s, ..."""
    base = min(2 ** attempt, 8)
    return base + random.uniform(0, 0.5)
