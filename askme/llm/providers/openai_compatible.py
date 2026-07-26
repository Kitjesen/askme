"""OpenAI-compatible LLM provider transport.

This class owns SDK clients, provider routing, and retry.  It deliberately
does not know about product prompts, conversation memory, task handoff, or UI
state.  Those live above it in the gateway/pipeline layers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext
from askme.llm.streaming.retry import RETRYABLE_STATUS, default_backoff

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """Transport wrapper for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        backoff_func: Callable[[int], float] = default_backoff,
    ) -> None:
        self._config = config
        self._max_retries = config.max_retries
        self._backoff = backoff_func

        self._client = _create_async_client(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
        )

        self._minimax_client: AsyncOpenAI | None = None
        if config.minimax_api_key:
            self._minimax_client = _create_async_client(
                api_key=config.minimax_api_key,
                base_url=config.minimax_base_url,
                model="MiniMax-M2.7-highspeed",
                timeout=config.timeout,
            )
            logger.info("MiniMax LLM client enabled: %s", config.minimax_base_url)

    @property
    def raw_client(self) -> AsyncOpenAI:
        return self._client

    @property
    def minimax_client(self) -> AsyncOpenAI | None:
        return self._minimax_client

    def client_for_model(self, model: str) -> AsyncOpenAI:
        """Return the client that should serve this model."""

        if self._minimax_client:
            model_lower = str(model or "").lower()
            if model_lower.startswith("minimax"):
                return self._minimax_client
        return self._client

    async def stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_token: asyncio.Event | None = None,
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        """Retry a streaming call until chunks begin flowing."""

        _ = context
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if cancel_token is not None and cancel_token.is_set():
                logger.info("[LLM] cancel_token set; aborting before attempt %d", attempt)
                return

            try:
                client = self.client_for_model(kwargs.get("model", ""))
                response = await self._invoke_with_timeout(client, kwargs)
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, exc)
                continue
            except APIStatusError as exc:
                if exc.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                    last_exc = exc
                    await self._sleep_before_retry(attempt, exc, status_code=exc.status_code)
                    continue
                raise

            async for chunk in response:
                if cancel_token is not None and cancel_token.is_set():
                    logger.info("[LLM] cancel_token set; stopping mid-stream")
                    return
                yield chunk
            return

        if last_exc is not None:
            raise last_exc

    async def completion_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        context: LLMCallContext | None = None,
    ) -> Any:
        """Retry a non-streaming completion call."""

        _ = context
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self.client_for_model(kwargs.get("model", ""))
                return await self._invoke_with_timeout(client, kwargs)
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, exc)
                continue
            except APIStatusError as exc:
                if exc.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                    last_exc = exc
                    await self._sleep_before_retry(attempt, exc, status_code=exc.status_code)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("completion_with_retry exhausted all attempts without an exception")

    @staticmethod
    def _extract_request_timeout(kwargs: dict[str, Any]) -> float | None:
        """Extract request timeout from kwargs when present."""

        timeout_value = kwargs.get("timeout")
        if timeout_value is None:
            return None
        try:
            timeout_value_float = float(timeout_value)
        except (TypeError, ValueError):
            return None
        return timeout_value_float if timeout_value_float > 0 else None

    async def _invoke_with_timeout(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """Run chat completion call with timeout applied per request."""

        timeout_seconds = self._extract_request_timeout(kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs.pop("timeout", None)
        if timeout_seconds is None:
            return await client.chat.completions.create(**call_kwargs)

        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client_for_call = with_options(timeout=timeout_seconds)
        else:
            client_for_call = client
        return await client_for_call.chat.completions.create(**call_kwargs)

    async def _sleep_before_retry(
        self,
        attempt: int,
        exc: Exception,
        *,
        status_code: int | None = None,
    ) -> None:
        wait = self._backoff(attempt)
        if status_code is None:
            logger.warning(
                "[LLM] Retry %d/%d after %.1fs (%s)", attempt + 1, self._max_retries, wait, exc
            )
        else:
            logger.warning(
                "[LLM] Retry %d/%d after %.1fs (HTTP %d)",
                attempt + 1,
                self._max_retries,
                wait,
                status_code,
            )
        await asyncio.sleep(wait)


def _create_async_client(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> AsyncOpenAI:
    try:
        from inovxio_llm import LLMClientConfig, create_async_openai_client

        client_config = LLMClientConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        return create_async_openai_client(client_config)
    except ModuleNotFoundError:
        return AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
