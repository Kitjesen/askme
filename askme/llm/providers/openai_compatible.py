"""OpenAI-compatible LLM provider transport.

This class owns SDK clients, provider routing, and retry.  It deliberately
does not know about product prompts, conversation memory, task handoff, or UI
state.  Those live above it in the gateway/pipeline layers.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, TypedDict

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext
from askme.llm.streaming.retry import RETRYABLE_STATUS, default_backoff

logger = logging.getLogger(__name__)

_DEFAULT_HTTP_KEEPALIVE_EXPIRY_SECONDS = 60.0
_DEFAULT_HTTP_MAX_CONNECTIONS = 100
_DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS = 20


class _HTTPPoolOptions(TypedDict):
    http_keepalive_expiry_seconds: float
    http_max_connections: int
    http_max_keepalive_connections: int


def _positive_float(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _http_pool_options(raw_options: Any) -> _HTTPPoolOptions:
    options = raw_options if isinstance(raw_options, dict) else {}
    max_connections = _positive_int(
        options.get("http_max_connections"),
        _DEFAULT_HTTP_MAX_CONNECTIONS,
    )
    max_keepalive = min(
        max_connections,
        _positive_int(
            options.get("http_max_keepalive_connections"),
            _DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )
    return {
        "http_keepalive_expiry_seconds": _positive_float(
            options.get("http_keepalive_expiry_seconds"),
            _DEFAULT_HTTP_KEEPALIVE_EXPIRY_SECONDS,
        ),
        "http_max_connections": max_connections,
        "http_max_keepalive_connections": max_keepalive,
    }


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
        http_pool_options = _http_pool_options(config.provider_options)

        self._client = _create_async_client(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            **http_pool_options,
            timeout=config.timeout,
        )

        self._minimax_client: AsyncOpenAI | None = None
        if config.minimax_api_key:
            self._minimax_client = _create_async_client(
                api_key=config.minimax_api_key,
                base_url=config.minimax_base_url,
                model="MiniMax-M2.7-highspeed",
                **http_pool_options,
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

    async def aclose(self) -> None:
        """Close all owned SDK clients without double-closing aliases."""

        seen: set[int] = set()
        for client in (self._client, self._minimax_client):
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning("[LLM] failed to close provider client", exc_info=True)

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


def _inovxio_pool_kwargs(
    config_factory: Any,
    *,
    http_keepalive_expiry_seconds: float,
    http_max_connections: int,
    http_max_keepalive_connections: int,
) -> dict[str, float | int]:
    requested: dict[str, float | int] = {
        "http_keepalive_expiry_seconds": http_keepalive_expiry_seconds,
        "http_max_connections": http_max_connections,
        "http_max_keepalive_connections": http_max_keepalive_connections,
    }
    try:
        parameters = inspect.signature(config_factory).parameters
        parameter_names = set(parameters)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        parameter_names = set()
        accepts_kwargs = False
    supported = (
        requested
        if accepts_kwargs
        else {name: value for name, value in requested.items() if name in parameter_names}
    )
    unsupported = sorted(set(requested) - set(supported))
    if unsupported:
        logger.warning(
            "[LLM] inovxio_llm does not expose HTTP pool option(s): %s; "
            "custom transport defaults remain authoritative",
            ", ".join(unsupported),
        )
    return supported


def _create_async_client(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    http_keepalive_expiry_seconds: float = _DEFAULT_HTTP_KEEPALIVE_EXPIRY_SECONDS,
    http_max_connections: int = _DEFAULT_HTTP_MAX_CONNECTIONS,
    http_max_keepalive_connections: int = _DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS,
) -> AsyncOpenAI:
    try:
        from inovxio_llm import LLMClientConfig, create_async_openai_client

        client_config = LLMClientConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            **_inovxio_pool_kwargs(
                LLMClientConfig,
                http_keepalive_expiry_seconds=http_keepalive_expiry_seconds,
                http_max_connections=http_max_connections,
                http_max_keepalive_connections=http_max_keepalive_connections,
            ),
        )
        return create_async_openai_client(client_config)
    except ModuleNotFoundError:
        http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=http_max_connections,
                max_keepalive_connections=http_max_keepalive_connections,
                keepalive_expiry=http_keepalive_expiry_seconds,
            ),
        )
        return AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
