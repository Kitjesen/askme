"""
LLM client wrapper for any OpenAI-compatible API (Claude Relay, DeepSeek, etc.).

Features:
  - Configurable timeout on all API calls
  - Automatic retry with exponential backoff for transient errors
  - Model fallback chain (e.g. Opus -> Sonnet -> Haiku)
  - cancel_token support: pass an asyncio.Event to chat_stream() for E-STOP

Usage::

    from askme.brain import LLMClient
    from askme.llm.config import LLMConfig

    cfg = LLMConfig(api_key="sk-...", model="MiniMax-M2.7-highspeed")
    client = LLMClient(llm_config=cfg)          # explicit config (preferred)
    client = LLMClient()                        # reads config.yaml (legacy)
    stream = client.chat_stream(messages)       # async generator
    response = await client.chat(messages)      # full response string
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from askme.config import get_config
from askme.interfaces.llm import LLMBackend
from askme.llm.config import LLMConfig
from askme.robot.ota_bridge import OTABridgeMetrics

logger = logging.getLogger(__name__)

# Retryable HTTP status codes (transient server errors)
_RETRYABLE_STATUS = {500, 502, 503, 504, 529}


class LLMClient(LLMBackend):
    """Async wrapper around ``AsyncOpenAI`` with timeout, retry, and fallback.

    Preferred construction (dependency-inverted)::

        cfg = LLMConfig.from_cfg(brain_section)
        client = LLMClient(llm_config=cfg, metrics=ota_metrics)

    Legacy construction (reads config.yaml — kept for backward-compat)::

        client = LLMClient(api_key=..., model=...)
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | None = None,
        # Legacy keyword overrides — used when llm_config is None
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        metrics: OTABridgeMetrics | None = None,
    ) -> None:
        if llm_config is not None:
            # Clean path: all config comes from LLMConfig; no config.yaml read.
            resolved_api_key = llm_config.api_key
            resolved_base_url = llm_config.base_url
            resolved_model = llm_config.model
            max_tokens = llm_config.max_tokens
            temperature = llm_config.temperature
            timeout = llm_config.timeout
            max_retries = llm_config.max_retries
            fallback_models = llm_config.fallback_models
            minimax_key = llm_config.minimax_api_key
            minimax_url = llm_config.minimax_base_url
        else:
            # Legacy path: read config.yaml, allow per-call keyword overrides.
            cfg = get_config().get("brain", {})
            resolved_api_key = api_key or cfg.get("api_key", "")
            resolved_base_url = base_url or cfg.get("base_url", "https://api.minimax.chat/v1")
            resolved_model = model or cfg.get("model", "MiniMax-M2.7-highspeed")
            max_tokens = cfg.get("max_tokens", 0)
            temperature = cfg.get("temperature", 0.7)
            timeout = cfg.get("timeout", 30.0)
            max_retries = cfg.get("max_retries", 2)
            fallback_models = cfg.get("fallback_models", [])
            minimax_key = cfg.get("minimax_api_key", "")
            minimax_url = cfg.get("minimax_base_url", "https://api.minimax.chat/v1")

        self.api_key: str = resolved_api_key
        self.base_url: str = resolved_base_url
        self.model: str = resolved_model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        # Retry / resilience config
        self._timeout: float = timeout
        self._max_retries: int = max_retries
        self._fallback_models: list[str] = fallback_models
        self._metrics = metrics

        # Disable SDK internal retry — we handle retry + model fallback ourselves
        # in _stream_with_retry / _completion_with_retry.  SDK retry just wastes
        # time retrying the same model instead of falling back to a faster one.
        try:
            from inovxio_llm import LLMClientConfig, create_async_openai_client

            _cfg = LLMClientConfig(
                api_key=self.api_key, base_url=self.base_url,
                model=self.model, timeout=self._timeout,
            )
            self._client = create_async_openai_client(_cfg)
        except ModuleNotFoundError:
            self._client = AsyncOpenAI(
                api_key=self.api_key or "dummy",
                base_url=self.base_url,
                timeout=self._timeout,
                max_retries=0,
            )

        # MiniMax client (optional — enabled when minimax_api_key is set)
        self._minimax_client: AsyncOpenAI | None = None
        if minimax_key:
            try:
                from inovxio_llm import LLMClientConfig, create_async_openai_client
                _mm_cfg = LLMClientConfig(
                    api_key=minimax_key, base_url=minimax_url,
                    model="MiniMax-M2.5-highspeed", timeout=self._timeout,
                )
                self._minimax_client = create_async_openai_client(_mm_cfg)
            except ModuleNotFoundError:
                self._minimax_client = AsyncOpenAI(
                    api_key=minimax_key,
                    base_url=minimax_url,
                    timeout=self._timeout,
                    max_retries=0,
                )
            logger.info("MiniMax LLM client enabled: %s", minimax_url)

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
        thinking: bool = False,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Return an async streaming iterator of ``ChatCompletionChunk``.

        Retries on transient errors and falls back to alternate models.
        ``thinking=False`` by default to skip <think> generation and reduce TTFT.

        Pass ``cancel_token`` (an asyncio.Event) to support mid-stream E-STOP.
        When the event is set the generator stops yielding and returns cleanly.
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
            # Disable MiniMax thinking for faster TTFT on conversational turns
            if not thinking:
                kwargs["extra_body"] = {"thinking": {"enabled": False}}

            models_to_try = self._model_chain(model)

            for model_name in models_to_try:
                last_model_name = model_name
                kwargs["model"] = model_name
                streaming_started = False
                try:
                    async for chunk in self._stream_with_retry(kwargs, cancel_token=cancel_token):
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
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> str:
        """Non-streaming convenience: return the full assistant reply as a string.

        Retries on transient errors and falls back to alternate models.
        Supports tool calling — pass ``tools`` and ``tool_choice`` as needed.
        For callers that only need text, omit both (defaults to None).
        """
        response = await self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
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
        thinking: bool = False,
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
            if not thinking:
                kwargs["extra_body"] = {"thinking": {"enabled": False}}

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
    # Internal: client routing & retry logic
    # ------------------------------------------------------------------

    def _client_for_model(self, model: str) -> AsyncOpenAI:
        """Return the secondary client for MiniMax/Qwen/DashScope models."""
        if self._minimax_client:
            _m = model.lower()
            if _m.startswith("minimax") or _m.startswith("qwen"):
                return self._minimax_client
        return self._client

    def _model_chain(self, override: str | None = None) -> list[str]:
        """Build ordered list of models to try: override/primary -> fallbacks.

        Cross-provider fallback is disabled — MiniMax models only fall back to
        other MiniMax models, relay models only to relay models.  Prevents
        wasting 10+ seconds trying a 429-limited relay when primary is MiniMax.
        """
        primary = override or self.model
        chain = [primary]
        primary_is_minimax = primary.lower().startswith("minimax")
        for fb in self._fallback_models:
            if fb == primary:
                continue
            fb_is_minimax = fb.lower().startswith("minimax")
            if primary_is_minimax == fb_is_minimax:
                chain.append(fb)
        return chain

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Retry streaming call up to ``_max_retries`` times with backoff.

        Only the connection phase is retried. Once chunks start flowing,
        mid-stream errors propagate to the caller because retrying would replay
        already-spoken audio.

        If ``cancel_token`` is set before or during streaming the generator
        stops yielding and returns cleanly — callers see a normal end-of-stream.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if cancel_token is not None and cancel_token.is_set():
                logger.info("[LLM] cancel_token set — aborting before attempt %d", attempt)
                return

            try:
                client = self._client_for_model(kwargs.get("model", ""))
                response = await client.chat.completions.create(**kwargs)
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
                if cancel_token is not None and cancel_token.is_set():
                    logger.info("[LLM] cancel_token set — stopping mid-stream")
                    return
                yield chunk
            return

        if last_exc:
            raise last_exc

    async def _completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Retry non-streaming call up to ``_max_retries`` times with backoff."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                client = self._client_for_model(kwargs.get("model", ""))
                return await client.chat.completions.create(**kwargs)
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
        raise RuntimeError("_completion_with_retry: exhausted all retries with no exception recorded")


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: 0.3s, 1s, 2s, ...

    First retry is fast (0.3s) to handle transient 503s without
    wasting time.  Subsequent retries back off normally.
    """
    if attempt == 0:
        return 0.3 + random.uniform(0, 0.2)
    base = min(2 ** (attempt - 1), 8)
    return base + random.uniform(0, 0.5)
