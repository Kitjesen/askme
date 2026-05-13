"""Product-level LLM gateway.

The gateway is the single runtime entrypoint for model calls.  It owns model
policy, request normalization, metrics, and provider fallback.  Business code
can keep using ``LLMClient`` for compatibility, but new code should depend on
``LLMGateway`` or the ``LLMBackend`` contract instead of provider SDK clients.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError

from askme.interfaces.llm import LLMBackend
from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext, LLMProvider
from askme.llm.core.factory import create_llm_provider, resolve_provider_name
from askme.llm.policy.model_policy import ModelPolicy
from askme.llm.providers.profiles import provider_profile
from askme.llm.streaming.retry import RETRYABLE_STATUS
from askme.robot.ota_bridge import OTABridgeMetrics


class LLMGateway(LLMBackend):
    """Provider-agnostic gateway for chat, streaming, and raw completions."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        metrics: OTABridgeMetrics | None = None,
        provider: LLMProvider | None = None,
        model_policy: ModelPolicy | None = None,
    ) -> None:
        self.config = llm_config
        self.api_key = llm_config.api_key
        self.base_url = llm_config.base_url
        self.model = llm_config.model
        self.provider_name = resolve_provider_name(llm_config)
        self.provider_profile = provider_profile(self.provider_name)
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self._metrics = metrics
        self._provider = provider or create_llm_provider(llm_config)
        self._model_policy = model_policy or ModelPolicy(
            primary_model=llm_config.model,
            fallback_models=list(llm_config.fallback_models),
        )

        # Backward-compatible escape hatches used by older tests and modules.
        self._client = self._provider.raw_client
        self._minimax_client = self._provider.minimax_client

    @property
    def raw_client(self) -> Any:
        """Direct access to the underlying provider client."""

        return self._provider.raw_client

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
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        """Stream assistant tokens with retry, fallback, and metrics."""

        started_at = time.perf_counter()
        success = False
        last_model_name = model or self.model
        _ = context  # Reserved for audit plumbing without changing call sites.

        try:
            kwargs = self._completion_kwargs(
                messages,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )
            for model_name in self._model_chain(model):
                last_model_name = model_name
                self._apply_model_policy(kwargs, model_name, thinking=thinking)
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
                    import logging

                    logging.getLogger(__name__).warning("[LLM] %s failed (%s), trying next model", model_name, exc)
                    continue
                except APIStatusError as exc:
                    if streaming_started:
                        raise
                    if exc.status_code in RETRYABLE_STATUS:
                        import logging

                        logging.getLogger(__name__).warning(
                            "[LLM] %s returned %d, trying next model",
                            model_name,
                            exc.status_code,
                        )
                        continue
                    raise
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        finally:
            self._record_metrics(started_at, success=success, model=last_model_name, mode="stream")

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
        """Return a single assistant text response."""

        response = await self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            context=context,
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
        context: LLMCallContext | None = None,
    ) -> Any:
        """Return the raw non-streaming completion object."""

        started_at = time.perf_counter()
        success = False
        last_model_name = model or self.model
        _ = context

        try:
            kwargs = self._completion_kwargs(
                messages,
                stream=False,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )
            for model_name in self._model_chain(model):
                last_model_name = model_name
                self._apply_model_policy(kwargs, model_name, thinking=thinking)
                try:
                    result = await self._completion_with_retry(kwargs)
                    success = True
                    return result
                except (APITimeoutError, APIConnectionError) as exc:
                    import logging

                    logging.getLogger(__name__).warning("[LLM] %s failed (%s), trying next model", model_name, exc)
                    continue
                except APIStatusError as exc:
                    if exc.status_code in RETRYABLE_STATUS:
                        import logging

                        logging.getLogger(__name__).warning(
                            "[LLM] %s returned %d, trying next model",
                            model_name,
                            exc.status_code,
                        )
                        continue
                    raise
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        finally:
            self._record_metrics(started_at, success=success, model=last_model_name, mode="completion")

    def supports_tools(self) -> bool:
        return self.provider_profile.supports_tools

    def supports_vision(self) -> bool:
        return self.provider_profile.supports_vision

    def provider_status(self) -> dict[str, Any]:
        """Return non-secret provider status for health/UI/debugging."""

        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
            "openai_compatible": self.provider_profile.openai_compatible,
            "domestic": self.provider_profile.domestic,
            "supports_tools": self.supports_tools(),
            "supports_vision": self.supports_vision(),
            "fallback_models": self._model_chain()[1:],
        }

    def _completion_kwargs(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        stream: bool,
        tools: list[dict] | None,
        tool_choice: str | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    def _apply_model_policy(self, kwargs: dict[str, Any], model: str, *, thinking: bool) -> None:
        kwargs["model"] = model
        extra_body = self._extra_body_for_model(model, thinking=thinking)
        if extra_body is None:
            kwargs.pop("extra_body", None)
        else:
            kwargs["extra_body"] = extra_body

    def _client_for_model(self, model: str) -> Any:
        return self._provider.client_for_model(model)

    def _model_chain(self, override: str | None = None) -> list[str]:
        return self._model_policy.model_chain(override)

    @staticmethod
    def _extra_body_for_model(model: str, *, thinking: bool) -> dict[str, Any] | None:
        return ModelPolicy.extra_body_for_model(model, thinking=thinking)

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncIterator[Any]:
        async for chunk in self._provider.stream_with_retry(kwargs, cancel_token=cancel_token):
            yield chunk

    async def _completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        return await self._provider.completion_with_retry(kwargs)

    def _record_metrics(self, started_at: float, *, success: bool, model: str, mode: str) -> None:
        if self._metrics is None:
            return
        self._metrics.record_llm_call(
            time.perf_counter() - started_at,
            success=success,
            model=model,
            mode=mode,
        )
