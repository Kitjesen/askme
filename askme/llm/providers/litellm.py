"""LiteLLM Proxy adapter for the AskMe LLM provider seam.

LiteLLM runs out of process as an authenticated OpenAI-compatible gateway.
AskMe deliberately does not import the LiteLLM Python SDK: the robot process
keeps only a scoped proxy key while provider credentials, routing, and retry
policy remain inside the proxy.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

from openai import AsyncOpenAI

from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext
from askme.llm.providers.openai_compatible import OpenAICompatibleProvider
from askme.llm.streaming.retry import default_backoff

logger = logging.getLogger(__name__)

_STREAM_CANCELLED = object()
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_OPAQUE_CALL_ID_RE = re.compile(r"^(?:[0-9a-f]{32}|sha256:[0-9a-f]{24})$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
_PURPOSES = frozenset(
    {
        "assistant_response",
        "tool_followup",
        "memory_compact",
        "health_probe",
        "vision_grounding",
        "general",
    }
)
_CHANNELS = frozenset({"voice", "text", "vision", "background", "system"})
_REQUEST_CLASSES = frozenset(
    {"voice_fast", "robot_action", "memory", "vision", "health_probe", "text"}
)
_PRIVACY_CLASSES = frozenset({"public", "conversation", "sensitive", "restricted", "operational"})


def _allowlisted(value: str, allowed: frozenset[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback


def _safe_identifier(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if _SAFE_ID_RE.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}"


def _safe_call_identifier(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return secrets.token_hex(16)
    if _OPAQUE_CALL_ID_RE.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}"


def _safe_turn_identifier(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}"


def _traceparent(trace_id: str | None) -> tuple[str, str]:
    normalized = str(trace_id or "").strip().lower()
    if not normalized or normalized == "0" * 32:
        normalized = secrets.token_hex(16)
    elif not _TRACE_ID_RE.fullmatch(normalized):
        normalized = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return normalized, f"00-{normalized}-{secrets.token_hex(8)}-01"


def build_litellm_proxy_request(
    kwargs: dict[str, Any],
    context: LLMCallContext | None,
) -> dict[str, Any]:
    """Return a proxy request enriched only with allowlisted correlation data."""

    request = dict(kwargs)
    request.pop("extra_headers", None)
    request.pop("metadata", None)
    if context is None:
        return request

    trace_id, traceparent = _traceparent(context.trace_id)
    model_alias = _safe_identifier(str(request.get("model") or "")) or "unknown"
    metadata: dict[str, str | int] = {
        "trace_id": trace_id,
        "purpose": _allowlisted(context.purpose, _PURPOSES, "general"),
        "channel": _allowlisted(context.channel, _CHANNELS, "text"),
        "request_class": _allowlisted(context.request_class, _REQUEST_CLASSES, "text"),
        "privacy_class": _allowlisted(
            context.privacy_class,
            _PRIVACY_CLASSES,
            "restricted",
        ),
        "model_alias": model_alias,
        "allow_cache": "true" if context.allow_cache else "false",
    }
    safe_call_id = _safe_call_identifier(context.call_id)
    metadata["call_id"] = safe_call_id
    if context.latency_budget_ms is not None:
        budget_ms = int(context.latency_budget_ms)
        if 0 < budget_ms <= 600_000:
            metadata["latency_budget_ms"] = budget_ms
    turn_id = _safe_turn_identifier(context.turn_id)
    if turn_id is not None:
        metadata["turn_id"] = turn_id
    request["metadata"] = metadata

    # Do not forward arbitrary caller headers across the proxy boundary.
    # Credentials/cookies belong to the scoped AsyncOpenAI client, while these
    # two correlation headers are generated and owned by AskMe.
    headers = {
        "traceparent": traceparent,
        "x-litellm-call-id": safe_call_id,
    }
    request["extra_headers"] = headers

    if not context.allow_cache:
        extra_body = dict(request.get("extra_body") or {})
        extra_body["cache"] = {"no-cache": True, "no-store": True}
        request["extra_body"] = extra_body
    return request


# Backward-compatible name for older focused tests and local diagnostics.
_request_with_context = build_litellm_proxy_request


async def _await_or_cancel(
    awaitable: Awaitable[Any],
    cancel_token: asyncio.Event | None,
    *,
    prefer_operation_on_race: bool = False,
) -> Any:
    """Wait for an upstream operation while making turn cancellation immediate."""

    if cancel_token is None:
        return await awaitable

    operation = asyncio.ensure_future(awaitable)
    cancellation = asyncio.create_task(cancel_token.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, cancellation},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation in done and (cancellation not in done or prefer_operation_on_race):
            return operation.result()
        return _STREAM_CANCELLED
    finally:
        for task in (operation, cancellation):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation, cancellation, return_exceptions=True)


async def _next_stream_chunk(
    iterator: AsyncIterator[Any],
    cancel_token: asyncio.Event | None,
) -> Any:
    """Wait for either the next proxy chunk or an explicit turn cancellation."""

    return await _await_or_cancel(anext(iterator), cancel_token)


class LiteLLMProxyProvider(OpenAICompatibleProvider):
    """OpenAI-compatible transport whose routing owner is LiteLLM Proxy."""

    provider_name = "litellm"

    def __init__(
        self,
        config: LLMConfig,
        *,
        backoff_func: Callable[[int], float] = default_backoff,
    ) -> None:
        proxy_config = replace(config, max_retries=0, minimax_api_key="")
        super().__init__(proxy_config, backoff_func=backoff_func)
        logger.info("LiteLLM Proxy enabled: %s", config.base_url)

    async def stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        cancel_token: asyncio.Event | None = None,
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        """Stream through LiteLLM and close the proxy response on interruption."""

        if cancel_token is not None and cancel_token.is_set():
            logger.info("[LLM] cancel_token set; aborting LiteLLM request")
            return

        kwargs = build_litellm_proxy_request(kwargs, context)
        client = self.client_for_model(str(kwargs.get("model", "")))
        response = await _await_or_cancel(
            client.chat.completions.create(**kwargs),
            cancel_token,
            prefer_operation_on_race=True,
        )
        if response is _STREAM_CANCELLED:
            logger.info("[LLM] cancel_token set; aborting pending LiteLLM request")
            return
        iterator = response.__aiter__()
        try:
            while True:
                try:
                    chunk = await _next_stream_chunk(iterator, cancel_token)
                except StopAsyncIteration:
                    return
                if chunk is _STREAM_CANCELLED:
                    logger.info("[LLM] cancel_token set; closing LiteLLM stream")
                    return
                yield chunk
        finally:
            close = getattr(response, "close", None) or getattr(response, "aclose", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.warning("[LLM] failed to close LiteLLM stream", exc_info=True)

    async def completion_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        context: LLMCallContext | None = None,
    ) -> Any:
        """Send non-streaming calls through the same safe control-plane envelope."""

        request = build_litellm_proxy_request(kwargs, context)
        return await super().completion_with_retry(request, context=context)

    def client_for_model(self, model: str) -> AsyncOpenAI:
        """Route every model through the proxy, never a direct provider client."""

        _ = model
        return self.raw_client
