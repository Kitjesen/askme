"""Product-level LLM gateway.

The gateway is the single runtime entrypoint for model calls.  It owns model
policy, request normalization, metrics, and provider fallback.  Business code
can keep using ``LLMClient`` for compatibility, but new code should depend on
``LLMGateway`` or the ``LLMBackend`` contract instead of provider SDK clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError

from askme.interfaces.llm import LLMBackend
from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import (
    LLMCallContext,
    LLMDeadlineExceeded,
    LLMNoSemanticResponse,
    LLMProvider,
)
from askme.llm.core.factory import create_llm_provider, resolve_provider_name
from askme.llm.policy.model_policy import ModelPolicy
from askme.llm.providers.profiles import provider_profile
from askme.llm.streaming.retry import RETRYABLE_STATUS
from askme.telemetry.ota_bridge import OTABridgeMetrics

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_OPAQUE_CALL_ID_RE = re.compile(r"^(?:[0-9a-f]{32}|sha256:[0-9a-f]{24})$")
_SAFE_MODEL_LABEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
_DIAGNOSTIC_PURPOSES = frozenset(
    {
        "assistant_response",
        "tool_followup",
        "memory_compact",
        "health_probe",
        "vision_grounding",
        "general",
    }
)
_DIAGNOSTIC_REQUEST_CLASSES = frozenset(
    {"voice_fast", "robot_action", "memory", "vision", "health_probe", "text"}
)


def _hashed_identifier(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:24]
    return f"sha256:{digest}"


def _normalized_trace_id(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "0" * 32:
        return secrets.token_hex(16)
    if _TRACE_ID_RE.fullmatch(normalized):
        return normalized
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def _normalized_call_id(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return secrets.token_hex(16)
    if _OPAQUE_CALL_ID_RE.fullmatch(normalized):
        return normalized
    return _hashed_identifier(normalized)


def _safe_model_label(value: str) -> str:
    normalized = str(value or "").strip()
    if _SAFE_MODEL_LABEL_RE.fullmatch(normalized):
        return normalized
    return _hashed_identifier(normalized) if normalized else "unknown"


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
        if provider is not None and not str(llm_config.provider or "").strip():
            self.provider_name = "fake"
        else:
            self.provider_name = resolve_provider_name(llm_config)
        self.provider_profile = provider_profile(self.provider_name)
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self._metrics = metrics
        self._provider = provider or create_llm_provider(llm_config)
        if self.provider_profile.manages_model_routing:
            self._model_policy = ModelPolicy(
                primary_model=llm_config.model,
                fallback_models=[],
            )
        else:
            self._model_policy = model_policy or ModelPolicy(
                primary_model=llm_config.model,
                fallback_models=list(llm_config.fallback_models),
            )

        # Backward-compatible escape hatches used by older tests and modules.
        self._client = self._provider.raw_client
        self._minimax_client = self._provider.minimax_client
        self._call_diagnostics: deque[dict[str, Any]] = deque(maxlen=100)
        self._call_diagnostics_lock = threading.Lock()
        self._request_activity_lock = threading.Lock()
        self._active_business_requests = 0
        self._active_warm_probes: set[asyncio.Event] = set()
        self._transport_started = False
        self._closed = False

    @property
    def raw_client(self) -> Any:
        """Direct access to the underlying provider client."""

        return self._provider.raw_client

    def request_activity(self) -> dict[str, int]:
        """Return an allowlisted snapshot used by warm-session admission."""

        with self._request_activity_lock:
            return {
                "active_business_requests": self._active_business_requests,
                "active_warm_probes": len(self._active_warm_probes),
            }

    def cancel_warm_probes(self) -> int:
        """Signal only in-flight health probes, leaving business requests untouched."""

        with self._request_activity_lock:
            warm_tokens = tuple(self._active_warm_probes)
            for cancel_token in warm_tokens:
                cancel_token.set()
            return len(warm_tokens)

    def close_sync(self) -> bool:
        """Logically close an unused gateway without running async transport code.

        Once a provider request has started, its transport may be bound to an
        event loop and must be closed through :meth:`aclose` on that loop.
        """

        with self._request_activity_lock:
            if self._closed:
                return True
            if (
                self._transport_started
                or self._active_business_requests
                or self._active_warm_probes
            ):
                return False
            self._closed = True
            return True

    async def aclose(self) -> None:
        """Release provider transports owned by this gateway."""

        with self._request_activity_lock:
            self._closed = True
            for cancel_token in tuple(self._active_warm_probes):
                cancel_token.set()
        close = getattr(self._provider, "aclose", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _begin_request_activity(
        self,
        context: LLMCallContext | None,
        cancel_token: asyncio.Event | None,
    ) -> str:
        is_warm_probe = bool(
            context is not None
            and (context.request_class == "health_probe" or context.purpose == "health_probe")
        )
        with self._request_activity_lock:
            if self._closed:
                raise RuntimeError("LLM gateway is closed")
            if is_warm_probe:
                if self._active_business_requests:
                    if cancel_token is not None:
                        cancel_token.set()
                    return "deferred"
                if cancel_token is not None:
                    self._active_warm_probes.add(cancel_token)
                self._transport_started = True
                return "warm"

            self._transport_started = True
            self._active_business_requests += 1
            for warm_cancel in tuple(self._active_warm_probes):
                warm_cancel.set()
            return "business"

    def _end_request_activity(
        self,
        activity_kind: str,
        cancel_token: asyncio.Event | None,
    ) -> None:
        with self._request_activity_lock:
            if activity_kind == "business":
                self._active_business_requests = max(
                    0,
                    self._active_business_requests - 1,
                )
            elif activity_kind == "warm" and cancel_token is not None:
                self._active_warm_probes.discard(cancel_token)

    async def chat_stream(
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
        """Stream assistant tokens with retry, fallback, and metrics."""

        context = self._ensure_context_identity(context)
        started_at = time.perf_counter()
        success = False
        requested_model = model or self.model
        last_model_name = requested_model
        resolved_model = requested_model
        outcome = "failed"
        semantic_observed = False
        deadline_at = self._deadline_at(started_at, context)
        last_failure: Exception | None = None
        activity_kind = self._begin_request_activity(context, cancel_token)
        if activity_kind == "deferred":
            return

        try:
            kwargs = self._completion_kwargs(
                messages,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            for model_name in self._model_chain(model):
                last_model_name = model_name
                self._apply_model_policy(kwargs, model_name, thinking=thinking)
                self._apply_remaining_timeout(kwargs, deadline_at, context)
                semantic_started = False
                try:
                    stream = self._stream_with_retry(
                        kwargs,
                        cancel_token=cancel_token,
                        context=context,
                    )
                    iterator = stream.__aiter__()
                    try:
                        while True:
                            try:
                                if semantic_started or deadline_at is None:
                                    chunk = await anext(iterator)
                                else:
                                    remaining = self._remaining_seconds(
                                        deadline_at,
                                        context,
                                        phase="first_semantic",
                                    )
                                    chunk = await asyncio.wait_for(
                                        anext(iterator),
                                        timeout=remaining,
                                    )
                            except StopAsyncIteration:
                                break
                            except TimeoutError as exc:
                                if context is None:
                                    raise
                                raise LLMDeadlineExceeded(
                                    phase="first_semantic",
                                    context=context,
                                ) from exc
                            if self._chunk_has_semantic_payload(chunk):
                                semantic_started = True
                                semantic_observed = True
                                resolved_model = (
                                    str(getattr(chunk, "model", "") or "") or model_name
                                )
                            yield chunk
                    finally:
                        close = getattr(iterator, "aclose", None)
                        if callable(close):
                            await close()
                    if cancel_token is not None and cancel_token.is_set():
                        outcome = "cancelled"
                        return
                    if not semantic_started:
                        last_failure = LLMNoSemanticResponse(
                            model_alias=model_name,
                            context=context,
                        )
                        if self.provider_profile.manages_model_routing:
                            raise last_failure
                        import logging

                        logging.getLogger(__name__).warning(
                            "[LLM] %s returned no semantic payload, trying next model",
                            model_name,
                        )
                        continue
                    success = True
                    outcome = "success"
                    if not resolved_model:
                        resolved_model = model_name
                    return
                except (APITimeoutError, APIConnectionError) as exc:
                    last_failure = exc
                    if semantic_started or self.provider_profile.manages_model_routing:
                        raise
                    import logging

                    logging.getLogger(__name__).warning(
                        "[LLM] %s failed (%s), trying next model", model_name, exc
                    )
                    continue
                except APIStatusError as exc:
                    last_failure = exc
                    if semantic_started or self.provider_profile.manages_model_routing:
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
            if last_failure is not None:
                raise last_failure
            raise LLMNoSemanticResponse(
                model_alias=last_model_name,
                context=context,
            )
        except BaseException as exc:
            outcome = self._diagnostic_outcome(exc)
            raise
        finally:
            self._end_request_activity(activity_kind, cancel_token)
            self._record_metrics(started_at, success=success, model=last_model_name, mode="stream")
            self._record_call_diagnostic(
                started_at=started_at,
                context=context,
                model_alias=requested_model,
                resolved_model=resolved_model or last_model_name,
                mode="stream",
                outcome=outcome,
                semantic_started=semantic_observed,
            )

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
        """Return the raw non-streaming completion object.

        Health probes wait outside the activity counters while business work is
        active, then enter as a tracked warm probe immediately before transport.
        Business completions continue to be admitted and counted immediately.
        """

        context = self._ensure_context_identity(context)
        started_at = time.perf_counter()
        success = False
        requested_model = model or self.model
        last_model_name = requested_model
        resolved_model = requested_model
        outcome = "failed"
        deadline_at = self._deadline_at(started_at, context)
        activity_kind = "deferred"
        activity_cancel = asyncio.Event()

        try:
            while activity_kind == "deferred":
                activity_cancel = asyncio.Event()
                activity_kind = self._begin_request_activity(context, activity_cancel)
                if activity_kind != "deferred":
                    break
                if deadline_at is None:
                    await asyncio.sleep(0.01)
                    continue
                remaining = self._remaining_seconds(
                    deadline_at,
                    context,
                    phase="admission",
                )
                await asyncio.sleep(min(0.01, remaining))

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
                self._apply_remaining_timeout(kwargs, deadline_at, context)
                try:
                    if deadline_at is None:
                        result = await self._completion_with_retry(
                            kwargs,
                            context=context,
                        )
                    else:
                        remaining = self._remaining_seconds(
                            deadline_at,
                            context,
                            phase="completion",
                        )
                        try:
                            result = await asyncio.wait_for(
                                self._completion_with_retry(kwargs, context=context),
                                timeout=remaining,
                            )
                        except TimeoutError as exc:
                            if context is None:
                                raise
                            raise LLMDeadlineExceeded(
                                phase="completion",
                                context=context,
                            ) from exc
                    success = True
                    outcome = "success"
                    resolved_model = str(getattr(result, "model", "") or "") or model_name
                    return result
                except (APITimeoutError, APIConnectionError) as exc:
                    if self.provider_profile.manages_model_routing:
                        raise
                    import logging

                    logging.getLogger(__name__).warning(
                        "[LLM] %s failed (%s), trying next model", model_name, exc
                    )
                    continue
                except APIStatusError as exc:
                    if self.provider_profile.manages_model_routing:
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
        except BaseException as exc:
            outcome = self._diagnostic_outcome(exc)
            raise
        finally:
            self._end_request_activity(activity_kind, activity_cancel)
            self._record_metrics(
                started_at, success=success, model=last_model_name, mode="completion"
            )
            self._record_call_diagnostic(
                started_at=started_at,
                context=context,
                model_alias=requested_model,
                resolved_model=resolved_model or last_model_name,
                mode="completion",
                outcome=outcome,
                semantic_started=success,
            )

    def supports_tools(self) -> bool:
        return self.provider_profile.supports_tools

    def supports_vision(self) -> bool:
        return self.provider_profile.supports_vision

    def provider_status(self) -> dict[str, Any]:
        """Return non-secret provider status for health/UI/debugging."""

        status = {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
            "openai_compatible": self.provider_profile.openai_compatible,
            "domestic": self.provider_profile.domestic,
            "supports_tools": self.supports_tools(),
            "supports_vision": self.supports_vision(),
            "fallback_models": self._model_chain()[1:],
            "routing_owner": (
                self.provider_name if self.provider_profile.manages_model_routing else "askme"
            ),
        }
        if self.provider_name == "litellm":
            diagnostics = self.recent_call_diagnostics(limit=1)
            status["call_diagnostics"] = {
                "count": self._call_diagnostic_count(),
                "last_outcome": diagnostics[-1]["outcome"] if diagnostics else None,
            }
        return status

    def recent_call_diagnostics(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return a bounded, message-free copy of recent contextual model calls."""

        safe_limit = max(1, min(int(limit), self._call_diagnostics.maxlen or 100))
        with self._call_diagnostics_lock:
            records = list(self._call_diagnostics)[-safe_limit:]
        return [dict(record) for record in records]

    @staticmethod
    def _ensure_context_identity(
        context: LLMCallContext | None,
    ) -> LLMCallContext | None:
        if context is None:
            return None
        trace_id = _normalized_trace_id(context.trace_id)
        call_id = _normalized_call_id(context.call_id)
        if trace_id == context.trace_id and call_id == context.call_id:
            return context
        return replace(context, trace_id=trace_id, call_id=call_id)

    @staticmethod
    def _diagnostic_outcome(exc: BaseException) -> str:
        if isinstance(exc, asyncio.CancelledError):
            return "cancelled"
        if isinstance(exc, GeneratorExit):
            return "abandoned"
        if isinstance(exc, LLMDeadlineExceeded):
            return "deadline_exceeded"
        if isinstance(exc, LLMNoSemanticResponse):
            return "no_semantic_response"
        if isinstance(exc, APITimeoutError):
            return "transport_timeout"
        if isinstance(exc, APIConnectionError):
            return "provider_unavailable"
        if isinstance(exc, APIStatusError):
            if exc.status_code == 429:
                return "rate_limited"
            if exc.status_code in {401, 403}:
                return "authentication_failed"
            return "upstream_http_error"
        return "failed"

    def _record_call_diagnostic(
        self,
        *,
        started_at: float,
        context: LLMCallContext | None,
        model_alias: str,
        resolved_model: str,
        mode: str,
        outcome: str,
        semantic_started: bool,
    ) -> None:
        if context is None:
            return
        raw_turn_id = str(context.turn_id or "").strip()
        purpose = str(context.purpose or "").strip().lower()
        request_class = str(context.request_class or "").strip().lower()
        record = {
            "call_id": _normalized_call_id(context.call_id),
            "trace_id": _normalized_trace_id(context.trace_id),
            "turn_id": _hashed_identifier(raw_turn_id) if raw_turn_id else "",
            "purpose": purpose if purpose in _DIAGNOSTIC_PURPOSES else "general",
            "request_class": (
                request_class if request_class in _DIAGNOSTIC_REQUEST_CLASSES else "text"
            ),
            "model_alias": _safe_model_label(model_alias),
            "resolved_model": _safe_model_label(resolved_model),
            "mode": str(mode),
            "outcome": str(outcome),
            "semantic_started": bool(semantic_started),
            "duration_ms": round(
                max(time.perf_counter() - started_at, 0.0) * 1000.0,
                2,
            ),
        }
        with self._call_diagnostics_lock:
            self._call_diagnostics.append(record)

    def _call_diagnostic_count(self) -> int:
        with self._call_diagnostics_lock:
            return len(self._call_diagnostics)

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

    @staticmethod
    def _deadline_at(
        started_at: float,
        context: LLMCallContext | None,
    ) -> float | None:
        if context is None or context.latency_budget_ms is None:
            return None
        budget_ms = int(context.latency_budget_ms)
        if budget_ms <= 0:
            return None
        return started_at + (budget_ms / 1000.0)

    @staticmethod
    def _remaining_seconds(
        deadline_at: float,
        context: LLMCallContext | None,
        *,
        phase: str,
    ) -> float:
        remaining = deadline_at - time.perf_counter()
        if remaining > 0:
            return remaining
        if context is None:
            raise TimeoutError(f"LLM deadline exceeded during {phase}")
        raise LLMDeadlineExceeded(phase=phase, context=context)

    @classmethod
    def _apply_remaining_timeout(
        cls,
        kwargs: dict[str, Any],
        deadline_at: float | None,
        context: LLMCallContext | None,
    ) -> None:
        if deadline_at is None:
            return
        kwargs["timeout"] = cls._remaining_seconds(
            deadline_at,
            context,
            phase="request_start",
        )

    @staticmethod
    def _chunk_has_semantic_payload(chunk: Any) -> bool:
        for choice in getattr(chunk, "choices", None) or ():
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content is not None and str(content).strip():
                return True
            if getattr(delta, "tool_calls", None):
                return True
        return False

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
        context: LLMCallContext | None = None,
    ) -> AsyncIterator[Any]:
        if context is None:
            stream = self._provider.stream_with_retry(kwargs, cancel_token=cancel_token)
        else:
            stream = self._provider.stream_with_retry(
                kwargs,
                cancel_token=cancel_token,
                context=context,
            )
        async for chunk in stream:
            yield chunk

    async def _completion_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        context: LLMCallContext | None = None,
    ) -> Any:
        if context is None:
            return await self._provider.completion_with_retry(kwargs)
        return await self._provider.completion_with_retry(kwargs, context=context)

    def _record_metrics(self, started_at: float, *, success: bool, model: str, mode: str) -> None:
        if self._metrics is None:
            return
        self._metrics.record_llm_call(
            time.perf_counter() - started_at,
            success=success,
            model=model,
            mode=mode,
        )
