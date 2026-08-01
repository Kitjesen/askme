from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import Request
from openai import APIConnectionError

from askme.llm.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext
from askme.llm.factory import available_llm_providers, create_llm_provider, resolve_provider_name
from askme.llm.gateway import LLMGateway
from askme.llm.model_policy import ModelPolicy
from askme.llm.providers.litellm import (
    LiteLLMProxyProvider,
    _await_or_cancel,
    _request_with_context,
)


def test_factory_builds_litellm_proxy_provider() -> None:
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="deepseek-v4-flash",
    )

    provider = create_llm_provider(config)

    assert resolve_provider_name(config) == "litellm"
    assert "litellm" in available_llm_providers()
    assert isinstance(provider, LiteLLMProxyProvider)
    assert str(provider.raw_client.base_url) == "http://127.0.0.1:4000/v1/"


def test_context_envelope_generates_unique_trace_and_replaces_reserved_headers() -> None:
    untrusted = {
        "model": "voice-fast",
        "extra_headers": {
            "TraceParent": "untrusted-trace",
            "X-LiteLLM-Call-ID": "untrusted-call",
            "Authorization": "Bearer provider-secret",
            "x-api-key": "provider-secret",
            "Cookie": "session=person-secret",
            "x-client-feature": "must-not-cross-boundary",
        },
        "metadata": {
            "session_id": "person-secret",
            "raw_face_id": "face-secret",
        },
    }

    first = _request_with_context(untrusted, LLMCallContext())
    second = _request_with_context(untrusted, LLMCallContext())
    without_context = _request_with_context(untrusted, None)
    operational = _request_with_context(
        untrusted,
        LLMCallContext(
            purpose="health_probe",
            channel="system",
            request_class="health_probe",
            privacy_class="operational",
        ),
    )

    first_traceparent = first["extra_headers"]["traceparent"]
    second_traceparent = second["extra_headers"]["traceparent"]
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", first_traceparent)
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", second_traceparent)
    assert first_traceparent.split("-")[1] != second_traceparent.split("-")[1]
    assert first["extra_headers"]["x-litellm-call-id"]
    assert second["extra_headers"]["x-litellm-call-id"]
    assert (
        first["extra_headers"]["x-litellm-call-id"] != second["extra_headers"]["x-litellm-call-id"]
    )
    assert first["metadata"]["call_id"] == first["extra_headers"]["x-litellm-call-id"]
    assert set(first["extra_headers"]) == {"traceparent", "x-litellm-call-id"}
    serialized = json.dumps(first, ensure_ascii=False)
    assert "provider-secret" not in serialized
    assert "person-secret" not in serialized
    assert "face-secret" not in serialized
    assert "must-not-cross-boundary" not in serialized
    assert "extra_headers" not in without_context
    assert "metadata" not in without_context
    assert "provider-secret" not in json.dumps(without_context, ensure_ascii=False)
    assert operational["metadata"]["channel"] == "system"
    assert operational["metadata"]["privacy_class"] == "operational"


def test_litellm_provider_imports_in_a_clean_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from askme.llm.providers import LiteLLMProxyProvider; "
            "print(LiteLLMProxyProvider.__name__)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LiteLLMProxyProvider"


class _NoopProvider:
    raw_client = SimpleNamespace(name="litellm-proxy")
    minimax_client = None

    def client_for_model(self, model: str):
        return self.raw_client

    async def stream_with_retry(
        self, kwargs: dict[str, Any], *, cancel_token: asyncio.Event | None = None
    ) -> AsyncIterator[Any]:
        _ = (kwargs, cancel_token)
        if False:
            yield None

    async def completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        _ = kwargs
        return None


class _FailingProvider(_NoopProvider):
    def __init__(self) -> None:
        self.stream_error = APIConnectionError(request=Request("POST", "https://proxy.invalid"))
        self.completion_error = APIConnectionError(request=Request("POST", "https://proxy.invalid"))

    async def stream_with_retry(self, kwargs, *, cancel_token=None):
        raise self.stream_error
        yield

    async def completion_with_retry(self, kwargs):
        raise self.completion_error


def test_gateway_delegates_model_fallback_to_litellm_proxy() -> None:
    gateway = LLMGateway(
        llm_config=LLMConfig(
            provider="litellm",
            model="deepseek-v4-flash",
            fallback_models=["deepseek-v4-pro"],
        ),
        provider=_NoopProvider(),
        model_policy=ModelPolicy(
            primary_model="deepseek-v4-flash", fallback_models=["shadow-bypass"]
        ),
    )

    assert gateway.provider_status()["fallback_models"] == []
    assert gateway.provider_status()["routing_owner"] == "litellm"
    assert gateway.provider_status()["supports_vision"] is False


@pytest.mark.asyncio
async def test_gateway_sends_safe_turn_context_to_litellm_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Stream:
        def __init__(self) -> None:
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._sent:
                raise StopAsyncIteration
            self._sent = True
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="好", tool_calls=None))]
            )

        async def close(self) -> None:
            return None

    async def _create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    proxy_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="voice-fast",
    )
    provider = create_llm_provider(config)
    monkeypatch.setattr(provider, "client_for_model", lambda model: proxy_client)
    gateway = LLMGateway(llm_config=config, provider=provider)
    context = LLMCallContext(
        trace_id="0123456789abcdef0123456789abcdef",
        session_id="person-sensitive-session",
        turn_id="turn-42",
        call_id="call|42",
        purpose="assistant_response",
        channel="voice",
        request_class="voice_fast",
        latency_budget_ms=900,
        privacy_class="conversation",
        allow_cache=False,
        operator_id="operator-secret",
        evidence_ids=("face-frame-secret",),
    )

    chunks = [
        chunk
        async for chunk in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
            context=context,
        )
    ]

    assert len(chunks) == 1
    assert captured["metadata"]["trace_id"] == "0123456789abcdef0123456789abcdef"
    assert captured["metadata"]["turn_id"] == "sha256:8388af242902285fb7292fdc"
    assert captured["metadata"]["purpose"] == "assistant_response"
    assert captured["metadata"]["channel"] == "voice"
    assert captured["metadata"]["request_class"] == "voice_fast"
    assert captured["metadata"]["privacy_class"] == "conversation"
    assert captured["metadata"]["model_alias"] == "voice-fast"
    assert captured["metadata"]["allow_cache"] == "false"
    assert captured["metadata"]["latency_budget_ms"] == 900
    assert captured["metadata"]["call_id"] == "sha256:01a1af6f749020fe363fb382"
    assert re.fullmatch(r"sha256:[0-9a-f]{24}", captured["metadata"]["call_id"])
    assert re.fullmatch(
        r"00-0123456789abcdef0123456789abcdef-[0-9a-f]{16}-01",
        captured["extra_headers"]["traceparent"],
    )
    assert captured["extra_headers"]["x-litellm-call-id"] == captured["metadata"]["call_id"]
    assert captured["extra_body"]["cache"] == {
        "no-cache": True,
        "no-store": True,
    }
    serialized = json.dumps(captured, ensure_ascii=False, default=str)
    assert "turn-42" not in serialized
    assert "person-sensitive-session" not in serialized
    assert "operator-secret" not in serialized
    assert "face-frame-secret" not in serialized


@pytest.mark.asyncio
async def test_nonstream_litellm_call_keeps_context_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    transport_options: dict[str, Any] = {}
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="完成"))])

    async def _create(**kwargs):
        captured.update(kwargs)
        return response

    class _ProxyClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

        def with_options(self, **kwargs):
            transport_options.update(kwargs)
            return self

    proxy_client = _ProxyClient()
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="memory-fast",
    )
    provider = create_llm_provider(config)
    monkeypatch.setattr(provider, "client_for_model", lambda model: proxy_client)
    gateway = LLMGateway(llm_config=config, provider=provider)
    context = LLMCallContext(
        trace_id="fedcba9876543210fedcba9876543210",
        turn_id="person@example.com/turn-memory-7",
        call_id="person@example.com|memory-call-7",
        purpose="memory_compact",
        channel="background",
        request_class="memory",
        latency_budget_ms=250,
        privacy_class="sensitive",
        allow_cache=False,
    )

    result = await gateway.chat_completion(
        [{"role": "user", "content": "压缩记忆"}],
        context=context,
    )

    assert result is response
    assert captured["metadata"]["purpose"] == "memory_compact"
    assert captured["metadata"]["request_class"] == "memory"
    assert captured["metadata"]["privacy_class"] == "sensitive"
    assert re.fullmatch(r"sha256:[0-9a-f]{24}", captured["metadata"]["turn_id"])
    assert captured["metadata"]["latency_budget_ms"] == 250
    assert captured["extra_body"]["cache"] == {
        "no-cache": True,
        "no-store": True,
    }
    assert re.fullmatch(
        r"00-fedcba9876543210fedcba9876543210-[0-9a-f]{16}-01",
        captured["extra_headers"]["traceparent"],
    )
    assert re.fullmatch(
        r"sha256:[0-9a-f]{24}",
        captured["extra_headers"]["x-litellm-call-id"],
    )
    assert 0 < transport_options["timeout"] <= 0.25
    assert "timeout" not in captured
    diagnostics = gateway.recent_call_diagnostics()
    assert diagnostics == [
        {
            "call_id": captured["extra_headers"]["x-litellm-call-id"],
            "trace_id": "fedcba9876543210fedcba9876543210",
            "turn_id": captured["metadata"]["turn_id"],
            "purpose": "memory_compact",
            "request_class": "memory",
            "model_alias": "memory-fast",
            "resolved_model": "memory-fast",
            "mode": "completion",
            "outcome": "success",
            "semantic_started": True,
            "duration_ms": diagnostics[0]["duration_ms"],
        }
    ]
    assert diagnostics[0]["duration_ms"] >= 0
    serialized_diagnostics = json.dumps(diagnostics, ensure_ascii=False)
    assert "压缩记忆" not in serialized_diagnostics
    assert "person@example.com" not in serialized_diagnostics
    assert gateway.provider_status()["call_diagnostics"] == {
        "count": 1,
        "last_outcome": "success",
    }


@pytest.mark.asyncio
async def test_cancel_closes_litellm_stream_and_stops_future_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stream:
        def __init__(self) -> None:
            self.index = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= 2:
                raise StopAsyncIteration
            self.index += 1
            return SimpleNamespace(index=self.index)

        async def close(self) -> None:
            self.closed = True

    stream = _Stream()
    completions = SimpleNamespace(create=lambda **kwargs: None)

    async def _create(**kwargs):
        return stream

    completions.create = _create
    proxy_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="deepseek-v4-flash",
    )
    provider = create_llm_provider(config)
    monkeypatch.setattr(provider, "client_for_model", lambda model: proxy_client)
    gateway = LLMGateway(llm_config=config, provider=provider)
    cancel_token = asyncio.Event()
    chunks = []

    async for chunk in gateway.chat_stream(
        [{"role": "user", "content": "你好"}],
        cancel_token=cancel_token,
    ):
        chunks.append(chunk)
        cancel_token.set()

    assert len(chunks) == 1
    assert stream.closed is True


@pytest.mark.asyncio
async def test_cancel_interrupts_a_stalled_litellm_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StalledStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(index=1)

        async def close(self) -> None:
            self.closed = True
            self.release.set()

    stream = _StalledStream()

    async def _create(**kwargs):
        _ = kwargs
        return stream

    proxy_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        )
    )
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="deepseek-v4-flash",
    )
    provider = create_llm_provider(config)
    monkeypatch.setattr(provider, "client_for_model", lambda model: proxy_client)
    gateway = LLMGateway(llm_config=config, provider=provider)
    cancel_token = asyncio.Event()
    iterator = gateway.chat_stream(
        [{"role": "user", "content": "你好"}],
        cancel_token=cancel_token,
    )

    next_chunk = asyncio.ensure_future(anext(iterator))
    await asyncio.wait_for(stream.started.wait(), timeout=0.5)
    cancel_token.set()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(next_chunk, timeout=0.2)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_cancel_interrupts_a_stalled_litellm_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    request_was_cancelled = asyncio.Event()

    async def _create(**kwargs):
        _ = kwargs
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            request_was_cancelled.set()
            raise
        raise AssertionError("cancelled request must not produce a stream")

    proxy_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        )
    )
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="deepseek-v4-flash",
    )
    provider = create_llm_provider(config)
    monkeypatch.setattr(provider, "client_for_model", lambda model: proxy_client)
    gateway = LLMGateway(llm_config=config, provider=provider)
    cancel_token = asyncio.Event()
    iterator = gateway.chat_stream(
        [{"role": "user", "content": "你好"}],
        cancel_token=cancel_token,
    )

    next_chunk = asyncio.ensure_future(anext(iterator))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    cancel_token.set()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(next_chunk, timeout=0.2)
    assert request_was_cancelled.is_set()


def test_litellm_never_bypasses_proxy_for_minimax_models() -> None:
    config = LLMConfig(
        provider="litellm",
        api_key="sk-virtual-key",
        base_url="http://127.0.0.1:4000/v1",
        model="deepseek-v4-flash",
        max_retries=4,
        minimax_api_key="direct-provider-secret",
    )

    provider = create_llm_provider(config)

    assert isinstance(provider, LiteLLMProxyProvider)
    assert provider.minimax_client is None
    assert provider.client_for_model("MiniMax-M2.7-highspeed") is provider.raw_client


@pytest.mark.asyncio
async def test_litellm_preserves_proxy_errors_instead_of_synthetic_timeout() -> None:
    config = LLMConfig(
        provider="litellm",
        model="deepseek-v4-flash",
    )
    provider = _FailingProvider()
    gateway = LLMGateway(llm_config=config, provider=provider)

    with pytest.raises(APIConnectionError) as stream_error:
        async for _ in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
        ):
            pass

    assert stream_error.value is provider.stream_error

    with pytest.raises(APIConnectionError) as completion_error:
        await gateway.chat_completion(
            [{"role": "user", "content": "你好"}],
        )

    assert completion_error.value is provider.completion_error


@pytest.mark.asyncio
async def test_cancel_race_prefers_a_completed_upstream_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_token = asyncio.Event()
    expected = object()

    async def _operation():
        cancel_token.set()
        return expected

    original_wait = asyncio.wait

    async def _complete_both(tasks, **kwargs):
        await asyncio.gather(*tasks)
        return set(tasks), set()

    monkeypatch.setattr(asyncio, "wait", _complete_both)
    try:
        result = await _await_or_cancel(
            _operation(),
            cancel_token,
            prefer_operation_on_race=True,
        )
    finally:
        monkeypatch.setattr(asyncio, "wait", original_wait)

    assert result is expected
