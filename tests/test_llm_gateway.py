from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import Request
from openai import APIConnectionError

from askme.interfaces.llm import LLMBackend
from askme.llm.config import LLMConfig
from askme.llm.core.contracts import (
    LLMCallContext,
    LLMDeadlineExceeded,
    LLMNoSemanticResponse,
)
from askme.llm.factory import available_llm_providers, create_llm_provider, resolve_provider_name
from askme.llm.gateway import LLMGateway
from askme.llm.model_policy import ModelPolicy
from askme.llm.providers.domestic import DashScopeProvider, DoubaoProvider, MiniMaxProvider
from askme.llm.providers.fake import FakeLLMProvider


class _FakeProvider:
    def __init__(self) -> None:
        self.raw_client = SimpleNamespace(name="raw")
        self.minimax_client = None
        self.calls: list[dict[str, Any]] = []

    def client_for_model(self, model: str) -> Any:
        return self.raw_client

    async def stream_with_retry(self, kwargs: dict[str, Any], *, cancel_token=None):
        self.calls.append(dict(kwargs))
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None))]
        )

    async def completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )


def test_llm_backend_contract_exposes_control_plane_parameters() -> None:
    chat_parameters = inspect.signature(LLMBackend.chat).parameters
    stream_parameters = inspect.signature(LLMBackend.chat_stream).parameters

    assert "context" in chat_parameters
    assert {"context", "cancel_token", "max_tokens"} <= stream_parameters.keys()


def test_model_policy_keeps_provider_fallbacks_separate() -> None:
    minimax = ModelPolicy(
        primary_model="MiniMax-M2.7-highspeed",
        fallback_models=["claude-3-5-sonnet", "MiniMax-M2.5-highspeed"],
    )
    relay = ModelPolicy(
        primary_model="claude-3-5-sonnet",
        fallback_models=["MiniMax-M2.7-highspeed", "claude-3-haiku"],
    )

    assert minimax.model_chain() == ["MiniMax-M2.7-highspeed", "MiniMax-M2.5-highspeed"]
    assert relay.model_chain() == ["claude-3-5-sonnet", "claude-3-haiku"]


@pytest.mark.asyncio
async def test_gateway_uses_injected_provider_and_records_request_policy() -> None:
    provider = _FakeProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="MiniMax-M2.7-highspeed", temperature=0.2),
        provider=provider,
    )

    result = await gateway.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert result == "ok"
    assert provider.calls[0]["model"] == "MiniMax-M2.7-highspeed"
    assert provider.calls[0]["temperature"] == 0.2
    assert provider.calls[0]["tools"] == [{"type": "function"}]
    assert provider.calls[0]["extra_body"] == {"reasoning_split": True}


@pytest.mark.asyncio
async def test_gateway_stream_supports_thinking_mode_without_minimax_extra_body() -> None:
    provider = _FakeProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="MiniMax-M2.7-highspeed"),
        provider=provider,
    )

    chunks = []
    async for chunk in gateway.chat_stream([{"role": "user", "content": "hi"}], thinking=True):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "extra_body" not in provider.calls[0]


@pytest.mark.asyncio
async def test_gateway_disables_deepseek_thinking_for_realtime_turns() -> None:
    provider = _FakeProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="deepseek-v4-flash"),
        provider=provider,
    )

    await gateway.chat([{"role": "user", "content": "hi"}])

    assert provider.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_gateway_can_explicitly_enable_deepseek_thinking() -> None:
    provider = _FakeProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="deepseek-v4-pro"),
        provider=provider,
    )

    chunks = []
    async for chunk in gateway.chat_stream(
        [{"role": "user", "content": "hi"}],
        thinking=True,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert provider.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_factory_resolves_domestic_provider_aliases() -> None:
    assert resolve_provider_name(LLMConfig(provider="minimax")) == "minimax"
    assert resolve_provider_name(LLMConfig(provider="ark")) == "doubao"
    assert resolve_provider_name(LLMConfig(provider="qwen")) == "dashscope"

    assert isinstance(
        create_llm_provider(LLMConfig(provider="minimax", api_key="x")), MiniMaxProvider
    )
    assert isinstance(
        create_llm_provider(LLMConfig(provider="doubao", api_key="x")), DoubaoProvider
    )
    assert isinstance(
        create_llm_provider(LLMConfig(provider="dashscope", api_key="x")), DashScopeProvider
    )


def test_explicit_minimax_config_builds_client_with_canonical_endpoint() -> None:
    config = LLMConfig.from_cfg(
        {
            "provider": "minimax",
            "api_key": "test-key",
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.7-highspeed",
        }
    )

    provider = create_llm_provider(config)

    assert isinstance(provider, MiniMaxProvider)
    assert str(provider.raw_client.base_url) == "https://api.minimaxi.com/v1/"


@pytest.mark.parametrize(
    "config",
    [
        LLMConfig(model="MiniMax-M2.7-highspeed"),
        LLMConfig(model="doubao-seed-1-6"),
        LLMConfig(model="qwen-max"),
        LLMConfig(base_url="https://api.deepseek.com/v1"),
    ],
)
def test_factory_rejects_implicit_provider_inference(config: LLMConfig) -> None:
    with pytest.raises(ValueError, match="provider.*required"):
        resolve_provider_name(config)


def test_cancel_warm_probes_never_changes_business_request_accounting() -> None:
    gateway = LLMGateway(
        llm_config=LLMConfig(provider="openai_compatible", model="voice-fast"),
        provider=_FakeProvider(),
    )
    first = asyncio.Event()
    second = asyncio.Event()
    gateway._active_warm_probes.update({first, second})
    gateway._active_business_requests = 2

    assert gateway.cancel_warm_probes() == 2
    assert first.is_set() and second.is_set()
    assert gateway.request_activity() == {
        "active_business_requests": 2,
        "active_warm_probes": 2,
    }


def test_unused_idle_gateway_can_close_synchronously_without_async_transport() -> None:
    provider = _FakeProvider()
    provider.aclose = AsyncMock()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )

    assert gateway.close_sync() is True
    assert gateway.close_sync() is True
    provider.aclose.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(gateway.chat([{"role": "user", "content": "hello"}]))


@pytest.mark.asyncio
async def test_used_idle_gateway_requires_async_close_on_owner_loop() -> None:
    provider = _FakeProvider()
    provider.aclose = AsyncMock()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )

    assert await gateway.chat([{"role": "user", "content": "hello"}]) == "ok"
    assert gateway.request_activity() == {
        "active_business_requests": 0,
        "active_warm_probes": 0,
    }
    assert gateway.close_sync() is False

    await gateway.aclose()
    provider.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_fake_provider_is_network_free() -> None:
    provider = create_llm_provider(
        LLMConfig(
            provider="fake", model="fake-offline", provider_options={"response_text": "offline-ok"}
        )
    )
    gateway = LLMGateway(
        llm_config=LLMConfig(
            provider="fake", model="fake-offline", provider_options={"response_text": "offline-ok"}
        ),
        provider=provider,
    )

    assert isinstance(provider, FakeLLMProvider)
    assert await gateway.chat([{"role": "user", "content": "hi"}]) == "offline-ok"
    assert provider.calls[0]["model"] == "fake-offline"


def test_provider_status_exposes_non_secret_metadata() -> None:
    gateway = LLMGateway(
        llm_config=LLMConfig(
            provider="doubao",
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="doubao-seed-1-6",
            fallback_models=["doubao-lite"],
        ),
        provider=_FakeProvider(),
    )

    assert gateway.provider_status() == {
        "provider": "doubao",
        "model": "doubao-seed-1-6",
        "base_url": "https://example.invalid/v1",
        "openai_compatible": True,
        "domestic": True,
        "supports_tools": True,
        "supports_vision": False,
        "fallback_models": ["doubao-lite"],
        "routing_owner": "askme",
    }


def test_available_provider_list_contains_product_targets() -> None:
    providers = set(available_llm_providers())
    assert {
        "minimax",
        "doubao",
        "dashscope",
        "deepseek",
        "zhipu",
        "litellm",
        "openai_compatible",
        "fake",
    } <= providers


@pytest.mark.asyncio
async def test_first_semantic_deadline_ignores_empty_keepalive_chunks() -> None:
    class _EmptyThenStalledProvider(_FakeProvider):
        async def stream_with_retry(self, kwargs, *, cancel_token=None, context=None):
            self.calls.append(dict(kwargs))
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None))]
            )
            await asyncio.Event().wait()

    provider = _EmptyThenStalledProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )
    context = LLMCallContext(
        trace_id="0123456789abcdef0123456789abcdef",
        turn_id="turn-deadline",
        channel="voice",
        request_class="voice_fast",
        latency_budget_ms=100,
    )
    chunks = []

    async def _consume() -> None:
        async for chunk in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
            context=context,
        ):
            chunks.append(chunk)

    with pytest.raises(LLMDeadlineExceeded) as caught:
        await asyncio.wait_for(_consume(), timeout=0.5)

    assert len(chunks) == 1
    assert caught.value.phase == "first_semantic"
    assert caught.value.trace_id == context.trace_id
    assert 0 < provider.calls[0]["timeout"] <= 0.1


@pytest.mark.asyncio
async def test_fallback_before_semantic_uses_only_remaining_latency_budget() -> None:
    class _FallbackProvider(_FakeProvider):
        async def stream_with_retry(self, kwargs, *, cancel_token=None, context=None):
            self.calls.append(dict(kwargs))
            if kwargs["model"] == "primary":
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None))]
                )
                await asyncio.sleep(0.03)
                raise APIConnectionError(request=Request("POST", "https://primary.invalid"))
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="备用回答", tool_calls=None))
                ]
            )

    provider = _FallbackProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(
            model="primary",
            fallback_models=["fallback"],
        ),
        provider=provider,
    )
    context = LLMCallContext(
        request_class="voice_fast",
        latency_budget_ms=500,
    )

    chunks = [
        chunk
        async for chunk in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
            context=context,
        )
    ]

    assert chunks[-1].choices[0].delta.content == "备用回答"
    assert [call["model"] for call in provider.calls] == ["primary", "fallback"]
    assert 0 < provider.calls[1]["timeout"] < provider.calls[0]["timeout"] <= 0.5


@pytest.mark.asyncio
async def test_stream_ending_without_semantic_payload_uses_fallback() -> None:
    class _EmptyPrimaryProvider(_FakeProvider):
        async def stream_with_retry(self, kwargs, *, cancel_token=None, context=None):
            self.calls.append(dict(kwargs))
            if kwargs["model"] == "primary":
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None))]
                )
                return
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="备用语义", tool_calls=None))
                ]
            )

    provider = _EmptyPrimaryProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="primary", fallback_models=["fallback"]),
        provider=provider,
    )

    chunks = [
        chunk
        async for chunk in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
        )
    ]

    assert chunks[-1].choices[0].delta.content == "备用语义"
    assert [call["model"] for call in provider.calls] == ["primary", "fallback"]


@pytest.mark.asyncio
async def test_litellm_empty_response_is_explicit_failure_without_hidden_bypass() -> None:
    class _EmptyProxyProvider(_FakeProvider):
        async def stream_with_retry(self, kwargs, *, cancel_token=None, context=None):
            self.calls.append(dict(kwargs))
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None))]
            )

    provider = _EmptyProxyProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(
            provider="litellm",
            model="voice-fast",
            fallback_models=["forbidden-direct-model"],
        ),
        provider=provider,
    )

    with pytest.raises(LLMNoSemanticResponse):
        async for _ in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
        ):
            pass

    assert [call["model"] for call in provider.calls] == ["voice-fast"]


@pytest.mark.asyncio
async def test_nonstream_completion_enforces_product_deadline() -> None:
    class _StalledProvider(_FakeProvider):
        async def completion_with_retry(self, kwargs, *, context=None):
            self.calls.append(dict(kwargs))
            await asyncio.Event().wait()

    provider = _StalledProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="memory-fast"),
        provider=provider,
    )
    context = LLMCallContext(
        trace_id="0123456789abcdef0123456789abcdef",
        turn_id="turn-memory-timeout",
        request_class="memory",
        latency_budget_ms=50,
    )

    with pytest.raises(LLMDeadlineExceeded) as caught:
        await asyncio.wait_for(
            gateway.chat_completion(
                [{"role": "user", "content": "压缩记忆"}],
                context=context,
            ),
            timeout=0.3,
        )

    assert caught.value.phase == "completion"
    assert caught.value.turn_id == "turn-memory-timeout"
    assert 0 < provider.calls[0]["timeout"] <= 0.05
    assert gateway.recent_call_diagnostics()[-1]["outcome"] == "deadline_exceeded"


@pytest.mark.asyncio
async def test_business_stream_preempts_inflight_health_probe() -> None:
    class _PriorityProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.warm_started = asyncio.Event()
            self.warm_cancelled = asyncio.Event()

        async def stream_with_retry(
            self,
            kwargs,
            *,
            cancel_token=None,
            context=None,
        ):
            if context is not None and context.request_class == "health_probe":
                self.warm_started.set()
                while cancel_token is not None and not cancel_token.is_set():
                    await asyncio.sleep(0.005)
                self.warm_cancelled.set()
                return
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="business", tool_calls=None))
                ]
            )

    provider = _PriorityProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )
    warm_cancel = asyncio.Event()
    warm_context = LLMCallContext(
        purpose="health_probe",
        request_class="health_probe",
    )

    async def _consume_warm() -> None:
        async for _chunk in gateway.chat_stream(
            [{"role": "user", "content": "好"}],
            cancel_token=warm_cancel,
            context=warm_context,
        ):
            pass

    warm_task = asyncio.create_task(_consume_warm())
    try:
        await asyncio.wait_for(provider.warm_started.wait(), timeout=1.0)
        chunks = [
            chunk
            async for chunk in gateway.chat_stream(
                [{"role": "user", "content": "你好"}],
            )
        ]
        await asyncio.wait_for(provider.warm_cancelled.wait(), timeout=0.15)
        await asyncio.wait_for(warm_task, timeout=0.15)

        assert len(chunks) == 1
        assert warm_cancel.is_set()
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }
    finally:
        warm_cancel.set()
        await asyncio.gather(warm_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_health_probe_defers_without_transport_while_business_stream_is_active() -> None:
    class _PriorityProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.business_started = asyncio.Event()
            self.release_business = asyncio.Event()
            self.health_calls = 0

        async def stream_with_retry(
            self,
            kwargs,
            *,
            cancel_token=None,
            context=None,
        ):
            if context is not None and context.request_class == "health_probe":
                self.health_calls += 1
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="warm"))]
                )
                return
            self.business_started.set()
            await self.release_business.wait()
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="business"))]
            )

    provider = _PriorityProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )

    async def _consume_business():
        return [
            chunk
            async for chunk in gateway.chat_stream(
                [{"role": "user", "content": "你好"}],
            )
        ]

    business_task = asyncio.create_task(_consume_business())
    try:
        await asyncio.wait_for(provider.business_started.wait(), timeout=1.0)
        warm_cancel = asyncio.Event()
        warm_chunks = [
            chunk
            async for chunk in gateway.chat_stream(
                [{"role": "user", "content": "好"}],
                cancel_token=warm_cancel,
                context=LLMCallContext(
                    purpose="health_probe",
                    request_class="health_probe",
                ),
            )
        ]

        assert warm_chunks == []
        assert warm_cancel.is_set()
        assert provider.health_calls == 0
        assert gateway.request_activity() == {
            "active_business_requests": 1,
            "active_warm_probes": 0,
        }

        provider.release_business.set()
        business_chunks = await asyncio.wait_for(business_task, timeout=1.0)
        assert len(business_chunks) == 1
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }
    finally:
        provider.release_business.set()
        await asyncio.gather(business_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_business_completion_also_preempts_inflight_health_probe() -> None:
    class _PriorityProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.warm_started = asyncio.Event()
            self.warm_cancelled = asyncio.Event()

        async def stream_with_retry(
            self,
            kwargs,
            *,
            cancel_token=None,
            context=None,
        ):
            self.warm_started.set()
            while cancel_token is not None and not cancel_token.is_set():
                await asyncio.sleep(0.005)
            self.warm_cancelled.set()
            if False:
                yield None

        async def completion_with_retry(self, kwargs, *, context=None):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="business"))]
            )

    provider = _PriorityProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )
    warm_cancel = asyncio.Event()
    warm_context = LLMCallContext(
        purpose="health_probe",
        request_class="health_probe",
    )

    async def _consume_warm() -> None:
        async for _chunk in gateway.chat_stream(
            [{"role": "user", "content": "好"}],
            cancel_token=warm_cancel,
            context=warm_context,
        ):
            pass

    warm_task = asyncio.create_task(_consume_warm())
    try:
        await asyncio.wait_for(provider.warm_started.wait(), timeout=1.0)
        assert await gateway.chat([{"role": "user", "content": "你好"}]) == "business"
        await asyncio.wait_for(provider.warm_cancelled.wait(), timeout=0.15)
        await asyncio.wait_for(warm_task, timeout=0.15)
        assert warm_cancel.is_set()
    finally:
        warm_cancel.set()
        await asyncio.gather(warm_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_nonstream_health_probe_is_counted_as_warm_without_cancelling_peer() -> None:
    class _ProbeProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.stream_started = asyncio.Event()
            self.release_stream = asyncio.Event()
            self.completion_started = asyncio.Event()
            self.release_completion = asyncio.Event()

        async def stream_with_retry(
            self,
            kwargs,
            *,
            cancel_token=None,
            context=None,
        ):
            self.stream_started.set()
            while not self.release_stream.is_set():
                if cancel_token is not None and cancel_token.is_set():
                    return
                await asyncio.sleep(0.005)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="warm"))])

        async def completion_with_retry(self, kwargs, *, context=None):
            self.completion_started.set()
            await self.release_completion.wait()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="warm"))]
            )

    provider = _ProbeProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="health-probe"),
        provider=provider,
    )
    context = LLMCallContext(
        purpose="health_probe",
        request_class="health_probe",
    )
    stream_cancel = asyncio.Event()

    async def _consume_stream() -> None:
        async for _chunk in gateway.chat_stream(
            [{"role": "user", "content": "好"}],
            cancel_token=stream_cancel,
            context=context,
        ):
            pass

    stream_task = asyncio.create_task(_consume_stream())
    completion_task = None
    try:
        await asyncio.wait_for(provider.stream_started.wait(), timeout=1.0)
        completion_task = asyncio.create_task(
            gateway.chat_completion(
                [{"role": "user", "content": "好"}],
                context=context,
            )
        )
        await asyncio.wait_for(provider.completion_started.wait(), timeout=1.0)

        assert not stream_cancel.is_set()
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 2,
        }

        provider.release_stream.set()
        provider.release_completion.set()
        result = await asyncio.wait_for(completion_task, timeout=1.0)
        await asyncio.wait_for(stream_task, timeout=1.0)
        assert result.choices[0].message.content == "warm"
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }
    finally:
        provider.release_stream.set()
        provider.release_completion.set()
        if completion_task is not None:
            await asyncio.gather(completion_task, return_exceptions=True)
        await asyncio.gather(stream_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_nonstream_health_probe_waits_for_business_drain_before_transport() -> None:
    class _AdmissionProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.business_started = asyncio.Event()
            self.release_business = asyncio.Event()
            self.probe_started = asyncio.Event()
            self.release_probe = asyncio.Event()

        async def stream_with_retry(
            self,
            kwargs,
            *,
            cancel_token=None,
            context=None,
        ):
            self.business_started.set()
            await self.release_business.wait()
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="business"))]
            )

        async def completion_with_retry(self, kwargs, *, context=None):
            self.probe_started.set()
            await self.release_probe.wait()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="warm"))]
            )

    provider = _AdmissionProvider()
    gateway = LLMGateway(
        llm_config=LLMConfig(model="voice-fast"),
        provider=provider,
    )

    async def _consume_business() -> None:
        async for _chunk in gateway.chat_stream(
            [{"role": "user", "content": "你好"}],
        ):
            pass

    business_task = asyncio.create_task(_consume_business())
    probe_task = None
    try:
        await asyncio.wait_for(provider.business_started.wait(), timeout=1.0)
        probe_task = asyncio.create_task(
            gateway.chat_completion(
                [{"role": "user", "content": "好"}],
                context=LLMCallContext(
                    purpose="health_probe",
                    request_class="health_probe",
                ),
            )
        )
        await asyncio.sleep(0.03)

        assert not provider.probe_started.is_set()
        assert not probe_task.done()
        assert gateway.request_activity() == {
            "active_business_requests": 1,
            "active_warm_probes": 0,
        }

        provider.release_business.set()
        await asyncio.wait_for(business_task, timeout=1.0)
        await asyncio.wait_for(provider.probe_started.wait(), timeout=1.0)
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 1,
        }

        provider.release_probe.set()
        result = await asyncio.wait_for(probe_task, timeout=1.0)
        assert result.choices[0].message.content == "warm"
        assert gateway.request_activity() == {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }
    finally:
        provider.release_business.set()
        provider.release_probe.set()
        if probe_task is not None:
            await asyncio.gather(probe_task, return_exceptions=True)
        await asyncio.gather(business_task, return_exceptions=True)
