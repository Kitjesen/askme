from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from askme.llm.config import LLMConfig
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
        yield SimpleNamespace(choices=[])

    async def completion_with_retry(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )


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

    assert isinstance(create_llm_provider(LLMConfig(provider="minimax", api_key="x")), MiniMaxProvider)
    assert isinstance(create_llm_provider(LLMConfig(provider="doubao", api_key="x")), DoubaoProvider)
    assert isinstance(create_llm_provider(LLMConfig(provider="dashscope", api_key="x")), DashScopeProvider)


def test_factory_infers_provider_from_model_and_url() -> None:
    assert resolve_provider_name(LLMConfig(model="MiniMax-M2.7-highspeed")) == "minimax"
    assert resolve_provider_name(LLMConfig(model="doubao-seed-1-6")) == "doubao"
    assert resolve_provider_name(LLMConfig(model="qwen-max")) == "dashscope"
    assert resolve_provider_name(LLMConfig(base_url="https://api.deepseek.com/v1")) == "deepseek"


@pytest.mark.asyncio
async def test_fake_provider_is_network_free() -> None:
    provider = create_llm_provider(
        LLMConfig(provider="fake", provider_options={"response_text": "offline-ok"})
    )
    gateway = LLMGateway(
        llm_config=LLMConfig(provider="fake", provider_options={"response_text": "offline-ok"}),
        provider=provider,
    )

    assert isinstance(provider, FakeLLMProvider)
    assert await gateway.chat([{"role": "user", "content": "hi"}]) == "offline-ok"
    assert provider.calls[0]["model"] == "MiniMax-M2.7-highspeed"


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
    }


def test_available_provider_list_contains_product_targets() -> None:
    providers = set(available_llm_providers())
    assert {"minimax", "doubao", "dashscope", "deepseek", "zhipu", "openai_compatible", "fake"} <= providers
