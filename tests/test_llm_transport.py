"""Transport lifecycle tests for warm LLM connections."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from askme.llm.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _create_async_client,
)


def test_async_openai_client_uses_explicit_warm_keepalive_pool() -> None:
    transport = MagicMock(spec=httpx.AsyncClient)
    sdk_client = object()

    with (
        patch.dict(sys.modules, {"inovxio_llm": None}),
        patch("httpx.AsyncClient", return_value=transport) as http_client_cls,
        patch(
            "askme.llm.providers.openai_compatible.AsyncOpenAI",
            return_value=sdk_client,
        ) as sdk_client_cls,
    ):
        result = _create_async_client(
            api_key="test-key",
            base_url="https://provider.invalid/v1",
            model="test-model",
            timeout=12.0,
            http_keepalive_expiry_seconds=65.0,
            http_max_connections=40,
            http_max_keepalive_connections=10,
        )

    assert result is sdk_client
    limits = http_client_cls.call_args.kwargs["limits"]
    assert limits.keepalive_expiry == 65.0
    assert limits.max_connections == 40
    assert limits.max_keepalive_connections == 10
    assert sdk_client_cls.call_args.kwargs["http_client"] is transport


def test_inovxio_client_receives_supported_warm_pool_options() -> None:
    captured: dict[str, object] = {}
    sdk_client = object()

    class FakeConfig:
        def __init__(
            self,
            *,
            api_key,
            base_url,
            model,
            timeout,
            http_keepalive_expiry_seconds,
            http_max_connections,
            http_max_keepalive_connections,
        ):
            captured.update(locals())
            captured.pop("self", None)

    module = ModuleType("inovxio_llm")
    module.LLMClientConfig = FakeConfig
    module.create_async_openai_client = lambda config: sdk_client

    with patch.dict(sys.modules, {"inovxio_llm": module}):
        result = _create_async_client(
            api_key="test-key",
            base_url="https://provider.invalid/v1",
            model="test-model",
            timeout=12.0,
            http_keepalive_expiry_seconds=65.0,
            http_max_connections=40,
            http_max_keepalive_connections=10,
        )

    assert result is sdk_client
    assert captured["http_keepalive_expiry_seconds"] == 65.0
    assert captured["http_max_connections"] == 40
    assert captured["http_max_keepalive_connections"] == 10


@pytest.mark.asyncio
async def test_provider_close_releases_primary_and_secondary_clients() -> None:
    primary = MagicMock()
    primary.close = AsyncMock()
    secondary = MagicMock()
    secondary.close = AsyncMock()
    provider = object.__new__(OpenAICompatibleProvider)
    provider._client = primary
    provider._minimax_client = secondary

    await provider.aclose()

    primary.close.assert_awaited_once_with()
    secondary.close.assert_awaited_once_with()
