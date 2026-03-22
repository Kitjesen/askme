"""Tests for LLMClient retry and fallback logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from openai import APITimeoutError, APIStatusError


def _make_client(monkeypatch, **overrides):
    """Create an LLMClient with test defaults."""
    monkeypatch.setattr(
        "askme.llm.client.get_config",
        lambda: {"brain": {
            "api_key": "test-key",
            "base_url": "https://test.example.com/v1",
            "model": "primary-model",
            "max_retries": 1,
            "timeout": 5.0,
            "fallback_models": ["fallback-1", "fallback-2"],
            **overrides,
        }},
    )
    from askme.llm.client import LLMClient
    return LLMClient()


async def test_chat_success(monkeypatch):
    """Normal chat call succeeds on first try."""
    client = _make_client(monkeypatch)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hello"

    client._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert client._client.chat.completions.create.call_count == 1


async def test_chat_retries_on_timeout(monkeypatch):
    """chat() retries on APITimeoutError then succeeds."""
    client = _make_client(monkeypatch, max_retries=2)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            APITimeoutError(request=MagicMock()),
            mock_response,
        ]
    )

    with patch("askme.llm.client._backoff", return_value=0.01):
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert client._client.chat.completions.create.call_count == 2


async def test_chat_falls_back_to_next_model(monkeypatch):
    """chat() falls back to next model when primary exhausts retries."""
    client = _make_client(monkeypatch, max_retries=0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "from fallback"

    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            APITimeoutError(request=MagicMock()),  # primary fails
            mock_response,  # fallback-1 succeeds
        ]
    )

    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "from fallback"

    # Check that fallback model was used
    calls = client._client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] == "primary-model"
    assert calls[1].kwargs["model"] == "fallback-1"


async def test_chat_completion_falls_back_to_next_model(monkeypatch):
    """chat_completion() falls back to next model when primary fails."""
    client = _make_client(monkeypatch, max_retries=0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "from fallback"

    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            APITimeoutError(request=MagicMock()),
            mock_response,
        ]
    )

    result = await client.chat_completion([{"role": "user", "content": "hi"}])

    assert result is mock_response
    calls = client._client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] == "primary-model"
    assert calls[1].kwargs["model"] == "fallback-1"


async def test_chat_retries_on_500(monkeypatch):
    """chat() retries on HTTP 500."""
    client = _make_client(monkeypatch, max_retries=1)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "recovered"

    error_response = MagicMock()
    error_response.status_code = 500
    error_response.headers = {}

    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            APIStatusError(
                message="Internal error",
                response=error_response,
                body=None,
            ),
            mock_response,
        ]
    )

    with patch("askme.llm.client._backoff", return_value=0.01):
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "recovered"


async def test_chat_raises_on_all_models_exhausted(monkeypatch):
    """chat() raises when all models are exhausted."""
    client = _make_client(monkeypatch, max_retries=0, fallback_models=[])

    client._client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    with pytest.raises(APITimeoutError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_model_chain_no_duplicates(monkeypatch):
    """_model_chain() doesn't duplicate primary model in fallbacks."""
    client = _make_client(
        monkeypatch, fallback_models=["primary-model", "other-model"]
    )
    chain = client._model_chain()
    assert chain == ["primary-model", "other-model"]
