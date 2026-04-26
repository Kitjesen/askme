"""Tests for LLMClient: streaming retry/fallback, metrics, client routing, backoff.

Complements test_llm_retry.py which covers non-streaming chat() basics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError


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


# ---------------------------------------------------------------------------
# Streaming retry
# ---------------------------------------------------------------------------


async def test_stream_retries_on_timeout(monkeypatch):
    """chat_stream retries on APITimeoutError during connection phase."""
    client = _make_client(monkeypatch, max_retries=2)

    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "hi"

    async def fake_chunks():
        yield chunk

    call_count = 0

    async def create_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise APITimeoutError(request=MagicMock())
        return fake_chunks()

    client._client.chat.completions.create = create_side_effect

    with patch("askme.llm.client._backoff", return_value=0.001):
        chunks = []
        async for c in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(c)

    assert len(chunks) == 1
    assert call_count == 2


async def test_stream_retries_on_503(monkeypatch):
    """chat_stream retries on HTTP 503 (retryable status)."""
    client = _make_client(monkeypatch, max_retries=1)

    chunk = MagicMock()

    async def fake_chunks():
        yield chunk

    error_response = MagicMock()
    error_response.status_code = 503
    error_response.headers = {}

    call_count = 0

    async def create_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise APIStatusError(
                message="Service unavailable",
                response=error_response,
                body=None,
            )
        return fake_chunks()

    client._client.chat.completions.create = create_side_effect

    with patch("askme.llm.client._backoff", return_value=0.001):
        chunks = []
        async for c in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(c)

    assert len(chunks) == 1


async def test_stream_no_retry_on_400(monkeypatch):
    """chat_stream does NOT retry on non-retryable status (400)."""
    client = _make_client(monkeypatch, max_retries=2)

    error_response = MagicMock()
    error_response.status_code = 400
    error_response.headers = {}

    client._client.chat.completions.create = AsyncMock(
        side_effect=APIStatusError(
            message="Bad request",
            response=error_response,
            body=None,
        )
    )

    with pytest.raises(APIStatusError):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass


async def test_stream_mid_stream_error_propagates(monkeypatch):
    """Mid-stream timeout propagates to caller (no retry after chunks started)."""
    client = _make_client(monkeypatch, max_retries=2)

    async def exploding_chunks():
        yield MagicMock()  # first chunk OK
        raise APITimeoutError(request=MagicMock())  # mid-stream failure

    client._client.chat.completions.create = AsyncMock(
        return_value=exploding_chunks()
    )

    with pytest.raises(APITimeoutError):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass


# ---------------------------------------------------------------------------
# Streaming fallback
# ---------------------------------------------------------------------------


async def test_stream_falls_back_to_next_model(monkeypatch):
    """chat_stream falls back to next model when primary fails."""
    client = _make_client(monkeypatch, max_retries=0)

    chunk = MagicMock()

    async def fake_chunks():
        yield chunk

    models_tried = []

    async def create_side_effect(**kwargs):
        model = kwargs.get("model", "")
        models_tried.append(model)
        if model == "primary-model":
            raise APITimeoutError(request=MagicMock())
        return fake_chunks()

    client._client.chat.completions.create = create_side_effect

    chunks = []
    async for c in client.chat_stream([{"role": "user", "content": "hi"}]):
        chunks.append(c)

    assert models_tried == ["primary-model", "fallback-1"]
    assert len(chunks) == 1


async def test_stream_exhausts_all_models_raises(monkeypatch):
    """chat_stream raises APITimeoutError when all models exhausted."""
    client = _make_client(monkeypatch, max_retries=0, fallback_models=[])

    client._client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    with pytest.raises(APITimeoutError):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass


# ---------------------------------------------------------------------------
# APIConnectionError handling
# ---------------------------------------------------------------------------


async def test_chat_retries_on_connection_error(monkeypatch):
    """chat() retries on APIConnectionError."""
    client = _make_client(monkeypatch, max_retries=1)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "recovered"

    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),
            mock_response,
        ]
    )

    with patch("askme.llm.client._backoff", return_value=0.001):
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "recovered"


async def test_stream_retries_on_connection_error(monkeypatch):
    """chat_stream retries on APIConnectionError during connection."""
    client = _make_client(monkeypatch, max_retries=1)

    chunk = MagicMock()

    async def fake_chunks():
        yield chunk

    call_count = 0

    async def create_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise APIConnectionError(request=MagicMock())
        return fake_chunks()

    client._client.chat.completions.create = create_side_effect

    with patch("askme.llm.client._backoff", return_value=0.001):
        chunks = []
        async for c in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(c)

    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Metrics recording
# ---------------------------------------------------------------------------


async def test_chat_records_metrics_on_success(monkeypatch):
    """Metrics are recorded after successful chat()."""
    from askme.robot.ota_bridge import OTABridgeMetrics
    metrics = OTABridgeMetrics()

    client = _make_client(monkeypatch)
    client._metrics = metrics

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    client._client.chat.completions.create = AsyncMock(return_value=mock_response)

    await client.chat([{"role": "user", "content": "hi"}])

    snap = metrics.snapshot()
    assert snap["llm"]["call_count"] == 1
    assert snap["llm"]["success_count"] == 1


async def test_chat_records_metrics_on_failure(monkeypatch):
    """Metrics are recorded even after chat() failure."""
    from askme.robot.ota_bridge import OTABridgeMetrics
    metrics = OTABridgeMetrics()

    client = _make_client(monkeypatch, max_retries=0, fallback_models=[])
    client._metrics = metrics

    client._client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    with pytest.raises(APITimeoutError):
        await client.chat([{"role": "user", "content": "hi"}])

    snap = metrics.snapshot()
    assert snap["llm"]["call_count"] == 1
    assert snap["llm"]["failure_count"] == 1


async def test_stream_records_metrics(monkeypatch):
    """Metrics recorded for streaming calls."""
    from askme.robot.ota_bridge import OTABridgeMetrics
    metrics = OTABridgeMetrics()

    client = _make_client(monkeypatch)
    client._metrics = metrics

    chunk = MagicMock()

    async def fake_chunks():
        yield chunk

    client._client.chat.completions.create = AsyncMock(
        return_value=fake_chunks()
    )

    async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
        pass

    snap = metrics.snapshot()
    assert snap["llm"]["call_count"] == 1
    assert snap["llm"]["success_count"] == 1
    assert snap["llm"]["last_mode"] == "stream"


# ---------------------------------------------------------------------------
# MiniMax client routing
# ---------------------------------------------------------------------------


def test_client_for_model_routes_minimax(monkeypatch):
    """_client_for_model returns MiniMax client for 'MiniMax-*' models."""
    client = _make_client(monkeypatch, minimax_api_key="mm-key",
                          minimax_base_url="https://api.minimax.chat/v1")

    mm_client = client._client_for_model("MiniMax-M2.5-highspeed")
    default_client = client._client_for_model("claude-opus-4-6")

    assert mm_client is client._minimax_client
    assert default_client is client._client


def test_client_for_model_default_when_no_minimax(monkeypatch):
    """Without MiniMax config, all models use default client."""
    client = _make_client(monkeypatch)

    assert client._client_for_model("MiniMax-M2.5") is client._client


# ---------------------------------------------------------------------------
# Model chain
# ---------------------------------------------------------------------------


def test_model_chain_with_override(monkeypatch):
    """_model_chain with override puts override first."""
    client = _make_client(monkeypatch)

    chain = client._model_chain("custom-model")
    assert chain[0] == "custom-model"
    assert "fallback-1" in chain
    assert "fallback-2" in chain


def test_model_chain_deduplicates(monkeypatch):
    """_model_chain deduplicates primary from fallback list."""
    client = _make_client(
        monkeypatch,
        fallback_models=["primary-model", "fallback-1"],
    )
    chain = client._model_chain()
    # primary-model should appear only once
    assert chain.count("primary-model") == 1


# ---------------------------------------------------------------------------
# Backoff function
# ---------------------------------------------------------------------------


def test_backoff_first_attempt_is_fast():
    """First retry (attempt=0) is fast: ~0.3-0.5s."""
    from askme.llm.client import _backoff

    for _ in range(10):
        val = _backoff(0)
        assert 0.3 <= val <= 0.6


def test_backoff_grows_with_attempt():
    """Later attempts have longer backoff."""
    from askme.llm.client import _backoff

    val0 = _backoff(0)
    val2 = _backoff(2)
    # attempt=2: base = 2^1 = 2, so should be ~2.0-2.5
    assert val2 > val0


def test_backoff_caps_at_8():
    """Backoff base is capped at 8 seconds."""
    from askme.llm.client import _backoff

    for _ in range(10):
        val = _backoff(10)
        assert val <= 8.5  # 8 + 0.5 jitter max


# ---------------------------------------------------------------------------
# Tool/temperature passthrough
# ---------------------------------------------------------------------------


async def test_chat_passes_tools_and_temperature(monkeypatch):
    """chat() passes tools and temperature to create()."""
    client = _make_client(monkeypatch)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    client._client.chat.completions.create = AsyncMock(return_value=mock_response)

    tools = [{"type": "function", "function": {"name": "test"}}]
    await client.chat(
        [{"role": "user", "content": "hi"}],
        tools=tools,
        temperature=0.0,
    )

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["temperature"] == 0.0


async def test_chat_stream_passes_tool_choice(monkeypatch):
    """chat_stream passes tool_choice to create()."""
    client = _make_client(monkeypatch)

    async def fake_chunks():
        yield MagicMock()

    client._client.chat.completions.create = AsyncMock(
        return_value=fake_chunks()
    )

    async for _ in client.chat_stream(
        [{"role": "user", "content": "hi"}],
        tool_choice="auto",
    ):
        pass

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == "auto"
