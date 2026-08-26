from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from askme.voice.diagnostics import online_smoke


@pytest.mark.asyncio
async def test_llm_check_reports_configured_provider(monkeypatch) -> None:
    raw_client = SimpleNamespace(close=AsyncMock())

    class FakeLLMClient:
        def __init__(self) -> None:
            self.raw_client = raw_client

        def provider_status(self) -> dict[str, str]:
            return {"provider": "deepseek", "model": "deepseek-v4-flash"}

        async def chat(self, messages, *, temperature):
            return "OK"

    monkeypatch.setattr(online_smoke, "LLMClient", FakeLLMClient)

    result = await online_smoke._check_llm()

    assert result["status"] == "ok"
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-v4-flash"
    raw_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_online_smoke_uses_provider_neutral_llm_key(monkeypatch) -> None:
    monkeypatch.setattr(
        online_smoke,
        "get_config",
        lambda reload: {
            "brain": {"api_key": "configured"},
            "voice": {
                "tts": {"backend": "minimax", "minimax_api_key": "configured"},
                "cloud_asr": {
                    "provider": "volcengine",
                    "app_id": "configured",
                    "access_token": "configured",
                },
            },
        },
    )
    monkeypatch.setattr(
        online_smoke,
        "_check_llm",
        AsyncMock(return_value={"status": "ok"}),
    )
    monkeypatch.setattr(
        online_smoke,
        "_check_minimax_tts",
        AsyncMock(return_value={"status": "ok"}),
    )
    monkeypatch.setattr(
        online_smoke,
        "_check_cloud_asr",
        lambda config, *, silence_seconds: {"status": "ok"},
    )

    result = await online_smoke.run_voice_online_smoke()

    assert result["status"] == "ok"
    assert "llm" in result["checks"]
    assert "tts" in result["checks"]
    assert "minimax_tts" not in result["checks"]
    assert "minimax_llm" not in result["checks"]
    assert result["keys_present"]["llm"] is True
    assert result["keys_present"]["cloud_asr"] is True


@pytest.mark.asyncio
async def test_online_smoke_routes_volcengine_tts_without_minimax(monkeypatch) -> None:
    monkeypatch.setattr(
        online_smoke,
        "get_config",
        lambda reload: {
            "brain": {"api_key": "configured"},
            "voice": {
                "tts": {
                    "backend": "volcengine",
                    "volcengine_tts_api_key": "configured",
                },
                "cloud_asr": {
                    "provider": "volcengine",
                    "app_id": "configured",
                    "access_token": "configured",
                },
            },
        },
    )
    monkeypatch.setattr(
        online_smoke,
        "_check_llm",
        AsyncMock(return_value={"status": "ok"}),
    )
    volc_check = AsyncMock(return_value={"status": "ok", "provider": "volcengine"})
    minimax_check = AsyncMock(return_value={"status": "error"})
    monkeypatch.setattr(online_smoke, "_check_volcengine_tts", volc_check)
    monkeypatch.setattr(online_smoke, "_check_minimax_tts", minimax_check)
    monkeypatch.setattr(
        online_smoke,
        "_check_cloud_asr",
        lambda config, *, silence_seconds: {"status": "ok"},
    )

    result = await online_smoke.run_voice_online_smoke()

    assert result["status"] == "ok"
    assert result["checks"]["tts"]["provider"] == "volcengine"
    assert result["keys_present"]["tts"] is True
    volc_check.assert_awaited_once()
    minimax_check.assert_not_awaited()
