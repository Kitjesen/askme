from __future__ import annotations

import pytest

from askme.voice.realtime.config import (
    RealtimeVoiceMode,
    resolve_realtime_voice_config,
)
from askme.voice.realtime.readiness import check_realtime_voice


def test_realtime_voice_is_disabled_and_cascade_safe_by_default() -> None:
    config = resolve_realtime_voice_config({})

    assert config.enabled is False
    assert config.mode is RealtimeVoiceMode.SPLIT
    assert config.provider == "volcengine_s2s"
    assert config.fallback == "cascade"
    assert config.endpoint == "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
    assert config.resource_id == "volc.speech.dialog"
    assert config.model == "1.2.1.1"
    assert config.input_sample_rate == 16_000
    assert config.output_sample_rate == 24_000
    assert config.output_format == "pcm_s16le"
    assert config.chunk_ms == 20
    assert config.available is False


def test_enabled_volcengine_realtime_config_uses_dedicated_credentials() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "provider": "volcengine",
                    "app_id": "app-1",
                    "access_token": "token-1",
                    "speaker": "zh_male_yunzhou_jupiter_bigtts",
                }
            }
        }
    )

    assert config.enabled is True
    assert config.mode is RealtimeVoiceMode.GENERAL_CHAT
    assert config.provider == "volcengine_s2s"
    assert config.credentials_configured is True
    assert config.available is True
    assert config.validation_errors() == []

    snapshot = config.status_snapshot()
    assert snapshot["credentials_configured"] is True
    assert "token-1" not in repr(snapshot)
    assert "app-1" not in repr(snapshot)


def test_qwen35_realtime_resolves_official_defaults_and_api_key() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "provider": "qwen3_5_omni",
                    "api_key": "dashscope-secret",
                    "workspace_id": "workspace-123",
                }
            }
        }
    )

    assert config.provider == "qwen3_5_omni"
    assert config.endpoint == (
        "wss://workspace-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
    )
    assert config.model == "qwen3.5-omni-flash-realtime"
    assert config.speaker == "Tina"
    assert config.credentials_configured is True
    assert config.available is True
    assert config.validation_errors() == []
    assert "dashscope-secret" not in repr(config.status_snapshot())
    assert "dashscope-secret" not in repr(config)


def test_qwen35_enabled_config_requires_workspace_id() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "provider": "qwen3_5_omni",
                    "api_key": "dashscope-secret",
                }
            }
        }
    )

    assert config.available is False
    assert config.validation_errors() == [
        "voice.realtime.workspace_id is required for qwen3_5_omni"
    ]


@pytest.mark.parametrize(
    ("region", "expected_host"),
    [
        ("cn-beijing", "ws-123.cn-beijing.maas.aliyuncs.com"),
        ("ap-southeast-1", "ws-123.ap-southeast-1.maas.aliyuncs.com"),
    ],
)
def test_qwen35_workspace_endpoint_is_derived_from_region(
    region: str,
    expected_host: str,
) -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "provider": "qwen3_5_omni",
                    "api_key": "dashscope-secret",
                    "workspace_id": "ws-123",
                    "region": region,
                }
            }
        }
    )

    assert config.endpoint == f"wss://{expected_host}/api-ws/v1/realtime"
    assert config.available is True


@pytest.mark.parametrize(
    "endpoint",
    [
        "wss://attacker.example/api-ws/v1/realtime",
        "wss://workspace.cn-beijing.maas.aliyuncs.com.attacker.example/api-ws/v1/realtime",
        "wss://user@workspace.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        "wss://workspace.cn-beijing.maas.aliyuncs.com:bad/api-ws/v1/realtime",
    ],
)
def test_qwen35_never_sends_api_key_to_untrusted_endpoint(endpoint: str) -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "provider": "qwen3_5_omni",
                    "api_key": "dashscope-secret",
                    "endpoint": endpoint,
                }
            }
        }
    )

    assert config.available is False
    assert any("official WSS endpoint" in error for error in config.validation_errors())


def test_doubao_30_resolves_duplex_endpoint_model_and_api_key() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "provider": "doubao_3_0",
                    "api_key": "volcengine-secret",
                }
            }
        }
    )

    assert config.provider == "volcengine_duplex"
    assert config.endpoint == (
        "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
    )
    assert config.model == "1.2.6.1"
    assert config.credentials_configured is True
    assert config.available is True
    assert config.validation_errors() == []
    assert "volcengine-secret" not in repr(config.status_snapshot())
    assert "volcengine-secret" not in repr(config)


def test_enabled_realtime_config_fails_closed_without_credentials() -> None:
    config = resolve_realtime_voice_config(
        {"voice": {"realtime": {"enabled": True, "mode": "general_chat"}}}
    )

    assert config.available is False
    assert config.validation_errors() == [
        "voice.realtime requires app_id and access_token when enabled"
    ]


@pytest.mark.parametrize("mode", ["full", "unsafe", "realtime_only"])
def test_unknown_realtime_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="voice.realtime.mode"):
        resolve_realtime_voice_config(
            {"voice": {"realtime": {"enabled": True, "mode": mode}}}
        )


def test_unsafe_audio_shape_is_rejected() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                    "input_sample_rate": 48_000,
                    "output_format": "ogg_opus",
                }
            }
        }
    )

    assert config.available is False
    assert config.validation_errors() == [
        "voice.realtime.input_sample_rate must be 16000 for PCM input",
        "voice.realtime.output_format must be pcm_s16le for the local robot player",
    ]


def test_unimplemented_sc2_model_is_rejected_fail_closed() -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                    "model": "2.2.0.0",
                }
            }
        }
    )

    assert config.available is False
    assert config.validation_errors() == [
        "voice.realtime.model must be 1.2.1.1 (O2.0)"
    ]


@pytest.mark.parametrize("window_ms", [500, 50_000])
def test_official_end_smooth_window_boundaries_are_accepted(window_ms: int) -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                    "end_smooth_window_ms": window_ms,
                }
            }
        }
    )

    assert config.available is True
    assert config.validation_errors() == []
    assert config.status_snapshot()["end_smooth_window_ms"] == window_ms


@pytest.mark.parametrize("window_ms", [499, 50_001])
def test_end_smooth_window_outside_official_range_is_rejected(window_ms: int) -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                    "end_smooth_window_ms": window_ms,
                }
            }
        }
    )

    assert config.available is False
    assert config.validation_errors() == [
        "voice.realtime.end_smooth_window_ms must be between 500 and 50000"
    ]


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://openspeech.bytedance.com/api/v3/realtime/dialogue",
        "wss://attacker.example/api/v3/realtime/dialogue",
        "https://openspeech.bytedance.com/api/v3/realtime/dialogue",
    ],
)
def test_realtime_voice_config_never_sends_credentials_to_untrusted_endpoint(
    endpoint: str,
) -> None:
    config = resolve_realtime_voice_config(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                    "endpoint": endpoint,
                }
            }
        }
    )

    assert config.available is False
    assert any("official WSS endpoint" in error for error in config.validation_errors())


def test_realtime_readiness_is_skipped_when_disabled() -> None:
    payload = check_realtime_voice({}, deps={"websocket_client": False})

    assert payload["status"] == "skipped"
    assert payload["ok"] is True
    assert payload["enabled"] is False


def test_realtime_readiness_requires_transport_dependency() -> None:
    payload = check_realtime_voice(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "app_id": "app",
                    "access_token": "token",
                }
            }
        },
        deps={"websocket_client": False},
    )

    assert payload["status"] == "degraded"
    assert payload["ok"] is False
    assert payload["checks"]["websocket_client"] is False
    assert payload["errors"] == [
        "Realtime S2S dependency missing: websocket-client"
    ]


def test_realtime_readiness_reports_safe_general_chat_route() -> None:
    payload = check_realtime_voice(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "app_id": "app",
                    "access_token": "token",
                }
            }
        },
        deps={"websocket_client": True},
    )

    assert payload["status"] == "ok"
    assert payload["ok"] is True
    assert payload["fallback"] == "cascade"
    assert payload["robot_control_allowed"] is False
    assert "token" not in repr(payload)
