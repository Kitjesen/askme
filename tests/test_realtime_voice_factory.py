from __future__ import annotations

from askme.voice.realtime.factory import build_realtime_dialogue
from askme.voice.realtime.qwen import QwenRealtimeDialogue
from askme.voice.realtime.volcengine import VolcengineRealtimeDialogue
from askme.voice.realtime.volcengine_duplex import VolcengineDuplexDialogue


def test_factory_keeps_existing_cascade_when_realtime_is_disabled() -> None:
    assert build_realtime_dialogue({}) is None
    assert (
        build_realtime_dialogue(
            {"voice": {"realtime": {"enabled": True, "mode": "split"}}}
        )
        is None
    )


def test_factory_builds_provider_without_opening_network_connection() -> None:
    provider = build_realtime_dialogue(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "app_id": "app",
                    "access_token": "token",
                }
            }
        }
    )

    assert isinstance(provider, VolcengineRealtimeDialogue)
    assert provider.status_snapshot()["active"] is False


def test_factory_fails_closed_to_cascade_for_invalid_provider_config() -> None:
    provider = build_realtime_dialogue(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "app_id": "",
                    "access_token": "",
                }
            }
        }
    )

    assert provider is None


def test_factory_builds_qwen35_without_opening_network_connection() -> None:
    provider = build_realtime_dialogue(
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

    assert isinstance(provider, QwenRealtimeDialogue)
    snapshot = provider.status_snapshot()
    assert snapshot["provider"] == "qwen3_5_omni"
    assert snapshot["model"] == "qwen3.5-omni-flash-realtime"
    assert "dashscope-secret" not in repr(snapshot)


def test_factory_builds_doubao_30_without_opening_network_connection() -> None:
    provider = build_realtime_dialogue(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "shadow",
                    "provider": "volcengine_duplex",
                    "api_key": "volcengine-secret",
                }
            }
        }
    )

    assert isinstance(provider, VolcengineDuplexDialogue)
    snapshot = provider.status_snapshot()
    assert snapshot["provider"] == "volcengine_duplex"
    assert snapshot["model"] == "1.2.6.1"
    assert "volcengine-secret" not in repr(snapshot)
