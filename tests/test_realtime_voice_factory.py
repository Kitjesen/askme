from __future__ import annotations

from askme.voice.realtime.factory import build_realtime_dialogue
from askme.voice.realtime.volcengine import VolcengineRealtimeDialogue


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
