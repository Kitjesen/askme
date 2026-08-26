"""Privacy contract for transcript-bearing voice health snapshots."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from askme.runtime.module import ModuleRegistry

from askme.voice.diagnostics.status_privacy import sanitize_voice_status


def test_voice_status_hides_transcript_text_by_default() -> None:
    status = {
        "asr": {
            "cloud": {
                "partial_text": "请带我去服务中心",
                "final_text": "请带我去服务中心。",
                "partial_age_ms": 125.0,
                "final_age_ms": 25.0,
            }
        }
    }

    public_status = sanitize_voice_status(status)

    cloud = public_status["asr"]["cloud"]
    assert "partial_text" not in cloud
    assert "final_text" not in cloud
    assert cloud["partial_text_present"] is True
    assert cloud["partial_text_chars"] == 8
    assert cloud["final_text_present"] is True
    assert cloud["final_text_chars"] == 9
    assert cloud["partial_age_ms"] == 125.0
    assert cloud["final_age_ms"] == 25.0


def test_voice_status_shows_transcripts_only_when_debug_is_explicit() -> None:
    status = {
        "asr": {
            "cloud": {
                "partial_text": "调试中的半句",
                "final_text": "调试中的完整句。",
            }
        }
    }

    debug_status = sanitize_voice_status(status, include_transcripts=True)

    cloud = debug_status["asr"]["cloud"]
    assert cloud["partial_text"] == "调试中的半句"
    assert cloud["final_text"] == "调试中的完整句。"
    assert cloud["partial_text_present"] is True
    assert cloud["partial_text_chars"] == 6
    assert cloud["final_text_present"] is True
    assert cloud["final_text_chars"] == 8


def test_voice_status_redacts_transcripts_nested_in_lists_without_mutation() -> None:
    status = {
        "providers": [
            {
                "name": "cloud",
                "final_text": "列表里的识别结果",
                "final_age_ms": 12.0,
            }
        ]
    }

    public_status = sanitize_voice_status(status)

    provider = public_status["providers"][0]
    assert "final_text" not in provider
    assert provider["final_text_present"] is True
    assert provider["final_text_chars"] == 8
    assert provider["final_age_ms"] == 12.0
    assert status["providers"][0]["final_text"] == "列表里的识别结果"


def test_voice_status_redacts_generic_turn_text_contracts_by_default() -> None:
    status = {
        "interaction": {
            "last_input_contract": {
                "transcript": "带我去仓库",
                "confidence": 0.92,
            },
            "last_action_contract": {
                "user_text": "带我去仓库",
                "reply": "需要先确认。",
            },
        },
        "trace": {
            "metadata": {
                "raw_text": "原始识别文本",
                "content": "模型输出内容",
            }
        },
    }

    public_status = sanitize_voice_status(status)

    assert "带我去仓库" not in repr(public_status)
    assert "需要先确认。" not in repr(public_status)
    assert "原始识别文本" not in repr(public_status)
    assert "模型输出内容" not in repr(public_status)
    assert public_status["interaction"]["last_input_contract"]["transcript_chars"] == 5
    assert public_status["interaction"]["last_action_contract"]["user_text_chars"] == 5
    assert public_status["interaction"]["last_action_contract"]["reply_present"] is True
    assert public_status["trace"]["metadata"]["raw_text_present"] is True
    assert public_status["trace"]["metadata"]["content_chars"] == 6


def test_public_health_snapshot_redacts_all_voice_transcript_copies() -> None:
    from askme.runtime.modules.health_module import HealthModule

    voice_status = {
        "mode": "voice",
        "enabled": True,
        "input_ready": True,
        "output_ready": True,
        "pipeline_ok": True,
        "asr": {
            "cloud": {
                "partial_text": "公共探针不应返回这句",
                "final_text": "公共探针也不应返回这句。",
                "partial_age_ms": 70.0,
                "final_age_ms": 15.0,
            }
        },
    }

    class VoiceModule:
        name = "voice"

        def __init__(self) -> None:
            self.audio = MagicMock()
            self.audio.status_snapshot.return_value = voice_status

        def health(self) -> dict[str, object]:
            return {"status": "ok", "audio": voice_status}

    registry = ModuleRegistry()
    registry.register(VoiceModule())  # type: ignore[arg-type]
    mock_server = MagicMock(enabled=True, port=8080)

    with patch(
        "askme.health_server.AskmeHealthServer",
        return_value=mock_server,
    ) as server_cls:
        HealthModule().build({}, registry)

    snapshot = server_cls.call_args.kwargs["snapshot_provider"]()

    assert "公共探针不应返回这句" not in repr(snapshot)
    assert "公共探针也不应返回这句。" not in repr(snapshot)
    pipeline_cloud = snapshot["voice_pipeline_status"]["asr"]["cloud"]
    component_cloud = snapshot["components"]["voice"]["audio"]["asr"]["cloud"]
    assert pipeline_cloud["partial_text_present"] is True
    assert pipeline_cloud["partial_text_chars"] == 10
    assert pipeline_cloud["final_text_present"] is True
    assert pipeline_cloud["final_text_chars"] == 12
    assert component_cloud == pipeline_cloud


@pytest.mark.asyncio
async def test_public_readiness_details_redact_asr_transcripts() -> None:
    from askme.runtime.modules.health_module import HealthModule

    voice_status = {
        "input_ready": True,
        "asr": {
            "cloud": {
                "partial_text": "就绪探针半句",
                "final_text": "就绪探针完整句",
                "partial_age_ms": 40.0,
                "final_age_ms": 10.0,
            }
        },
    }

    class VoiceModule:
        name = "voice"
        asr_provider = object()

        def health(self) -> dict[str, object]:
            return {"status": "ok", "audio": voice_status}

    registry = ModuleRegistry()
    registry.register(VoiceModule())  # type: ignore[arg-type]
    mock_server = MagicMock(enabled=True, port=8080)
    module = HealthModule()

    with patch(
        "askme.health_server.AskmeHealthServer",
        return_value=mock_server,
    ):
        module.build({}, registry)

    readiness = await module.health_service.check_all()

    details = readiness["components"]["asr"]["details"]
    assert "就绪探针半句" not in repr(details)
    assert "就绪探针完整句" not in repr(details)
    assert details["cloud"]["partial_text_present"] is True
    assert details["cloud"]["partial_text_chars"] == 6
    assert details["cloud"]["final_text_present"] is True
    assert details["cloud"]["final_text_chars"] == 7


@pytest.mark.parametrize(
    ("debug_setting", "expects_transcript"),
    [
        (True, True),
        ("true", False),
    ],
)
def test_public_health_requires_boolean_debug_opt_in_for_transcripts(
    debug_setting: object,
    expects_transcript: bool,
) -> None:
    from askme.runtime.modules.health_module import HealthModule

    transcript = "仅供显式调试查看"
    voice_status = {
        "pipeline_ok": True,
        "asr": {"cloud": {"final_text": transcript, "final_age_ms": 5.0}},
    }

    class VoiceModule:
        name = "voice"

        def __init__(self) -> None:
            self.audio = MagicMock()
            self.audio.status_snapshot.return_value = voice_status

        def health(self) -> dict[str, object]:
            return {"status": "ok", "audio": voice_status}

    registry = ModuleRegistry()
    registry.register(VoiceModule())  # type: ignore[arg-type]
    mock_server = MagicMock(enabled=True, port=8080)

    with patch(
        "askme.health_server.AskmeHealthServer",
        return_value=mock_server,
    ) as server_cls:
        HealthModule().build(
            {
                "health_server": {
                    "debug_include_voice_transcripts": debug_setting,
                }
            },
            registry,
        )

    snapshot = server_cls.call_args.kwargs["snapshot_provider"]()
    cloud = snapshot["voice_pipeline_status"]["asr"]["cloud"]

    assert (cloud.get("final_text") == transcript) is expects_transcript
    assert cloud["final_text_present"] is True
    assert cloud["final_text_chars"] == 8
