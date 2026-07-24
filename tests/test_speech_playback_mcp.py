from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.mcp.context import AppContext
from askme.ports import (
    PlaybackTarget,
    SpeechDelivery,
    SpeechPlaybackJob,
    SpeechPlaybackState,
    SpeechPlaybackTimestamps,
    SpeechPriority,
)


def _job(playback_id="spk_mcp", state=SpeechPlaybackState.QUEUED):
    return SpeechPlaybackJob(
        playback_id=playback_id,
        state=state,
        target=PlaybackTarget(robot_id="robot-1", device_id="speaker-1"),
        delivery=SpeechDelivery.PLAYBACK,
        priority=SpeechPriority.NORMAL,
        text_chars=5,
        request_hash="hash",
        idempotency_key="mcp-1",
        timestamps=SpeechPlaybackTimestamps(queued_at="now"),
    )


def _ctx(app):
    ctx = AsyncMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.info = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_mcp_voice_speak_submits_same_targeted_playback_contract() -> None:
    from askme.mcp.tools.voice_tools import voice_speak

    app = AppContext(
        config={
            "voice": {
                "playback": {
                    "mcp_enabled": True,
                    "mcp_operator_id": "askme.mcp",
                    "mcp_roles": ["operator"],
                }
            }
        }
    )
    app.speech_playback = AsyncMock()
    app.speech_playback.submit = AsyncMock(return_value=_job())

    result = json.loads(
        await voice_speak(
            "hello",
            _ctx(app),
            robot_id="robot-1",
            device_id="speaker-1",
            idempotency_key="mcp-1",
        )
    )

    assert result["playback_id"] == "spk_mcp"
    request = app.speech_playback.submit.await_args.args[0]
    assert request.text == "hello"
    assert request.target == PlaybackTarget(robot_id="robot-1", device_id="speaker-1")
    assert request.actor.operator_id == "askme.mcp"
    assert request.actor.surface == "mcp"


@pytest.mark.asyncio
async def test_mcp_playback_status_and_stop_use_shared_port() -> None:
    from askme.mcp.tools.voice_tools import voice_playback_status, voice_stop

    app = AppContext(config={"voice": {"playback": {"mcp_enabled": True}}})
    app.speech_playback = AsyncMock()
    app.speech_playback.status = AsyncMock(
        return_value=_job(state=SpeechPlaybackState.PLAYING)
    )
    app.speech_playback.cancel = AsyncMock(
        return_value=_job(state=SpeechPlaybackState.CANCELLED)
    )
    ctx = _ctx(app)

    status = json.loads(await voice_playback_status("spk_mcp", ctx))
    stopped = json.loads(await voice_stop("spk_mcp", ctx, reason="operator_cancelled"))

    assert status["state"] == "playing"
    assert stopped["state"] == "cancelled"
    app.speech_playback.cancel.assert_awaited_once()


def test_runtime_adapter_binds_shared_speech_playback_port() -> None:
    from askme.mcp.runtime_adapter import app_context_from_runtime_app

    playback = object()
    runtime = SimpleNamespace(
        modules={"voice": SimpleNamespace(speech_playback=playback)}
    )

    app = app_context_from_runtime_app(runtime)

    assert app.speech_playback is playback
