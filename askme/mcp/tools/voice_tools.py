"""MCP tools for voice I/O."""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server.fastmcp import Context

from askme.errors import VOICE_NOT_AVAILABLE, error_response
from askme.mcp.context import AppContext
from askme.mcp.registration import mcp
from askme.ports import (
    PlaybackTarget,
    SpeechActor,
    SpeechDelivery,
    SpeechPlaybackError,
    SpeechPlaybackRequest,
    SpeechPriority,
)

logger = logging.getLogger(__name__)

_voice_lock = asyncio.Lock()


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def voice_listen(ctx: Context) -> str:
    """Record one utterance and transcribe it to text."""
    app = _get_app(ctx)
    if app.voice_io is None:
        return error_response(VOICE_NOT_AVAILABLE, "Voice I/O not initialised")

    await ctx.info("Listening for speech...")

    async with _voice_lock:
        text = await asyncio.to_thread(app.voice_io.listen_once)

    if text:
        await ctx.info(f"Transcribed: {text}")
        return text
    return "[No speech detected within timeout]"



@mcp.tool()
async def voice_speak(
    text: str,
    ctx: Context,
    robot_id: str = "",
    device_id: str = "",
    site_id: str = "",
    idempotency_key: str = "",
    priority: str = "normal",
    queue_policy: str = "enqueue",
    voice_profile_id: str = "",
    speed: float | None = None,
    pitch: float | None = None,
    volume: float | None = None,
    ttl_s: float = 60.0,
) -> str:
    """Queue literal text for one explicit robot/device; never invokes an LLM."""
    app = _get_app(ctx)
    if app.speech_playback is not None:
        denied = _mcp_playback_denial(app)
        if denied is not None:
            return denied
        if not robot_id or not device_id:
            return error_response(
                "target_required",
                "robot_id and device_id are required for product speech playback.",
            )
        if not idempotency_key:
            return error_response(
                "idempotency_key_required",
                "idempotency_key is required so MCP retries cannot duplicate speech.",
            )
        try:
            request = SpeechPlaybackRequest(
                text=text,
                target=PlaybackTarget(
                    robot_id=robot_id,
                    device_id=device_id,
                    site_id=site_id,
                ),
                actor=_mcp_actor(app),
                idempotency_key=idempotency_key,
                delivery=SpeechDelivery.PLAYBACK,
                priority=SpeechPriority(priority),
                queue_policy=queue_policy,
                voice_profile_id=voice_profile_id,
                speed=speed,
                pitch=pitch,
                volume=volume,
                ttl_s=ttl_s,
            )
            job = await app.speech_playback.submit(request)
            await ctx.info(f"Speech playback queued: {job.playback_id}")
            return json.dumps(job.to_payload(), ensure_ascii=False)
        except SpeechPlaybackError as exc:
            return error_response(exc.code, str(exc), {"status_code": exc.status_code})
        except ValueError as exc:
            return error_response("invalid_playback_request", str(exc))

    # Backward-compatible lab fallback. Product runtimes always bind the job port.
    if app.voice_io is None and app.tts_engine is None:
        return error_response(VOICE_NOT_AVAILABLE, "TTS engine not initialised")
    await ctx.info(f"Speaking legacy text ({len(text)} chars)")
    async with _voice_lock:
        if app.voice_io is not None:
            await asyncio.to_thread(app.voice_io.speak_and_wait, text)
        else:
            app.tts_engine.speak(text)
            app.tts_engine.start_playback()
            await asyncio.to_thread(app.tts_engine.wait_done)
            app.tts_engine.stop_playback()
    return f"[Spoken] {text}"


@mcp.tool()
async def voice_playback_status(playback_id: str, ctx: Context) -> str:
    """Return the lifecycle state for one speech playback job."""
    app = _get_app(ctx)
    if app.speech_playback is None:
        return error_response(VOICE_NOT_AVAILABLE, "Speech playback service not initialised")
    denied = _mcp_playback_denial(app)
    if denied is not None:
        return denied
    try:
        return json.dumps(
            (await app.speech_playback.status(playback_id)).to_payload(),
            ensure_ascii=False,
        )
    except SpeechPlaybackError as exc:
        return error_response(exc.code, str(exc), {"status_code": exc.status_code})


@mcp.tool()
async def voice_stop(
    playback_id: str,
    ctx: Context,
    reason: str = "operator_cancelled",
) -> str:
    """Cancel one queued or active playback by playback_id."""
    app = _get_app(ctx)
    if app.speech_playback is None:
        return error_response(VOICE_NOT_AVAILABLE, "Speech playback service not initialised")
    denied = _mcp_playback_denial(app)
    if denied is not None:
        return denied
    try:
        job = await app.speech_playback.cancel(
            playback_id,
            reason=reason,
            actor=_mcp_actor(app),
        )
        return json.dumps(job.to_payload(), ensure_ascii=False)
    except SpeechPlaybackError as exc:
        return error_response(exc.code, str(exc), {"status_code": exc.status_code})


@mcp.tool()
async def voice_synthesize(
    text: str,
    ctx: Context,
    robot_id: str = "",
    device_id: str = "",
    site_id: str = "",
    idempotency_key: str = "",
) -> str:
    """Synthesize a WAV artifact without playing it."""
    app = _get_app(ctx)
    if app.speech_playback is None:
        return error_response(VOICE_NOT_AVAILABLE, "Speech playback service not initialised")
    denied = _mcp_playback_denial(app)
    if denied is not None:
        return denied
    if not robot_id or not device_id or not idempotency_key:
        return error_response(
            "target_or_idempotency_missing",
            "robot_id, device_id, and idempotency_key are required.",
        )
    try:
        job = await app.speech_playback.submit(
            SpeechPlaybackRequest(
                text=text,
                target=PlaybackTarget(
                    robot_id=robot_id,
                    device_id=device_id,
                    site_id=site_id,
                ),
                actor=_mcp_actor(app),
                idempotency_key=idempotency_key,
                delivery=SpeechDelivery.SYNTHESIZE_ONLY,
            )
        )
        return json.dumps(job.to_payload(), ensure_ascii=False)
    except SpeechPlaybackError as exc:
        return error_response(exc.code, str(exc), {"status_code": exc.status_code})


def _playback_config(app: AppContext) -> dict:
    voice_cfg = app.config.get("voice") if isinstance(app.config.get("voice"), dict) else {}
    value = voice_cfg.get("playback") if isinstance(voice_cfg.get("playback"), dict) else {}
    return value


def _mcp_playback_denial(app: AppContext) -> str | None:
    if not bool(_playback_config(app).get("mcp_enabled", False)):
        return error_response(
            "mcp_voice_playback_disabled",
            "MCP voice playback is disabled by local robot policy.",
        )
    return None


def _mcp_actor(app: AppContext) -> SpeechActor:
    cfg = _playback_config(app)
    roles = cfg.get("mcp_roles") if isinstance(cfg.get("mcp_roles"), list) else ["operator"]
    return SpeechActor(
        operator_id=str(cfg.get("mcp_operator_id") or "askme.mcp"),
        roles=frozenset(str(role) for role in roles),
        surface="mcp",
    )
