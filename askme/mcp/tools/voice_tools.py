"""MCP tools for voice I/O."""

from __future__ import annotations

import asyncio
import logging

from mcp.server.fastmcp import Context

from askme.errors import VOICE_NOT_AVAILABLE, error_response
from askme.mcp.context import AppContext
from askme.mcp.registration import mcp

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
async def voice_speak(text: str, ctx: Context) -> str:
    """Synthesise *text* to speech and play it through the speakers."""
    app = _get_app(ctx)
    if app.voice_io is None and app.tts_engine is None:
        return error_response(VOICE_NOT_AVAILABLE, "TTS engine not initialised")

    await ctx.info(f"Speaking: {text[:50]}...")

    async with _voice_lock:
        if app.voice_io is not None:
            await asyncio.to_thread(app.voice_io.speak_and_wait, text)
        else:
            app.tts_engine.speak(text)
            app.tts_engine.start_playback()
            await asyncio.to_thread(app.tts_engine.wait_done)
            app.tts_engine.stop_playback()

    return f"[Spoken] {text}"
