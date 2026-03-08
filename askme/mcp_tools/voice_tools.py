"""MCP tools for voice I/O (ASR + TTS)."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from mcp.server.fastmcp import Context

from askme.errors import VOICE_NOT_AVAILABLE, INTERNAL_ERROR, error_response
from askme.mcp_server import AppContext, mcp

logger = logging.getLogger(__name__)

# Serialise voice operations — only one listen/speak at a time.
_voice_lock = asyncio.Lock()


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def voice_listen(ctx: Context) -> str:
    """Record audio from the microphone, detect speech via VAD, and
    transcribe it to text using streaming ASR.

    Returns the transcribed text, or an error message.
    """
    app = _get_app(ctx)
    if app.asr_engine is None or app.vad_engine is None:
        return error_response(VOICE_NOT_AVAILABLE, "ASR/VAD engines not initialised")

    await ctx.info("Listening for speech...")

    async with _voice_lock:
        text = await asyncio.to_thread(_listen_sync, app)

    if text:
        await ctx.info(f"Transcribed: {text}")
        return text
    return "[No speech detected within timeout]"


def _listen_sync(app: AppContext) -> str | None:
    """Blocking microphone listen — runs in a worker thread."""
    import sounddevice as sd

    asr = app.asr_engine
    vad = app.vad_engine
    sample_rate: int = asr.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 100 ms chunks
    speech_active = False

    stream = asr.create_stream()

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        for _ in range(300):  # 300 × 100 ms = 30 s timeout
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)

            samples_int16 = (samples * 32768).astype(np.int16)
            vad.accept_waveform(samples_int16)

            if vad.is_speech_detected():
                speech_active = True
                stream.accept_waveform(sample_rate, samples)
                while asr.is_ready(stream):
                    asr.decode_stream(stream)
            elif speech_active:
                speech_active = False
                stream.accept_waveform(sample_rate, samples)
                while asr.is_ready(stream):
                    asr.decode_stream(stream)

            if asr.is_endpoint(stream):
                text = asr.get_result(stream).strip()
                if text:
                    return text

    return None


@mcp.tool()
async def voice_speak(text: str, ctx: Context) -> str:
    """Synthesise *text* to speech via TTS and play it through the speakers.

    Args:
        text: The text to speak aloud.
    """
    app = _get_app(ctx)
    if app.tts_engine is None:
        return error_response(VOICE_NOT_AVAILABLE, "TTS engine not initialised")

    await ctx.info(f"Speaking: {text[:50]}...")

    async with _voice_lock:
        app.tts_engine.speak(text)
        app.tts_engine.start_playback()
        await asyncio.to_thread(app.tts_engine.wait_done)
        app.tts_engine.stop_playback()

    return f"[Spoken] {text}"
