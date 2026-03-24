"""Full voice loop test without microphone.

Uses pre-recorded WAV files to simulate ASR input, then runs the
complete pipeline: ASR → IntentRouter → BrainPipeline → TTS playback.
"""

import asyncio
import io
import sys
import time
import wave

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Injected text queries — simulates what a user would say
INJECTED_QUERIES = [
    "Thunder 报告状态",
    "三号区域发现异常，怎么处理",
    "现在几点了",
    "退出",  # should trigger quit
]


def asr_on_wavfile(wav_path: str, asr) -> str:
    """Run ASR on a WAV file and return the recognised text."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to ASR model sample rate if needed
    if sr != asr.sample_rate:
        ratio = asr.sample_rate / sr
        new_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    stream = asr.create_stream()
    chunk = int(0.1 * asr.sample_rate)
    for i in range(0, len(samples), chunk):
        stream.accept_waveform(asr.sample_rate, samples[i: i + chunk])
        while asr.is_ready(stream):
            asr.decode_stream(stream)
    # Flush
    tail = np.zeros(int(0.5 * asr.sample_rate), dtype=np.float32)
    stream.accept_waveform(asr.sample_rate, tail)
    while asr.is_ready(stream):
        asr.decode_stream(stream)

    return asr.get_result(stream).strip()


async def run_voice_loop_test():
    from askme.blueprints.voice import voice as voice_blueprint
    from askme.brain.intent_router import IntentType
    from askme.voice.asr import ASREngine
    from askme.config import get_config

    cfg = get_config()

    print("=" * 55)
    print("Full Voice Loop Test (no microphone)")
    print("=" * 55)

    # Init full app (loads SOUL.md, TTS, LLM, etc.)
    app = await voice_blueprint.build(cfg)

    # Init ASR for wav file test
    asr_cfg = cfg.get("voice", {}).get("asr", {})
    asr = ASREngine(asr_cfg)

    # ── Phase 1: ASR on bundled WAV files ────────────────────
    import os
    wav_dir = os.path.join(
        asr_cfg.get("model_dir",
                    "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"),
        "test_wavs",
    )
    wav_files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))[:2]

    print(f"\n[Phase 1] ASR on {len(wav_files)} test wav files")
    print("─" * 50)
    for wav_name in wav_files:
        text = asr_on_wavfile(os.path.join(wav_dir, wav_name), asr)
        print(f"  {wav_name} → \"{text}\"")

    # ── Phase 2: Injected text → full pipeline ─────────────
    print(f"\n[Phase 2] Injected queries → Router → LLM → TTS")
    print("─" * 50)

    for query in INJECTED_QUERIES:
        print(f"\n  User: {query}")

        voice_mod = app.modules.get("voice")
        router = getattr(voice_mod, "router", None)
        audio = getattr(voice_mod, "audio", None)
        pipeline_mod = app.modules.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None)

        intent = router.route(query)
        print(f"  Intent: {intent.type.name}")

        if intent.type == IntentType.COMMAND and intent.command in ("quit", "exit", "/quit", "/exit"):
            audio.speak("收到，结束巡检。")
            audio.start_playback()
            audio.wait_speaking_done()
            audio.stop_playback()
            print("  Thunder: 收到，结束巡检。")
            break

        if intent.type == IntentType.ESTOP:
            pipeline.handle_estop()
            print("  [ESTOP triggered]")
            continue

        t0 = time.monotonic()
        response = await pipeline.process(query)
        elapsed = time.monotonic() - t0

        print(f"  Thunder: {response[:80]}")
        print(f"  ({elapsed:.1f}s total, waiting for TTS...)")

        # Wait for TTS to finish
        audio.wait_speaking_done()
        audio.stop_playback()
        print(f"  TTS done.")
        await asyncio.sleep(0.5)

    await app.stop()
    print(f"\n{'=' * 55}")
    print("Voice loop test complete!")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(run_voice_loop_test())
