#!/usr/bin/env python3
"""Debug TTS playback in full pipeline."""
import asyncio
import time
import sys
sys.path.insert(0, ".")

from askme.config import get_config
from askme.blueprints.voice import voice


async def main():
    cfg = get_config()
    app = await voice.build(cfg)
    await app.start()

    audio = app.modules["voice"].audio
    pipeline = app.modules["pipeline"].brain_pipeline
    tts = getattr(audio, "tts", None)

    print(f"Audio type: {type(audio).__name__}")
    print(f"TTS type: {type(tts).__name__ if tts else 'None'}")
    if tts:
        backend = getattr(tts, "_backend", "?")
        print(f"TTS backend: {backend}")
    print(f"pipeline._audio is voice.audio: {pipeline._audio is audio}")
    print(f"audio._voice_mode: {getattr(audio, '_voice_mode', '?')}")
    print(f"audio._woken_up: {getattr(audio, '_woken_up', '?')}")

    # Step 1: start playback
    audio.start_playback()
    is_playing = getattr(audio, "_is_playing", "?")
    print(f"\nPlayback started, is_playing={is_playing}")

    # Step 2: AudioAgent.speak (this is what pipeline calls)
    print("\n=== audio.speak() [AudioAgent] ===")
    audio.speak("通过AudioAgent说话，一二三。")
    print("audio.speak() called, waiting 6s...")
    time.sleep(6)

    # Step 3: direct TTSEngine.speak (this worked before)
    if tts:
        print("\n=== tts.speak() [TTSEngine directly] ===")
        tts.speak("直接TTSEngine说话，四五六。")
        print("tts.speak() called, waiting 6s...")
        time.sleep(6)

    # Step 4: check states
    if tts:
        worker = getattr(tts, "_worker_thread", None)
        print(f"\nTTS worker alive: {worker.is_alive() if worker else 'no thread'}")

    audio.stop_playback()
    await app.stop()
    print("\n=== done ===")


if __name__ == "__main__":
    asyncio.run(main())
