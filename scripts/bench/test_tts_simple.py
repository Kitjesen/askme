#!/usr/bin/env python3
"""Minimal TTS test — no blueprint, no asyncio, just speak."""
import sys
import time
sys.path.insert(0, ".")

from askme.config import get_config
from askme.voice.tts import TTSEngine

cfg = get_config()
tts = TTSEngine(cfg.get("voice", {}).get("tts", {}))
print(f"Backend: {tts._backend}")

tts.start_playback()
print("Playback started")

tts.speak("测试一，直接说话。")
print("Speak 1 queued, waiting 5s...")
time.sleep(5)

tts.speak("测试二，第二句话。")
print("Speak 2 queued, waiting 5s...")
time.sleep(5)

tts.stop_playback()
print("Done")
