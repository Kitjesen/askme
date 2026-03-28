#!/usr/bin/env python3
"""Test TTS → Speaker on S100P."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def log(msg):
    print(msg, flush=True)

from askme.config import get_config
cfg = get_config()
tts_cfg = cfg.get("voice", {}).get("tts", {})
log(f"TTS backend={tts_cfg.get('backend')}")

from askme.voice.tts import TTSEngine
tts = TTSEngine(tts_cfg)

log("start_playback()...")
tts.start_playback()

log("speak('你好森哥，语音测试，一二三四五')...")
t0 = time.time()
tts.speak("你好森哥，语音测试，一二三四五。")

log("wait_done()...")
tts.wait_done()
log(f"Playback finished in {time.time()-t0:.1f}s")

tts.stop_playback()
log("Done! Did you hear it?")
