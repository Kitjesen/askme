#!/usr/bin/env python3
"""Live ASR test: compare native 48kHz→resample vs forced 16kHz."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sounddevice as sd
from scipy.signal import resample_poly

DEVICE = 1  # HK-MIC
DURATION = 5
TARGET_RATE = 16000

from askme.config import get_config
cfg = get_config()
asr_cfg = cfg.get("voice", {}).get("asr", {})
from askme.voice.asr import ASREngine
engine = ASREngine(asr_cfg)

def run_asr(audio_16k):
    stream = engine.recognizer.create_stream()
    chunk_size = TARGET_RATE // 10
    for i in range(0, len(audio_16k), chunk_size):
        chunk = audio_16k[i:i + chunk_size]
        stream.accept_waveform(TARGET_RATE, chunk)
        while engine.recognizer.is_ready(stream):
            engine.recognizer.decode_stream(stream)
    return engine.recognizer.get_result(stream).strip()

# Method 1: Record at native 48kHz, high-quality resample to 16kHz
print("=== Method 1: Native 48kHz → polyphase resample → 16kHz ===", flush=True)
print(f"Recording {DURATION}s... SPEAK NOW!", flush=True)
rec48 = sd.rec(int(DURATION * 48000), samplerate=48000, channels=1, dtype="float32", device=DEVICE)
sd.wait()
peak = int(np.max(np.abs(rec48)) * 32768)
print(f"Done. peak={peak}", flush=True)

# Polyphase resample: 48000→16000 = factor 1/3
audio_16k = resample_poly(rec48.flatten(), up=1, down=3).astype(np.float32)
t0 = time.perf_counter()
text1 = run_asr(audio_16k)
ms1 = (time.perf_counter() - t0) * 1000
print(f"Result: '{text1}' ({ms1:.0f}ms)", flush=True)

# Method 2: Record directly at 16kHz (ALSA auto-resample)
print(f"\n=== Method 2: Force 16kHz (ALSA resample) ===", flush=True)
print(f"Recording {DURATION}s... SPEAK NOW!", flush=True)
try:
    rec16 = sd.rec(int(DURATION * 16000), samplerate=16000, channels=1, dtype="float32", device=DEVICE)
    sd.wait()
    peak2 = int(np.max(np.abs(rec16)) * 32768)
    print(f"Done. peak={peak2}", flush=True)
    t0 = time.perf_counter()
    text2 = run_asr(rec16.flatten())
    ms2 = (time.perf_counter() - t0) * 1000
    print(f"Result: '{text2}' ({ms2:.0f}ms)", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)
    text2 = "N/A"

print(f"\n=== Comparison ===", flush=True)
print(f"  48kHz→resample: '{text1}'", flush=True)
print(f"  Force 16kHz:    '{text2}'", flush=True)
