#!/usr/bin/env python3
"""Loopback test: play speech wav → HKMIC records → MicInput pipeline → ASR.

Tests the full audio capture pipeline without a human speaker.
"""
import os
import sys
import subprocess
import threading
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.config import get_config
from askme.voice.mic_input import MicInput
from askme.voice.asr import ASREngine

cfg = get_config()

# --- Step 1: Record while playing speech ---
print("=== Loopback: play speech + record from HKMIC ===", flush=True)

mic = MicInput.from_config(cfg)
print(f"  Pipeline: {mic._native_rate}Hz {mic._native_channels}ch → {mic._sample_rate}Hz mono", flush=True)

chunks = []
recording_done = threading.Event()

def record_thread():
    with mic.open():
        # Record ~8 seconds (80 x 100ms chunks)
        for _ in range(80):
            chunks.append(mic.read_chunk())
    recording_done.set()

# Start recording, then play speech after a short delay
t = threading.Thread(target=record_thread)
t.start()
time.sleep(0.5)  # let mic settle

print("  Playing /tmp/test_speech_48k.wav on plughw:1,0...", flush=True)
r = subprocess.run(
    ["aplay", "-D", "plughw:1,0", "/tmp/test_speech_48k.wav"],
    capture_output=True, text=True, timeout=15,
)
if r.returncode != 0:
    print(f"  aplay failed: {r.stderr.strip()[:100]}", flush=True)
else:
    print("  Playback done.", flush=True)

t.join(timeout=12)

audio = np.concatenate(chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
dur = len(audio) / 16000
print(f"  Recorded: {len(audio)} samples ({dur:.1f}s)", flush=True)
print(f"  peak={peak} rms={rms}", flush=True)

# Save for inspection
pcm16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/loopback_result.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm16.tobytes())
print("  Saved: /tmp/loopback_result.wav", flush=True)

# --- Step 2: Local ASR ---
print("\n=== Local ASR (sherpa-onnx) ===", flush=True)
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(audio), 1600):
    stream.accept_waveform(16000, audio[i:i + 1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
local_text = engine.recognizer.get_result(stream).strip()
print(f"  Local: '{local_text}'", flush=True)

# --- Step 3: Also test raw 48kHz → direct resample (control) ---
print("\n=== Control: raw sd.rec 48kHz → scipy resample ===", flush=True)
import sounddevice as sd
from scipy.signal import resample_poly

print("  Playing again + recording raw 48kHz...", flush=True)
raw_chunks = []
raw_done = threading.Event()

def raw_record():
    rec = sd.rec(int(8 * 48000), samplerate=48000, channels=2, dtype="float32", device=0)
    sd.wait()
    raw_chunks.append(rec[:, 0])  # ch0
    raw_done.set()

t2 = threading.Thread(target=raw_record)
t2.start()
time.sleep(0.5)
subprocess.run(["aplay", "-D", "plughw:1,0", "/tmp/test_speech_48k.wav"],
               capture_output=True, timeout=15)
t2.join(timeout=12)

raw_audio = raw_chunks[0]
raw_16k = resample_poly(raw_audio, up=1, down=3).astype(np.float32)
raw_peak = int(np.max(np.abs(raw_16k)) * 32768)
print(f"  Raw peak={raw_peak}", flush=True)

stream2 = engine.recognizer.create_stream()
for i in range(0, len(raw_16k), 1600):
    stream2.accept_waveform(16000, raw_16k[i:i + 1600])
    while engine.recognizer.is_ready(stream2):
        engine.recognizer.decode_stream(stream2)
raw_text = engine.recognizer.get_result(stream2).strip()
print(f"  Raw ASR: '{raw_text}'", flush=True)

# --- Summary ---
print("\n=== Summary ===", flush=True)
print(f"  MicInput pipeline: peak={peak} ASR='{local_text}'", flush=True)
print(f"  Raw control:       peak={raw_peak} ASR='{raw_text}'", flush=True)
if local_text:
    print("  ✓ MicInput pipeline works!", flush=True)
elif raw_text:
    print("  △ Raw works but pipeline doesn't — check HPF/AGC/resample", flush=True)
else:
    print("  ✗ Neither worked — mic can't pick up speaker (distance/volume?)", flush=True)
