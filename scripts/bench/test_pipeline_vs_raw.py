#!/usr/bin/env python3
"""Compare MicInput pipeline output vs raw recording — diagnose quality loss.

Speak during recording. Saves both versions for comparison.
"""
import os
import sys
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sounddevice as sd
from scipy.signal import resample_poly
from askme.config import get_config
from askme.voice.mic_input import MicInput
from askme.voice.asr import ASREngine

cfg = get_config()
DURATION = 6

print("=== Pipeline vs Raw comparison ===", flush=True)
print(f"Recording {DURATION}s — SPEAK NOW after beep!", flush=True)
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)
print(">>> RECORDING <<<", flush=True)

# Method A: MicInput pipeline (what blueprint uses)
mic = MicInput.from_config(cfg)
n_chunks = int(DURATION * 1000 / 100)
pipeline_chunks = []
chunk_info = []
with mic.open():
    for i in range(n_chunks):
        chunk = mic.read_chunk()
        pipeline_chunks.append(chunk)
        chunk_info.append((len(chunk), int(np.max(np.abs(chunk)) * 32768)))

print("\nRecording done.", flush=True)

# Analyze pipeline chunks
print("\n=== Pipeline Chunk Analysis ===", flush=True)
lengths = [c[0] for c in chunk_info]
peaks = [c[1] for c in chunk_info]
print(f"  Chunks: {len(chunk_info)}", flush=True)
print(f"  Lengths: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.0f} std={np.std(lengths):.0f}", flush=True)
print(f"  Peaks: min={min(peaks)} max={max(peaks)}", flush=True)
print(f"  First 10 chunks: {chunk_info[:10]}", flush=True)
print(f"  Zero-peak chunks: {sum(1 for p in peaks if p < 50)}/{len(peaks)}", flush=True)

pipeline_audio = np.concatenate(pipeline_chunks)
p_peak = int(np.max(np.abs(pipeline_audio)) * 32768)
p_rms = int(np.sqrt(np.mean(pipeline_audio ** 2)) * 32768)
print(f"  Total: {len(pipeline_audio)} samples ({len(pipeline_audio)/16000:.1f}s) peak={p_peak} rms={p_rms}", flush=True)

# Save pipeline wav
pcm = (pipeline_audio * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/pipeline_output.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm.tobytes())

# Method B: Raw recording + scipy resample (what raw test does)
print("\n=== Now raw recording (same duration, speak again!) ===", flush=True)
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)
print(">>> RECORDING <<<", flush=True)

raw = sd.rec(int(DURATION * 48000), samplerate=48000, channels=2, dtype="float32", device=0)
sd.wait()
raw_ch0 = raw[:, 0]
raw_ch0 = raw_ch0 - np.mean(raw_ch0)  # DC removal
raw_16k = resample_poly(raw_ch0, up=1, down=3).astype(np.float32)
r_peak = int(np.max(np.abs(raw_16k)) * 32768)
r_rms = int(np.sqrt(np.mean(raw_16k ** 2)) * 32768)
print(f"  Raw: {len(raw_16k)} samples ({len(raw_16k)/16000:.1f}s) peak={r_peak} rms={r_rms}", flush=True)

# Save raw wav
pcm_r = (raw_16k * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/raw_output.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm_r.tobytes())

# ASR both
print("\n=== ASR Comparison ===", flush=True)
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))

for label, audio in [("Pipeline", pipeline_audio), ("Raw", raw_16k)]:
    stream = engine.recognizer.create_stream()
    for i in range(0, len(audio), 1600):
        stream.accept_waveform(16000, audio[i:i + 1600])
        while engine.recognizer.is_ready(stream):
            engine.recognizer.decode_stream(stream)
    text = engine.recognizer.get_result(stream).strip()
    print(f"  {label}: '{text}'", flush=True)

# Play both back
print("\n=== Playback ===", flush=True)
import subprocess
for label, path in [("Pipeline", "/tmp/pipeline_output.wav"), ("Raw", "/tmp/raw_output.wav")]:
    print(f"  Playing {label}...", flush=True)
    # Resample to 48k for aplay
    subprocess.run(["aplay", "-D", "plughw:1,0", path], capture_output=True, timeout=10)
    time.sleep(0.5)

print("\nDone.", flush=True)
