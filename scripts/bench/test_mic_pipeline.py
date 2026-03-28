#!/usr/bin/env python3
"""Test MicInput resampling pipeline: HKMIC 48kHz 2ch → HPF → AGC → 16kHz mono → ASR."""
import os
import sys
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.config import get_config
from askme.voice.mic_input import MicInput
from askme.voice.asr import ASREngine

cfg = get_config()
voice_cfg = cfg.get("voice", {})

print("=== MicInput Pipeline Test ===", flush=True)
print(f"  input_device: {voice_cfg.get('input_device')}", flush=True)
print(f"  mic_native_rate: {voice_cfg.get('mic_native_rate')}", flush=True)
print(f"  mic_channels: {voice_cfg.get('mic_channels')}", flush=True)
print(f"  mic_channel_select: {voice_cfg.get('mic_channel_select')}", flush=True)

mic = MicInput.from_config(cfg)
print(f"  MicInput: native={mic._native_rate}Hz → target={mic._sample_rate}Hz", flush=True)
print(f"  needs_resample: {mic._needs_resample}", flush=True)
print(f"  chunk_samples: {mic._chunk_samples} ({mic._chunk_samples/mic._sample_rate*1000:.0f}ms)", flush=True)

# Record 5 seconds
DURATION = 5
n_chunks = int(DURATION * 1000 / 100)  # 100ms chunks
print(f"\nRecording {DURATION}s ({n_chunks} chunks)... SPEAK NOW!", flush=True)

all_chunks = []
with mic.open():
    for i in range(n_chunks):
        chunk = mic.read_chunk()
        all_chunks.append(chunk)
        if i % 10 == 0:
            peak = int(np.max(np.abs(chunk)) * 32768)
            print(f"  chunk {i}: len={len(chunk)} peak={peak}", flush=True)

audio = np.concatenate(all_chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
print(f"\nRecorded: {len(audio)} samples ({len(audio)/16000:.1f}s) at 16kHz", flush=True)
print(f"  peak={peak} rms={rms}", flush=True)

# Save wav for inspection
wav_path = "/tmp/mic_pipeline_test.wav"
pcm16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm16.tobytes())
print(f"  Saved: {wav_path}", flush=True)

# Local ASR
print("\n=== Local ASR ===", flush=True)
asr_cfg = voice_cfg.get("asr", {})
engine = ASREngine(asr_cfg)
stream = engine.recognizer.create_stream()
for i in range(0, len(audio), 1600):
    stream.accept_waveform(16000, audio[i:i + 1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
text = engine.recognizer.get_result(stream).strip()
t_asr = time.perf_counter()
print(f"  Result: '{text}'", flush=True)

if text:
    print("\n✓ Pipeline WORKS: HKMIC → resample → ASR recognized speech!", flush=True)
else:
    if peak < 200:
        print("\n✗ No speech detected (peak too low — try speaking louder/closer)", flush=True)
    else:
        print("\n? Signal present but ASR returned empty — check audio quality", flush=True)
