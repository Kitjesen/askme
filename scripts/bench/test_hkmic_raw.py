#!/usr/bin/env python3
"""Raw HKMIC recording test — no pipeline, just sounddevice + save wav.

Run on S100P, speak into HKMIC during countdown.
Saves raw 48kHz and resampled 16kHz wavs for inspection.
"""
import time
import wave
import numpy as np
import sounddevice as sd

DEVICE = 0  # HKMIC
DURATION = 8

# List devices first
print("=== Audio Input Devices ===", flush=True)
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']} in={d['max_input_channels']}ch rate={d['default_samplerate']}", flush=True)

# Countdown
print(f"\nWill record {DURATION}s from device {DEVICE}", flush=True)
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)
print(">>> RECORDING — SPEAK NOW! <<<", flush=True)

# Record raw at native 48kHz 2ch
rec = sd.rec(int(DURATION * 48000), samplerate=48000, channels=2, dtype="float32", device=DEVICE)
sd.wait()
print("Recording done.", flush=True)

# Analyze each channel
print("\n=== Channel Analysis ===", flush=True)
for ch in range(2):
    data = rec[:, ch]
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    dc = np.mean(data)
    # Find segments with signal above noise floor
    frame_len = 4800  # 100ms frames
    loud_frames = 0
    for i in range(0, len(data) - frame_len, frame_len):
        frame_rms = np.sqrt(np.mean(data[i:i+frame_len] ** 2))
        if frame_rms > 0.005:  # above noise floor
            loud_frames += 1
    total_frames = len(data) // frame_len
    print(f"  ch{ch}: peak={peak:.4f}({int(peak*32768)}) rms={rms:.4f}({int(rms*32768)}) dc={dc:.6f}", flush=True)
    print(f"        loud_frames={loud_frames}/{total_frames} ({100*loud_frames/max(total_frames,1):.0f}%)", flush=True)

# Save raw 48kHz ch0
best_ch = 0
raw = rec[:, best_ch]
raw = raw - np.mean(raw)  # DC removal
pcm48 = (raw * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/hkmic_raw_48k.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(48000)
    wf.writeframes(pcm48.tobytes())
print(f"\nSaved: /tmp/hkmic_raw_48k.wav ({DURATION}s, 48kHz)", flush=True)

# Resample to 16kHz
from scipy.signal import resample_poly
audio_16k = resample_poly(raw, up=1, down=3).astype(np.float32)
pcm16 = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/hkmic_raw_16k.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm16.tobytes())
print(f"Saved: /tmp/hkmic_raw_16k.wav ({len(audio_16k)/16000:.1f}s, 16kHz)", flush=True)

# Quick ASR test on the resampled audio
print("\n=== Quick ASR (sherpa-onnx) ===", flush=True)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from askme.config import get_config
from askme.voice.asr import ASREngine
cfg = get_config()
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(audio_16k), 1600):
    stream.accept_waveform(16000, audio_16k[i:i+1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
text = engine.recognizer.get_result(stream).strip()
print(f"  ASR: '{text}'", flush=True)

# Playback test — play back the recording through speaker
print("\n=== Playback (play recording back) ===", flush=True)
import subprocess
r = subprocess.run(["aplay", "-D", "plughw:1,0", "/tmp/hkmic_raw_48k.wav"],
                   capture_output=True, text=True, timeout=15)
if r.returncode == 0:
    print("  Playback OK — did you hear your voice?", flush=True)
else:
    print(f"  Playback failed: {r.stderr.strip()[:80]}", flush=True)

print("\nDone.", flush=True)
