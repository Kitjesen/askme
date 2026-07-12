#!/usr/bin/env python3
"""Loopback test: play speech wav → HKMIC records → MicInput pipeline → ASR.

Tests the full audio capture pipeline without a human speaker.
"""
import os
import subprocess
import sys
import threading
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.voice.asr import ASREngine
from askme.voice.mic_input import MicInput
from askme.voice.tts import TTSEngine

from askme.config import get_config

cfg = get_config()
DEFAULT_TEST_TEXT = "现在开始测试，看看麦克风能不能听到。"


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"unsupported sample width: {sample_width}")
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), sample_rate


def play_fixture(path: Path) -> bool:
    if not path.exists():
        print(f"  {path} not found; synthesizing test speech via TTSEngine.", flush=True)
        tts = TTSEngine(cfg.get("voice", {}).get("tts", {}))
        try:
            tts.speak(DEFAULT_TEST_TEXT)
            tts.start_playback()
            tts.wait_done(timeout=30)
            tts.stop_playback()
            print("  TTS playback done.", flush=True)
            return True
        finally:
            tts.shutdown()

    audio, sample_rate = read_wav_mono(path)
    tts = TTSEngine(cfg.get("voice", {}).get("tts", {}))
    try:
        if tts.play_feedback_audio(audio, sample_rate):
            print("  Playback done via TTSEngine USB direct.", flush=True)
            return True
    finally:
        tts.shutdown()

    print("  USB direct inactive; falling back to aplay...", flush=True)
    r = subprocess.run(
        ["aplay", "-D", "plughw:1,0", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        print(f"  aplay failed: {r.stderr.strip()[:100]}", flush=True)
        return False
    print("  Playback done via aplay.", flush=True)
    return True

# --- Step 1: Record while playing speech ---
print("=== Loopback: play speech + record from HKMIC ===", flush=True)

mic = MicInput.from_config(cfg)
asr_rate = mic.sample_rate
asr_step = max(1, int(asr_rate * 0.1))
print(f"  Pipeline: {mic._native_rate}Hz {mic._native_channels}ch → {asr_rate}Hz mono", flush=True)

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

fixture = Path("/tmp/test_speech_48k.wav")
print(f"  Playing {fixture} through product output path...", flush=True)
playback_ok = play_fixture(fixture)

t.join(timeout=12)

if not playback_ok:
    print("  Playback failed; loopback result would be misleading.", flush=True)
    sys.exit(3)

if not chunks:
    print("  No MicInput chunks captured; cannot run loopback ASR.", flush=True)
    sys.exit(2)

audio = np.concatenate(chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
dur = len(audio) / asr_rate
print(f"  Recorded: {len(audio)} samples ({dur:.1f}s)", flush=True)
print(f"  peak={peak} rms={rms}", flush=True)

# Save for inspection
pcm16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
with wave.open("/tmp/loopback_result.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(asr_rate)
    wf.writeframes(pcm16.tobytes())
print("  Saved: /tmp/loopback_result.wav", flush=True)

# --- Step 2: Local ASR ---
print("\n=== Local ASR (sherpa-onnx) ===", flush=True)
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(audio), asr_step):
    stream.accept_waveform(asr_rate, audio[i:i + asr_step])
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
    try:
        rec = sd.rec(int(8 * 48000), samplerate=48000, channels=2, dtype="float32", device=0)
        sd.wait()
        raw_chunks.append(rec[:, 0])  # ch0
    except Exception as exc:
        print(f"  raw sd.rec skipped: {exc}", flush=True)
    finally:
        raw_done.set()

t2 = threading.Thread(target=raw_record)
t2.start()
time.sleep(0.5)
raw_playback_ok = play_fixture(fixture)
t2.join(timeout=12)

if not raw_playback_ok:
    print("  Playback failed during raw control; skipping raw ASR.", flush=True)

raw_peak = 0
raw_text = ""
if raw_playback_ok and raw_chunks:
    raw_audio = raw_chunks[0]
    rate_gcd = gcd(asr_rate, 48000)
    raw_asr = resample_poly(
        raw_audio,
        up=asr_rate // rate_gcd,
        down=48000 // rate_gcd,
    ).astype(np.float32)
    raw_peak = int(np.max(np.abs(raw_asr)) * 32768)
    print(f"  Raw peak={raw_peak}", flush=True)

    stream2 = engine.recognizer.create_stream()
    for i in range(0, len(raw_asr), asr_step):
        stream2.accept_waveform(asr_rate, raw_asr[i:i + asr_step])
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
