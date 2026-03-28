#!/usr/bin/env python3
"""E2E voice loopback: play wav → HKMIC → MicInput pipeline → ASR → LLM → TTS → Speaker.

No human needed — uses speaker→mic loopback for automated testing.
"""
import asyncio
import os
import subprocess
import sys
import threading
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.config import get_config
from askme.voice.mic_input import MicInput
from askme.voice.asr import ASREngine

cfg = get_config()

# ============================================================
# Step 1: Loopback capture
# ============================================================
print("=" * 50, flush=True)
print("Step 1: Loopback capture (speaker → HKMIC)", flush=True)
print("=" * 50, flush=True)

mic = MicInput.from_config(cfg)
chunks = []

def record_fn():
    with mic.open():
        for _ in range(80):  # 8 seconds
            chunks.append(mic.read_chunk())

t = threading.Thread(target=record_fn)
t.start()
time.sleep(0.5)

subprocess.run(["aplay", "-D", "plughw:1,0", "/tmp/test_speech_48k.wav"],
               capture_output=True, timeout=15)
t.join(timeout=12)

audio = np.concatenate(chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
print(f"  Captured: {len(audio)/16000:.1f}s, peak={peak}, rms={rms}", flush=True)

# ============================================================
# Step 2: Local ASR
# ============================================================
print("\n" + "=" * 50, flush=True)
print("Step 2: Local ASR (sherpa-onnx)", flush=True)
print("=" * 50, flush=True)

t0 = time.perf_counter()
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(audio), 1600):
    stream.accept_waveform(16000, audio[i:i + 1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
asr_text = engine.recognizer.get_result(stream).strip()
asr_ms = (time.perf_counter() - t0) * 1000
print(f"  ASR ({asr_ms:.0f}ms): '{asr_text}'", flush=True)

if not asr_text:
    print("  No speech recognized — aborting.", flush=True)
    sys.exit(1)

# ============================================================
# Step 3: LLM (MiniMax)
# ============================================================
print("\n" + "=" * 50, flush=True)
print("Step 3: LLM (MiniMax)", flush=True)
print("=" * 50, flush=True)

# Use the brain module directly for a single query
from openai import OpenAI

brain_cfg = cfg.get("brain", {})
client = OpenAI(
    api_key=brain_cfg.get("api_key", os.environ.get("MINIMAX_API_KEY", "")),
    base_url=brain_cfg.get("base_url", "https://api.minimax.chat/v1"),
)

system_prompt = brain_cfg.get("system_prompt", "你是Thunder，工业巡检机器人。中文简洁回答。")
user_msg = f"[语音输入] {asr_text}"

t0 = time.perf_counter()
resp = client.chat.completions.create(
    model=brain_cfg.get("voice_model", "MiniMax-M2.7-highspeed"),
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ],
    max_tokens=200,
)
llm_text = resp.choices[0].message.content.strip()
llm_ms = (time.perf_counter() - t0) * 1000
print(f"  LLM ({llm_ms:.0f}ms): '{llm_text[:120]}'", flush=True)

# ============================================================
# Step 4: TTS → Speaker
# ============================================================
print("\n" + "=" * 50, flush=True)
print("Step 4: TTS → Speaker (MiniMax Speech)", flush=True)
print("=" * 50, flush=True)

# Filter out [SILENT] response
if "[SILENT]" in llm_text:
    print("  LLM returned [SILENT] — skipping TTS.", flush=True)
else:
    from askme.voice.tts import TTSEngine
    tts_cfg = cfg.get("voice", {}).get("tts", {})
    tts = TTSEngine(tts_cfg)
    tts.start_playback()

    t0 = time.perf_counter()
    tts.speak(llm_text[:200])  # cap at 200 chars
    tts.wait_done()
    tts_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTS ({tts_ms:.0f}ms): played '{llm_text[:60]}...'", flush=True)
    tts.stop_playback()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50, flush=True)
print("E2E Summary", flush=True)
print("=" * 50, flush=True)
print(f"  Mic:  HKMIC 48kHz→16kHz (peak={peak})", flush=True)
print(f"  ASR:  '{asr_text}' ({asr_ms:.0f}ms)", flush=True)
print(f"  LLM:  '{llm_text[:80]}' ({llm_ms:.0f}ms)", flush=True)
silent = "[SILENT]" in llm_text
print(f"  TTS:  {'skipped (SILENT)' if silent else 'OK'}", flush=True)
print(f"  Total: {asr_ms + llm_ms:.0f}ms (ASR+LLM)", flush=True)
print("\n  ✓ Voice loop complete!" if not silent else "\n  △ LLM said SILENT (loopback not addressed to robot)", flush=True)
