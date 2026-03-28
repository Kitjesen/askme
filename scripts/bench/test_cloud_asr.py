#!/usr/bin/env python3
"""Cloud ASR: HK-MIC (card0, 48kHz 2ch) → resample → DashScope + local ASR."""
import os
import sys
import json
import time
import threading
import uuid
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sounddevice as sd
import soxr
import websocket

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if not api_key:
    print("DASHSCOPE_API_KEY not set", flush=True)
    sys.exit(1)

DEVICE = 0       # card0 = HK-MIC
NATIVE_RATE = 48000
TARGET_RATE = 16000
DURATION = 5

# Countdown
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)

print(f"RECORDING {DURATION}s! SPEAK NOW!", flush=True)
rec = sd.rec(int(DURATION * NATIVE_RATE), samplerate=NATIVE_RATE, channels=2, dtype="float32", device=DEVICE)
sd.wait()

# Use channel 0 (stronger signal)
raw = rec[:, 0]
dc = np.mean(raw)
raw = raw - dc
peak = np.max(np.abs(raw))
rms = np.sqrt(np.mean(raw ** 2))
print(f"Ch0: DC={dc:.4f} peak={peak:.4f}({int(peak*32768)}) rms={rms:.4f}", flush=True)

# Resample 48k → 16k with soxr
audio_16k = soxr.resample(raw, NATIVE_RATE, TARGET_RATE, quality="VHQ").astype(np.float32)
pcm16 = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
print(f"Resampled: peak={int(np.max(np.abs(pcm16)))}", flush=True)

# Save wav
with wave.open("/tmp/mic_hkmic.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(TARGET_RATE)
    wf.writeframes(pcm16.tobytes())

# Local ASR
print("\n=== Local ASR ===", flush=True)
from askme.voice.asr import ASREngine
from askme.config import get_config
cfg = get_config()
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(audio_16k), 1600):
    stream.accept_waveform(TARGET_RATE, audio_16k[i:i+1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
local_text = engine.recognizer.get_result(stream).strip()
print(f"Local: '{local_text}'", flush=True)

# Cloud ASR
print("\n=== Cloud ASR ===", flush=True)
ws = websocket.WebSocket()
ws.settimeout(10)
ws.connect("wss://dashscope.aliyuncs.com/api-ws/v1/inference/",
           header=[f"Authorization: bearer {api_key}"])

task_id = str(uuid.uuid4())
ws.send(json.dumps({
    "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
    "payload": {
        "task_group": "audio", "task": "asr", "function": "recognition",
        "model": "paraformer-realtime-v2",
        "parameters": {"sample_rate": TARGET_RATE, "format": "pcm", "language_hints": ["zh", "en"]},
        "input": {},
    },
}))
assert json.loads(ws.recv())["header"]["event"] == "task-started"

result_text = ""
result_ready = threading.Event()

def recv_loop():
    global result_text
    while True:
        try:
            raw = ws.recv()
        except Exception:
            break
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        event = msg.get("header", {}).get("event", "")
        if event == "result-generated":
            s = msg.get("payload", {}).get("output", {}).get("sentence", {})
            if s.get("sentence_end") and s.get("text"):
                result_text += s["text"]
        elif event in ("task-finished", "task-failed"):
            break
    result_ready.set()

threading.Thread(target=recv_loop, daemon=True).start()

pcm_bytes = pcm16.tobytes()
for i in range(0, len(pcm_bytes), 6400):
    ws.send_binary(pcm_bytes[i:i + 6400])
    time.sleep(0.01)

ws.send(json.dumps({
    "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
    "payload": {"input": {}},
}))
result_ready.wait(timeout=10)
ws.close()

print(f"Cloud: '{result_text}'", flush=True)
print(f"\n=== Result ===", flush=True)
print(f"  Local: '{local_text}'", flush=True)
print(f"  Cloud: '{result_text}'", flush=True)
