#!/usr/bin/env python3
"""Cloud ASR: product MicInput path -> DashScope + local ASR."""
import json
import os
import sys
import threading
import time
import uuid
import wave
from math import gcd

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import websocket
from askme.voice.asr import ASREngine
from askme.voice.mic_input import MicInput

from askme.config import get_config

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if not api_key:
    print("DASHSCOPE_API_KEY not set", flush=True)
    sys.exit(1)

TARGET_RATE = 16000
DURATION = 5
cfg = get_config()

# Countdown
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)

print(f"RECORDING {DURATION}s through MicInput! SPEAK NOW!", flush=True)
mic = MicInput.from_config(cfg)
chunks: list[np.ndarray] = []
deadline = time.monotonic() + DURATION
with mic.open():
    while time.monotonic() < deadline:
        chunks.append(mic.read_chunk())

raw = np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)
dc = float(np.mean(raw)) if len(raw) else 0.0
raw = raw - dc
peak = float(np.max(np.abs(raw))) if len(raw) else 0.0
rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) else 0.0
print(
    f"MicInput: {mic._native_rate}Hz {mic._native_channels}ch -> {mic.sample_rate}Hz mono",
    flush=True,
)
print(f"Audio: DC={dc:.4f} peak={peak:.4f}({int(peak*32768)}) rms={rms:.4f}", flush=True)
if len(raw) == 0:
    print("No MicInput samples captured; cannot run ASR.", flush=True)
    sys.exit(2)

asr_audio = raw
if mic.sample_rate != TARGET_RATE:
    from scipy.signal import resample_poly

    rate_gcd = gcd(TARGET_RATE, mic.sample_rate)
    asr_audio = resample_poly(
        raw,
        TARGET_RATE // rate_gcd,
        mic.sample_rate // rate_gcd,
    ).astype(np.float32)
    asr_audio = np.clip(asr_audio, -1.0, 1.0)
    print(f"Resampled for ASR: {mic.sample_rate}Hz -> {TARGET_RATE}Hz", flush=True)
else:
    print(f"ASR audio rate: {TARGET_RATE}Hz", flush=True)

pcm16 = (asr_audio * 32768).clip(-32768, 32767).astype(np.int16)
print(f"PCM16: peak={int(np.max(np.abs(pcm16)))}", flush=True)

# Save wav
with wave.open("/tmp/mic_hkmic.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(TARGET_RATE)
    wf.writeframes(pcm16.tobytes())

# Local ASR
print("\n=== Local ASR ===", flush=True)
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
stream = engine.recognizer.create_stream()
for i in range(0, len(asr_audio), 1600):
    stream.accept_waveform(TARGET_RATE, asr_audio[i:i + 1600])
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
print("\n=== Result ===", flush=True)
print(f"  Local: '{local_text}'", flush=True)
print(f"  Cloud: '{result_text}'", flush=True)
