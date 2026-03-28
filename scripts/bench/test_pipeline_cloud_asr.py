#!/usr/bin/env python3
"""Test MicInput pipeline → Cloud ASR directly (bypass VoiceLoop/VAD).

Proves whether the pipeline audio can be recognized by DashScope.
"""
import json
import os
import sys
import threading
import time
import uuid
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.config import get_config
from askme.voice.mic_input import MicInput

cfg = get_config()
api_key = os.environ.get("DASHSCOPE_API_KEY", cfg.get("voice", {}).get("cloud_asr", {}).get("api_key", ""))

if not api_key or api_key.startswith("${"):
    # Try .env
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("DASHSCOPE_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
    if not api_key or api_key.startswith("${"):
        print("DASHSCOPE_API_KEY not set", flush=True)
        sys.exit(1)

import websocket

DURATION = 6

print("=== MicInput Pipeline → Cloud ASR (Direct) ===", flush=True)
mic = MicInput.from_config(cfg)
print(f"  Pipeline: {mic._native_rate}Hz → {mic._sample_rate}Hz", flush=True)

# Record using MicInput pipeline
n_chunks = int(DURATION * 1000 / 100)
print(f"\nRecording {DURATION}s — SPEAK NOW!", flush=True)
for i in range(3, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)
print(">>> RECORDING <<<", flush=True)

all_chunks = []
with mic.open():
    for _ in range(n_chunks):
        chunk = mic.read_chunk()
        all_chunks.append(chunk)

audio = np.concatenate(all_chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
print(f"  Captured: {len(audio)/16000:.1f}s peak={peak} rms={rms}", flush=True)

# Convert to int16 PCM
pcm16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
pcm_bytes = pcm16.tobytes()
print(f"  PCM bytes: {len(pcm_bytes)}", flush=True)

# Send to DashScope Cloud ASR
print("\n=== Sending to DashScope ===", flush=True)
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
        "parameters": {"sample_rate": 16000, "format": "pcm",
                       "language_hints": ["zh", "en"]},
        "input": {},
    },
}))
ack = json.loads(ws.recv())
assert ack["header"]["event"] == "task-started", f"Unexpected: {ack}"
print("  Session started", flush=True)

result_text = ""
done = threading.Event()

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
                print(f"  Partial: '{s['text']}'", flush=True)
        elif event in ("task-finished", "task-failed"):
            break
    done.set()

threading.Thread(target=recv_loop, daemon=True).start()

# Send audio in chunks (simulate streaming)
chunk_size = 6400  # 200ms of pcm16 at 16kHz
sent = 0
for i in range(0, len(pcm_bytes), chunk_size):
    ws.send_binary(pcm_bytes[i:i + chunk_size])
    sent += min(chunk_size, len(pcm_bytes) - i)
    time.sleep(0.01)
print(f"  Sent {sent} bytes", flush=True)

ws.send(json.dumps({
    "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
    "payload": {"input": {}},
}))
done.wait(timeout=10)
ws.close()

print(f"\n=== Result ===", flush=True)
print(f"  Cloud ASR: '{result_text}'", flush=True)
if result_text:
    print("  ✓ Pipeline audio recognized by DashScope!", flush=True)
else:
    print("  ✗ DashScope returned empty — audio format issue?", flush=True)
