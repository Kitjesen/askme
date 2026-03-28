#!/usr/bin/env python3
"""Full voice E2E test: loopback + live mic + Cloud ASR + LLM + TTS."""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.config import get_config
from askme.voice.mic_input import MicInput
from askme.voice.asr import ASREngine

cfg = get_config()
mic = MicInput.from_config(cfg)
engine = ASREngine(cfg.get("voice", {}).get("asr", {}))

# ---- Test 1: Automated Loopback ----
print("=== Test 1: Automated Loopback ===", flush=True)
chunks = []

def record_loopback():
    with mic.open():
        for _ in range(70):
            chunks.append(mic.read_chunk())

t = threading.Thread(target=record_loopback)
t.start()
time.sleep(0.3)
subprocess.run(["aplay", "-D", "default", "/tmp/test_speech_48k.wav"],
               capture_output=True, timeout=15)
t.join(timeout=10)

audio = np.concatenate(chunks)
peak = int(np.max(np.abs(audio)) * 32768)
rms = int(np.sqrt(np.mean(audio ** 2)) * 32768)
print("  Signal: peak=%d rms=%d" % (peak, rms), flush=True)

stream = engine.recognizer.create_stream()
for i in range(0, len(audio), 1600):
    stream.accept_waveform(16000, audio[i:i + 1600])
    while engine.recognizer.is_ready(stream):
        engine.recognizer.decode_stream(stream)
local1 = engine.recognizer.get_result(stream).strip()
print("  Local ASR: '%s'" % local1, flush=True)

# ---- Test 2: Beep + record + dual ASR ----
print("\n=== Test 2: Live Mic (beep cue) ===", flush=True)
subprocess.run(["aplay", "-D", "default", "/tmp/tone1k.wav"],
               capture_output=True, timeout=10)
time.sleep(0.2)

chunks2 = []
with mic.open():
    for _ in range(60):
        chunks2.append(mic.read_chunk())

audio2 = np.concatenate(chunks2)
peak2 = int(np.max(np.abs(audio2)) * 32768)
rms2 = int(np.sqrt(np.mean(audio2 ** 2)) * 32768)
print("  Signal: peak=%d rms=%d" % (peak2, rms2), flush=True)

stream2 = engine.recognizer.create_stream()
for i in range(0, len(audio2), 1600):
    stream2.accept_waveform(16000, audio2[i:i + 1600])
    while engine.recognizer.is_ready(stream2):
        engine.recognizer.decode_stream(stream2)
local2 = engine.recognizer.get_result(stream2).strip()
print("  Local ASR: '%s'" % local2, flush=True)

# Cloud ASR
cloud_text = ""
api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if api_key:
    try:
        import websocket
        pcm16 = (audio2 * 32768).clip(-32768, 32767).astype(np.int16)
        ws = websocket.WebSocket()
        ws.settimeout(10)
        ws.connect("wss://dashscope.aliyuncs.com/api-ws/v1/inference/",
                   header=["Authorization: bearer " + api_key])
        tid = str(uuid.uuid4())
        ws.send(json.dumps({
            "header": {"action": "run-task", "task_id": tid, "streaming": "duplex"},
            "payload": {
                "task_group": "audio", "task": "asr", "function": "recognition",
                "model": "paraformer-realtime-v2",
                "parameters": {"sample_rate": 16000, "format": "pcm",
                               "language_hints": ["zh", "en"]},
                "input": {},
            },
        }))
        assert json.loads(ws.recv())["header"]["event"] == "task-started"
        done = threading.Event()

        def rx():
            global cloud_text
            while True:
                try:
                    r = ws.recv()
                except Exception:
                    break
                if isinstance(r, bytes):
                    continue
                m = json.loads(r)
                e = m.get("header", {}).get("event", "")
                if e == "result-generated":
                    s = m.get("payload", {}).get("output", {}).get("sentence", {})
                    if s.get("sentence_end") and s.get("text"):
                        cloud_text += s["text"]
                elif e in ("task-finished", "task-failed"):
                    break
            done.set()

        threading.Thread(target=rx, daemon=True).start()
        b = pcm16.tobytes()
        for i in range(0, len(b), 6400):
            ws.send_binary(b[i:i + 6400])
            time.sleep(0.005)
        ws.send(json.dumps({
            "header": {"action": "finish-task", "task_id": tid, "streaming": "duplex"},
            "payload": {"input": {}},
        }))
        done.wait(timeout=10)
        ws.close()
        print("  Cloud ASR: '%s'" % cloud_text, flush=True)
    except Exception as exc:
        print("  Cloud ASR error: %s" % exc, flush=True)

# ---- Test 3: LLM + TTS ----
asr_text = cloud_text or local2
if asr_text:
    print("\n=== Test 3: LLM + TTS ===", flush=True)
    print("  Input: '%s'" % asr_text, flush=True)

    from askme.blueprints.voice import voice

    async def run_llm_tts():
        app = await voice.build(cfg)
        await app.start()
        pipeline = app.get("pipeline").brain_pipeline
        resp = await asyncio.wait_for(pipeline.process(asr_text, source="voice"), timeout=30)
        print("  LLM: '%s'" % resp[:100], flush=True)
        # TTS is handled by pipeline automatically via voice module
        await asyncio.sleep(3)  # let TTS finish playing
        await app.stop()
        return resp

    llm_resp = asyncio.run(run_llm_tts())
    print("  TTS done.", flush=True)
else:
    print("\nNo speech detected - skipping LLM+TTS", flush=True)

# ---- Summary ----
tts_status = "OK" if asr_text else "skipped"
print("\n=== Summary ===", flush=True)
print("  MicInput: 48kHz 2ch -> HPF -> AGC -> 16kHz mono", flush=True)
print("  Loopback ASR:  '%s'" % local1, flush=True)
print("  Live Local:    '%s'" % local2, flush=True)
print("  Live Cloud:    '%s'" % cloud_text, flush=True)
print("  LLM+TTS:       %s" % tts_status, flush=True)
