"""Measure real MiniMax API latency: TTS + LLM TTFT."""
import json
import os
import sys
import time

import requests
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
api_key = os.environ.get("MINIMAX_API_KEY", cfg.get("brain", {}).get("minimax_api_key", ""))

# ── TTS latency ──
print("=" * 50)
print("TTS (speech-2.8-turbo, PCM, short sentence)")
t0 = time.perf_counter()
resp = requests.post(
    "https://api.minimax.chat/v1/t2a_v2",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "speech-2.8-turbo",
        "text": "你好，前方路口左转。",
        "voice_setting": {"voice_id": "male-qn-qingse", "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "format": "pcm"},
    },
    timeout=10,
)
t1 = time.perf_counter()
if resp.status_code == 200:
    data = resp.json()
    audio_b64 = data.get("data", {}).get("audio", "")
    audio_bytes = len(audio_b64) * 3 // 4  # base64 decode estimate
    duration_s = audio_bytes / (32000 * 2)  # 16-bit PCM
    print(f"  API latency: {(t1-t0)*1000:.0f}ms")
    print(f"  Audio duration: {duration_s:.1f}s")
    print(f"  Stream factor: {duration_s/(t1-t0):.1f}x" if (t1-t0) > 0 else "")
else:
    print(f"  FAILED: {resp.status_code} {resp.text[:200]}")

# ── LLM TTFT (streaming, short reply) ──
print()
print("=" * 50)
print("LLM TTFT (MiniMax-M2.7-highspeed, max_tokens=30, stream=True)")
t0 = time.perf_counter()
resp = requests.post(
    "https://api.minimax.chat/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "MiniMax-M2.7-highspeed",
        "messages": [{"role": "user", "content": "用最短的话回复：前方路口怎么走"}],
        "max_tokens": 30,
        "temperature": 0.7,
        "stream": True,
    },
    stream=True,
    timeout=30,
)
if resp.status_code != 200:
    print(f"  FAILED: {resp.status_code} {resp.text[:300]}")
    sys.exit(1)
ttft = None
total_chunks = 0
full_text = ""
for line in resp.iter_lines():
    if line and line.startswith(b"data: "):
        data = line[6:]
        if data == b"[DONE]":
            break
        try:
            chunk = json.loads(data)
            total_chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            full_text += delta
        except Exception:
            pass
total = time.perf_counter() - t0
print(f"  TTFT (first token): {ttft*1000:.0f}ms")
print(f"  Total time: {total*1000:.0f}ms")
print(f"  Chunks: {total_chunks}")
print(f"  Reply: {full_text}")
print(f"  Generation speed: {len(full_text)/max(0.001,total-ttft):.0f} chars/s")

# ── Summary ──
print()
print("=" * 50)
print("VOICE PIPELINE ESTIMATE (streaming, short reply)")
print("  ASR:  1-2s")
if ttft:
    print(f"  LLM TTFT: {ttft*1000:.0f}ms (first token)")
    print(f"  LLM generation: {(total-ttft)*1000:.0f}ms")
print(f"  TTS (PCM stream): {(t1-t0)*1000:.0f}ms (first sentence, overlaps with generation)")
print("  ---")
if ttft:
    print(f"  USER PERCEIVED: ~{ttft + (t1-t0):.0f}ms (first word spoken)")
    print(f"  FULL RESPONSE: ~{total*1000:.0f}ms")
