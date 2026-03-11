"""End-to-end test: MiniMax M2.5 conversation + TTS pipeline.

Tests realistic askme scenarios:
  1. Basic greeting (Thunder identity)
  2. Multi-turn conversation
  3. Tool/function calling
  4. Think-tag filtering
  5. TTS streaming latency
  6. Comparison with relay

Usage: python scripts/test_e2e_minimax.py
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import httpx
from openai import AsyncOpenAI

# ── Config ────────────────────────────────────────────────────

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "MiniMax-M2.5-highspeed"

RELAY_KEY = os.environ.get("LLM_API_KEY", "")
RELAY_URL = os.environ.get("LLM_BASE_URL", "https://cursor.scihub.edu.kg/api/v1")
RELAY_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = "工业巡检机器人Thunder。中文口语，简洁汇报，80字以内。不用markdown。"

SEED_MESSAGES = [
    {"role": "user", "content": "你是Thunder，穹沛科技的工业巡检机器人，用中文简洁回答"},
    {"role": "assistant", "content": "收到。我是Thunder，穹沛的巡检机器人。等待指令。"},
]

# Tool definition for testing function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sensor_data",
            "description": "获取指定传感器的实时数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor_type": {
                        "type": "string",
                        "enum": ["temperature", "humidity", "vibration", "pressure"],
                        "description": "传感器类型"
                    },
                    "location": {
                        "type": "string",
                        "description": "传感器位置"
                    }
                },
                "required": ["sensor_type"]
            }
        }
    }
]


# ── Think filter (from brain_pipeline) ────────────────────────

class ThinkFilter:
    def __init__(self):
        self._in_think = False
        self._buf = ""

    def feed(self, text):
        self._buf += text
        out = []
        while True:
            if self._in_think:
                idx = self._buf.find("</think>")
                if idx < 0:
                    if len(self._buf) > 8:
                        self._buf = self._buf[-8:]
                    return "".join(out)
                self._buf = self._buf[idx + 8:]
                self._in_think = False
            else:
                idx = self._buf.find("<think>")
                if idx < 0:
                    safe = max(0, len(self._buf) - 7)
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    return "".join(out)
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + 7:]
                self._in_think = True

    def flush(self):
        if self._in_think:
            self._buf = ""
            return ""
        r = self._buf
        self._buf = ""
        return r


# ── Test helpers ──────────────────────────────────────────────

async def stream_chat(client, model, messages, tools=None, label=""):
    """Stream a chat and return (ttft, total_time, raw_text, clean_text, tool_calls)."""
    t0 = time.perf_counter()
    ttft = None
    raw = ""
    tokens = 0
    tool_calls_acc = {}
    filt = ThinkFilter()

    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 300,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        resp = await client.chat.completions.create(**kwargs)
        async for chunk in resp:
            if ttft is None:
                ttft = time.perf_counter() - t0
            delta = chunk.choices[0].delta

            if delta.content:
                raw += delta.content
                tokens += 1

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

        total = time.perf_counter() - t0
        clean = filt.feed(raw) + filt.flush()
        return {
            "ttft": ttft or total,
            "total": total,
            "tokens": tokens,
            "raw": raw,
            "clean": clean.strip(),
            "tool_calls": list(tool_calls_acc.values()) if tool_calls_acc else None,
            "error": None,
        }
    except Exception as e:
        return {
            "ttft": -1,
            "total": time.perf_counter() - t0,
            "tokens": 0,
            "raw": "",
            "clean": "",
            "tool_calls": None,
            "error": str(e)[:150],
        }


async def test_tts_stream(text):
    """Test MiniMax TTS streaming and return (ttft, total, chunks, bytes)."""
    url = f"{MINIMAX_URL}/t2a_v2"
    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "speech-2.8-turbo",
        "text": text,
        "stream": True,
        "voice_setting": {"voice_id": "male-qn-qingse"},
        "audio_setting": {"sample_rate": 24000, "format": "pcm", "channel": 1},
        "output_format": "hex",
    }

    t0 = time.perf_counter()
    ttft = None
    total_bytes = 0
    chunk_count = 0

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    return {"ttft": -1, "total": 0, "chunks": 0, "bytes": 0,
                            "error": f"HTTP {resp.status_code}"}
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        payload = json.loads(data_str)
                        hex_audio = payload.get("data", {}).get("audio", "")
                        if hex_audio:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            pcm = bytes.fromhex(hex_audio)
                            total_bytes += len(pcm)
                            chunk_count += 1
                    except (json.JSONDecodeError, ValueError):
                        pass

        total = time.perf_counter() - t0
        audio_sec = total_bytes / (24000 * 2)
        return {"ttft": ttft or total, "total": total, "chunks": chunk_count,
                "bytes": total_bytes, "audio_sec": audio_sec, "error": None}
    except Exception as e:
        return {"ttft": -1, "total": 0, "chunks": 0, "bytes": 0, "error": str(e)[:100]}


def print_result(label, r):
    if r["error"]:
        print(f"  {label}: ERROR - {r['error']}")
        return
    print(f"  {label}:")
    print(f"    TTFT={r['ttft']:.2f}s  Total={r['total']:.2f}s  Tokens={r['tokens']}")
    if r["clean"]:
        print(f"    Clean: {r['clean'][:120]}")
    if r["tool_calls"]:
        for tc in r["tool_calls"]:
            print(f"    Tool: {tc['name']}({tc['arguments'][:80]})")


# ── Test scenarios ────────────────────────────────────────────

async def main():
    if not MINIMAX_KEY:
        print("ERROR: MINIMAX_API_KEY not set")
        sys.exit(1)

    mm_client = AsyncOpenAI(api_key=MINIMAX_KEY, base_url=MINIMAX_URL, timeout=30.0)
    relay_client = AsyncOpenAI(api_key=RELAY_KEY, base_url=RELAY_URL, timeout=30.0)

    print("=" * 70)
    print("Askme E2E Test: MiniMax M2.5 vs Claude Relay")
    print("=" * 70)

    # ── Test 1: Basic greeting ────────────────────────────────
    print("\n[Test 1] Basic greeting: '你好'")
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *SEED_MESSAGES,
        {"role": "user", "content": "[工业巡检模式：中文口语，80字以内] 你好"},
    ]

    r_mm = await stream_chat(mm_client, MINIMAX_MODEL, msgs)
    r_relay = await stream_chat(relay_client, RELAY_MODEL, msgs)
    print_result("MiniMax", r_mm)
    print_result("Relay Haiku", r_relay)

    # ── Test 2: Domain question ───────────────────────────────
    print("\n[Test 2] Domain question: '3号电机温度怎么样'")
    msgs2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *SEED_MESSAGES,
        {"role": "user", "content": "[工业巡检模式：中文口语，80字以内] 3号电机温度怎么样"},
    ]

    r_mm2 = await stream_chat(mm_client, MINIMAX_MODEL, msgs2)
    r_relay2 = await stream_chat(relay_client, RELAY_MODEL, msgs2)
    print_result("MiniMax", r_mm2)
    print_result("Relay Haiku", r_relay2)

    # ── Test 3: Tool calling ──────────────────────────────────
    print("\n[Test 3] Tool calling: '查一下厨房的温度'")
    msgs3 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *SEED_MESSAGES,
        {"role": "user", "content": "[工业巡检模式：中文口语，80字以内] 查一下厨房的温度"},
    ]

    r_mm3 = await stream_chat(mm_client, MINIMAX_MODEL, msgs3, tools=TOOLS)
    r_relay3 = await stream_chat(relay_client, RELAY_MODEL, msgs3, tools=TOOLS)
    print_result("MiniMax", r_mm3)
    print_result("Relay Haiku", r_relay3)

    # ── Test 4: Multi-turn ────────────────────────────────────
    print("\n[Test 4] Multi-turn: follow-up question")
    msgs4 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *SEED_MESSAGES,
        {"role": "user", "content": "[工业巡检模式：中文口语，80字以内] 今天巡检了哪些区域"},
        {"role": "assistant", "content": "今天巡检了A区配电室、B区冷却塔和C区仓库。A区正常，B区2号冷却塔振动偏高，已记录。"},
        {"role": "user", "content": "[工业巡检模式：中文口语，80字以内] B区那个振动具体多少"},
    ]

    r_mm4 = await stream_chat(mm_client, MINIMAX_MODEL, msgs4)
    r_relay4 = await stream_chat(relay_client, RELAY_MODEL, msgs4)
    print_result("MiniMax", r_mm4)
    print_result("Relay Haiku", r_relay4)

    # ── Test 5: TTS pipeline ─────────────────────────────────
    print("\n[Test 5] TTS streaming: short + long text")
    tts_short = r_mm.get("clean", "") or "你好，我是Thunder巡检机器人。"
    tts_long = "今天巡检了A区配电室、B区冷却塔和C区仓库。A区设备运行正常，温度湿度在标准范围内。B区2号冷却塔振动值偏高，达到4.2毫米每秒，超过预警阈值，建议安排检修。C区仓库一切正常。"

    r_tts1 = await test_tts_stream(tts_short[:80])
    r_tts2 = await test_tts_stream(tts_long)

    if r_tts1["error"]:
        print(f"  TTS short: ERROR - {r_tts1['error']}")
    else:
        print(f"  TTS short: TTFT={r_tts1['ttft']:.2f}s  Total={r_tts1['total']:.2f}s  "
              f"Chunks={r_tts1['chunks']}  Audio={r_tts1['audio_sec']:.1f}s")

    if r_tts2["error"]:
        print(f"  TTS long:  ERROR - {r_tts2['error']}")
    else:
        print(f"  TTS long:  TTFT={r_tts2['ttft']:.2f}s  Total={r_tts2['total']:.2f}s  "
              f"Chunks={r_tts2['chunks']}  Audio={r_tts2['audio_sec']:.1f}s")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Test':<30s}  {'MiniMax TTFT':>14s}  {'Relay TTFT':>14s}  {'Speedup':>10s}")
    print("-" * 70)

    tests = [
        ("1. Greeting", r_mm, r_relay),
        ("2. Domain Q&A", r_mm2, r_relay2),
        ("3. Tool calling", r_mm3, r_relay3),
        ("4. Multi-turn", r_mm4, r_relay4),
    ]

    for name, mm, rl in tests:
        mm_t = f"{mm['ttft']:.2f}s" if not mm["error"] else "FAIL"
        rl_t = f"{rl['ttft']:.2f}s" if not rl["error"] else "FAIL"
        if not mm["error"] and not rl["error"] and mm["ttft"] > 0 and rl["ttft"] > 0:
            speedup = f"{rl['ttft'] / mm['ttft']:.1f}x"
        else:
            speedup = "-"
        print(f"  {name:<28s}  {mm_t:>14s}  {rl_t:>14s}  {speedup:>10s}")

    print("-" * 70)
    if not r_tts1["error"]:
        print(f"  TTS TTFT: {r_tts1['ttft']:.2f}s (short)  {r_tts2['ttft']:.2f}s (long)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
