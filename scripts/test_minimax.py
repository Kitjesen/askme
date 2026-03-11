"""Quick smoke test for MiniMax LLM + TTS integration.

Usage: python scripts/test_minimax.py
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


API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.chat/v1"
LLM_MODEL = "MiniMax-M2.5-highspeed"
TTS_MODEL = "speech-2.8-turbo"


async def test_llm():
    """Test MiniMax LLM streaming."""
    print("=" * 60)
    print("MiniMax LLM Test")
    print("=" * 60)

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)

    t0 = time.perf_counter()
    ttft = None
    full_text = ""
    token_count = 0

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "你好，用一句话介绍自己。"}],
            stream=True,
            temperature=0.3,
            max_tokens=100,
        )
        async for chunk in response:
            if ttft is None:
                ttft = time.perf_counter() - t0
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                token_count += 1

        total = time.perf_counter() - t0
        print(f"  TTFT:   {ttft:.2f}s")
        print(f"  Total:  {total:.2f}s")
        print(f"  Tokens: {token_count}")
        print(f"  Text:   {full_text.strip()[:120]}")
        print(f"  TPS:    {token_count / total:.0f}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_tts():
    """Test MiniMax TTS streaming."""
    print()
    print("=" * 60)
    print("MiniMax TTS Test")
    print("=" * 60)

    url = f"{BASE_URL}/t2a_v2"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": TTS_MODEL,
        "text": "你好，我是Thunder，穹沛科技的巡检机器人。",
        "stream": True,
        "voice_setting": {"voice_id": "male-qn-qingse"},
        "audio_setting": {
            "sample_rate": 24000,
            "format": "pcm",
            "channel": 1,
        },
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
                    body_text = await resp.aread()
                    print(f"  ERROR: HTTP {resp.status_code}: {body_text[:200]}")
                    return False

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
                            pcm_bytes = bytes.fromhex(hex_audio)
                            total_bytes += len(pcm_bytes)
                            chunk_count += 1
                    except (json.JSONDecodeError, ValueError):
                        pass

        total = time.perf_counter() - t0
        duration_sec = total_bytes / (24000 * 2)  # 24kHz 16-bit mono
        print(f"  TTFT:     {ttft:.2f}s" if ttft else "  TTFT:     N/A")
        print(f"  Total:    {total:.2f}s")
        print(f"  Chunks:   {chunk_count}")
        print(f"  PCM:      {total_bytes / 1024:.1f} KB ({duration_sec:.1f}s audio)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    if not API_KEY:
        print("ERROR: MINIMAX_API_KEY not set in .env")
        sys.exit(1)

    print(f"API Key: {API_KEY[:12]}...{API_KEY[-8:]}")
    print()

    llm_ok = await test_llm()
    tts_ok = await test_tts()

    print()
    print("=" * 60)
    print(f"LLM: {'OK' if llm_ok else 'FAIL'}  |  TTS: {'OK' if tts_ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
