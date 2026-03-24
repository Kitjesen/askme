#!/usr/bin/env python3
"""Latency breakdown: where does each millisecond go in a conversation turn."""
import asyncio
import sys
import time
sys.path.insert(0, ".")

from askme.config import get_config
from askme.llm.client import LLMClient

async def main():
    cfg = get_config()
    client = LLMClient()

    messages = [
        {"role": "system", "content": "你是Thunder巡检机器人。回答简短。"},
        {"role": "user", "content": "你是谁"},
    ]

    print("=== LLM latency breakdown (5 runs) ===\n")

    for i in range(5):
        t0 = time.perf_counter()
        first_token = None
        tokens = []

        async for chunk in client.chat_stream(messages):
            if first_token is None:
                first_token = time.perf_counter()
            text = ""
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", "") or ""
            elif isinstance(chunk, str):
                text = chunk
            if text:
                tokens.append(text)

        t1 = time.perf_counter()
        reply = "".join(tokens)
        ttft = (first_token - t0) * 1000 if first_token else 0
        total = (t1 - t0) * 1000
        streaming = total - ttft

        print(f"  Run {i+1}: TTFT={ttft:.0f}ms + streaming={streaming:.0f}ms = total={total:.0f}ms")
        print(f"          reply='{reply[:50]}' ({len(reply)} chars)")

    print(f"\n  Model: {client.model}")
    print(f"  Base URL: {client._base_url[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())
