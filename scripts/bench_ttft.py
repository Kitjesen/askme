"""Benchmark TTFT (Time To First Token) across models on the relay.

Usage: python scripts/bench_ttft.py
"""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from openai import AsyncOpenAI

RELAY_URL = os.environ.get("LLM_BASE_URL", "https://cursor.scihub.edu.kg/api/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MINIMAX_URL = "https://api.minimax.chat/v1"
MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")

# (model_name, base_url, api_key) tuples
MODELS = [
    ("MiniMax-M2.5-highspeed", MINIMAX_URL, MINIMAX_KEY),
    ("claude-haiku-4-5-20251001", RELAY_URL, API_KEY),
    ("claude-sonnet-4-5-20250929", RELAY_URL, API_KEY),
    ("claude-opus-4-6", RELAY_URL, API_KEY),
]

PROMPT = [
    {"role": "user", "content": "你好，现在几点了？用一句话回答。"},
]

ROUNDS = 3


async def measure_ttft(client: AsyncOpenAI, model: str) -> dict:
    """Measure TTFT and total time for one streaming call."""
    t0 = time.perf_counter()
    ttft = None
    full_text = ""
    token_count = 0

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=PROMPT,
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
        return {
            "model": model,
            "ttft": ttft or total,
            "total": total,
            "tokens": token_count,
            "text": full_text.strip()[:60],
            "error": None,
        }
    except Exception as e:
        total = time.perf_counter() - t0
        return {
            "model": model,
            "ttft": -1,
            "total": total,
            "tokens": 0,
            "text": "",
            "error": str(e)[:80],
        }


async def main():
    # Build clients keyed by base_url
    clients: dict[str, AsyncOpenAI] = {}
    for _, url, key in MODELS:
        if url not in clients:
            clients[url] = AsyncOpenAI(api_key=key, base_url=url, timeout=30.0)

    print(f"Relay:   {RELAY_URL}")
    print(f"MiniMax: {MINIMAX_URL}")
    print(f"Rounds:  {ROUNDS} per model")
    print(f"Prompt:  {PROMPT[0]['content']}")
    print("=" * 80)

    results: dict[str, list[dict]] = {m[0]: [] for m in MODELS}

    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")
        for model_name, url, _ in MODELS:
            r = await measure_ttft(clients[url], model_name)
            results[model_name].append(r)
            if r["error"]:
                print(f"  {model_name:40s}  ERROR: {r['error']}")
            else:
                print(
                    f"  {model_name:40s}  TTFT={r['ttft']:.2f}s  "
                    f"Total={r['total']:.2f}s  "
                    f"Tokens={r['tokens']}  "
                    f'"{r["text"]}"'
                )
            await asyncio.sleep(1.0)

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Model':40s}  {'TTFT avg':>10s}  {'TTFT min':>10s}  {'Total avg':>10s}  {'OK':>4s}")
    print("-" * 80)
    for model_name, _, _ in MODELS:
        ok_runs = [r for r in results[model_name] if r["error"] is None]
        if not ok_runs:
            print(f"  {model_name:40s}  ALL FAILED")
            continue
        ttfts = [r["ttft"] for r in ok_runs]
        totals = [r["total"] for r in ok_runs]
        print(
            f"  {model_name:40s}  "
            f"{sum(ttfts)/len(ttfts):8.2f}s  "
            f"{min(ttfts):8.2f}s  "
            f"{sum(totals)/len(totals):8.2f}s  "
            f"{len(ok_runs):3d}/{len(results[model_name])}"
        )


if __name__ == "__main__":
    asyncio.run(main())
