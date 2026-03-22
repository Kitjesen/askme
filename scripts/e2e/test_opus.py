"""Quick test: compare models for voice assistant quality + latency."""

import asyncio
import io
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient

SYSTEM = (
    "你是一个有用的 AI 语音助手。用中文简洁口语化回答。"
    "不要用markdown格式，像朋友聊天一样，100字以内。"
)

QUERIES = [
    "你好啊，你是谁？",
    "帮我解释一下什么是强化学习，要简单易懂",
    "讲个笑话吧",
    "四足机器人的运动控制用什么算法比较好？",
]


async def test_model(model_name: str):
    client = LLMClient(model=model_name)
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    history = []
    total_time = 0.0

    for q in QUERIES:
        history.append({"role": "user", "content": q})
        msgs = [{"role": "system", "content": SYSTEM}] + history

        t0 = time.monotonic()
        ttft = None
        full = ""

        try:
            async for chunk in client.chat_stream(msgs):
                d = chunk.choices[0].delta
                if d.content:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    full += d.content
        except Exception as e:
            print(f"   [Stream error: {e}]")

        dur = time.monotonic() - t0
        total_time += dur
        history.append({"role": "assistant", "content": full})

        ft = f"{ttft:.2f}" if ttft else "N/A"
        print(f"Q: {q}")
        print(f"A: {full}")
        print(f"   TTFT={ft}s  Total={dur:.2f}s  Len={len(full)}")
        print()

    print(f"--- {model_name}: Total={total_time:.1f}s ---")
    print()
    return total_time


async def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["claude-opus-4-6"]
    for m in models:
        await test_model(m)


if __name__ == "__main__":
    asyncio.run(main())
