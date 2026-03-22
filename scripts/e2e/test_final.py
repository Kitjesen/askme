"""Final test: verify the prompt seed + user prefix strategy works end-to-end."""

import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient
from askme.config import get_config

MODEL = "claude-opus-4-6"

QUESTIONS = [
    "你好，你是谁？",
    "讲个笑话",
    "什么是强化学习？",
    "四足机器人用什么运动控制算法？",
    "今天心情不好，安慰我一下",
]


def build_messages(history, user_text, cfg):
    """Simulate what BrainPipeline._prepare_messages does."""
    brain = cfg.get("brain", {})
    system_prompt = brain.get("system_prompt", "")
    seed = brain.get("prompt_seed", [])
    prefix = brain.get("user_prefix", "")

    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(seed)
    msgs.extend(history)

    # Add user message with prefix
    tagged = f"{prefix}\n{user_text}" if prefix else user_text
    msgs.append({"role": "user", "content": tagged})

    return msgs


async def main():
    cfg = get_config(reload=True)
    client = LLMClient(model=MODEL)

    brain = cfg.get("brain", {})
    print(f"Model: {MODEL}")
    print(f"System: {brain.get('system_prompt', '')[:60]}")
    print(f"Seed: {len(brain.get('prompt_seed', []))} messages")
    print(f"Prefix: {brain.get('user_prefix', '')[:40]}...")
    print()

    history = []
    total_score = 0
    total_time = 0.0

    for q in QUESTIONS:
        msgs = build_messages(history, q, cfg)

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
            full = f"[ERROR: {e}]"

        dur = time.monotonic() - t0
        total_time += dur

        # Save to history (raw, without prefix)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": full})

        # Score
        has_md = any(s in full for s in ["**", "- ", "```", "# "])
        is_chinese = not any(s in full for s in ["I'm ", "I can ", "help you"])
        short = len(full) <= 120
        not_dev = not any(w in full for w in ["开发者", "写代码", "调试程序"])

        score = sum([not has_md, is_chinese, short, not_dev])
        total_score += score

        ft = f"{ttft:.1f}" if ttft else "N/A"
        flags = []
        if has_md: flags.append("MD!")
        if not is_chinese: flags.append("EN!")
        if not short: flags.append(f"长({len(full)})")
        if not not_dev: flags.append("DEV!")
        flag_str = f"  {flags}" if flags else ""

        print(f"Q: {q}")
        print(f"A: {full[:150]}{'...' if len(full)>150 else ''}")
        print(f"   TTFT={ft}s  Total={dur:.1f}s  Score={score}/4{flag_str}")
        print()

    n = len(QUESTIONS)
    print(f"{'='*50}")
    print(f"Total: {total_score}/{n*4}  Avg={total_score/n:.1f}/4")
    print(f"Time: {total_time:.1f}s  Avg={total_time/n:.1f}s/q")


if __name__ == "__main__":
    asyncio.run(main())
