"""Live chat test: MiniMax LLM + TTS through askme pipeline.

Simulates the text_loop flow with real MiniMax backend.
Usage: python scripts/test_live_chat.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ASKME_CONFIG_PATH", "config.yaml")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from askme.config import get_config
from askme.brain.llm_client import LLMClient
from askme.pipeline.brain_pipeline import _ThinkFilter


async def chat_round(client, messages, label="", model=None):
    """One round of streaming chat, print incrementally."""
    filt = _ThinkFilter()
    t0 = time.perf_counter()
    ttft = None
    raw = ""
    clean_acc = ""
    tool_calls = {}
    use_model = model or client.model

    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  User: {messages[-1]['content'][:80]}")
    print(f"  Model: {use_model}")
    sys.stdout.write("  Thunder: ")
    sys.stdout.flush()

    try:
        async for chunk in client.chat_stream(messages, model=use_model):
            if ttft is None:
                ttft = time.perf_counter() - t0

            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": ""}
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments

            if delta.content:
                raw += delta.content
                piece = filt.feed(delta.content)
                if piece:
                    clean_acc += piece
                    sys.stdout.write(piece)
                    sys.stdout.flush()

        # Flush remaining
        tail = filt.flush()
        if tail:
            clean_acc += tail
            sys.stdout.write(tail)
        print()

        total = time.perf_counter() - t0
        print(f"  ---")
        print(f"  TTFT: {ttft:.2f}s | Total: {total:.2f}s | Raw: {len(raw)} chars | Clean: {len(clean_acc.strip())} chars")

        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({tc['arguments'][:80]})")

        return clean_acc.strip(), ttft, total

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return "", -1, time.perf_counter() - t0


async def main():
    cfg = get_config()
    brain_cfg = cfg.get("brain", {})

    client = LLMClient()
    print(f"Primary model: {client.model}")
    print(f"MiniMax client: {'enabled' if client._minimax_client else 'disabled'}")

    voice_model = brain_cfg.get("voice_model", client.model)
    system_prompt = brain_cfg.get("system_prompt", "")
    user_prefix = brain_cfg.get("user_prefix", "")
    seed = brain_cfg.get("prompt_seed", [])

    # Load SOUL.md if available
    soul_file = brain_cfg.get("soul_file", "")
    soul_content = ""
    if soul_file and os.path.isfile(soul_file):
        with open(soul_file, encoding="utf-8") as f:
            soul_content = f.read().strip()

    def build_messages(user_text, history=None):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if soul_content:
            msgs.append({"role": "user", "content": f"[角色设定]\n{soul_content}"})
            msgs.append({"role": "assistant", "content": "收到，已加载角色设定。"})
        msgs.extend(seed)
        if history:
            msgs.extend(history)
        prefixed = f"{user_prefix} {user_text}" if user_prefix else user_text
        msgs.append({"role": "user", "content": prefixed})
        return msgs

    history = []

    # ── Test conversations ────────────────────────────────────

    test_inputs = [
        ("1. 问候", "你好"),
        ("2. 身份确认", "你是谁？叫什么名字"),
        ("3. 领域问答", "3号电机温度怎么样"),
        ("4. 多轮跟进", "那振动数据呢"),
        ("5. 指令", "去B区巡检"),
    ]

    results = []

    for label, user_text in test_inputs:
        msgs = build_messages(user_text, history)
        # Override model to voice_model for all test rounds
        reply, ttft, total = await chat_round(client, msgs, label, model=voice_model)

        # Add to history for multi-turn
        prefixed = f"{user_prefix} {user_text}" if user_prefix else user_text
        history.append({"role": "user", "content": prefixed})
        if reply:
            history.append({"role": "assistant", "content": reply})

        results.append((label, ttft, total, len(reply)))
        await asyncio.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Summary (voice_model={voice_model})")
    print(f"{'Test':<25s}  {'TTFT':>8s}  {'Total':>8s}  {'Reply':>8s}")
    print("-" * 55)
    for label, ttft, total, rlen in results:
        print(f"  {label:<23s}  {ttft:6.2f}s  {total:6.2f}s  {rlen:5d} ch")
    avg_ttft = sum(r[1] for r in results if r[1] > 0) / max(1, sum(1 for r in results if r[1] > 0))
    print(f"\n  Average TTFT: {avg_ttft:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
