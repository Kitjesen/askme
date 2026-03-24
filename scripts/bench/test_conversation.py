#!/usr/bin/env python3
"""Test full conversation: text input → LLM → response."""
import asyncio
import sys
import time
sys.path.insert(0, ".")

from askme.config import get_config
from askme.blueprints.text import text

async def main():
    cfg = get_config()
    print("Building text blueprint...")
    app = await text.build(cfg)
    await app.start()
    print(f"Started {len(app.modules)} modules\n")

    text_mod = app.modules.get("text")
    if not text_mod or not hasattr(text_mod, "text_loop"):
        print("ERROR: text_loop not found")
        await app.stop()
        return

    tl = text_mod.text_loop

    questions = ["你好", "你是谁", "现在几点了"]
    for q in questions:
        print(f"[你] {q}")
        t0 = time.perf_counter()
        reply = await tl.process_turn(q)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[askme] {reply}")
        print(f"  ({elapsed:.0f}ms)\n")

    await app.stop()
    print("=== 对话测试完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
