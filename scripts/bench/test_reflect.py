#!/usr/bin/env python3
"""Test L3 Episodic reflect on S100P."""
import asyncio
import os
import sys
import shutil
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class MockLLMClient:
    """Wraps MiniMax API with the LLMClient.chat() interface."""

    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.chat/v1",
        )

    async def chat(self, messages, **kwargs):
        resp = await self._client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=messages,
        )
        return resp.choices[0].message.content or ""


async def main():
    # Use unique test dir to avoid stale data
    test_data = f"/tmp/askme_reflect_{uuid.uuid4().hex[:8]}"
    shutil.rmtree(test_data, ignore_errors=True)

    # Patch config BEFORE importing EpisodicMemory
    import askme.config as cfg
    original_get = cfg.get_config

    def patched_config():
        c = original_get()
        c["app"] = {"data_dir": test_data}
        c["memory"] = c.get("memory", {})
        c["memory"]["episodic"] = {
            "reflect_min_events": 3,
            "importance_threshold": 1.0,
            "reflect_cooldown_seconds": 0,
            "admission_threshold": 0.0,
        }
        return c

    cfg.get_config = patched_config

    from askme.memory.episodic_memory import EpisodicMemory

    llm = MockLLMClient()
    mem = EpisodicMemory(llm=llm)
    print(f"Created, buffer={mem.buffer_size} (should be 0)")

    # Log events
    events = [
        ("perception", "检测到一个人站在仓库A门口", {"label": "person", "confidence": 0.92}),
        ("perception", "仓库A温度传感器读数异常: 38.5°C", {"sensor": "temp_A", "value": 38.5}),
        ("action", "向操作员报告温度异常", {"target": "operator"}),
        ("outcome", "操作员确认收到，安排检查", {}),
        ("perception", "仓库B一切正常", {"sensor": "all", "status": "normal"}),
        ("perception", "3号巡检点发现地面积水", {"label": "water", "confidence": 0.87}),
        ("action", "拍照记录并上报3号巡检点积水情况", {"photo": True}),
        ("outcome", "维修人员已派出", {}),
    ]

    for etype, desc, ctx in events:
        ep = mem.log(etype, desc, context=ctx)
        print(f"  [{etype}] imp={ep.importance:.2f} {desc[:40]}")

    print(f"\nBuffer: {mem.buffer_size}, cumulative_imp={mem.cumulative_importance:.2f}")
    print(f"Should reflect: {mem.should_reflect()}")

    # Reflect
    print("\n=== Reflecting... ===")
    try:
        result = await mem.reflect(force=True)
        if result:
            print(f"Summary: {result[:200]}")
        else:
            print("Reflection returned None")
    except Exception as e:
        print(f"Reflect exception: {type(e).__name__}: {e}")

    # Check outputs
    knowledge_dir = os.path.join(test_data, "memory", "knowledge")
    if os.path.isdir(knowledge_dir):
        for kf in sorted(os.listdir(knowledge_dir)):
            path = os.path.join(knowledge_dir, kf)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"\n--- knowledge/{kf} ---")
            print(content[:300])

    digests_dir = os.path.join(test_data, "memory", "digests")
    if os.path.isdir(digests_dir):
        for df in sorted(os.listdir(digests_dir)):
            path = os.path.join(digests_dir, df)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"\n--- digest/{df} ---")
            print(content[:400])

    print(f"\nPost-reflect buffer: {mem.buffer_size}")

    # Cleanup
    shutil.rmtree(test_data, ignore_errors=True)
    cfg.get_config = original_get
    print("Done!")


asyncio.run(main())
