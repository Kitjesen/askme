#!/usr/bin/env python3
"""Test L5 Semantic Index on S100P — indexes L3 knowledge, unified search."""
import asyncio
import os
import sys
import shutil
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


async def main():
    test_data = f"/tmp/askme_l5_{uuid.uuid4().hex[:8]}"

    import askme.config as cfg
    original_get = cfg.get_config

    def patched():
        c = original_get()
        c["app"] = {"data_dir": test_data}
        c["memory"] = c.get("memory", {})
        c["memory"]["episodic"] = {
            "reflect_min_events": 1, "importance_threshold": 0.1,
            "reflect_cooldown_seconds": 0, "admission_threshold": 0.0,
        }
        return c

    cfg.get_config = patched

    from openai import AsyncOpenAI
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.semantic_index import SemanticIndex

    # LLM client
    client = AsyncOpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.chat/v1",
    )

    class FakeLLM:
        async def chat(self, messages, **kw):
            resp = await client.chat.completions.create(
                model="MiniMax-M2.7-highspeed", messages=messages,
            )
            return resp.choices[0].message.content or ""

    # Step 1: Create episodic memory and log events
    llm = FakeLLM()
    mem = EpisodicMemory(llm=llm)
    print(f"EpisodicMemory created, buffer={mem.buffer_size}")

    events = [
        ("perception", "仓库A温度传感器读数38.5度异常", {"sensor": "temp_A"}),
        ("action", "向操作员报告温度异常", {}),
        ("perception", "3号巡检点发现地面积水", {"label": "water"}),
        ("outcome", "维修人员已派出处理积水", {}),
        ("perception", "检测到操作员张三进入仓库B", {"label": "person"}),
    ]
    for etype, desc, ctx in events:
        mem.log(etype, desc, context=ctx)
    print(f"Logged {mem.buffer_size} events")

    # Step 2: Reflect to generate L3 knowledge
    print("\n=== L3 Reflect ===")
    result = await mem.reflect(force=True)
    print(f"Summary: {(result or 'None')[:100]}")

    # Step 3: Create L5 and sync knowledge
    print("\n=== L5 Sync ===")
    idx = SemanticIndex()
    n = await idx.sync(episodic=mem)
    print(f"Indexed {n} entries from L3")

    # Step 4: Search
    print("\n=== L5 Search ===")
    for q in ["温度异常", "漏水维修", "操作员", "巡检"]:
        results = await idx.search(q, n=5)
        print(f"\n  [{q}]: {len(results)} results")
        for r in results[:3]:
            src = r.get("source", "?")
            cat = r.get("category", "")
            score = r.get("score", 0)
            text = r.get("text", "")[:60]
            print(f"    [{src}/{cat}] score={score:.3f} {text}")

    # Step 5: Filter by source
    print("\n=== L5 Search (knowledge only) ===")
    results = await idx.search("温度", n=5, source_filter="knowledge")
    for r in results[:3]:
        print(f"  [{r.get('category','')}] {r.get('text','')[:60]}")

    idx.close()
    shutil.rmtree(test_data, ignore_errors=True)
    cfg.get_config = original_get
    print("\nL5 Semantic Index test passed!")


asyncio.run(main())
