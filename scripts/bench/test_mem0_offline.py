#!/usr/bin/env python3
"""Test Mem0 in fully offline mode (infer=False, no LLM calls)."""
import os
import shutil
import time
from mem0 import Memory

# Clean slate
shutil.rmtree("/tmp/mem0_offline", ignore_errors=True)

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "askme_offline",
            "path": "/tmp/mem0_offline",
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {
            "model": "BAAI/bge-small-en-v1.5",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": "not-needed-offline",
            "openai_base_url": "http://localhost:1/fake",
            "model": "unused",
        },
    },
}

m = Memory.from_config(config)
print("Init OK - LLM is fake, will not be called")

# infer=False skips LLM, directly embeds + stores
entries = [
    "仓库A温度传感器报警，需要检查",
    "仓库B正常无异常",
    "温度传感器校准完成恢复正常",
    "3号巡检点发现漏水需要维修",
    "操作员要求增加仓库A巡检频率",
]

print("\n=== Learn ===")
for text in entries:
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": text}]
    r = m.add(msgs, user_id="robot", infer=False)
    ms = (time.perf_counter() - t0) * 1000
    n = len(r.get("results", []))
    print(f"  {ms:.0f}ms: stored={n} | {text[:30]}")

print("\n=== Search ===")
for q in ["温度", "漏水", "巡检", "仓库A怎么了", "异常"]:
    t0 = time.perf_counter()
    results = m.search(q, user_id="robot")
    ms = (time.perf_counter() - t0) * 1000
    items = results.get("results", [])
    print(f"\n  [{q}] {ms:.0f}ms, {len(items)} results")
    for r in items[:3]:
        mem = r.get("memory", "?")
        score = r.get("score", 0)
        print(f"    [{score:.3f}] {mem[:50]}")

print("\nMem0 OFFLINE E2E passed!")
