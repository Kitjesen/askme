#!/usr/bin/env python3
"""Test Mem0 integration on S100P with real API key."""
import os
import sys
import time
sys.path.insert(0, ".")

from askme.config import get_config

cfg = get_config()
brain = cfg.get("brain", {})
api_key = brain.get("api_key") or os.environ.get("LLM_API_KEY", "")
base_url = brain.get("base_url") or os.environ.get("LLM_BASE_URL", "")
model = "claude-haiku-4-5-20251001"  # Haiku is more stable on relay

print(f"API key: {api_key[:10]}...")
print(f"Base URL: {base_url}")
print(f"Model: {model}")

from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "askme", "path": "data/memory/mem0_store"},
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": api_key,
            "openai_base_url": base_url,
            "model": model,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": api_key,
            "openai_base_url": base_url,
            "model": "text-embedding-3-small",
        },
    },
}

print("\nCreating Mem0...")
t0 = time.perf_counter()
m = Memory.from_config(config)
print(f"Created in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n=== Test 1: Add memory ===")
t0 = time.perf_counter()
result = m.add("仓库A的温度传感器今天又报警了，这已经是本周第三次了", user_id="operator")
print(f"Add: {result}")
print(f"Time: {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n=== Test 2: Add another ===")
t0 = time.perf_counter()
result = m.add("仓库B一切正常，上午巡检没有发现异常", user_id="operator")
print(f"Add: {result}")
print(f"Time: {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n=== Test 3: Search ===")
t0 = time.perf_counter()
results = m.search("仓库A有什么问题", user_id="operator")
print(f"Search results: {results}")
print(f"Time: {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n=== Test 4: Get all ===")
all_memories = m.get_all(user_id="operator")
print(f"Total memories: {len(all_memories.get('results', []))}")
for mem in all_memories.get("results", []):
    print(f"  - {mem.get('memory', '')}")

print("\n=== Mem0 test complete ===")
