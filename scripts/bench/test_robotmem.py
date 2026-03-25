#!/usr/bin/env python3
"""Test RobotMem on S100P."""
import time
from robotmem import RobotMemory

mem = RobotMemory(collection="askme_test7", embed_backend="onnx")
print("Created (hybrid mode: BM25 + vector)")

print("\n=== Learn ===")
entries = [
    ("仓库A温度传感器报警", {"location": "仓库A"}),
    ("仓库B正常无异常", {"location": "仓库B"}),
    ("温度传感器校准完成恢复正常", {"location": "仓库A"}),
    ("3号巡检点发现漏水", {"location": "3号点"}),
    ("操作员要求增加仓库A巡检频率", {}),
]
for text, ctx in entries:
    t0 = time.perf_counter()
    mem.learn(text, context=ctx)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {ms:.0f}ms: {text[:30]}")

queries = ["温度", "仓库A", "异常", "巡检"]
for q in queries:
    print(f"\n=== Recall: {q} ===")
    t0 = time.perf_counter()
    results = mem.recall(q, n=5, min_confidence=0.0)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {ms:.0f}ms, {len(results)} results")
    for r in results:
        content = r.get("content", "")
        conf = r.get("confidence", 0)
        rrf = r.get("_rrf_score", 0)
        print(f"  [{conf:.2f}|rrf={rrf:.3f}] {content[:50]}")

print("\nDone!")
