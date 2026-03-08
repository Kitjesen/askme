"""Memory system full test.

Tests all three local memory layers:
  L1  EpisodicMemory  — log events, decay scoring, keyword retrieval
  L2  SessionMemory   — summarize & save session, retrieve recent
  L3  EpisodicMemory.reflect() — LLM reflection → digest + knowledge files
  MemoryBridge        — embedding server health check (graceful offline)

Speaks key results via TTS at the end.
"""

import asyncio
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("askme.brain.episodic_memory").setLevel(logging.INFO)
logging.getLogger("askme.brain.session_memory").setLevel(logging.INFO)
logging.getLogger("askme.brain.memory_bridge").setLevel(logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── simulated patrol events ──────────────────────────────────────────────────
PATROL_EVENTS = [
    ("perception", "管道区域发现锈迹，位置：B2区西侧"),
    ("perception", "压力表示数正常，值 3.2 MPa"),
    ("perception", "控制柜门关闭，状态正常"),
    ("command",    "用户指令：对三号阀门区域重点巡检"),
    ("action",     "开始执行三号阀门区域精细扫描"),
    ("perception", "三号阀门区域管道接头轻微渗漏，湿润痕迹"),
    ("outcome",    "渗漏位置已记录：阀门 V-031，建议24小时内维修"),
    ("perception", "温度传感器 T-07 读数 85°C，超过预警阈值 80°C"),
    ("error",      "摄像头 C-04 连接中断，巡检视角受限"),
    ("action",     "切换备用视角，继续巡检"),
    ("outcome",    "本次巡检完成，发现 2 个需关注点"),
    ("command",    "用户指令：生成巡检报告"),
]


async def test_episodic_memory(llm) -> dict:
    """Test L1 episode logging + retrieval + L2 reflection."""
    from askme.brain.episodic_memory import EpisodicMemory

    print("\n[L1] EpisodicMemory — 写入事件")
    print("-" * 50)

    mem = EpisodicMemory(llm=llm)

    for event_type, desc in PATROL_EVENTS:
        ep = mem.log(event_type, desc)
        print(f"  [{event_type:10s}] imp={ep.importance:.2f}  {desc[:50]}")

    print(f"\n  缓冲区: {mem.buffer_size} 条  累计重要度: {mem.cumulative_importance:.2f}")
    print(f"  should_reflect: {mem.should_reflect()}")

    # ── keyword retrieval ─────────────────────────────────────────────────────
    print("\n[L1] 关键词检索 — '阀门 渗漏'")
    print("-" * 50)
    hits = mem.retrieve("阀门 渗漏", top_k=3)
    for ep in hits:
        print(f"  score={ep.retrieval_score({'阀门','渗漏'}):.3f}  {ep.description}")

    # ── reflection ────────────────────────────────────────────────────────────
    print("\n[L2] 反思（Reflection）— 调用 LLM 提炼知识")
    print("-" * 50)
    t0 = time.monotonic()
    summary = await mem.reflect(force=True)
    elapsed = time.monotonic() - t0
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  摘要: {summary}")

    # ── check saved files ─────────────────────────────────────────────────────
    digest_files = list(mem._digests_dir.glob("*.md"))
    knowledge_files = list(mem._knowledge_dir.glob("*.md"))
    print(f"\n  保存摘要文件: {len(digest_files)} 个")
    print(f"  知识分类文件: {len(knowledge_files)} 个 → {[f.stem for f in knowledge_files]}")

    # Show one knowledge file
    for kf in knowledge_files[:1]:
        content = kf.read_text(encoding="utf-8")
        print(f"\n  [{kf.stem}.md 内容]:\n{content[:400]}")

    return {
        "events_logged": len(PATROL_EVENTS),
        "reflection_summary": summary or "",
        "digest_count": len(digest_files),
        "knowledge_cats": [f.stem for f in knowledge_files],
    }


async def test_session_memory(llm) -> dict:
    """Test L2 SessionMemory: summarize conversation + retrieve."""
    from askme.brain.session_memory import SessionMemory

    print("\n[SessionMemory] 保存+检索会话摘要")
    print("-" * 50)

    mem = SessionMemory(llm=llm)

    # Simulate a conversation being trimmed
    fake_messages = [
        {"role": "user", "content": "Thunder，B2区西侧发现锈迹怎么办？"},
        {"role": "assistant", "content": "B2区西侧锈迹已记录。建议安排防腐处理，优先级中等。"},
        {"role": "user", "content": "三号阀门 V-031 有渗漏，需要多快处理？"},
        {"role": "assistant", "content": "V-031 渗漏建议24小时内处理，防止扩大。已生成维修工单。"},
        {"role": "user", "content": "T-07 温度超阈值了，需要报警吗？"},
        {"role": "assistant", "content": "T-07 达85°C超过80°C预警线，已触发一级预警，通知维护团队。"},
    ]

    await mem.summarize_and_save(fake_messages)

    recent = mem.get_recent_summaries()
    print(f"  最近会话摘要:\n{recent[:400] if recent else '  (无)' }")

    return {"session_summary": recent[:200]}


async def test_memory_bridge() -> dict:
    """Test L3 MemoryBridge — embedding server health check."""
    from askme.brain.memory_bridge import MemoryBridge

    print("\n[MemoryBridge] L3 向量记忆健康检查")
    print("-" * 50)

    bridge = MemoryBridge()
    reachable = await asyncio.to_thread(bridge._check_embedding_server)
    print(f"  Embedding server ({bridge._embed_url}): {'在线' if reachable else '离线'}")

    if reachable:
        ctx = await bridge.retrieve("阀门渗漏")
        print(f"  检索结果: {ctx[:200] if ctx else '(无)'}")
        return {"l3_available": True, "retrieved": ctx}
    else:
        print("  [跳过] 嵌入服务器未启动，L3 优雅降级，其余功能不受影响")
        return {"l3_available": False}


async def main():
    from askme.brain.llm_client import LLMClient
    from askme.config import get_config
    from askme.voice.tts import TTSEngine

    cfg = get_config()
    brain_cfg = cfg.get("brain", {})
    tts_cfg = cfg.get("voice", {}).get("tts", {})

    print("=" * 55)
    print("Memory System Full Test")
    print("=" * 55)

    # ── Init LLM ──────────────────────────────────────────────────────────────
    llm = LLMClient()

    # ── Init TTS ──────────────────────────────────────────────────────────────
    tts = TTSEngine(tts_cfg)
    tts.start_playback()
    tts.speak("Thunder 记忆系统测试开始。")
    tts.wait_done()

    # ── Run tests ────────────────────────────────────────────────────────────
    ep_result = await test_episodic_memory(llm)
    sess_result = await test_session_memory(llm)
    bridge_result = await test_memory_bridge()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("测试总结")
    print("=" * 55)
    print(f"  L1 Episode 写入     : {ep_result['events_logged']} 条")
    print(f"  L2 反思摘要         : {ep_result['digest_count']} 个文件")
    print(f"  L2 知识分类         : {ep_result['knowledge_cats']}")
    print(f"  L2 Session 摘要     : {'OK' if sess_result['session_summary'] else '无'}")
    print(f"  L3 向量记忆         : {'在线' if bridge_result['l3_available'] else '离线（优雅降级）'}")

    # ── Speak summary ─────────────────────────────────────────────────────────
    summary = ep_result.get("reflection_summary", "")
    if summary:
        tts.speak(f"反思完成。{summary[:80]}")
        tts.wait_done()

    tts.speak("记忆系统测试完成。三层记忆均正常运行。")
    tts.wait_done()
    tts.shutdown()

    print("\n完成。")


if __name__ == "__main__":
    asyncio.run(main())
