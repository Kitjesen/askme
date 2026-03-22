"""Integration test: simulate a robot dog's day and verify episodic memory behavior.

This test simulates realistic robot events (YOLO detections, patrol actions,
user commands, etc.) flowing through the episodic memory system, triggering
reflection, and building categorized world knowledge.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest


def _make_memory_with_llm(tmp_path, monkeypatch, reflection_response=None):
    """Create an EpisodicMemory with mock LLM and paths in tmp_path."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_MIN_EVENTS", 5)
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_COOLDOWN_S", 0)
    monkeypatch.setattr("askme.memory.episodic_memory.IMPORTANCE_THRESHOLD", 1.0)

    from askme.brain.episodic_memory import EpisodicMemory

    mock_llm = AsyncMock()
    if reflection_response is None:
        reflection_response = json.dumps({
            "summary": "机器人完成了客厅到走廊的巡逻，检测到主人和一只猫",
            "new_facts": [
                {"fact": "客厅有一张沙发和一台电视", "category": "environment"},
                {"fact": "走廊尽头有一扇关着的门", "category": "environment"},
                {"fact": "主人通常在下午出现在客厅", "category": "entities"},
                {"fact": "家里有一只橘猫", "category": "entities"},
            ],
            "patterns": [
                {"pattern": "主人每次出现时都会对机器人说你好", "category": "interactions", "confidence": 0.8},
                {"pattern": "猫看到机器人会跑开", "category": "entities", "confidence": 0.6},
            ],
            "updates": [],
            "importance": "high",
        })
    mock_llm.chat.return_value = reflection_response

    mem = EpisodicMemory(llm=mock_llm)
    return mem, mock_llm


# ── Scenario: A robot dog's afternoon patrol ──────────────

PATROL_EVENTS = [
    ("perception", "YOLO检测: 沙发(0.95), 电视(0.91)", {"detections": [{"label": "sofa", "conf": 0.95}, {"label": "tv", "conf": 0.91}]}),
    ("action", "开始巡逻路径A: 客厅到走廊", {"path": "A", "start": "living_room"}),
    ("perception", "YOLO检测: person(0.88)", {"detections": [{"label": "person", "conf": 0.88}]}),
    ("command", "用户说: 你好，过来", {"source": "voice"}),
    ("action", "移动到用户位置", {"target": [1.2, 0.5, 0.0]}),
    ("outcome", "到达用户位置，距离0.3m", {"distance": 0.3}),
    ("perception", "YOLO检测: cat(0.82)", {"detections": [{"label": "cat", "conf": 0.82}]}),
    ("action", "继续巡逻: 走廊段", {"path": "A", "segment": "hallway"}),
    ("perception", "检测到走廊尽头有门(关闭)", {"type": "door", "state": "closed"}),
    ("outcome", "巡逻路径A完成", {"duration_s": 120, "detections_total": 4}),
]


async def test_full_patrol_scenario(tmp_path, monkeypatch):
    """Simulate a patrol: log events -> trigger reflection -> verify categorized knowledge."""
    mem, mock_llm = _make_memory_with_llm(tmp_path, monkeypatch)

    # Log all patrol events
    for event_type, desc, ctx in PATROL_EVENTS:
        ep = mem.log(event_type, desc, ctx)

    assert mem.buffer_size == 10

    # Verify importance scoring: person detection should be higher than routine action
    recent = mem.get_recent(10)
    person_ep = next(ep for ep in recent if "person" in ep.description)
    action_ep = next(ep for ep in recent if "开始巡逻" in ep.description)
    assert person_ep.importance > action_ep.importance

    # Command should have highest importance
    cmd_ep = next(ep for ep in recent if ep.event_type == "command")
    assert cmd_ep.importance >= 0.7

    # Trigger reflection
    summary = await mem.reflect(force=True)
    assert summary is not None
    assert "巡逻" in summary

    # Verify categorized knowledge
    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"

    env = (knowledge_dir / "environment.md").read_text(encoding="utf-8")
    assert "沙发" in env
    assert "门" in env

    ent = (knowledge_dir / "entities.md").read_text(encoding="utf-8")
    assert "橘猫" in ent

    # Patterns go to categorized files
    interactions = (knowledge_dir / "interactions.md").read_text(encoding="utf-8")
    assert "说你好" in interactions

    # Digest saved
    digests = list((tmp_path / "data" / "memory" / "digests").glob("*.md"))
    assert len(digests) == 1
    digest = digests[0].read_text(encoding="utf-8")
    assert "[high]" in digest

    assert mem.buffer_size == 0
    assert mem.cumulative_importance == 0.0


async def test_importance_based_reflection_trigger(tmp_path, monkeypatch):
    """Reflection triggers based on cumulative importance, not just count."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_COOLDOWN_S", 0)
    monkeypatch.setattr("askme.memory.episodic_memory.IMPORTANCE_THRESHOLD", 3.0)
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_MIN_EVENTS", 100)

    from askme.brain.episodic_memory import EpisodicMemory

    mem = EpisodicMemory()

    # Low importance events don't trigger reflection
    for i in range(20):
        mem.log("system", f"heartbeat {i}")  # importance ~0.1 each
    assert mem.should_reflect() is False  # ~2.0 < 3.0

    # High importance event pushes over threshold
    mem.log("error", "电机过流警告")  # ~0.8
    mem.log("command", "紧急停止")    # ~0.7
    assert mem.should_reflect() is True


async def test_knowledge_accumulates_across_reflections(tmp_path, monkeypatch):
    """Multiple reflection cycles build up categorized world knowledge."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_MIN_EVENTS", 3)
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_COOLDOWN_S", 0)

    from askme.brain.episodic_memory import EpisodicMemory

    mock_llm = AsyncMock()
    mem = EpisodicMemory(llm=mock_llm)

    # Reflection 1: morning patrol
    mock_llm.chat.return_value = json.dumps({
        "summary": "早晨巡逻",
        "new_facts": [
            {"fact": "厨房有冰箱", "category": "environment"},
            {"fact": "阳台有植物", "category": "environment"},
        ],
        "patterns": [],
        "updates": [],
        "importance": "medium",
    })
    for i in range(5):
        mem.log("action", f"早晨事件{i}")
    await mem.reflect(force=True)

    # Reflection 2: afternoon patrol (updates a fact)
    mock_llm.chat.return_value = json.dumps({
        "summary": "下午巡逻",
        "new_facts": [
            {"fact": "卧室有书桌", "category": "environment"},
        ],
        "patterns": [
            {"pattern": "主人下午在书桌前工作", "category": "routines", "confidence": 0.7},
        ],
        "updates": [{"old": "阳台有植物", "new": "阳台有三盆多肉植物", "category": "environment"}],
        "importance": "medium",
    })
    for i in range(5):
        mem.log("perception", f"下午事件{i}")
    await mem.reflect(force=True)

    # Verify accumulated knowledge
    env_file = tmp_path / "data" / "memory" / "knowledge" / "environment.md"
    env = env_file.read_text(encoding="utf-8")
    assert "厨房有冰箱" in env
    assert "三盆多肉植物" in env
    assert "阳台有植物" not in env  # replaced
    assert "卧室有书桌" in env

    # Routines got pattern
    routines = (tmp_path / "data" / "memory" / "knowledge" / "routines.md").read_text(encoding="utf-8")
    assert "书桌前工作" in routines

    # Two digests
    digests = list((tmp_path / "data" / "memory" / "digests").glob("*.md"))
    assert len(digests) == 2


async def test_retrieval_with_decay(tmp_path, monkeypatch):
    """Recent episodes score higher than old ones in retrieval."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    # Log an old event
    old_ep = mem.log("perception", "检测到猫在客厅")
    old_ep.timestamp -= 7 * 24 * 3600  # 7 days ago
    old_ep.last_accessed = old_ep.timestamp

    # Log a recent event
    mem.log("perception", "检测到猫在阳台")

    results = mem.retrieve("猫", top_k=2)
    assert len(results) == 2
    # Recent one should rank first (higher activation)
    assert "阳台" in results[0].description


async def test_system_prompt_injection(tmp_path, monkeypatch):
    """Knowledge and digests are injected into system prompt context."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "environment.md").write_text(
        "# environment: 环境布局\n- 客厅面积约20平米",
        encoding="utf-8",
    )
    (knowledge_dir / "entities.md").write_text(
        "# entities: 识别的人、动物、物体\n- 家里有两个人和一只猫",
        encoding="utf-8",
    )

    digests_dir = tmp_path / "data" / "memory" / "digests"
    (digests_dir / "2026-03-08_140000.md").write_text(
        "## 2026-03-08 14:00 [high]\n机器人完成了下午巡逻",
        encoding="utf-8",
    )

    ctx = mem.get_knowledge_context()
    assert "世界知识" in ctx
    assert "20平米" in ctx
    assert "猫" in ctx

    digest = mem.get_recent_digest()
    assert "近期经历" in digest
    assert "下午巡逻" in digest


async def test_concurrent_logging_safety(tmp_path, monkeypatch):
    """Concurrent logging doesn't corrupt the buffer."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    async def log_events(prefix: str, count: int):
        for i in range(count):
            mem.log("action", f"{prefix}_{i}")
            await asyncio.sleep(0)

    await asyncio.gather(
        log_events("yolo", 20),
        log_events("nav", 20),
        log_events("cmd", 10),
    )
    assert mem.buffer_size == 50


async def test_reflection_with_existing_knowledge(tmp_path, monkeypatch):
    """Reflection incorporates existing knowledge in prompt."""
    mem, mock_llm = _make_memory_with_llm(tmp_path, monkeypatch)

    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "environment.md").write_text(
        "# environment\n- 客厅有沙发", encoding="utf-8"
    )
    (knowledge_dir / "entities.md").write_text(
        "# entities\n- 主人养了一只猫", encoding="utf-8"
    )

    for i in range(6):
        mem.log("action", f"事件{i}")
    await mem.reflect(force=True)

    call_args = mock_llm.chat.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "客厅有沙发" in user_content
    assert "主人养了一只猫" in user_content


async def test_event_types_importance_ordering(tmp_path, monkeypatch):
    """Different event types produce expected importance ordering."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    events = {
        "system": mem.log("system", "电量80%"),
        "action": mem.log("action", "移动到A点"),
        "perception": mem.log("perception", "YOLO检测物体"),
        "command": mem.log("command", "用户说停下"),
        "error": mem.log("error", "电机过流"),
    }

    # Expected ordering: error > command > perception >= action > system
    assert events["error"].importance > events["command"].importance
    assert events["command"].importance > events["perception"].importance
    assert events["perception"].importance >= events["action"].importance
    assert events["action"].importance > events["system"].importance


async def test_vision_perception_logging(tmp_path, monkeypatch):
    """Vision scene descriptions get logged as perception episodes."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    # Simulate what BrainPipeline does: log a vision scene description
    scene = "我看到了: 1个person, 2个cup"
    ep = mem.log("perception", scene)

    assert ep.event_type == "perception"
    assert "person" in ep.description
    assert ep.importance > 0  # base perception importance

    recent = mem.get_recent(5)
    assert len(recent) == 1
    assert recent[0].description == scene


async def test_surprise_boosts_perception_importance(tmp_path, monkeypatch):
    """Surprise/novel context flag gives perception events higher importance."""
    mem, _ = _make_memory_with_llm(tmp_path, monkeypatch)

    normal_ep = mem.log("perception", "检测到杯子")
    surprise_ep = mem.log("perception", "检测到杯子", context={"surprise": True})

    assert surprise_ep.importance > normal_ep.importance
