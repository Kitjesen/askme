"""Tests for the episodic memory system: importance, decay, reflection, knowledge."""

from __future__ import annotations

import json
import math
import time
from unittest.mock import AsyncMock


def _make_memory(tmp_path, monkeypatch):
    """Helper: create an EpisodicMemory with paths redirected to tmp_path."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {
            "app": {"data_dir": str(tmp_path / "data")},
            "memory": {"episodic": {"reflect_min_events": 5}},
        },
    )
    from askme.memory.episodic_memory import EpisodicMemory

    return EpisodicMemory()


# ── Importance Scoring ─────────────────────────────────────


def test_importance_command_high():
    """User commands get high importance."""
    from askme.memory.episodic_memory import score_importance

    score = score_importance("command", "用户说: 过来")
    assert score >= 0.7


def test_importance_error_high():
    """Errors get high importance."""
    from askme.memory.episodic_memory import score_importance

    score = score_importance("error", "电机过流")
    assert score >= 0.8


def test_importance_system_low():
    """System events get low importance."""
    from askme.memory.episodic_memory import score_importance

    score = score_importance("system", "电量 80%")
    assert score <= 0.2


def test_importance_person_detection_boost():
    """Detecting a person boosts importance."""
    from askme.memory.episodic_memory import score_importance

    base = score_importance("perception", "YOLO检测")
    boosted = score_importance("perception", "YOLO检测: person",
                               {"detections": [{"label": "person", "conf": 0.9}]})
    assert boosted > base
    assert boosted >= 0.7


def test_importance_danger_keywords():
    """Danger keywords boost importance."""
    from askme.memory.episodic_memory import score_importance

    score = score_importance("perception", "检测到危险物体")
    assert score > 0.5


def test_importance_new_entity_boost():
    """New/unknown entity keywords boost importance."""
    from askme.memory.episodic_memory import score_importance

    score = score_importance("perception", "发现新的物体")
    assert score > 0.5


def test_importance_clamped_to_one():
    """Importance never exceeds 1.0."""
    from askme.memory.episodic_memory import score_importance

    # Stack all boosts
    score = score_importance(
        "error", "新的危险person失败",
        {"detections": [{"label": "person", "conf": 1.0}, {"label": "fire", "conf": 1.0}]},
    )
    assert score <= 1.0


def test_importance_surprise_boost():
    """Surprise/novelty flag in context boosts importance."""
    from askme.memory.episodic_memory import score_importance

    base = score_importance("perception", "检测到一只猫")
    surprise = score_importance("perception", "检测到一只猫", {"surprise": True})
    assert surprise > base
    assert surprise - base >= 0.3  # surprise boost is 0.4, may be clamped


def test_importance_novel_flag_equivalent():
    """Both 'surprise' and 'novel' context flags trigger the same boost."""
    from askme.memory.episodic_memory import score_importance

    s1 = score_importance("perception", "检测到一只猫", {"surprise": True})
    s2 = score_importance("perception", "检测到一只猫", {"novel": True})
    assert s1 == s2


def test_importance_person_keyword_in_text():
    """Person keyword in description triggers person boost even without structured detections."""
    from askme.memory.episodic_memory import score_importance

    no_person = score_importance("perception", "我看到了: 2个cup")
    with_person = score_importance("perception", "我看到了: 1个person, 2个cup")
    assert with_person > no_person


def test_importance_person_chinese_keyword():
    """Chinese person keywords also trigger the boost."""
    from askme.memory.episodic_memory import score_importance

    base = score_importance("perception", "检测到物体")
    with_person = score_importance("perception", "检测到一个人在门口")
    assert with_person > base


# ── Episode: Activation & Decay ───────────────────────────


def test_episode_initial_retrievability():
    """New episode starts with retrievability ~1.0."""
    from askme.memory.episodic_memory import Episode

    ep = Episode("action", "test", importance=0.5)
    assert ep.retrievability() > 0.99
    assert ep.access_count == 0


def test_episode_ebbinghaus_decay():
    """Retrievability decays following R = e^(-t/S)."""
    from askme.memory.episodic_memory import Episode, DEFAULT_STABILITY_S

    ep = Episode("action", "test", importance=0.5)
    # After one stability period, R = e^(-1) ≈ 0.368
    future = ep.timestamp + ep.stability
    r = ep.retrievability(now=future)
    assert 0.3 < r < 0.4  # e^(-1) ≈ 0.368


def test_episode_access_doubles_stability():
    """Each access doubles stability (spacing effect)."""
    from askme.memory.episodic_memory import Episode

    ep = Episode("action", "test", importance=0.5)
    initial_s = ep.stability

    ep.access()
    assert ep.stability > initial_s * 1.9  # ~2x

    ep.access()
    assert ep.stability > initial_s * 3.9  # ~4x


def test_episode_access_improves_retrievability():
    """Accessing an old episode improves its retrievability (via stability growth)."""
    from askme.memory.episodic_memory import Episode

    ep = Episode("action", "test", importance=0.5)
    ep.timestamp -= 7200  # 2 hours ago
    ep.last_accessed = ep.timestamp

    r_before = ep.retrievability()
    ep.access()  # doubles stability + resets last_accessed
    r_after = ep.retrievability()

    assert r_after > r_before


def test_importance_affects_initial_stability():
    """Higher importance gives higher initial stability."""
    from askme.memory.episodic_memory import Episode

    ep_low = Episode("system", "heartbeat", importance=0.1)
    ep_high = Episode("error", "motor overcurrent", importance=0.8)

    assert ep_high.stability > ep_low.stability


def test_retrieval_score_with_keywords():
    """Retrieval score increases with keyword relevance."""
    from askme.memory.episodic_memory import Episode

    ep1 = Episode("perception", "检测到猫在沙发上", importance=0.3)
    ep2 = Episode("action", "巡逻走廊", importance=0.2)

    s1 = ep1.retrieval_score({"猫", "沙发"})
    s2 = ep2.retrieval_score({"猫", "沙发"})
    assert s1 > s2


# ── L1: Episode Buffer ────────────────────────────────────


def test_log_and_get_recent(tmp_path, monkeypatch):
    """Events are logged and retrievable via get_recent()."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("perception", "看到一个人", {"label": "person"})
    mem.log("action", "执行巡逻")
    mem.log("outcome", "巡逻完成")

    recent = mem.get_recent(10)
    assert len(recent) == 3
    assert recent[0].event_type == "perception"
    assert recent[0].description == "看到一个人"
    assert recent[0].context == {"label": "person"}
    assert recent[2].event_type == "outcome"


def test_log_returns_episode(tmp_path, monkeypatch):
    """log() returns the created Episode."""
    mem = _make_memory(tmp_path, monkeypatch)
    ep = mem.log("command", "用户说你好")
    assert ep.event_type == "command"
    assert ep.importance >= 0.7


def test_log_auto_scores_importance(tmp_path, monkeypatch):
    """Logged episodes have auto-scored importance."""
    mem = _make_memory(tmp_path, monkeypatch)
    ep = mem.log("command", "用户指令")
    assert ep.importance > 0.5


def test_log_override_importance(tmp_path, monkeypatch):
    """Importance can be overridden manually."""
    mem = _make_memory(tmp_path, monkeypatch)
    ep = mem.log("system", "test", importance=0.99)
    assert ep.importance == 0.99


def test_cumulative_importance_tracking(tmp_path, monkeypatch):
    """Cumulative importance is tracked across logs."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("command", "a")
    mem.log("command", "b")
    assert mem.cumulative_importance >= 1.4  # 0.7 + 0.7


def test_buffer_size_property(tmp_path, monkeypatch):
    """buffer_size reflects current buffer length."""
    mem = _make_memory(tmp_path, monkeypatch)
    assert mem.buffer_size == 0
    mem.log("action", "test")
    assert mem.buffer_size == 1


def test_retrieve_relevance(tmp_path, monkeypatch):
    """retrieve() returns episodes sorted by relevance."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("perception", "检测到猫在客厅")
    mem.log("action", "巡逻走廊")
    mem.log("perception", "客厅有一只猫在睡觉")

    results = mem.retrieve("猫", top_k=2)
    assert len(results) == 2
    # Both cat-related should rank higher
    assert "猫" in results[0].description


def test_episode_to_log_line():
    """Episode.to_log_line() produces readable format with importance."""
    from askme.memory.episodic_memory import Episode

    ep = Episode("perception", "检测到障碍物", {"distance": 1.5}, importance=0.6)
    line = ep.to_log_line()
    assert "[perception]" in line
    assert "检测到障碍物" in line
    assert "imp=0.6" in line


def test_episode_to_dict():
    """Episode.to_dict() contains all fields including importance and stability."""
    from askme.memory.episodic_memory import Episode

    ep = Episode("action", "移动到A点", importance=0.4)
    d = ep.to_dict()
    assert d["type"] == "action"
    assert d["desc"] == "移动到A点"
    assert d["importance"] == 0.4
    assert "stability" in d
    assert "retrievability" in d
    assert "ts" in d


def test_flush_to_disk(tmp_path, monkeypatch):
    """Buffer is flushed to JSONL on disk when threshold reached."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {
            "app": {"data_dir": str(tmp_path / "data")},
            "memory": {"episodic": {"reflect_min_events": 5}},
        },
    )
    monkeypatch.setattr("askme.memory.episodic_memory.FLUSH_THRESHOLD", 3)

    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory()
    mem.log("a", "one")
    mem.log("b", "two")
    mem.log("c", "three")

    jsonl_files = list((tmp_path / "data" / "memory" / "episodes").glob("*.jsonl"))
    assert len(jsonl_files) >= 1
    content = jsonl_files[0].read_text(encoding="utf-8")
    lines = [l for l in content.strip().split("\n") if l]
    assert len(lines) == 3


# ── L2: Reflection ────────────────────────────────────────


def test_should_reflect_importance_trigger(tmp_path, monkeypatch):
    """should_reflect triggers on cumulative importance."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )
    monkeypatch.setattr("askme.memory.episodic_memory.IMPORTANCE_THRESHOLD", 2.0)
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_MIN_EVENTS", 100)

    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory()
    # Log high-importance events
    mem.log("command", "a")  # ~0.7
    mem.log("command", "b")  # ~0.7
    assert mem.should_reflect() is False  # ~1.4 < 2.0
    mem.log("command", "c")  # ~0.7 → total ~2.1
    assert mem.should_reflect() is True


def test_should_reflect_cooldown(tmp_path, monkeypatch):
    """should_reflect respects cooldown period."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )
    monkeypatch.setattr("askme.memory.episodic_memory.REFLECT_MIN_EVENTS", 2)

    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory()
    mem.log("a", "one")
    mem.log("b", "two")
    assert mem.should_reflect() is True

    mem._last_reflect_time = time.time()
    assert mem.should_reflect() is False


def test_should_reflect_min_events(tmp_path, monkeypatch):
    """should_reflect requires minimum events."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("a", "one")
    assert mem.should_reflect() is False


async def test_reflect_no_llm(tmp_path, monkeypatch):
    """reflect() returns None when no LLM is configured."""
    mem = _make_memory(tmp_path, monkeypatch)
    for i in range(15):
        mem.log("a", f"event {i}")
    result = await mem.reflect(force=True)
    assert result is None


async def test_reflect_resets_cumulative_importance(tmp_path, monkeypatch):
    """After reflection, cumulative importance resets to 0."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.episodic_memory import EpisodicMemory

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = json.dumps({
        "summary": "test",
        "new_facts": [],
        "patterns": [],
        "updates": [],
        "importance": "low",
    })
    mem = EpisodicMemory(llm=mock_llm)
    for i in range(15):
        mem.log("command", f"event {i}")

    assert mem.cumulative_importance > 0
    await mem.reflect(force=True)
    assert mem.cumulative_importance == 0.0


async def test_reflect_with_mock_llm(tmp_path, monkeypatch):
    """reflect() processes LLM response and saves digest + knowledge."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.episodic_memory import EpisodicMemory

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = json.dumps({
        "summary": "机器人执行了巡逻任务并发现了一个新区域",
        "new_facts": [
            {"fact": "走廊尽头有一扇门", "category": "environment"},
            {"fact": "家里有一只橘猫", "category": "entities"},
        ],
        "patterns": [
            {"pattern": "每天下午3点走廊比较安静", "category": "routines", "confidence": 0.7},
        ],
        "updates": [],
        "importance": "medium",
    })

    mem = EpisodicMemory(llm=mock_llm)
    for i in range(15):
        mem.log("action", f"巡逻步骤 {i}")

    result = await mem.reflect(force=True)
    assert result is not None
    assert "巡逻" in result

    # Check categorized knowledge files
    env_file = tmp_path / "data" / "memory" / "knowledge" / "environment.md"
    assert env_file.exists()
    assert "走廊尽头有一扇门" in env_file.read_text(encoding="utf-8")

    ent_file = tmp_path / "data" / "memory" / "knowledge" / "entities.md"
    assert ent_file.exists()
    assert "橘猫" in ent_file.read_text(encoding="utf-8")

    # Patterns go to routines
    routines_file = tmp_path / "data" / "memory" / "knowledge" / "routines.md"
    assert routines_file.exists()
    assert "下午3点" in routines_file.read_text(encoding="utf-8")

    # Digest saved
    digests = list((tmp_path / "data" / "memory" / "digests").glob("*.md"))
    assert len(digests) == 1

    assert mem.buffer_size == 0


async def test_reflect_failure_does_not_start_cooldown(tmp_path, monkeypatch):
    """A failed reflection should be retriable immediately."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {
            "app": {"data_dir": str(tmp_path / "data")},
            "memory": {"episodic": {"reflect_min_events": 5}},
        },
    )

    from askme.memory.episodic_memory import EpisodicMemory

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "not json"
    mem = EpisodicMemory(llm=mock_llm)
    for i in range(6):
        mem.log("command", f"event {i}")

    before = mem._last_reflect_time  # noqa: SLF001
    result = await mem.reflect(force=True)
    assert result is None
    assert mem._last_reflect_time == before  # noqa: SLF001
    assert mem.should_reflect() is True


def test_restore_active_journal_on_restart(tmp_path, monkeypatch):
    """Unreflected episodes survive a restart via the active journal."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("command", "first")
    mem.log("action", "second")

    restored = _make_memory(tmp_path, monkeypatch)
    recent = restored.get_recent(10)
    assert len(recent) == 2
    assert recent[0].description == "first"
    assert restored.cumulative_importance > 0


# ── L3: World Knowledge ──────────────────────────────────


def test_get_knowledge_context_empty(tmp_path, monkeypatch):
    """Empty knowledge returns empty string."""
    mem = _make_memory(tmp_path, monkeypatch)
    assert mem.get_knowledge_context() == ""


def test_get_knowledge_context_with_data(tmp_path, monkeypatch):
    """Knowledge context is returned when files exist."""
    mem = _make_memory(tmp_path, monkeypatch)
    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "environment.md").write_text(
        "# environment: 环境布局\n- 走廊长10米", encoding="utf-8"
    )
    ctx = mem.get_knowledge_context()
    assert "世界知识" in ctx
    assert "走廊长10米" in ctx


def test_get_knowledge_context_truncation(tmp_path, monkeypatch):
    """Long knowledge is truncated to max_chars."""
    mem = _make_memory(tmp_path, monkeypatch)
    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "general.md").write_text("x" * 2000, encoding="utf-8")
    ctx = mem.get_knowledge_context(max_chars=100)
    assert len(ctx) < 200
    assert "..." in ctx


def test_get_recent_digest(tmp_path, monkeypatch):
    """Recent digests are loaded for system prompt."""
    mem = _make_memory(tmp_path, monkeypatch)
    digests_dir = tmp_path / "data" / "memory" / "digests"
    (digests_dir / "2026-03-08_120000.md").write_text(
        "## 2026-03-08 12:00 [medium]\n巡逻完成", encoding="utf-8"
    )
    digest = mem.get_recent_digest()
    assert "近期经历" in digest
    assert "巡逻完成" in digest


def test_get_recent_digest_empty(tmp_path, monkeypatch):
    """No digests returns empty string."""
    mem = _make_memory(tmp_path, monkeypatch)
    assert mem.get_recent_digest() == ""


def test_get_relevant_context_returns_ranked_episodes(tmp_path, monkeypatch):
    """Relevant episodic context is query-aware and formatted for prompt injection."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem.log("perception", "检测到猫在走廊")
    mem.log("action", "执行巡逻")
    ctx = mem.get_relevant_context("猫", top_k=1)
    assert "Relevant episodes" in ctx
    assert "猫" in ctx


# ── Knowledge Updates ─────────────────────────────────────


def test_update_knowledge_categorized(tmp_path, monkeypatch):
    """Knowledge updates go to correct category files."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem._update_knowledge({
        "new_facts": [
            {"fact": "客厅面积20平米", "category": "environment"},
            {"fact": "主人叫森哥", "category": "entities"},
        ],
        "patterns": [],
        "updates": [],
    })

    env = (tmp_path / "data" / "memory" / "knowledge" / "environment.md").read_text(encoding="utf-8")
    assert "客厅面积20平米" in env

    ent = (tmp_path / "data" / "memory" / "knowledge" / "entities.md").read_text(encoding="utf-8")
    assert "森哥" in ent


def test_update_knowledge_replaces_old_facts(tmp_path, monkeypatch):
    """_update_knowledge handles UPDATE operations within categories."""
    mem = _make_memory(tmp_path, monkeypatch)
    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "environment.md").write_text(
        "# environment: 环境\n- 门是开着的", encoding="utf-8"
    )
    mem._update_knowledge({
        "new_facts": [],
        "patterns": [],
        "updates": [{"old": "门是开着的", "new": "门是关着的", "category": "environment"}],
    })
    content = (knowledge_dir / "environment.md").read_text(encoding="utf-8")
    assert "门是关着的" in content
    assert "门是开着的" not in content


def test_update_knowledge_deduplicates(tmp_path, monkeypatch):
    """New facts that already exist are not duplicated."""
    mem = _make_memory(tmp_path, monkeypatch)
    knowledge_dir = tmp_path / "data" / "memory" / "knowledge"
    (knowledge_dir / "general.md").write_text(
        "# general\n- 走廊长10米", encoding="utf-8"
    )
    mem._update_knowledge({
        "new_facts": [{"fact": "走廊长10米", "category": "general"}],
        "patterns": [],
        "updates": [],
    })
    content = (knowledge_dir / "general.md").read_text(encoding="utf-8")
    assert content.count("走廊长10米") == 1


def test_update_knowledge_fallback_plain_strings(tmp_path, monkeypatch):
    """Plain string facts (not dicts) go to general category."""
    mem = _make_memory(tmp_path, monkeypatch)
    mem._update_knowledge({
        "new_facts": ["这是一个普通事实"],
        "patterns": ["这是一个规律"],
        "updates": [],
    })
    general = tmp_path / "data" / "memory" / "knowledge" / "general.md"
    assert general.exists()
    assert "普通事实" in general.read_text(encoding="utf-8")

    routines = tmp_path / "data" / "memory" / "knowledge" / "routines.md"
    assert routines.exists()
    assert "规律" in routines.read_text(encoding="utf-8")


# ── Cleanup ───────────────────────────────────────────────


def test_cleanup_old_episodes(tmp_path, monkeypatch):
    """Old episode files are removed by cleanup."""
    mem = _make_memory(tmp_path, monkeypatch)
    episodes_dir = tmp_path / "data" / "memory" / "episodes"

    old_file = episodes_dir / "old.jsonl"
    old_file.write_text("{}\n", encoding="utf-8")
    import os
    old_mtime = time.time() - 48 * 3600
    os.utime(old_file, (old_mtime, old_mtime))

    new_file = episodes_dir / "new.jsonl"
    new_file.write_text("{}\n", encoding="utf-8")

    removed = mem.cleanup_old_episodes()
    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()


# ── Parse Reflection ──────────────────────────────────────


def test_parse_reflection_valid_json():
    """_parse_reflection extracts JSON from LLM response."""
    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory.__new__(EpisodicMemory)
    result = mem._parse_reflection(
        '好的，这是分析结果：\n{"summary": "测试", "new_facts": [], "patterns": [], "updates": [], "importance": "low"}'
    )
    assert result is not None
    assert result["summary"] == "测试"


def test_parse_reflection_invalid():
    """_parse_reflection returns None for invalid input."""
    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory.__new__(EpisodicMemory)
    assert mem._parse_reflection("no json here") is None
