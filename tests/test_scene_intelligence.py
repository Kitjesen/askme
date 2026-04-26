"""Tests for SceneIntelligence — unified scene-awareness API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from askme.perception.scene_intelligence import SceneIntelligence

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_episodic(
    world_knowledge: dict | None = None,
    buffer: list | None = None,
    knowledge_context: str = "",
    recent_digest: str = "",
    relevant_context: str = "",
) -> MagicMock:
    ep = MagicMock()
    ep._world_knowledge = world_knowledge or {}
    ep._buffer = buffer or []
    ep.get_knowledge_context.return_value = knowledge_context
    ep.get_recent_digest.return_value = recent_digest
    ep.get_relevant_context.return_value = relevant_context
    return ep


def _make_episode(
    event_type: str = "command",
    importance: float = 0.0,
    content: str = "event content",
    timestamp: str = "2026-01-01T00:00:00",
) -> dict:
    return {
        "event_type": event_type,
        "importance": importance,
        "content": content,
        "timestamp": timestamp,
    }


def _make_scene(
    world_knowledge: dict | None = None,
    buffer: list | None = None,
    knowledge_context: str = "",
    recent_digest: str = "",
    session: MagicMock | None = None,
) -> SceneIntelligence:
    ep = _make_episodic(
        world_knowledge=world_knowledge,
        buffer=buffer,
        knowledge_context=knowledge_context,
        recent_digest=recent_digest,
    )
    return SceneIntelligence(episodic=ep, session=session)


# ── TestWhoIsAround ───────────────────────────────────────────────────────────

class TestWhoIsAround:
    def test_empty_world_knowledge_returns_empty(self):
        scene = _make_scene(world_knowledge={})
        assert scene.who_is_around() == []

    def test_no_entities_key_returns_empty(self):
        scene = _make_scene(world_knowledge={"location": "room A"})
        assert scene.who_is_around() == []

    def test_entities_dict_returns_sorted_keys(self):
        entities = {"charlie": {}, "alice": {}, "bob": {}}
        scene = _make_scene(world_knowledge={"entities": entities})
        result = scene.who_is_around()
        assert result == ["alice", "bob", "charlie"]

    def test_entities_non_dict_returns_empty(self):
        # If entities is a list instead of dict
        scene = _make_scene(world_knowledge={"entities": ["alice", "bob"]})
        assert scene.who_is_around() == []

    def test_single_entity(self):
        scene = _make_scene(world_knowledge={"entities": {"robot": {}}})
        assert scene.who_is_around() == ["robot"]


# ── TestAnomalies ─────────────────────────────────────────────────────────────

class TestAnomalies:
    def test_empty_buffer_returns_empty(self):
        scene = _make_scene(buffer=[])
        assert scene.anomalies() == []

    def test_low_importance_error_excluded(self):
        ep = _make_episode(event_type="error", importance=0.3)
        scene = _make_scene(buffer=[ep])
        assert scene.anomalies() == []

    def test_high_importance_error_included(self):
        ep = _make_episode(event_type="error", importance=0.8, content="disk full")
        scene = _make_scene(buffer=[ep])
        anomalies = scene.anomalies()
        assert len(anomalies) == 1
        assert anomalies[0]["description"] == "disk full"
        assert anomalies[0]["type"] == "error"

    def test_high_importance_outcome_included(self):
        ep = _make_episode(event_type="outcome", importance=0.7, content="patrol done")
        scene = _make_scene(buffer=[ep])
        anomalies = scene.anomalies()
        assert len(anomalies) == 1

    def test_command_type_excluded_even_if_important(self):
        ep = _make_episode(event_type="command", importance=0.9)
        scene = _make_scene(buffer=[ep])
        assert scene.anomalies() == []

    def test_exactly_at_threshold_included(self):
        ep = _make_episode(event_type="error", importance=0.5)
        scene = _make_scene(buffer=[ep])
        assert len(scene.anomalies()) == 1

    def test_sorted_by_importance_descending(self):
        buf = [
            _make_episode(event_type="error", importance=0.6, content="low"),
            _make_episode(event_type="error", importance=0.9, content="high"),
            _make_episode(event_type="error", importance=0.7, content="mid"),
        ]
        scene = _make_scene(buffer=buf)
        anomalies = scene.anomalies()
        importances = [a["importance"] for a in anomalies]
        assert importances == sorted(importances, reverse=True)

    def test_anomaly_dict_has_required_keys(self):
        ep = _make_episode(event_type="error", importance=0.8)
        scene = _make_scene(buffer=[ep])
        anomaly = scene.anomalies()[0]
        for key in ("time", "type", "description", "importance"):
            assert key in anomaly

    def test_importance_rounded(self):
        ep = _make_episode(event_type="error", importance=0.789123)
        scene = _make_scene(buffer=[ep])
        assert scene.anomalies()[0]["importance"] == round(0.789123, 2)


# ── TestBriefing ──────────────────────────────────────────────────────────────

class TestBriefing:
    async def test_empty_everything_returns_no_records(self):
        scene = _make_scene()
        result = await scene.briefing()
        assert result == "暂无场景记录。"

    async def test_knowledge_context_included(self):
        scene = _make_scene(knowledge_context="场景知识内容")
        result = await scene.briefing()
        assert "场景知识内容" in result

    async def test_recent_digest_included(self):
        scene = _make_scene(recent_digest="最近事件摘要")
        result = await scene.briefing()
        assert "最近事件摘要" in result

    async def test_session_summaries_included(self):
        session = MagicMock()
        session.get_recent_summaries.return_value = "Session summary"
        scene = _make_scene(knowledge_context="知识", session=session)
        result = await scene.briefing()
        assert "Session summary" in result

    async def test_empty_session_summaries_not_added(self):
        session = MagicMock()
        session.get_recent_summaries.return_value = ""
        scene = _make_scene(knowledge_context="知识", session=session)
        result = await scene.briefing()
        assert "Session summary" not in result

    async def test_anomalies_included_in_briefing(self):
        buf = [_make_episode(event_type="error", importance=0.8, content="fire detected")]
        scene = _make_scene(buffer=buf)
        result = await scene.briefing()
        assert "fire detected" in result
        assert "近期异常事件" in result

    async def test_at_most_5_anomalies_shown(self):
        # 8 high-importance errors — only 5 should appear in briefing
        buf = [
            _make_episode(event_type="error", importance=0.9, content=f"error_{i}")
            for i in range(8)
        ]
        scene = _make_scene(buffer=buf)
        result = await scene.briefing()
        # Count how many "error_" entries appear
        count = sum(1 for i in range(8) if f"error_{i}" in result)
        assert count <= 5

    async def test_no_session_no_crash(self):
        scene = _make_scene(knowledge_context="知识", session=None)
        result = await scene.briefing()
        assert isinstance(result, str)


# ── TestTodaySummary ──────────────────────────────────────────────────────────

class TestTodaySummary:
    async def test_no_records_returns_no_records_string(self):
        scene = _make_scene()
        llm = MagicMock()
        result = await scene.today_summary(llm)
        assert result == "暂无场景记录。"
        llm.chat.assert_not_called()

    async def test_with_content_calls_llm(self):
        scene = _make_scene(knowledge_context="有内容")
        llm = MagicMock()
        llm.chat = AsyncMock(return_value="LLM summary")
        result = await scene.today_summary(llm)
        assert result == "LLM summary"
        llm.chat.assert_called_once()

    async def test_llm_failure_returns_raw_briefing(self):
        scene = _make_scene(knowledge_context="有内容")
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await scene.today_summary(llm)
        assert "有内容" in result  # raw briefing returned

    async def test_llm_returns_empty_string_falls_back_to_raw(self):
        scene = _make_scene(knowledge_context="有内容")
        llm = MagicMock()
        llm.chat = AsyncMock(return_value="")
        result = await scene.today_summary(llm)
        assert "有内容" in result


# ── TestStatus ────────────────────────────────────────────────────────────────

class TestStatus:
    def test_has_required_keys(self):
        scene = _make_scene()
        status = scene.status()
        for key in ("known_entities", "anomaly_count", "episode_buffer_size",
                    "has_world_knowledge"):
            assert key in status

    def test_empty_everything(self):
        scene = _make_scene()
        status = scene.status()
        assert status["known_entities"] == []
        assert status["anomaly_count"] == 0
        assert status["episode_buffer_size"] == 0
        assert status["has_world_knowledge"] is False

    def test_entity_count_reflected(self):
        scene = _make_scene(world_knowledge={"entities": {"alice": {}, "bob": {}}})
        assert len(scene.status()["known_entities"]) == 2

    def test_anomaly_count_correct(self):
        buf = [
            _make_episode(event_type="error", importance=0.8),
            _make_episode(event_type="error", importance=0.3),  # below threshold
        ]
        scene = _make_scene(buffer=buf)
        assert scene.status()["anomaly_count"] == 1

    def test_episode_buffer_size_correct(self):
        buf = [_make_episode() for _ in range(5)]
        scene = _make_scene(buffer=buf)
        assert scene.status()["episode_buffer_size"] == 5

    def test_has_world_knowledge_true(self):
        scene = _make_scene(world_knowledge={"foo": "bar"})
        assert scene.status()["has_world_knowledge"] is True
