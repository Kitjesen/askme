"""Extended tests for Episode, score_importance, recency_boost."""

from __future__ import annotations

import math
import time

import pytest

from askme.memory.episode import (
    DEFAULT_STABILITY_S,
    MAX_STABILITY_S,
    STABILITY_GROWTH_FACTOR,
    Episode,
    recency_boost,
    score_importance,
)


# ── Episode ───────────────────────────────────────────────────────────────────

class TestEpisodeInit:
    def test_stability_scales_with_importance(self):
        e_low = Episode("action", "routine", importance=0.0)
        e_high = Episode("error", "critical failure", importance=0.8)
        assert e_high.stability > e_low.stability

    def test_default_importance_zero(self):
        e = Episode("system", "startup")
        assert e.importance == 0.0

    def test_access_count_starts_zero(self):
        e = Episode("command", "检查仓库")
        assert e.access_count == 0

    def test_context_defaults_to_empty_dict(self):
        e = Episode("command", "go")
        assert e.context == {}

    def test_context_stored(self):
        e = Episode("perception", "saw person", context={"room": "A"})
        assert e.context["room"] == "A"


class TestEpisodeAccess:
    def test_access_increments_count(self):
        e = Episode("command", "test")
        e.access()
        assert e.access_count == 1

    def test_access_doubles_stability(self):
        e = Episode("command", "test", importance=0.5)
        s_before = e.stability
        e.access()
        assert abs(e.stability - s_before * STABILITY_GROWTH_FACTOR) < 0.01

    def test_stability_capped_at_max(self):
        e = Episode("command", "test", importance=1.0)
        e.stability = MAX_STABILITY_S * 0.9
        e.access()
        assert e.stability == MAX_STABILITY_S

    def test_access_updates_last_accessed(self):
        e = Episode("command", "test")
        old = e.last_accessed
        time.sleep(0.01)
        e.access()
        assert e.last_accessed > old


class TestEpisodeRetrievability:
    def test_fresh_episode_near_one(self):
        e = Episode("command", "test", importance=0.5)
        r = e.retrievability(now=e.timestamp)
        assert r > 0.99

    def test_decays_over_time(self):
        e = Episode("command", "test", importance=0.5)
        r_now = e.retrievability(now=e.timestamp)
        r_later = e.retrievability(now=e.timestamp + e.stability)
        assert r_later < r_now

    def test_at_stability_time_approx_1_over_e(self):
        e = Episode("command", "test", importance=0.0)
        r = e.retrievability(now=e.timestamp + e.stability)
        assert abs(r - math.exp(-1)) < 0.01

    def test_access_slows_decay(self):
        e = Episode("command", "test", importance=0.0)
        e.access()  # doubles stability
        now = e.last_accessed + DEFAULT_STABILITY_S  # 1 original stability later
        r = e.retrievability(now=now)
        # With doubled stability, at t=S₀ we're only at half-decay
        assert r > math.exp(-1)


class TestEpisodeRetrievalScore:
    def test_relevant_keywords_boost_score(self):
        e = Episode("command", "navigate to warehouse A", importance=0.5)
        score_with = e.retrieval_score(query_keywords={"warehouse"})
        score_without = e.retrieval_score(query_keywords=set())
        assert score_with > score_without

    def test_no_keywords_no_relevance_boost(self):
        e = Episode("command", "test", importance=0.5)
        score = e.retrieval_score(query_keywords=None)
        # Only recency + importance components
        assert score >= 0.0

    def test_full_keyword_match_max_relevance(self):
        e = Episode("command", "warehouse A anomaly detected", importance=0.5)
        score = e.retrieval_score(query_keywords={"warehouse", "anomaly"})
        assert score > 0.0

    def test_score_is_non_negative(self):
        e = Episode("system", "boot", importance=0.0)
        assert e.retrieval_score() >= 0.0


class TestEpisodeSerialization:
    def test_to_dict_has_required_keys(self):
        e = Episode("command", "test")
        d = e.to_dict()
        for key in ("ts", "type", "desc", "importance", "stability", "access_count"):
            assert key in d

    def test_from_dict_roundtrip(self):
        e = Episode("error", "disk full", context={"disk": "/dev/sda"}, importance=0.8)
        d = e.to_dict()
        e2 = Episode.from_dict(d)
        assert e2.event_type == "error"
        assert e2.description == "disk full"
        assert e2.importance == pytest.approx(0.8, abs=0.01)

    def test_from_dict_handles_missing_fields(self):
        # Only minimal data
        e = Episode.from_dict({"type": "system", "desc": "boot"})
        assert e.event_type == "system"
        assert e.importance == 0.0

    def test_to_log_line_includes_type_and_desc(self):
        e = Episode("command", "go to warehouse")
        line = e.to_log_line()
        assert "command" in line
        assert "go to warehouse" in line


# ── score_importance ──────────────────────────────────────────────────────────

class TestScoreImportance:
    def test_command_higher_than_system(self):
        cmd = score_importance("command", "巡检任务")
        sys = score_importance("system", "内部日志")
        assert cmd > sys

    def test_error_type_high_base(self):
        score = score_importance("error", "连接失败")
        assert score >= 0.8

    def test_person_detection_boost(self):
        no_person = score_importance("perception", "空旷走廊")
        with_person = score_importance("perception", "I see a person here")
        assert with_person > no_person

    def test_danger_keyword_boost(self):
        normal = score_importance("action", "正常巡检")
        danger = score_importance("action", "发现危险物品")
        assert danger > normal

    def test_fail_keyword_boost(self):
        ok = score_importance("action", "任务完成")
        failed = score_importance("action", "任务失败异常")
        assert failed > ok

    def test_detection_person_boost(self):
        ctx = {"detections": [{"label": "person", "conf": 1.0}]}
        score = score_importance("perception", "scan", ctx)
        assert score > score_importance("perception", "scan", {})

    def test_fire_detection_danger_boost(self):
        ctx = {"detections": [{"label": "fire", "conf": 0.9}]}
        score = score_importance("perception", "thermal scan", ctx)
        base = score_importance("perception", "thermal scan", {})
        assert score > base

    def test_surprise_context_boost(self):
        base = score_importance("perception", "saw something")
        surprised = score_importance("perception", "saw something", {"surprise": True})
        assert surprised > base

    def test_high_conf_perception_boost(self):
        low_conf = score_importance("perception", "test", {
            "detections": [{"label": "box", "conf": 0.5}]
        })
        high_conf = score_importance("perception", "test", {
            "detections": [{"label": "box", "conf": 0.95}]
        })
        assert high_conf > low_conf

    def test_score_clamped_to_one(self):
        # Multiple boosts that would exceed 1.0
        ctx = {
            "detections": [{"label": "fire", "conf": 1.0}],
            "surprise": True,
        }
        score = score_importance("error", "fire detected danger fail", ctx)
        assert score <= 1.0

    def test_score_nonnegative(self):
        assert score_importance("system", "x") >= 0.0

    def test_unknown_event_type_defaults_to_0_3(self):
        score = score_importance("custom_type", "no boosts")
        assert abs(score - 0.3) < 0.01


# ── recency_boost ─────────────────────────────────────────────────────────────

class TestRecencyBoost:
    def test_age_zero_returns_importance(self):
        assert recency_boost(0.8, 0.0) == pytest.approx(0.8)

    def test_decays_with_age(self):
        fresh = recency_boost(1.0, 1.0)
        old = recency_boost(1.0, 100.0)
        assert fresh > old

    def test_floor_at_half_importance(self):
        very_old = recency_boost(1.0, 1e6)
        assert very_old >= 0.5  # floor prevents full decay

    def test_zero_importance_stays_zero(self):
        assert recency_boost(0.0, 24.0) == pytest.approx(0.0)

    def test_negative_age_treated_as_zero(self):
        """Negative age (future timestamp) should return importance unchanged."""
        assert recency_boost(0.7, -5.0) == pytest.approx(0.7)

    def test_at_half_life_approx_75_percent(self):
        """At one half-life, score = I * (0.5 + 0.5 * 0.5) = 0.75 * I."""
        result = recency_boost(1.0, 72.0, half_life_hours=72.0)
        assert abs(result - 0.75) < 0.01
