"""Tests for ProactiveAgent helper methods — format_alert, adaptive_interval."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from askme.pipeline.proactive_agent import ProactiveAgent, _ALERT_TEMPLATES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_agent(base_interval: float = 60.0) -> ProactiveAgent:
    """Create a minimal ProactiveAgent with all dependencies mocked."""
    llm = MagicMock()
    audio = MagicMock()
    episodic = MagicMock()
    episodic.log.return_value = None
    episodic.should_reflect.return_value = False

    agent = ProactiveAgent(
        llm=llm,
        vision=None,
        audio=audio,
        episodic=episodic,
        config={
            "proactive": {
                "enabled": False,
                "patrol_interval": base_interval,
            },
        },
    )
    # Set _base_interval explicitly (derived from patrol_interval)
    agent._base_interval = base_interval
    return agent


# ── TestFormatAlert ───────────────────────────────────────────────────────────

class TestFormatAlert:
    def test_known_topic_with_payload(self):
        result = ProactiveAgent._format_alert(
            "navigation.stall_detected",
            {"stall_duration_s": 5.0},
        )
        assert result is not None
        assert "5" in result  # stall_duration_s formatted

    def test_known_topic_no_params(self):
        result = ProactiveAgent._format_alert("mission.failed", {})
        assert result is not None
        assert "失败" in result

    def test_unknown_topic_returns_none(self):
        result = ProactiveAgent._format_alert("custom.unknown_topic", {})
        assert result is None

    def test_arrival_topic_with_distance(self):
        result = ProactiveAgent._format_alert(
            "navigation.arrival",
            {"distance_remaining_m": 0.3},
        )
        assert result is not None
        assert "0.3" in result

    def test_milestone_topic(self):
        result = ProactiveAgent._format_alert(
            "navigation.milestone",
            {"progress_pct": 50, "waypoint_current": 3, "waypoint_total": 6},
        )
        assert result is not None
        assert "50" in result

    def test_stall_cleared_no_params_needed(self):
        result = ProactiveAgent._format_alert("navigation.stall_cleared", {})
        assert result is not None
        assert "恢复" in result

    def test_mission_completed(self):
        result = ProactiveAgent._format_alert("mission.completed", {})
        assert result is not None

    def test_mission_canceled(self):
        result = ProactiveAgent._format_alert("mission.canceled", {})
        assert result is not None

    def test_missing_placeholder_falls_back(self):
        # Known topic but payload missing a required key
        result = ProactiveAgent._format_alert(
            "navigation.stall_detected",
            {},  # missing stall_duration_s
        )
        # Falls back to first clause of template
        assert result is not None
        assert result.endswith("。")

    def test_all_known_topics_produce_non_none(self):
        for topic in _ALERT_TEMPLATES:
            result = ProactiveAgent._format_alert(topic, {})
            # Either formatted or fell back — never None for known topics
            assert result is not None


# ── TestAdaptiveInterval ──────────────────────────────────────────────────────

class TestAdaptiveInterval:
    def _at_hour(self, agent: ProactiveAgent, hour: int) -> float:
        """Run _adaptive_interval with a mocked hour."""
        import datetime as dt_module
        fake_now = dt_module.datetime(2026, 1, 1, hour, 0, 0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            return agent._adaptive_interval()

    def test_returns_float(self):
        agent = _make_agent(base_interval=60.0)
        result = self._at_hour(agent, 12)
        assert isinstance(result, float)
        assert result > 0

    def test_night_hours_longer_interval(self):
        agent = _make_agent(base_interval=60.0)
        night = self._at_hour(agent, 23)
        normal = self._at_hour(agent, 12)
        assert night > normal

    def test_peak_hours_shorter_interval(self):
        agent = _make_agent(base_interval=60.0)
        peak = self._at_hour(agent, 10)
        normal = self._at_hour(agent, 12)
        assert peak < normal

    def test_post_anomaly_shorter_interval(self):
        agent = _make_agent(base_interval=60.0)
        agent._last_anomaly_time = time.monotonic()  # just happened
        post_anomaly = self._at_hour(agent, 12)

        agent._last_anomaly_time = 0  # reset
        normal = self._at_hour(agent, 12)

        assert post_anomaly < normal

    def test_consecutive_normal_scans_relaxes(self):
        agent = _make_agent(base_interval=60.0)
        agent._consecutive_normal = 15  # many normal scans
        relaxed = self._at_hour(agent, 12)

        agent._consecutive_normal = 0
        normal = self._at_hour(agent, 12)

        assert relaxed >= normal

    def test_early_morning_night_hours(self):
        agent = _make_agent(base_interval=60.0)
        early = self._at_hour(agent, 3)
        normal = self._at_hour(agent, 12)
        assert early > normal  # 03:00 should be night = longer interval
