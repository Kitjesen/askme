"""Tests for TrendAnalyzer — frequency spike detection in episodic streams."""

import time

import pytest

from askme.memory.episode import Episode
from askme.memory.trend_analyzer import TrendAnalyzer, Trend


def _make_episode(event_type: str, hours_ago: float = 0.0, description: str = "") -> Episode:
    """Create an Episode with a controlled timestamp."""
    ep = Episode(
        event_type=event_type,
        description=description or f"{event_type} event",
        context=None,
        importance=0.5,
    )
    # Override timestamp to simulate past events
    ep.timestamp = time.time() - hours_ago * 3600
    return ep


class TestAnalyze:
    def test_empty_episodes_returns_empty(self):
        analyzer = TrendAnalyzer()
        assert analyzer.analyze([]) == []

    def test_uniform_distribution_no_spikes(self):
        """Events spread evenly over 24h should not produce spikes."""
        analyzer = TrendAnalyzer()
        episodes = []
        # 1 event per hour for 24 hours — uniform
        for h in range(24):
            episodes.append(_make_episode("perception", hours_ago=h))
        trends = analyzer.analyze(episodes, window_hours=1, baseline_hours=24)
        assert len(trends) == 0

    def test_spike_detected(self):
        """10 events in the last hour vs 1/hour baseline should spike."""
        analyzer = TrendAnalyzer()
        episodes = []
        # Baseline: 1 event per hour for hours 2-24
        for h in range(2, 24):
            episodes.append(_make_episode("perception", hours_ago=h))
        # Recent: 10 events in the last hour
        for _ in range(10):
            episodes.append(_make_episode("perception", hours_ago=0.1))

        trends = analyzer.analyze(episodes, window_hours=1, baseline_hours=24)
        assert len(trends) == 1
        assert trends[0].event_type == "perception"
        assert trends[0].count == 10
        assert trends[0].spike_ratio >= 2.0

    def test_description_contains_event_type_and_ratio(self):
        analyzer = TrendAnalyzer()
        episodes = []
        for h in range(2, 24):
            episodes.append(_make_episode("error", hours_ago=h))
        for _ in range(8):
            episodes.append(_make_episode("error", hours_ago=0.1))

        trends = analyzer.analyze(episodes, window_hours=1, baseline_hours=24)
        assert len(trends) >= 1
        desc = trends[0].description_zh
        assert "error" in desc
        assert "倍" in desc

    def test_multiple_event_types(self):
        """Different event types tracked independently."""
        analyzer = TrendAnalyzer()
        episodes = []
        # Baseline: 1 per hour each type
        for h in range(2, 24):
            episodes.append(_make_episode("perception", hours_ago=h))
            episodes.append(_make_episode("command", hours_ago=h))
        # Spike only in perception
        for _ in range(15):
            episodes.append(_make_episode("perception", hours_ago=0.05))
        # Normal for command
        episodes.append(_make_episode("command", hours_ago=0.05))

        trends = analyzer.analyze(episodes, window_hours=1, baseline_hours=24)
        event_types = [t.event_type for t in trends]
        assert "perception" in event_types
        # command should not spike (only 1 in recent window)

    def test_no_baseline_uses_floor(self):
        """When there's no baseline data, use 0.5 floor — still detect if count > 1."""
        analyzer = TrendAnalyzer()
        # Only recent events, no baseline
        episodes = [_make_episode("perception", hours_ago=0.1) for _ in range(5)]
        trends = analyzer.analyze(episodes, window_hours=1, baseline_hours=24)
        # 5 / 0.5 = 10x spike
        assert len(trends) == 1
        assert trends[0].spike_ratio >= 2.0

    def test_frozen_dataclass(self):
        """Trend should be immutable."""
        t = Trend(
            window_start=0.0,
            window_end=1.0,
            event_type="test",
            count=5,
            baseline=1.0,
            spike_ratio=5.0,
            description_zh="test trend",
        )
        with pytest.raises(AttributeError):
            t.count = 10  # type: ignore[misc]


class TestGetSummary:
    def test_empty_when_no_trends(self):
        analyzer = TrendAnalyzer()
        episodes = [_make_episode("perception", hours_ago=h) for h in range(24)]
        assert analyzer.get_summary(episodes) == ""

    def test_summary_contains_descriptions(self):
        analyzer = TrendAnalyzer()
        episodes = []
        for h in range(2, 24):
            episodes.append(_make_episode("perception", hours_ago=h))
        for _ in range(10):
            episodes.append(_make_episode("perception", hours_ago=0.1))

        summary = analyzer.get_summary(episodes)
        assert "perception" in summary
        assert "倍" in summary
