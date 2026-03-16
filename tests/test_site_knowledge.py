"""Tests for SiteKnowledge — spatial memory module."""

from __future__ import annotations

from askme.brain.site_knowledge import SiteKnowledge, Location


def test_record_visit_creates_location(tmp_path):
    """record_visit creates a new location entry."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_visit("仓库A", coords=(10.0, 20.0), description="东北角仓库")
    loc = sk.get_location("仓库A")
    assert loc is not None
    assert loc.name == "仓库A"
    assert loc.coords == (10.0, 20.0)
    assert loc.description == "东北角仓库"


def test_record_visit_increments_count(tmp_path):
    """Repeated visits increment visit_count."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_visit("3号门")
    sk.record_visit("3号门")
    sk.record_visit("3号门")
    loc = sk.get_location("3号门")
    assert loc is not None
    assert loc.visit_count == 3


def test_record_anomaly_increments_count(tmp_path):
    """record_anomaly increments anomaly_count on the location."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_anomaly("仓库B", "温度异常")
    sk.record_anomaly("仓库B", "门未关闭")
    loc = sk.get_location("仓库B")
    assert loc is not None
    assert loc.anomaly_count == 2


def test_find_nearby_with_coords(tmp_path):
    """find_nearby returns locations within radius."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_visit("A", coords=(0.0, 0.0))
    sk.record_visit("B", coords=(5.0, 0.0))
    sk.record_visit("C", coords=(100.0, 100.0))
    nearby = sk.find_nearby((0.0, 0.0), radius=10.0)
    names = {loc.name for loc in nearby}
    assert "A" in names
    assert "B" in names
    assert "C" not in names


def test_get_anomaly_hotspots(tmp_path):
    """get_anomaly_hotspots returns locations exceeding min_count."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_anomaly("热点X", "issue1")
    sk.record_anomaly("热点X", "issue2")
    sk.record_anomaly("热点X", "issue3")
    sk.record_anomaly("冷点Y", "issue1")
    hotspots = sk.get_anomaly_hotspots(min_count=2)
    names = [loc.name for loc in hotspots]
    assert "热点X" in names
    assert "冷点Y" not in names


def test_get_location_history(tmp_path):
    """get_location_history returns events for a specific location."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_visit("大门")
    sk.record_anomaly("大门", "灯坏了")
    sk.record_observation("大门", "有快递")
    history = sk.get_location_history("大门")
    assert len(history) == 3
    types = [e.event_type for e in history]
    assert types == ["visit", "anomaly", "observation"]


def test_get_context_returns_string(tmp_path):
    """get_context returns a non-empty string when locations exist."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_visit("充电桩", description="走廊尽头")
    ctx = sk.get_context()
    assert isinstance(ctx, str)
    assert "充电桩" in ctx
    assert "访问1次" in ctx


def test_get_context_empty(tmp_path):
    """get_context returns empty string when no locations recorded."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    assert sk.get_context() == ""


def test_persistence_save_load(tmp_path):
    """Locations and events survive save/load round-trip."""
    sk1 = SiteKnowledge(data_dir=str(tmp_path))
    sk1.record_visit("仓库A", coords=(1.0, 2.0), description="test")
    sk1.record_anomaly("仓库A", "温度过高")

    # Create a fresh instance from the same directory
    sk2 = SiteKnowledge(data_dir=str(tmp_path))
    loc = sk2.get_location("仓库A")
    assert loc is not None
    assert loc.visit_count == 1
    assert loc.anomaly_count == 1
    assert loc.coords == (1.0, 2.0)
    history = sk2.get_location_history("仓库A")
    assert len(history) == 2


def test_record_observation(tmp_path):
    """record_observation adds an event without changing visit/anomaly counts."""
    sk = SiteKnowledge(data_dir=str(tmp_path))
    sk.record_observation("走廊", "地面湿滑", coords=(3.0, 4.0))
    loc = sk.get_location("走廊")
    assert loc is not None
    assert loc.visit_count == 0
    assert loc.anomaly_count == 0
    assert loc.coords == (3.0, 4.0)
