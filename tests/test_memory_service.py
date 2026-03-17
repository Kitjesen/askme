"""Tests for MemoryService -- standalone unified memory API."""

from __future__ import annotations

import json

from askme.brain.memory_service import MemoryService, get_memory_service, reset_memory_service


def test_record_visit_delegates_to_site(tmp_path):
    """record_visit creates a location in SiteKnowledge."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_visit("仓库A", coords=(10.0, 20.0), description="东北角")
    loc = mem.site.get_location("仓库A")
    assert loc is not None
    assert loc.coords == (10.0, 20.0)
    assert loc.visit_count == 1


def test_record_anomaly_and_get_hotspots(tmp_path):
    """record_anomaly + get_anomaly_hotspots work together."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_anomaly("区域X", "温度异常")
    mem.record_anomaly("区域X", "烟雾报警")
    mem.record_anomaly("区域Y", "门未关闭")
    hotspots = mem.get_anomaly_hotspots(min_count=2)
    names = [loc.name for loc in hotspots]
    assert "区域X" in names
    assert "区域Y" not in names


def test_record_task_outcome_and_get_best(tmp_path):
    """record_task_outcome + get_best_procedure work together."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    for _ in range(5):
        mem.record_task_outcome("route_north", "navigate", True, duration=60.0)
    mem.record_task_outcome("route_south", "navigate", True)
    for _ in range(3):
        mem.record_task_outcome("route_south", "navigate", False)
    best = mem.get_best_procedure("navigate")
    assert best is not None
    assert best.name == "route_north"


def test_should_remember_filters_duplicates(tmp_path):
    """should_remember returns False for near-duplicate text."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    # First time: novel -> admitted
    assert mem.should_remember("command", "导航到仓库A", importance=0.8) is True
    # Exact duplicate: low novelty -> rejected (with default threshold)
    assert mem.should_remember("system", "导航到仓库A", importance=0.0) is False


def test_get_full_context_returns_combined_string(tmp_path):
    """get_full_context returns combined spatial + procedural context."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_visit("大门", description="正门")
    mem.record_task_outcome("巡逻路线A", "patrol", True, description="东翼巡逻")
    ctx = mem.get_full_context()
    assert "大门" in ctx
    assert "巡逻路线A" in ctx


def test_get_full_context_empty_when_no_data(tmp_path):
    """get_full_context returns empty string when nothing recorded."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    assert mem.get_full_context() == ""


def test_sync_from_map_loads_topo_data(tmp_path):
    """sync_from_map loads topo data into spatial memory."""
    topo = {
        "nodes": {
            "0": {
                "position": [1.0, 2.0, 0.3],
                "visit_count": 10,
                "visible_labels": ["货架"],
                "neighbors": [],
                "edge_distances": {},
            },
        },
        "current_node_id": 0,
        "path_history": [0],
    }
    topo_path = tmp_path / "topo_memory.json"
    with open(topo_path, "w", encoding="utf-8") as f:
        json.dump(topo, f, ensure_ascii=False)

    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    count = mem.sync_from_map(topo_path=str(topo_path))
    assert count == 1
    loc = mem.site.get_location("node_0")
    assert loc is not None
    assert loc.coords == (1.0, 2.0)


def test_singleton_get_memory_service_returns_same():
    """get_memory_service returns the same instance on repeated calls."""
    reset_memory_service()
    try:
        mem1 = get_memory_service()
        mem2 = get_memory_service()
        assert mem1 is mem2
    finally:
        reset_memory_service()


def test_save_and_load_persistence(tmp_path):
    """save/load round-trip preserves data."""
    site_dir = str(tmp_path / "site")
    proc_dir = str(tmp_path / "proc")
    mem1 = MemoryService(config={"site_dir": site_dir, "procedures_dir": proc_dir})
    mem1.record_visit("充电桩", coords=(5.0, 5.0))
    mem1.record_task_outcome("route_x", "navigate", True)
    mem1.save()

    # Fresh instance from same dirs
    mem2 = MemoryService(config={"site_dir": site_dir, "procedures_dir": proc_dir})
    loc = mem2.site.get_location("充电桩")
    assert loc is not None
    assert loc.coords == (5.0, 5.0)
    procs = mem2.procedures.get_procedures("navigate")
    assert len(procs) == 1


def test_find_nearby_delegates(tmp_path):
    """find_nearby delegates to SiteKnowledge."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_visit("A", coords=(0.0, 0.0))
    mem.record_visit("B", coords=(100.0, 100.0))
    nearby = mem.find_nearby((0.0, 0.0), radius=5.0)
    names = [loc.name for loc in nearby]
    assert "A" in names
    assert "B" not in names


def test_record_observation_delegates(tmp_path):
    """record_observation delegates to SiteKnowledge."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_observation("走廊", "地面湿滑", coords=(3.0, 4.0))
    loc = mem.site.get_location("走廊")
    assert loc is not None
    assert loc.visit_count == 0
    assert loc.coords == (3.0, 4.0)


def test_get_location_history_delegates(tmp_path):
    """get_location_history delegates to SiteKnowledge."""
    mem = MemoryService(config={"site_dir": str(tmp_path / "site"),
                                "procedures_dir": str(tmp_path / "proc")})
    mem.record_visit("大门")
    mem.record_anomaly("大门", "灯坏了")
    history = mem.get_location_history("大门")
    assert len(history) == 2
    types = [e.event_type for e in history]
    assert "visit" in types
    assert "anomaly" in types
