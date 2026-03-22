"""Tests for MapAdapter -- LingTu map data sync into SiteKnowledge."""

from __future__ import annotations

import json

from askme.memory.map_adapter import MapAdapter
from askme.memory.site_knowledge import SiteKnowledge


SAMPLE_TOPO = {
    "nodes": {
        "0": {
            "position": [2.0, 3.0, 0.35],
            "visit_count": 193,
            "visible_labels": ["传送带", "工厂大门"],
            "neighbors": [1],
            "edge_distances": {"1": 2.83},
        },
        "1": {
            "position": [5.0, 6.0, 0.40],
            "visit_count": 42,
            "visible_labels": ["仓储区", "楼梯"],
            "neighbors": [0],
            "edge_distances": {"0": 2.83},
        },
    },
    "current_node_id": 1,
    "path_history": [0, 1],
}

SAMPLE_KG = {
    "仓储区": {
        "objects": ["货架", "叉车", "托盘"],
    },
    "工厂大门": ["安全门", "门禁"],
}


def _write_topo(tmp_path, data=None):
    """Helper: write topo_memory.json to tmp_path, return path string."""
    path = tmp_path / "topo_memory.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data or SAMPLE_TOPO, f, ensure_ascii=False)
    return str(path)


def _write_kg(tmp_path, data=None):
    """Helper: write room_object_kg.json to tmp_path, return path string."""
    path = tmp_path / "room_object_kg.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data or SAMPLE_KG, f, ensure_ascii=False)
    return str(path)


def test_load_topo_reads_valid_json(tmp_path):
    """load_topo reads valid JSON and returns dict with nodes."""
    topo_path = _write_topo(tmp_path)
    adapter = MapAdapter(topo_path)
    data = adapter.load_topo()
    assert "nodes" in data
    assert "0" in data["nodes"]
    assert data["nodes"]["0"]["visit_count"] == 193


def test_load_topo_missing_file_returns_empty(tmp_path):
    """load_topo returns empty dict when file does not exist."""
    adapter = MapAdapter(str(tmp_path / "nonexistent.json"))
    data = adapter.load_topo()
    assert data == {}


def test_sync_creates_locations_from_nodes(tmp_path):
    """sync creates a Location for each topo node."""
    topo_path = _write_topo(tmp_path)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path)
    count = adapter.sync(site)
    assert count == 2
    assert site.get_location("node_0") is not None
    assert site.get_location("node_1") is not None


def test_sync_sets_coords_from_position(tmp_path):
    """sync maps position[0:2] to Location.coords."""
    topo_path = _write_topo(tmp_path)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path)
    adapter.sync(site)
    loc = site.get_location("node_0")
    assert loc is not None
    assert loc.coords == (2.0, 3.0)


def test_sync_adds_visible_labels_as_tags(tmp_path):
    """sync stores visible_labels as Location.tags including elevation."""
    topo_path = _write_topo(tmp_path)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path)
    adapter.sync(site)
    loc = site.get_location("node_0")
    assert loc is not None
    assert "传送带" in loc.tags
    assert "工厂大门" in loc.tags
    assert "elevation:0.35" in loc.tags


def test_get_current_position_returns_coords(tmp_path):
    """get_current_position returns 3D position of current_node_id."""
    topo_path = _write_topo(tmp_path)
    adapter = MapAdapter(topo_path)
    adapter.load_topo()
    pos = adapter.get_current_position()
    assert pos is not None
    # current_node_id=1, node 1 position=[5.0, 6.0, 0.40]
    assert pos == (5.0, 6.0, 0.40)


def test_get_current_position_no_data(tmp_path):
    """get_current_position returns None when no topo loaded."""
    adapter = MapAdapter(str(tmp_path / "nonexistent.json"))
    pos = adapter.get_current_position()
    assert pos is None


def test_get_nearby_labels_at_current(tmp_path):
    """get_nearby_labels returns labels at current node."""
    topo_path = _write_topo(tmp_path)
    adapter = MapAdapter(topo_path)
    adapter.load_topo()
    labels = adapter.get_nearby_labels()
    assert "仓储区" in labels
    assert "楼梯" in labels


def test_get_nearby_labels_at_specific_node(tmp_path):
    """get_nearby_labels with explicit node_id returns that node's labels."""
    topo_path = _write_topo(tmp_path)
    adapter = MapAdapter(topo_path)
    adapter.load_topo()
    labels = adapter.get_nearby_labels(node_id=0)
    assert "传送带" in labels
    assert "工厂大门" in labels


def test_sync_with_kg_enriches_description(tmp_path):
    """sync with room_object_kg adds object info to description."""
    topo_path = _write_topo(tmp_path)
    kg_path = _write_kg(tmp_path)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path, kg_path)
    adapter.sync(site)
    # node_1 has "仓储区" which has objects in KG
    loc = site.get_location("node_1")
    assert loc is not None
    assert "货架" in loc.description
    # node_0 has "工厂大门" which has list-format objects in KG
    loc0 = site.get_location("node_0")
    assert loc0 is not None
    assert "安全门" in loc0.description


def test_sync_preserves_visit_count(tmp_path):
    """sync sets visit_count from topo data."""
    topo_path = _write_topo(tmp_path)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path)
    adapter.sync(site)
    loc = site.get_location("node_0")
    assert loc is not None
    assert loc.visit_count == 193


def test_sync_skips_nodes_without_position(tmp_path):
    """sync skips nodes that have insufficient position data."""
    bad_topo = {"nodes": {"99": {"visible_labels": ["test"]}}}
    topo_path = _write_topo(tmp_path, bad_topo)
    site_dir = tmp_path / "site"
    site = SiteKnowledge(data_dir=str(site_dir))
    adapter = MapAdapter(topo_path)
    count = adapter.sync(site)
    assert count == 0


def test_load_kg_missing_file_returns_empty(tmp_path):
    """load_kg returns empty dict when file does not exist."""
    adapter = MapAdapter(str(tmp_path / "topo.json"), str(tmp_path / "missing_kg.json"))
    data = adapter.load_kg()
    assert data == {}
