"""Tests for MapAdapter — topo loading, sync to SiteKnowledge, position queries."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from askme.memory.map_adapter import MapAdapter

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_topo(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_topo(nodes: dict | None = None, current_node_id: int | None = None) -> dict:
    d: dict = {"nodes": nodes or {}}
    if current_node_id is not None:
        d["current_node_id"] = current_node_id
    return d


def _make_node(x: float = 1.0, y: float = 2.0, z: float = 0.0,
               labels: list | None = None, visits: int = 0) -> dict:
    return {
        "position": [x, y, z],
        "visible_labels": labels or [],
        "visit_count": visits,
    }


# ── load_topo ─────────────────────────────────────────────────────────────────

class TestLoadTopo:
    def test_returns_empty_when_file_not_found(self, tmp_path):
        adapter = MapAdapter(str(tmp_path / "nonexistent.json"))
        result = adapter.load_topo()
        assert result == {}

    def test_loads_valid_json(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, {"nodes": {"1": _make_node()}})
        adapter = MapAdapter(str(topo_path))
        data = adapter.load_topo()
        assert "nodes" in data

    def test_returns_empty_on_invalid_json(self, tmp_path):
        topo_path = tmp_path / "bad.json"
        topo_path.write_text("{ invalid json", encoding="utf-8")
        adapter = MapAdapter(str(topo_path))
        result = adapter.load_topo()
        assert result == {}

    def test_caches_loaded_data(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, {"nodes": {"1": _make_node()}})
        adapter = MapAdapter(str(topo_path))
        adapter.load_topo()
        assert adapter._topo_data != {}


# ── load_kg ───────────────────────────────────────────────────────────────────

class TestLoadKg:
    def test_returns_empty_when_no_kg_path(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, {})
        adapter = MapAdapter(str(topo_path))  # no kg_path
        assert adapter.load_kg() == {}

    def test_returns_empty_when_kg_not_found(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, {})
        adapter = MapAdapter(str(topo_path), kg_path=str(tmp_path / "missing.json"))
        assert adapter.load_kg() == {}

    def test_loads_valid_kg(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        kg_path = tmp_path / "kg.json"
        _write_topo(topo_path, {})
        kg_path.write_text(json.dumps({"storage": {"objects": ["box", "pallet"]}}))
        adapter = MapAdapter(str(topo_path), kg_path=str(kg_path))
        data = adapter.load_kg()
        assert "storage" in data


# ── sync ──────────────────────────────────────────────────────────────────────

class TestSync:
    def _make_site(self):
        """Create a minimal SiteKnowledge mock."""
        site = MagicMock()
        site.get_location.return_value = None  # location doesn't exist yet
        site.record_visit.return_value = None
        return site

    def test_returns_zero_with_empty_topo(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, _make_topo())  # no nodes
        adapter = MapAdapter(str(topo_path))
        site = self._make_site()
        count = adapter.sync(site)
        assert count == 0

    def test_syncs_node_count(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={
            "1": _make_node(1.0, 2.0),
            "2": _make_node(3.0, 4.0),
        })
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        site = self._make_site()
        count = adapter.sync(site)
        assert count == 2

    def test_skips_node_without_position(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={
            "1": {"visible_labels": []},  # no position
        })
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        site = self._make_site()
        count = adapter.sync(site)
        assert count == 0

    def test_record_visit_called_for_new_location(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={"1": _make_node(1.0, 2.0, labels=["corridor"])})
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))

        site = self._make_site()
        adapter.sync(site)
        site.record_visit.assert_called_once()
        call_args = site.record_visit.call_args
        assert call_args[0][0] == "node_1"

    def test_updates_existing_location(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={"1": _make_node(1.0, 2.0)})
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))

        existing = MagicMock()
        existing.visit_count = 2
        existing.tags = []

        site = MagicMock()
        site.get_location.return_value = existing  # location exists
        adapter.sync(site)
        # record_visit should NOT be called (location exists)
        site.record_visit.assert_not_called()
        # coords should be updated
        assert existing.coords == (1.0, 2.0)

    def test_description_from_visible_labels(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={"1": _make_node(0, 0, labels=["corridor", "exit"])})
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))

        site = self._make_site()
        adapter.sync(site)
        call_kwargs = site.record_visit.call_args[1]
        description = call_kwargs.get("description", "")
        assert "corridor" in description or "exit" in description

    def test_kg_enriches_description(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        kg_path = tmp_path / "kg.json"
        topo = _make_topo(nodes={"1": _make_node(0, 0, labels=["storage"])})
        _write_topo(topo_path, topo)
        kg_path.write_text(json.dumps({"storage": {"objects": ["box", "pallet"]}}))
        adapter = MapAdapter(str(topo_path), kg_path=str(kg_path))

        site = self._make_site()
        adapter.sync(site)
        call_kwargs = site.record_visit.call_args[1]
        description = call_kwargs.get("description", "")
        assert "box" in description or "pallet" in description

    def test_elevation_tag_added(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={"1": _make_node(0, 0, z=1.5)})
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))

        existing = MagicMock()
        existing.visit_count = 0
        existing.tags = []
        site = MagicMock()
        site.get_location.return_value = existing
        adapter.sync(site)
        tags_added = [tag for call in site.method_calls for tag in []]
        # Check that elevation tag was set somewhere in existing.tags
        assert any("elevation" in tag for tag in existing.tags)


# ── get_current_position ──────────────────────────────────────────────────────

class TestGetCurrentPosition:
    def test_returns_none_when_no_topo(self, tmp_path):
        adapter = MapAdapter(str(tmp_path / "nonexistent.json"))
        assert adapter.get_current_position() is None

    def test_returns_none_when_no_current_node(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, _make_topo(nodes={"1": _make_node()}))
        adapter = MapAdapter(str(topo_path))
        assert adapter.get_current_position() is None

    def test_returns_xyz_tuple(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(
            nodes={"1": _make_node(3.0, 4.0, z=0.5)},
            current_node_id=1,
        )
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        pos = adapter.get_current_position()
        assert pos is not None
        assert pos == (3.0, 4.0, 0.5)

    def test_returns_none_when_position_incomplete(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = {
            "nodes": {"1": {"position": [1.0, 2.0]}},  # only 2D
            "current_node_id": 1,
        }
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        assert adapter.get_current_position() is None


# ── get_nearby_labels ─────────────────────────────────────────────────────────

class TestGetNearbyLabels:
    def test_returns_empty_when_no_topo(self, tmp_path):
        adapter = MapAdapter(str(tmp_path / "nonexistent.json"))
        assert adapter.get_nearby_labels() == []

    def test_returns_labels_for_current_node(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(
            nodes={"3": _make_node(labels=["box", "shelf"])},
            current_node_id=3,
        )
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        labels = adapter.get_nearby_labels()
        assert "box" in labels
        assert "shelf" in labels

    def test_returns_labels_for_specified_node(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        topo = _make_topo(nodes={"5": _make_node(labels=["door"])})
        _write_topo(topo_path, topo)
        adapter = MapAdapter(str(topo_path))
        labels = adapter.get_nearby_labels(node_id=5)
        assert "door" in labels

    def test_returns_empty_when_node_not_found(self, tmp_path):
        topo_path = tmp_path / "topo.json"
        _write_topo(topo_path, _make_topo(nodes={}))
        adapter = MapAdapter(str(topo_path))
        assert adapter.get_nearby_labels(node_id=99) == []
