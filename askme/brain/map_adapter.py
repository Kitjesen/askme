"""Non-invasive adapter: loads LingTu map data into SiteKnowledge.

Does NOT modify LingTu or SiteKnowledge. Just reads map files
and syncs them into the spatial memory layer.

Can be called by askme, by nav-gateway, by cortex, or by any other service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from askme.brain.site_knowledge import SiteKnowledge

logger = logging.getLogger(__name__)


class MapAdapter:
    """Sync LingTu topo_memory + room_object_kg into SiteKnowledge.

    Usage::

        adapter = MapAdapter(
            topo_path="maps/semantic/topo_memory.json",
            kg_path="maps/semantic/room_object_kg.json",
        )
        adapter.sync(site_knowledge)  # one-time or periodic sync
    """

    def __init__(self, topo_path: str, kg_path: str | None = None) -> None:
        self._topo_path = Path(topo_path)
        self._kg_path = Path(kg_path) if kg_path else None
        self._topo_data: dict[str, Any] = {}
        self._kg_data: dict[str, Any] = {}

    def load_topo(self) -> dict[str, Any]:
        """Load topo_memory.json, return raw data."""
        if not self._topo_path.exists():
            logger.warning("Topo file not found: %s", self._topo_path)
            return {}
        try:
            with open(self._topo_path, encoding="utf-8") as f:
                self._topo_data = json.load(f)
            return self._topo_data
        except Exception as e:
            logger.warning("Failed to load topo_memory: %s", e)
            return {}

    def load_kg(self) -> dict[str, Any]:
        """Load room_object_kg.json, return raw data."""
        if self._kg_path is None or not self._kg_path.exists():
            return {}
        try:
            with open(self._kg_path, encoding="utf-8") as f:
                self._kg_data = json.load(f)
            return self._kg_data
        except Exception as e:
            logger.warning("Failed to load room_object_kg: %s", e)
            return {}

    def sync(self, site: SiteKnowledge) -> int:
        """Sync map data into SiteKnowledge.

        - Each topo node becomes a Location (or updates existing)
        - visible_labels become description + tags
        - position[0:2] becomes coords (x,y)
        - position[2] stored as elevation in tags
        - visit_count synced
        - room_object_kg enriches location descriptions with object info
        - Returns number of locations synced.

        Non-destructive: only adds/updates, never deletes existing data.
        """
        topo = self.load_topo()
        kg = self.load_kg()

        nodes = topo.get("nodes", {})
        if not nodes:
            return 0

        # Build KG lookup: room_type -> object list
        kg_objects: dict[str, list[str]] = {}
        for room_type, info in kg.items():
            if isinstance(info, dict):
                kg_objects[room_type] = info.get("objects", [])
            elif isinstance(info, list):
                kg_objects[room_type] = info

        synced = 0
        for node_id, node in nodes.items():
            position = node.get("position", [])
            if len(position) < 2:
                continue

            coords = (float(position[0]), float(position[1]))
            labels = node.get("visible_labels", [])
            visit_count = node.get("visit_count", 0)

            # Build name from node_id
            name = f"node_{node_id}"

            # Build description from visible labels
            description = ""
            if labels:
                description = "可见: " + ", ".join(labels)

            # Enrich with KG data
            kg_extras: list[str] = []
            for label in labels:
                if label in kg_objects:
                    objs = kg_objects[label]
                    if objs:
                        kg_extras.append(f"{label}含{','.join(objs)}")
            if kg_extras:
                description += " | " + "; ".join(kg_extras)

            # Build tags
            tags = list(labels)
            if len(position) >= 3:
                tags.append(f"elevation:{position[2]}")

            # Update or create in SiteKnowledge
            existing = site.get_location(name)
            if existing is None:
                # Use record_visit to create and set visit_count
                site.record_visit(name, coords=coords, description=description)
                loc = site.get_location(name)
                if loc is not None:
                    # Adjust visit_count to match topo (record_visit set it to 1)
                    loc.visit_count = max(visit_count, loc.visit_count)
                    loc.tags = tags
            else:
                # Update existing
                existing.coords = coords
                if description:
                    existing.description = description
                existing.visit_count = max(visit_count, existing.visit_count)
                # Merge tags without duplicates
                for tag in tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)

            synced += 1

        return synced

    def get_current_position(self) -> tuple[float, float, float] | None:
        """Get robot's current position from topo_memory.current_node_id."""
        if not self._topo_data:
            self.load_topo()
        if not self._topo_data:
            return None

        current_id = self._topo_data.get("current_node_id")
        if current_id is None:
            return None

        nodes = self._topo_data.get("nodes", {})
        node = nodes.get(str(current_id))
        if node is None:
            return None

        position = node.get("position", [])
        if len(position) < 3:
            return None

        return (float(position[0]), float(position[1]), float(position[2]))

    def get_nearby_labels(self, node_id: int | None = None) -> list[str]:
        """Get visible object labels at current or specified node."""
        if not self._topo_data:
            self.load_topo()
        if not self._topo_data:
            return []

        if node_id is None:
            node_id = self._topo_data.get("current_node_id")
        if node_id is None:
            return []

        nodes = self._topo_data.get("nodes", {})
        node = nodes.get(str(node_id))
        if node is None:
            return []

        return node.get("visible_labels", [])
