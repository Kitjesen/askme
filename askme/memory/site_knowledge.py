"""Spatial memory — learns factory layout over time.

The robot builds a mental map of locations, landmarks, and spatial
relationships from patrol experiences. Enables queries like
"what's near the loading dock?" or "where was the last anomaly?"

Reference: Meta-Memory (arXiv 2509.20754, Sep 2025)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Location:
    name: str                    # "仓库A", "3号门", "充电桩"
    coords: tuple[float, float] | None = None  # (x, y) from LingTu
    description: str = ""        # "东北角的大仓库，旁边有红色消防栓"
    visit_count: int = 0
    last_visited: float = 0.0    # timestamp
    anomaly_count: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coords": list(self.coords) if self.coords else None,
            "description": self.description,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited,
            "anomaly_count": self.anomaly_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Location:
        coords = tuple(data["coords"]) if data.get("coords") else None
        return cls(
            name=data["name"],
            coords=coords,
            description=data.get("description", ""),
            visit_count=data.get("visit_count", 0),
            last_visited=data.get("last_visited", 0.0),
            anomaly_count=data.get("anomaly_count", 0),
            tags=data.get("tags", []),
        )


@dataclass
class SpatialEvent:
    location_name: str
    event_type: str              # "visit", "anomaly", "observation"
    description: str
    timestamp: float
    coords: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_name": self.location_name,
            "event_type": self.event_type,
            "description": self.description,
            "timestamp": self.timestamp,
            "coords": list(self.coords) if self.coords else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialEvent:
        coords = tuple(data["coords"]) if data.get("coords") else None
        return cls(
            location_name=data["location_name"],
            event_type=data["event_type"],
            description=data.get("description", ""),
            timestamp=data.get("timestamp", 0.0),
            coords=coords,
        )


class SiteKnowledge:
    """Accumulates spatial knowledge about the patrol site.

    Over time, the robot learns:
    - Where things are (location registry)
    - What happened where (spatial event log)
    - Spatial relationships ("仓库A is near 3号门")
    - Anomaly hotspots (locations with frequent anomalies)
    """

    def __init__(self, data_dir: str = "data/memory/site") -> None:
        self._locations: dict[str, Location] = {}
        self._events: list[SpatialEvent] = []
        self._max_events: int = 500  # prevent unbounded growth
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def record_visit(self, name: str, coords: tuple[float, float] | None = None,
                     description: str = "") -> None:
        """Record visiting a location. Creates it if new, increments visit_count."""
        loc = self._locations.get(name)
        if loc is None:
            loc = Location(name=name, coords=coords, description=description)
            self._locations[name] = loc
        loc.visit_count += 1
        loc.last_visited = time.time()
        if coords is not None:
            loc.coords = coords
        if description:
            loc.description = description
        self._events.append(SpatialEvent(
            location_name=name, event_type="visit",
            description=description or f"Visited {name}",
            timestamp=time.time(), coords=coords,
        ))
        self._trim_events()
        self._save()

    def record_anomaly(self, location_name: str, description: str,
                       coords: tuple[float, float] | None = None) -> None:
        """Record an anomaly at a location."""
        loc = self._locations.get(location_name)
        if loc is None:
            loc = Location(name=location_name, coords=coords)
            self._locations[location_name] = loc
        loc.anomaly_count += 1
        if coords is not None:
            loc.coords = coords
        self._events.append(SpatialEvent(
            location_name=location_name, event_type="anomaly",
            description=description,
            timestamp=time.time(), coords=coords,
        ))
        self._trim_events()
        self._save()

    def record_observation(self, location_name: str, description: str,
                           coords: tuple[float, float] | None = None) -> None:
        """Record a general observation at a location."""
        loc = self._locations.get(location_name)
        if loc is None:
            loc = Location(name=location_name, coords=coords)
            self._locations[location_name] = loc
        if coords is not None:
            loc.coords = coords
        self._events.append(SpatialEvent(
            location_name=location_name, event_type="observation",
            description=description,
            timestamp=time.time(), coords=coords,
        ))
        self._trim_events()
        self._save()

    def get_location(self, name: str) -> Location | None:
        """Get a known location by name."""
        return self._locations.get(name)

    def find_nearby(self, coords: tuple[float, float], radius: float = 10.0) -> list[Location]:
        """Find locations within radius of coordinates."""
        result: list[Location] = []
        for loc in self._locations.values():
            if loc.coords is None:
                continue
            dx = loc.coords[0] - coords[0]
            dy = loc.coords[1] - coords[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                result.append(loc)
        return result

    def get_anomaly_hotspots(self, min_count: int = 2) -> list[Location]:
        """Get locations with frequent anomalies."""
        return [loc for loc in self._locations.values()
                if loc.anomaly_count >= min_count]

    def get_location_history(self, name: str, limit: int = 10) -> list[SpatialEvent]:
        """Get recent events at a location."""
        events = [e for e in self._events if e.location_name == name]
        return events[-limit:]

    def get_context(self, query: str = "") -> str:
        """Get spatial context for system prompt injection.

        Returns summary of known locations + recent spatial events.
        """
        if not self._locations:
            return ""
        parts: list[str] = ["[已知地点]"]
        for loc in self._locations.values():
            line = f"- {loc.name}: 访问{loc.visit_count}次"
            if loc.anomaly_count:
                line += f", 异常{loc.anomaly_count}次"
            if loc.description:
                line += f" ({loc.description})"
            parts.append(line)

        recent = self._events[-5:]
        if recent:
            parts.append("[近期空间事件]")
            for ev in recent:
                ts = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
                parts.append(f"- [{ts}] {ev.location_name}: {ev.description}")

        return "\n".join(parts)

    def _trim_events(self) -> None:
        """Keep only the most recent events to prevent unbounded growth."""
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def _save(self) -> None:
        """Persist to JSON files."""
        loc_path = self._data_dir / "locations.json"
        evt_path = self._data_dir / "events.json"
        try:
            with open(loc_path, "w", encoding="utf-8") as f:
                json.dump([loc.to_dict() for loc in self._locations.values()],
                          f, ensure_ascii=False, indent=2)
            with open(evt_path, "w", encoding="utf-8") as f:
                json.dump([ev.to_dict() for ev in self._events],
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("SiteKnowledge save failed: %s", e)

    def _load(self) -> None:
        """Load from JSON files."""
        loc_path = self._data_dir / "locations.json"
        evt_path = self._data_dir / "events.json"
        if loc_path.exists():
            try:
                with open(loc_path, encoding="utf-8") as f:
                    for item in json.load(f):
                        loc = Location.from_dict(item)
                        self._locations[loc.name] = loc
            except Exception as e:
                logger.warning("SiteKnowledge load locations failed: %s", e)
        if evt_path.exists():
            try:
                with open(evt_path, encoding="utf-8") as f:
                    self._events = [SpatialEvent.from_dict(item) for item in json.load(f)]
            except Exception as e:
                logger.warning("SiteKnowledge load events failed: %s", e)
