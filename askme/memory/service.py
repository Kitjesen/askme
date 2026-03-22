"""Standalone memory service -- the single entry point for all memory operations.

Any module (askme, nav-gateway, cortex, skills, tools) can import and use this.
Does NOT depend on BrainPipeline, AudioAgent, or any voice/LLM code.

Usage::

    from askme.memory.service import get_memory_service

    mem = get_memory_service()
    mem.record_visit("仓库A", coords=(10, 20))
    mem.record_anomaly("仓库A", "温度异常")
    hotspots = mem.get_anomaly_hotspots()
    best_route = mem.get_best_procedure("navigate")
    context = mem.get_full_context("查询")  # for LLM system prompt
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from askme.memory.map_adapter import MapAdapter
from askme.memory.admission import MemoryAdmissionControl
from askme.memory.procedural import ProceduralMemory
from askme.memory.site_knowledge import Location, SiteKnowledge

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory API -- wraps all memory subsystems.

    Subsystems:
        - SiteKnowledge: spatial memory (locations, anomalies, hotspots)
        - ProceduralMemory: task learning (route success rates)
        - MemoryAdmissionControl: what to remember
        - MapAdapter: sync from LingTu maps (optional)

    Thread-safe. Singleton via get_memory_service().
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        data_base = config.get("data_dir", "data/memory")
        site_dir = config.get("site_dir", f"{data_base}/site")
        proc_dir = config.get("procedures_dir", f"{data_base}/procedures")
        threshold = config.get("admission_threshold", 0.4)

        self._site = SiteKnowledge(data_dir=site_dir)
        self._procedures = ProceduralMemory(data_dir=proc_dir)
        self._admission = MemoryAdmissionControl(threshold=threshold)
        self._map_adapter: MapAdapter | None = None
        self._lock = threading.Lock()

    # -- Spatial --

    def record_visit(self, name: str, coords: tuple[float, float] | None = None,
                     description: str = "") -> None:
        """Record visiting a location."""
        with self._lock:
            self._site.record_visit(name, coords=coords, description=description)

    def record_anomaly(self, location: str, description: str,
                       coords: tuple[float, float] | None = None) -> None:
        """Record an anomaly at a location."""
        with self._lock:
            self._site.record_anomaly(location, description, coords=coords)

    def record_observation(self, location: str, description: str,
                           coords: tuple[float, float] | None = None) -> None:
        """Record a general observation at a location."""
        with self._lock:
            self._site.record_observation(location, description, coords=coords)

    def get_anomaly_hotspots(self, min_count: int = 2) -> list[Location]:
        """Get locations with frequent anomalies."""
        return self._site.get_anomaly_hotspots(min_count=min_count)

    def find_nearby(self, coords: tuple[float, float],
                    radius: float = 10.0) -> list[Location]:
        """Find locations within radius of coordinates."""
        return self._site.find_nearby(coords, radius=radius)

    def get_location_history(self, name: str, limit: int = 10) -> list:
        """Get recent events at a location."""
        return self._site.get_location_history(name, limit=limit)

    # -- Procedural --

    def record_task_outcome(self, name: str, task_type: str, success: bool,
                            duration: float = 0.0, description: str = "") -> None:
        """Record a task outcome for procedural learning."""
        with self._lock:
            self._procedures.record_outcome(
                name, task_type, success=success,
                duration=duration, description=description,
            )

    def get_best_procedure(self, task_type: str):
        """Get the best-scoring procedure for a task type."""
        return self._procedures.get_best_procedure(task_type)

    # -- Admission --

    def should_remember(self, kind: str, text: str,
                        importance: float = 0.5) -> bool:
        """Check if an event is worth remembering."""
        admitted, _score = self._admission.should_admit(kind, text, importance)
        return admitted

    # -- Map sync --

    def sync_from_map(self, topo_path: str | None = None,
                      kg_path: str | None = None) -> int:
        """Sync LingTu map data into spatial memory.

        Returns number of locations synced.
        """
        if topo_path is None:
            if self._map_adapter is None:
                logger.warning("sync_from_map called without topo_path and no adapter configured")
                return 0
        else:
            self._map_adapter = MapAdapter(topo_path, kg_path)

        assert self._map_adapter is not None
        with self._lock:
            return self._map_adapter.sync(self._site)

    # -- Context for LLM --

    def get_full_context(self, query: str = "") -> str:
        """Assemble ALL memory context for system prompt injection.

        Combines spatial + procedural context.
        """
        parts: list[str] = []
        site_ctx = self._site.get_context(query)
        if site_ctx:
            parts.append(site_ctx)
        proc_ctx = self._procedures.get_context()
        if proc_ctx:
            parts.append(proc_ctx)
        return "\n".join(parts)

    # -- Persistence --

    def save(self) -> None:
        """Persist all subsystems."""
        with self._lock:
            self._site._save()
            self._procedures._save()

    def load(self) -> None:
        """Reload all subsystems from disk."""
        with self._lock:
            self._site._load()
            self._procedures._load()

    # -- Direct access (for advanced use) --

    @property
    def site(self) -> SiteKnowledge:
        """Direct access to SiteKnowledge."""
        return self._site

    @property
    def procedures(self) -> ProceduralMemory:
        """Direct access to ProceduralMemory."""
        return self._procedures

    @property
    def admission(self) -> MemoryAdmissionControl:
        """Direct access to MemoryAdmissionControl."""
        return self._admission


# Singleton
_instance: MemoryService | None = None
_singleton_lock = threading.Lock()


def get_memory_service(config: dict[str, Any] | None = None) -> MemoryService:
    """Get or create the singleton MemoryService instance."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = MemoryService(config)
        return _instance


def reset_memory_service() -> None:
    """Reset singleton (for testing only)."""
    global _instance
    with _singleton_lock:
        _instance = None
