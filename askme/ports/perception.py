"""Perception provider contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VisionPort(Protocol):
    """Vision capability consumed by runtime, pipeline, and tools."""

    @property
    def available(self) -> bool:
        """Whether a vision backend is usable."""

    def auto_capture_enabled(self) -> bool:
        """Whether turns should capture scene context automatically."""

    async def describe_scene(self, frame: Any = None) -> str:
        """Describe the current scene."""

    async def describe_scene_with_question(self, question: str, frame: Any = None) -> str:
        """Answer a targeted question about the current scene."""

    async def find_object(self, target: str, frame: Any = None) -> dict[str, Any] | None:
        """Find a target object in the current scene."""

    async def save_snapshot(
        self,
        frame: Any = None,
        *,
        label: str = "snapshot",
        output_dir: str = "data/captures",
    ) -> str | None:
        """Persist a current-frame snapshot."""

    def interaction_snapshot(self, max_age: float = 2.0) -> dict[str, Any]:
        """Return gate-ready interaction evidence."""


@runtime_checkable
class InteractionPerceptionPort(Protocol):
    """Interaction-gate perception capability from external sensor algorithms."""

    @property
    def enabled(self) -> bool:
        """Whether external interaction perception is configured."""

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """Return a merged interaction snapshot."""


@runtime_checkable
class ChangeMonitorPort(Protocol):
    """Background monitor that emits perception change events."""

    @property
    def is_active(self) -> bool:
        """Whether the monitor is currently producing events."""

    async def run(self, stop_event: Any) -> None:
        """Run until *stop_event* is set."""


@runtime_checkable
class SceneIntelligencePort(Protocol):
    """Scene-summary capability consumed by MCP/API composition code."""

    def who_is_around(self) -> list[str]:
        """Return known nearby entities."""

    def anomalies(self) -> list[dict[str, Any]]:
        """Return current anomaly records."""

    async def briefing(self) -> str:
        """Return a concise scene briefing."""

    async def today_summary(self, llm: Any) -> str:
        """Return an LLM-assisted summary for the current day."""


__all__ = [
    "ChangeMonitorPort",
    "InteractionPerceptionPort",
    "SceneIntelligencePort",
    "VisionPort",
]
