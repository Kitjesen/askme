"""CognitionModule - world state, working memory, and safe mission planning."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from askme.cognition import (
    ActivePerceptionResolver,
    CognitionPerceptionSync,
    CognitivePlanner,
    WorkingMemory,
    WorldStateService,
)
from askme.runtime.module import Module, ModuleRegistry, Out

logger = logging.getLogger(__name__)


class CognitionModule(Module):
    """Provide robot-aware context and planning without owning execution."""

    name = "cognition"
    depends_on = ("memory", "mission", "pulse", "perception")
    provides = ("world_state", "working_memory", "cognitive_planner")

    world_state: Out[WorldStateService]
    working_memory: Out[WorkingMemory]
    cognitive_planner: Out[CognitivePlanner]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        cog_cfg = cfg.get("cognition", {}) if isinstance(cfg, dict) else {}
        self.enabled = bool(cog_cfg.get("enabled", True))
        self._sync_enabled = bool(cog_cfg.get("sync_enabled", True))
        self._sync_interval_s = max(0.1, float(cog_cfg.get("sync_interval_seconds", 1.0)))
        self._scene_stale_after_s = max(0.1, float(cog_cfg.get("scene_stale_after_s", 3.0)))
        self._robot_stale_after_s = max(0.1, float(cog_cfg.get("robot_stale_after_s", 5.0)))

        self._world_state = WorldStateService(
            max_facts=int(cog_cfg.get("max_facts", 200)),
            max_events=int(cog_cfg.get("max_events", 100)),
            default_stale_after_s=float(cog_cfg.get("default_stale_after_s", 10.0)),
        )
        self._working_memory = WorkingMemory(
            enabled=bool(cog_cfg.get("working_memory_enabled", True)),
            max_items=int(cog_cfg.get("working_memory_max_items", 80)),
            retention_seconds=float(cog_cfg.get("working_memory_retention_seconds", 1800.0)),
            persist_enabled=bool(cog_cfg.get("working_memory_persist_enabled", False)),
        )

        mission_mod = registry.get("mission")
        mission_service = getattr(mission_mod, "mission_service", None) if mission_mod else None
        self._planner = CognitivePlanner(
            world_state=self._world_state,
            working_memory=self._working_memory,
            mission_service=mission_service,
            max_sessions=int(cog_cfg.get("planning_session_max_items", 20)),
        )

        perception_mod = registry.get("perception")
        self._vision_bridge = getattr(perception_mod, "vision_bridge", None) if perception_mod else None
        self._perception_world_state = getattr(perception_mod, "world_state", None) if perception_mod else None

        pulse_mod = registry.get("pulse")
        self._pulse_bus = getattr(pulse_mod, "bus", None) if pulse_mod else None

        change_detector = getattr(perception_mod, "change_detector", None) if perception_mod else None
        event_file = (
            cog_cfg.get("change_event_file")
            or getattr(change_detector, "_event_file", None)
            or None
        )
        self._perception_sync = CognitionPerceptionSync(
            self._world_state,
            event_file=event_file,
            scene_stale_after_s=self._scene_stale_after_s,
            robot_stale_after_s=self._robot_stale_after_s,
            max_event_lines_per_sync=int(cog_cfg.get("max_event_lines_per_sync", 100)),
        )
        self._active_perception = ActivePerceptionResolver(
            world_state=self._world_state,
            refresh=self._active_perception_refresh,
        )
        self._sync_task: asyncio.Task[None] | None = None
        self._last_sync: dict[str, Any] | None = None
        self._seed_runtime_context(cfg)
        logger.info(
            "CognitionModule: built (enabled=%s, mission=%s, vision=%s, pulse=%s, sync=%s)",
            self.enabled,
            mission_service is not None,
            self._vision_bridge is not None,
            self._pulse_bus is not None,
            self._sync_enabled,
        )

    @property
    def world_state(self) -> WorldStateService:  # type: ignore[override]
        return self._world_state

    @property
    def working_memory(self) -> WorkingMemory:  # type: ignore[override]
        return self._working_memory

    @property
    def cognitive_planner(self) -> CognitivePlanner:  # type: ignore[override]
        return self._planner

    async def start(self) -> None:
        if self.enabled and self._sync_enabled and self._sync_task is None:
            self._sync_task = asyncio.create_task(
                self._sync_loop(),
                name="askme-cognition-sync",
            )

    async def stop(self) -> None:
        task = self._sync_task
        self._sync_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def refresh_runtime_context(self) -> dict[str, Any]:
        """Pull Pulse/perception signals into the cognition world state."""
        if not self.enabled:
            return {"synced": False, "reason": "disabled"}
        self._last_sync = await self._perception_sync.sync_once(
            pulse_bus=self._pulse_bus,
            perception_world_state=self._perception_world_state,
        )
        return dict(self._last_sync)

    async def refresh_perception(self) -> dict[str, Any]:
        """Pull one lightweight scene summary into the world-state cache."""
        if not self.enabled:
            return {"refreshed": False, "reason": "disabled"}
        vision = self._vision_bridge
        if vision is None:
            return {"refreshed": False, "reason": "vision_not_configured"}
        describe = getattr(vision, "describe_scene", None)
        if not callable(describe):
            return {"refreshed": False, "reason": "vision_describe_unavailable"}
        try:
            summary = await describe()
        except Exception as exc:
            logger.debug("CognitionModule: perception refresh failed: %s", exc)
            return {"refreshed": False, "reason": str(exc)}
        if isinstance(summary, dict):
            objects = [
                item
                for item in summary.get("objects", [])
                if isinstance(item, dict)
            ]
            self._world_state.update_scene(
                summary=str(summary.get("summary", "")),
                objects=objects,
                source=str(summary.get("source") or "vision_bridge"),
                stale_after_s=self._scene_stale_after_s,
            )
            return {
                "refreshed": True,
                "summary": str(summary.get("summary", "")),
                "object_count": len(objects),
                "source": str(summary.get("source") or "vision_bridge"),
            }
        self._world_state.update_fact(
            "scene.summary",
            str(summary or ""),
            source="vision_bridge",
            stale_after_s=self._scene_stale_after_s,
        )
        return {"refreshed": True, "summary": str(summary or "")}

    async def context_payload(self, *, refresh_perception: bool = False) -> dict[str, Any]:
        sync = None
        refresh = None
        if refresh_perception:
            sync = await self.refresh_runtime_context()
            refresh = await self.refresh_perception()
        payload = self._planner.context_payload()
        if sync is not None:
            payload["sync"] = sync
        if refresh is not None:
            payload["refresh"] = refresh
        return payload

    async def plan_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("cognition module is disabled")
        sync = await self.refresh_runtime_context()
        if bool(payload.get("refresh_perception", False)):
            await self.refresh_perception()
        text = str(payload.get("text") or payload.get("message") or payload.get("goal") or "")
        planning_session_id = _clean_optional(
            payload.get("planning_session_id") or payload.get("session_id")
        )

        def _plan_once(session_id: str | None = planning_session_id) -> Any:
            return self._planner.plan_from_text(
                text,
                operator_id=_clean_optional(payload.get("operator_id")),
                robot_id=_clean_optional(payload.get("robot_id")),
                site_id=_clean_optional(payload.get("site_id")),
                channel=str(payload.get("channel", "http-cognition")).strip() or "http-cognition",
                metadata=_metadata(payload),
                planning_session_id=session_id,
                reply_to_plan_id=_clean_optional(payload.get("reply_to_plan_id")),
                operator_confirmation=payload.get("operator_confirmation", payload.get("confirm")),
                cancel=bool(payload.get("cancel", False)) or str(payload.get("action", "")).lower() == "cancel",
                revise_goal=_clean_optional(payload.get("revise_goal")),
            )

        plan = _plan_once()
        active = await self._active_perception.resolve(
            plan,
            replan=lambda: _plan_once(plan.planning_session_id),
            refresh_context={"payload": dict(payload), "sync": sync},
        )
        resolved_plan = active["plan"]
        return {
            "plan": resolved_plan.to_dict(),
            "planned": True,
            "sync": sync,
            "active_perception": active["active_perception"],
        }

    def health(self) -> dict[str, Any]:
        context = self._planner.context_payload()
        last_sync = self._last_sync or {}
        return {
            "status": "ok" if self.enabled else "disabled",
            "world_fact_count": context["world_state"]["fact_count"],
            "working_memory_items": context["working_memory"]["item_count"],
            "planning_session_count": len(context.get("planning_sessions", [])),
            "planner_ready": self._planner is not None,
            "sync_enabled": self._sync_enabled,
            "last_sync_at": last_sync.get("last_sync_at", 0.0),
            "synced_event_count": last_sync.get("synced_event_count", 0),
            "fresh_object_count": last_sync.get("fresh_object_count", len(context["world_state"]["scene"]["objects"])),
            "sync_errors": last_sync.get("errors", []),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "world_state": True,
            "working_memory": True,
            "pulse_context_sync": True,
            "change_event_sync": True,
            "perception_world_state_sync": True,
            "planning_sessions": True,
            "mission_draft_planning": True,
            "active_perception_refresh": True,
            "hardware_dispatch": False,
            "http_paths": [
                "GET /api/cognition/context",
                "POST /api/cognition/plan",
            ],
        }

    def _seed_runtime_context(self, cfg: dict[str, Any]) -> None:
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        voice_bridge = runtime_cfg.get("voice_bridge", {}) if isinstance(runtime_cfg, dict) else {}
        seed: dict[str, Any] = {}
        for key in ("robot_id", "site_id", "operator_id", "session_id", "channel"):
            value = voice_bridge.get(key)
            if value:
                seed[key] = value
        if seed:
            self._world_state.update_robot_state(seed, source="config", stale_after_s=None)

    async def _sync_loop(self) -> None:
        while True:
            try:
                await self.refresh_runtime_context()
            except Exception as exc:
                logger.debug("CognitionModule: runtime context sync failed: %s", exc)
            await asyncio.sleep(self._sync_interval_s)

    async def _active_perception_refresh(self, context: dict[str, Any]) -> dict[str, Any]:
        sync = await self.refresh_runtime_context()
        refresh = await self.refresh_perception()
        return {
            "refreshed": bool(refresh.get("refreshed")),
            "sync": sync,
            "perception": refresh,
            "request": dict(context.get("request", {})),
        }


def _clean_optional(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(key, str) and not key.startswith("_")
    }
