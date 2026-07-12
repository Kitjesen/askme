from __future__ import annotations

import asyncio
from typing import Any

from askme.runtime.mission import MissionService
from askme.runtime.module import Module, ModuleRegistry, Runtime

from askme.cognition import (
    ActivePerceptionResolver,
    CognitivePlanner,
    WorkingMemory,
    WorldStateService,
)
from askme.runtime.modules import CognitionModule, MemoryModule, MissionModule


class StubVisionBridge:
    def __init__(self) -> None:
        self.calls = 0

    async def describe_scene(self) -> dict[str, Any]:
        self.calls += 1
        return {
            "source": "stub_vision",
            "summary": "fresh valve in front",
            "objects": [
                {
                    "class_id": "valve",
                    "confidence": 0.94,
                    "distance_m": 1.2,
                    "track_id": "valve-1",
                }
            ],
        }


class StubPerceptionModule(Module):
    name = "perception"

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self._vision_bridge = StubVisionBridge()
        self._world_state = None

    @property
    def vision_bridge(self) -> StubVisionBridge:
        return self._vision_bridge

    @property
    def world_state(self) -> Any:
        return self._world_state


def test_active_perception_resolver_refreshes_and_replans() -> None:
    world = WorldStateService()
    planner = CognitivePlanner(
        world_state=world,
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    async def refresh(context: dict[str, Any]) -> dict[str, Any]:
        world.update_scene(
            summary="fresh valve in front",
            objects=[{"class_id": "valve", "confidence": 0.94, "track_id": "valve-1"}],
            source="stub_vision",
        )
        return {"refreshed": True, "request": context["request"]}

    initial = planner.plan_from_text("grab this", robot_id="dog-1")
    resolver = ActivePerceptionResolver(world_state=world, refresh=refresh)

    result = asyncio.run(
        resolver.resolve(
            initial,
            replan=lambda: planner.plan_from_text(
                "grab this",
                robot_id="dog-1",
                planning_session_id=initial.planning_session_id,
            ),
        )
    )
    plan = result["plan"].to_dict()

    assert result["active_perception"]["requested"] is True
    assert result["active_perception"]["refresh"]["refreshed"] is True
    assert result["active_perception"]["resolved_after_refresh"] is True
    assert plan["reference"]["resolved"]["label"] == "valve"
    assert plan["missing_inputs"] == ["operator_confirmation"]
    assert world.snapshot()["events"][0]["kind"] == "perception_refresh_requested"


def test_cognition_module_active_perception_uses_local_stub_refresh() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(StubPerceptionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    result = asyncio.run(mod.plan_from_payload({"text": "grab this", "robot_id": "dog-1"}))
    context = asyncio.run(mod.context_payload())

    assert result["active_perception"]["requested"] is True
    assert result["active_perception"]["refresh"]["refreshed"] is True
    assert result["active_perception"]["refresh"]["perception"]["object_count"] == 1
    assert result["plan"]["reference"]["resolved"]["label"] == "valve"
    assert result["plan"]["interaction_state"] == "awaiting_confirmation"
    assert context["world_state"]["events"][0]["kind"] == "perception_refresh_requested"
    assert app.modules["perception"].vision_bridge.calls == 1
