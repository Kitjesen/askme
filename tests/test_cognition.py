from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from askme.robot.mock_pulse import MockPulse
from askme.runtime.mission import MissionService
from askme.runtime.module import Module, ModuleRegistry, Runtime

from askme.cognition import (
    CognitionPerceptionSync,
    CognitivePlanner,
    WorkingMemory,
    WorldStateService,
)
from askme.perception.world_state import WorldState as PerceptionWorldState
from askme.runtime.modules import CognitionModule, MemoryModule, MissionModule
from askme.schemas.events import ChangeEvent, ChangeEventType


class MockPulseModule(Module):
    name = "pulse"

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self._bus = MockPulse()

    @property
    def bus(self) -> MockPulse:
        return self._bus


class _MalformedPerceptionWorld:
    async def snapshot(self) -> dict[str, Any]:
        return {"summary": "schema changed"}


class _FakeVisionBridge:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def describe_scene(self) -> dict[str, Any]:
        return self.payload


class _FakePerceptionModule(Module):
    name = "perception"
    payload: dict[str, Any] = {}

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.vision_bridge = _FakeVisionBridge(dict(self.payload))
        self.world_state = None


def test_world_state_resolves_fresh_deictic_reference() -> None:
    world = WorldStateService()
    world.update_scene(
        summary="operator is pointing at a valve",
        objects=[{"class_id": "valve", "confidence": 0.9, "distance_m": 1.2}],
    )

    result = world.resolve_reference("检查这个")

    assert result["needs_clarification"] is False
    assert result["resolved"]["label"] == "valve"


def test_world_state_requires_clarification_without_fresh_reference() -> None:
    world = WorldStateService()

    result = world.resolve_reference("拿那个")

    assert result["requires_reference"] is True
    assert result["needs_clarification"] is True
    assert result["reason"] == "no_fresh_scene_object"


def test_world_state_applies_change_event_appearance_and_departure() -> None:
    world = WorldStateService()
    world.apply_change_event(
        ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=time.time(),
            subject_class="person",
            confidence=0.91,
            track_id="person-1",
            distance_m=2.0,
        )
    )

    assert world.fresh_objects()[0]["label"] == "person"

    world.apply_change_event(
        ChangeEvent(
            event_type=ChangeEventType.PERSON_LEFT,
            timestamp=time.time(),
            subject_class="person",
            track_id="person-1",
        )
    )

    assert world.fresh_objects() == []


def test_world_state_tracks_area_device_and_map_catalogs() -> None:
    world = WorldStateService()
    world.update_area_catalog(
        [
            {
                "area_id": "area-a",
                "name": "Area A",
                "route_id": "route-a",
                "map_id": "map-main",
                "map_version": "v1",
            }
        ],
        map_id="map-main",
        map_version="v1",
    )
    world.update_device_catalog(
        [
            {
                "device_id": "panel-3",
                "area_id": "area-a",
                "device_type": "status_panel",
                "status": "normal",
            }
        ]
    )
    world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=True,
        localization_quality=0.91,
    )

    snapshot = world.snapshot()

    assert world.get_area("area-a")["route_id"] == "route-a"
    assert world.get_device("panel-3")["area_id"] == "area-a"
    assert snapshot["environment"]["areas"][0]["map_version"] == "v1"
    assert snapshot["environment"]["devices"][0]["device_type"] == "status_panel"
    assert snapshot["map"]["current_id"] == "map-main"
    assert snapshot["map"]["localization_quality"] == 0.91


def test_world_state_snapshot_exposes_scene_observed_at() -> None:
    world = WorldStateService()
    observed_at = time.time() - 0.5
    world.update_scene(
        objects=[{"class_id": "person", "confidence": 0.91}],
        observed_at=observed_at,
        stale_after_s=2.0,
    )

    snapshot = world.snapshot()

    assert snapshot["scene"]["observed_at"] == observed_at
    assert snapshot["scene"]["stale"] is False


def test_working_memory_keeps_ephemeral_turn_context() -> None:
    memory = WorkingMemory(retention_seconds=60)

    memory.record_turn("巡检 A 区", observations=["scene has a door"])
    memory.set_focus(route="A")
    snapshot = memory.snapshot()

    assert snapshot["persist_enabled"] is False
    assert snapshot["item_count"] == 2
    assert snapshot["focus"]["route"] == "A"
    assert "巡检 A 区" in memory.summary()


def test_cognitive_planner_drafts_mission_without_dispatching() -> None:
    world = WorldStateService()
    memory = WorkingMemory()
    mission = MissionService()
    planner = CognitivePlanner(
        world_state=world,
        working_memory=memory,
        mission_service=mission,
    )

    plan = planner.plan_from_text(
        "巡检 A 区",
        operator_id="operator-1",
        robot_id="dog-1",
        site_id="factory-1",
    )
    payload = plan.to_dict()

    assert payload["intent"] == "inspection_patrol"
    assert payload["interaction_state"] == "awaiting_confirmation"
    assert payload["mission"]["drafted"] is True
    assert payload["mission"]["mission"]["status"] == "pending_confirmation"
    assert payload["mission"]["mission"]["robot_id"] == "dog-1"
    assert any(step["step"] == "submit_to_arbiter" for step in payload["steps"])
    assert "Do not dispatch hardware actions" in " ".join(payload["safety_constraints"])
    assert payload["planning_session_id"]
    assert payload["handoff_ready"] is False
    assert payload["missing_inputs"] == ["operator_confirmation"]
    assert payload["readiness"]["status"] == "awaiting_operator_confirmation"
    assert payload["readiness"]["can_submit_to_runtime"] is False
    assert payload["readiness"]["blocked_by"] == ["operator_confirmation"]
    assert "confirm_plan" in payload["readiness"]["allowed_next_actions"]
    assert payload["handoff_contract"]["consumer"] == "runtime_handoff"
    assert payload["handoff_contract"]["can_dispatch_hardware"] is False
    assert payload["handoff_contract"]["dispatch_authority"] == "runtime_arbiter_only"
    assert payload["handoff_contract"]["requires_safety_preflight"] is True
    assert payload["world_state_snapshot_id"] == payload["handoff_contract"]["world_state_snapshot_id"]
    assert "确认" in payload["next_prompt"]


def test_cognitive_planner_keeps_visitor_wayfinding_out_of_runtime_handoff() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    plan = planner.plan_from_text("梵木咖啡怎么走？", robot_id="dog-1")
    payload = plan.to_dict()

    assert payload["intent"] == "visitor_wayfinding"
    assert payload["interaction_state"] == "answer_ready"
    assert payload["handoff_ready"] is False
    assert payload["mission"] is None
    assert payload["readiness"]["status"] == "ready_to_answer"
    assert payload["readiness"]["can_submit_to_runtime"] is False
    assert payload["readiness"]["requires_operator_confirmation"] is False
    assert payload["handoff_contract"]["can_dispatch_hardware"] is False
    assert payload["handoff_contract"]["submit_conditions"] == [
        "no_runtime_handoff_for_information_response",
        "answer_must_use_grounded_park_knowledge",
    ]
    assert any(step["step"] == "answer_with_grounded_park_knowledge" for step in payload["steps"])
    assert "不会启动机器狗移动" in payload["next_prompt"]


def test_cognitive_planner_treats_explicit_visitor_escort_as_physical_task() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    draft = planner.plan_from_text("请带我去梵木咖啡", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation="确认",
        robot_id="dog-1",
    )

    assert draft.intent == "visitor_escort"
    assert draft.interaction_state == "awaiting_confirmation"
    assert draft.mission["mission"]["mission_type"] == "navigate_to"
    assert any(step["target"] == "梵木咖啡" for step in draft.mission["mission"]["steps"])
    assert draft.handoff_ready is False
    assert draft.readiness["requires_operator_confirmation"] is True
    assert draft.readiness["requires_safety_preflight"] is True
    assert "确认" in draft.next_prompt
    assert confirmed.interaction_state == "ready_for_arbiter"
    assert confirmed.handoff_ready is True


def test_cognitive_planner_clarifies_ambiguous_manipulation() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    plan = planner.plan_from_text("拿那个")

    assert plan.requires_clarification is True
    assert plan.mission is None
    assert plan.interaction_state == "clarifying"
    assert "目标" in plan.clarification_question
    assert "scene_reference" in plan.missing_inputs


def test_cognitive_planner_confirms_existing_session_for_handoff() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )

    assert draft.interaction_state == "awaiting_confirmation"
    assert draft.handoff_ready is False
    assert confirmed.planning_session_id == draft.planning_session_id
    assert confirmed.interaction_state == "ready_for_arbiter"
    assert confirmed.handoff_ready is True
    assert confirmed.missing_inputs == []
    payload = confirmed.to_dict()
    assert payload["operator_id"] == ""
    assert payload["robot_id"] == "dog-1"
    assert payload["readiness"]["status"] == "ready_for_runtime_handoff"
    assert payload["readiness"]["can_submit_to_runtime"] is True
    assert payload["readiness"]["blocked_by"] == []
    assert "submit_to_runtime_arbiter" in payload["readiness"]["allowed_next_actions"]
    assert payload["handoff_contract"]["confirmed"] is True
    assert payload["handoff_contract"]["handoff_ready"] is True
    assert payload["handoff_contract"]["blocked_by"] == []
    assert payload["handoff_contract"]["world_state_snapshot_id"].startswith("world-")
    assert payload["world_state_snapshot_id"] == payload["handoff_contract"]["world_state_snapshot_id"]
    assert any(
        step["status"] == "ready"
        for step in confirmed.steps
        if step["step"] == "submit_to_arbiter"
    )


def test_cognitive_planner_blocks_handoff_without_mission_draft() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=None,
    )

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    payload = confirmed.to_dict()

    assert payload["mission"] is None
    assert payload["handoff_ready"] is False
    assert "mission_draft" in payload["missing_inputs"]
    assert payload["readiness"]["can_submit_to_runtime"] is False
    assert payload["handoff_contract"]["confirmed"] is False
    assert payload["handoff_contract"]["blocked_by"] == ["mission_draft"]
    assert any(
        step["status"] != "ready"
        for step in payload["steps"]
        if step["step"] == "submit_to_arbiter"
    )


def test_cognitive_planner_cancels_existing_session() -> None:
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=WorkingMemory(),
        mission_service=MissionService(),
    )

    draft = planner.plan_from_text("inspect area-a")
    cancelled = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        cancel=True,
    )

    assert cancelled.interaction_state == "cancelled"
    assert cancelled.handoff_ready is False
    assert cancelled.mission is None
    assert cancelled.session["confirmation_status"] == "cancelled"
    assert cancelled.readiness["status"] == "cancelled"
    assert cancelled.handoff_contract["can_dispatch_hardware"] is False


def test_perception_sync_reads_pulse_detections_and_robot_state() -> None:
    pulse = MockPulse()
    pulse.publish(
        "/thunder/detections",
        {
            "timestamp": time.time(),
            "frame_id": 1,
            "detections": [
                {
                    "class_id": "person",
                    "confidence": 0.88,
                    "bbox": [0, 0, 10, 10],
                    "distance_m": 1.5,
                }
            ],
        },
    )
    pulse.publish("/thunder/estop", {"active": True, "_ts": time.time()})
    pulse.publish(
        "/thunder/cms_state",
        {"state": "Standing", "addr": "robot-1", "_ts": time.time()},
    )
    world = WorldStateService()
    sync = CognitionPerceptionSync(world)

    result = asyncio.run(sync.sync_once(pulse_bus=pulse))
    snapshot = world.snapshot()

    assert "/thunder/detections" in result["pulse_topics"]
    assert snapshot["scene"]["objects"][0]["label"] == "person"
    assert snapshot["robot"]["estop_active"] is True
    assert snapshot["robot"]["cms_state"] == "Standing"


def test_perception_sync_keeps_robot_pulse_payload_timestamp_for_freshness() -> None:
    pulse = MockPulse()
    old_ts = time.time() - 10.0
    pulse.publish("/thunder/estop", {"active": True, "_ts": old_ts})
    world = WorldStateService()
    sync = CognitionPerceptionSync(world, robot_stale_after_s=1.0)

    asyncio.run(sync.sync_once(pulse_bus=pulse))

    assert world.get_fact("robot.estop_active") is None
    stale_fact = world.get_fact("robot.estop_active", include_stale=True)
    assert stale_fact is not None
    assert stale_fact.observed_at == old_ts
    assert "robot.estop_active" in world.snapshot()["stale_keys"]


def test_perception_sync_reads_change_event_jsonl(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event = ChangeEvent(
        event_type=ChangeEventType.OBJECT_APPEARED,
        timestamp=time.time(),
        subject_class="valve",
        confidence=0.84,
        track_id="valve-7",
    )
    event_file.write_text(json.dumps(event.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    world = WorldStateService()
    sync = CognitionPerceptionSync(world, event_file=event_file)

    result = asyncio.run(sync.sync_once())

    assert result["event_count"] == 1
    assert result["synced_event_count"] == 1
    assert world.fresh_objects()[0]["track_id"] == "valve-7"


def test_perception_sync_bridges_real_perception_world_state() -> None:
    perception = PerceptionWorldState()
    perception.apply_event_sync(
        ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=time.time(),
            subject_class="valve",
            confidence=0.91,
            bbox=(10, 20, 30, 40),
            distance_m=1.4,
            track_id="valve-9",
        )
    )
    world = WorldStateService()
    sync = CognitionPerceptionSync(world)

    first = asyncio.run(sync.sync_once(perception_world_state=perception))
    second = asyncio.run(sync.sync_once(perception_world_state=perception))
    objects = world.fresh_objects()

    assert first["perception_world_state"]["synced"] is True
    assert second["perception_world_state"]["object_count"] == 1
    assert len(objects) == 1
    assert objects[0]["track_id"] == "valve-9"
    assert objects[0]["label"] == "valve"
    assert objects[0]["bbox"] == [10, 20, 30, 40]
    assert objects[0]["distance_m"] == 1.4
    assert world.resolve_reference("grab this")["resolved"]["track_id"] == "valve-9"


def test_perception_sync_rejects_malformed_snapshot_without_clearing_scene() -> None:
    world = WorldStateService()
    world.update_scene(
        objects=[{"class_id": "person", "confidence": 0.91, "track_id": "person-1"}],
        stale_after_s=30.0,
    )
    sync = CognitionPerceptionSync(world)

    result = asyncio.run(sync.sync_once(perception_world_state=_MalformedPerceptionWorld()))
    objects = world.fresh_objects()

    assert result["perception_world_state"] == {
        "synced": False,
        "reason": "snapshot_objects_missing",
    }
    assert len(objects) == 1
    assert objects[0]["track_id"] == "person-1"


def test_perception_sync_respects_stale_perception_world_state() -> None:
    perception = PerceptionWorldState()
    perception.apply_event_sync(
        ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=time.time() - 10.0,
            subject_class="valve",
            confidence=0.91,
            track_id="stale-valve",
        )
    )
    world = WorldStateService()
    sync = CognitionPerceptionSync(world, scene_stale_after_s=0.5)

    result = asyncio.run(sync.sync_once(perception_world_state=perception))

    assert result["perception_world_state"]["synced"] is True
    assert world.fresh_objects() == []
    assert world.snapshot()["scene"]["stale"] is True
    assert world.resolve_reference("grab this")["reason"] == "no_fresh_scene_object"


def test_cognition_module_builds_with_memory_and_mission() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )

    app = asyncio.run(runtime.build({}))
    mod = app.modules["cognition"]

    assert mod.health()["planner_ready"] is True
    assert mod.capabilities()["hardware_dispatch"] is False
    assert mod.capabilities()["mission_draft_planning"] is True
    assert mod.capabilities()["pulse_context_sync"] is False
    assert mod.capabilities()["perception_world_state_sync"] is False


def test_cognition_module_plans_with_fresh_pulse_context() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(MockPulseModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    pulse = app.modules["pulse"].bus
    pulse.publish(
        "/thunder/detections",
        {
            "timestamp": time.time(),
            "frame_id": 8,
            "detections": [
                {
                    "class_id": "valve",
                    "confidence": 0.93,
                    "bbox": [4, 8, 42, 80],
                    "distance_m": 1.1,
                }
            ],
        },
    )

    mod = app.modules["cognition"]
    result = asyncio.run(mod.plan_from_payload({"text": "检查这个", "robot_id": "dog-1"}))

    assert result["sync"]["fresh_object_count"] == 1
    assert result["plan"]["reference"]["resolved"]["label"] == "valve"
    assert result["plan"]["mission"]["mission"]["robot_id"] == "dog-1"


def test_cognition_module_refresh_perception_preserves_snapshot_freshness() -> None:
    observed_at = time.time() - 10.0
    _FakePerceptionModule.payload = {
        "summary": "old frame",
        "objects": [
            {
                "class_id": "person",
                "confidence": 0.93,
                "last_seen": observed_at,
                "track_id": "p-old",
            }
        ],
        "source": "fake_vision",
    }
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(_FakePerceptionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"scene_stale_after_s": 0.5, "sync_enabled": False}}))
    mod = app.modules["cognition"]

    result = asyncio.run(mod.refresh_perception())
    snapshot = mod.world_state.snapshot()

    assert result["refreshed"] is True
    assert result["observed_at"] == observed_at
    assert snapshot["scene"]["observed_at"] == observed_at
    assert snapshot["scene"]["objects"] == []
    assert "scene.objects" in snapshot["stale_keys"]
    assert mod.capabilities()["active_perception_refresh"] is True


def test_cognition_module_payload_confirms_planning_session() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    draft = asyncio.run(mod.plan_from_payload({"text": "inspect area-a", "robot_id": "dog-1"}))
    confirmed = asyncio.run(
        mod.plan_from_payload(
            {
                "planning_session_id": draft["plan"]["planning_session_id"],
                "operator_confirmation": True,
                "robot_id": "dog-1",
            }
        )
    )
    context = asyncio.run(mod.context_payload())

    assert confirmed["plan"]["interaction_state"] == "ready_for_arbiter"
    assert confirmed["plan"]["handoff_ready"] is True
    assert confirmed["plan"]["missing_inputs"] == []
    assert context["planning_sessions"][0]["session_id"] == draft["plan"]["planning_session_id"]
    assert mod.health()["planning_session_count"] == 1


def test_cognition_module_payload_cancels_planning_session() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    draft = asyncio.run(mod.plan_from_payload({"text": "inspect area-a"}))
    cancelled = asyncio.run(
        mod.plan_from_payload(
            {
                "planning_session_id": draft["plan"]["planning_session_id"],
                "action": "cancel",
            }
        )
    )

    assert cancelled["plan"]["interaction_state"] == "cancelled"
    assert cancelled["plan"]["handoff_ready"] is False
    assert cancelled["plan"]["mission"] is None


def test_cognition_module_keeps_conversation_session_separate_from_planning_session() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    result = asyncio.run(
        mod.plan_from_payload(
            {
                "text": "inspect area-a",
                "conversation_session_id": "conv-1",
            }
        )
    )
    memory = mod.working_memory.snapshot()

    assert result["plan"]["conversation_session_id"] == "conv-1"
    assert result["plan"]["planning_session_id"] != "conv-1"
    assert memory["focus"]["conversation_session_id"] == "conv-1"
    assert {item["conversation_session_id"] for item in memory["items"]} == {"conv-1"}


def test_cognitive_planner_filters_working_memory_by_conversation_session() -> None:
    memory = WorkingMemory(retention_seconds=60)
    memory.record("note", "session A secret", conversation_session_id="conv-a")
    memory.record("note", "session B route", conversation_session_id="conv-b")
    planner = CognitivePlanner(
        world_state=WorldStateService(),
        working_memory=memory,
        mission_service=MissionService(),
    )

    plan = planner.plan_from_text(
        "inspect area-b",
        conversation_session_id="conv-b",
    )

    assert "session B route" in plan.context["working_memory"]
    assert "session A secret" not in plan.context["working_memory"]


def test_cognition_module_ignores_legacy_session_id_for_conversation_context() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    result = asyncio.run(
        mod.plan_from_payload({"text": "inspect area-a", "session_id": "legacy-session"})
    )

    assert result["plan"]["conversation_session_id"] == ""
    assert result["plan"]["planning_session_id"] != "legacy-session"


def test_cognition_continues_only_with_explicit_planning_session_id() -> None:
    runtime = (
        Runtime.use(MemoryModule)
        + Runtime.use(MissionModule)
        + Runtime.use(CognitionModule)
    )
    app = asyncio.run(runtime.build({"cognition": {"sync_enabled": False}}))
    mod = app.modules["cognition"]

    draft = asyncio.run(
        mod.plan_from_payload(
            {
                "text": "inspect area-a",
                "conversation_session_id": "conv-1",
            }
        )
    )
    continued = asyncio.run(
        mod.plan_from_payload(
            {
                "text": "confirm",
                "conversation_session_id": "conv-1",
                "operator_confirmation": True,
            }
        )
    )

    assert continued["plan"]["conversation_session_id"] == "conv-1"
    assert continued["plan"]["planning_session_id"] != draft["plan"]["planning_session_id"]
