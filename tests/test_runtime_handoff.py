from __future__ import annotations

import time

from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService
from askme.runtime.handoff import (
    RuntimeHandoffService,
    SafetyPreflightService,
    SkillRegistry,
    TaskHandoff,
    TaskStep,
)
from askme.runtime.mission import MissionService


def _confirmed_plan() -> tuple[WorldStateService, dict]:
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 86,
            "estop_active": False,
            "localized": True,
        },
        stale_after_s=60.0,
    )
    planner = CognitivePlanner(
        world_state=world,
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
    return world, confirmed.to_dict()


class _FakeDogSafetyClient:
    def __init__(self, *, configured: bool = True, estop_active: bool = False) -> None:
        self.configured = configured
        self.estop_active = estop_active
        self.is_estop_calls = 0
        self.query_estop_state_calls = 0

    def is_configured(self) -> bool:
        return self.configured

    def is_estop_active(self) -> bool:
        self.is_estop_calls += 1
        return self.estop_active

    def query_estop_state(self) -> dict:
        self.query_estop_state_calls += 1
        raise AssertionError("runtime preflight must not call dog-safety network query")


def _perception_reasons(payload: dict) -> set[str]:
    return {
        str(item.get("reason"))
        for item in payload.get("perception_requests", [])
        if isinstance(item, dict)
    }


def test_runtime_handoff_accepts_confirmed_plan_and_completes_fake_run() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["handoff"]["confirmation_status"] == "confirmed"
    assert result["handoff"]["task_type"] == "inspection_patrol"
    assert result["run"]["current_state"] == "completed"
    assert result["run"]["report"]["status"] == "completed"
    assert result["run"]["report"]["completed_steps"] == [
        "go_to_area",
        "follow_patrol_route",
        "inspect_equipment",
        "capture_image",
        "generate_report",
    ]
    assert len(result["run"]["skill_results"]) == 5
    assert result["run"]["skill_results"][2]["skill_name"] == "inspect_equipment"
    assert result["run"]["report"]["observations"]
    assert any(
        item["type"] == "image_ref"
        for item in result["run"]["report"]["artifacts"]
    )
    assert any(
        event["kind"] == "runtime.task_completed"
        for event in world.snapshot()["events"]
    )


def test_runtime_handoff_maps_field_incident_policy_to_high_level_skills() -> None:
    plan = {
        "plan_id": "field-event-1",
        "planning_session_id": "field-session-1",
        "intent": "field_incident_response",
        "goal": "Handle illegal parking at main road",
        "handoff_ready": True,
        "operator_id": "security-1",
        "operator_roles": ["operator"],
        "mission": {
            "mission": {
                "mission_type": "field_incident_response",
                "goal": "Handle illegal parking at main road",
                "risk_tier": "medium",
                "operator_id": "security-1",
                "operator_roles": ["operator"],
                "steps": [{"target": "zone-main-road"}],
                "field_event": {
                    "event_id": "field-event-1",
                    "scenario_id": "illegal_parking",
                    "location": "main road",
                    "robot_motion_policy": "retreat_to_safe_distance",
                },
            }
        },
        "reference": {"resolved": {"area_id": "zone-main-road"}},
        "missing_inputs": [],
    }
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 86,
            "estop_active": False,
            "localized": True,
        },
        stale_after_s=60.0,
    )
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["handoff"]["task_type"] == "field_incident_response"
    assert result["run"]["report"]["completed_steps"] == [
        "retreat_to_safe_distance",
        "capture_image",
        "generate_report",
    ]
    first_step = result["handoff"]["steps"][0]
    assert first_step["skill_name"] == "retreat_to_safe_distance"
    assert first_step["parameters"]["minimum_distance_m"] == 2.0
    assert first_step["requires_confirmation"] is True


def test_runtime_handoff_accepts_known_area_catalog() -> None:
    world, plan = _confirmed_plan()
    world.update_area_catalog(
        [
            {
                "area_id": "area-a",
                "allowed": True,
                "map_id": "map-main",
                "map_version": "v1",
            }
        ],
        map_id="map-main",
        map_version="v1",
    )
    world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=True,
        localization_quality=0.9,
    )
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["run"]["current_state"] == "completed"
    assert result["preflight"]["perception_requests"] == []


def test_safety_preflight_rejects_stale_world_state() -> None:
    world, plan = _confirmed_plan()
    snapshot = world.snapshot()
    snapshot["updated_at"] = time.time() - 120
    registry = SkillRegistry()
    handoff = TaskHandoff.from_plan(
        plan,
        world_state_snapshot=snapshot,
        skill_registry=registry,
    )
    assessment = SafetyPreflightService(max_world_state_age_s=1).assess(
        handoff,
        skill_registry=registry,
    )

    assert assessment.passed is False
    assert "world_state_stale" in assessment.failed_checks
    assert "Refresh perception/world state" in assessment.recommended_fix
    assert "refresh_world_state" in _perception_reasons(assessment.to_dict())


def test_safety_preflight_rejects_unregistered_skill() -> None:
    world, plan = _confirmed_plan()
    registry = SkillRegistry()
    handoff = TaskHandoff(
        handoff_id="handoff-test",
        plan_id=plan["plan_id"],
        session_id=plan["planning_session_id"],
        operator_id="operator-1",
        intent="navigation",
        task_type="navigate_to",
        target_area="area-a",
        target_object=None,
        constraints=[],
        steps=[
            TaskStep(
                step_id="step-1",
                sequence=1,
                skill_name="drive_motor_direct",
                parameters={"meters": 10},
            )
        ],
        risk_level="high",
        required_capabilities=[],
        missing_info=[],
        confirmation_status="confirmed",
        world_state_snapshot_id="world-test",
        safety_notes=[],
        created_at=time.time(),
        expires_at=time.time() + 60,
        planner_version="test",
        source_plan=plan,
        world_state_snapshot=world.snapshot(),
    )

    assessment = SafetyPreflightService().assess(handoff, skill_registry=registry)

    assert assessment.passed is False
    assert "unregistered_skill:drive_motor_direct" in assessment.failed_checks
    assert "registered high-level skills" in assessment.recommended_fix


def test_safety_preflight_rejects_unknown_target_area_when_catalog_exists() -> None:
    world, plan = _confirmed_plan()
    world.update_area_catalog([{"area_id": "area-b", "allowed": True}])
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert result["run"]["current_state"] == "blocked"
    assert "target_area_unknown" in result["preflight"]["failed_checks"]
    assert "site catalog" in result["preflight"]["recommended_fix"]
    assert "observe_or_register_area" in _perception_reasons(result["preflight"])
    assert result["replan_proposal"]["recommended_action"] == "load_site_catalog_or_clarify_area"
    assert result["run"]["replan_proposals"][0]["operator_confirmation_required"] is True
    assert any(
        event["event_type"] == "perception_requested"
        for event in result["run"]["runtime_events"]
    )
    assert any(
        event["event_type"] == "replan_proposed"
        for event in result["run"]["runtime_events"]
    )


def test_safety_preflight_requests_area_catalog_when_missing() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert "area_catalog_unavailable" in result["preflight"]["warnings"]
    assert "load_area_catalog" in _perception_reasons(result["preflight"])


def test_operator_viewer_role_cannot_submit_runtime_handoff() -> None:
    world, plan = _confirmed_plan()
    plan["operator_roles"] = ["viewer"]
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert "operator_not_authorized" in result["preflight"]["failed_checks"]
    assert result["handoff"]["operator_roles"] == ["viewer"]
    assert result["replan_proposal"]["recommended_action"] == "request_authorized_operator_review"


def test_high_risk_handoff_can_require_supervisor_role() -> None:
    world, plan = _confirmed_plan()
    plan["operator_roles"] = ["operator"]
    service = RuntimeHandoffService(
        world_state=world,
        require_supervisor_for_high_risk=True,
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert "supervisor_confirmation_required" in result["preflight"]["failed_checks"]
    assert "supervisor" in result["preflight"]["recommended_fix"]


def test_supervisor_role_can_submit_high_risk_handoff_when_required() -> None:
    world, plan = _confirmed_plan()
    plan["operator_roles"] = ["supervisor"]
    service = RuntimeHandoffService(
        world_state=world,
        require_supervisor_for_high_risk=True,
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["handoff"]["operator_roles"] == ["supervisor"]


def test_safety_preflight_rejects_blocked_target_area() -> None:
    world, plan = _confirmed_plan()
    world.update_area_catalog([{"area_id": "area-a", "allowed": False}])
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert "target_area_blocked" in result["preflight"]["failed_checks"]
    assert result["replan_proposal"]["recommended_action"] == "choose_allowed_area"


def test_safety_preflight_rejects_area_map_version_mismatch() -> None:
    world, plan = _confirmed_plan()
    world.update_area_catalog(
        [{"area_id": "area-a", "allowed": True, "map_id": "map-main", "map_version": "v2"}]
    )
    world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=True,
        localization_quality=0.93,
    )
    service = RuntimeHandoffService(world_state=world)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert "map_version_mismatch" in result["preflight"]["failed_checks"]
    assert "active map" in result["preflight"]["recommended_fix"]
    assert result["replan_proposal"]["recommended_action"] == "refresh_map_state_then_replan"


def test_dog_safety_estop_blocks_runtime_handoff_without_network_query() -> None:
    world, plan = _confirmed_plan()
    client = _FakeDogSafetyClient(estop_active=True)
    service = RuntimeHandoffService(world_state=world, dog_safety_client=client)

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert result["reason"] == "preflight_failed"
    assert "dog_safety_estop_active" in result["preflight"]["failed_checks"]
    assert "Clear E-STOP" in result["preflight"]["recommended_fix"]
    assert result["run"]["report"]["replan_proposals"][0]["recommended_action"] == "clear_estop_then_retry"
    assert client.is_estop_calls == 1
    assert client.query_estop_state_calls == 0


def test_unconfigured_dog_safety_warns_by_default() -> None:
    world, plan = _confirmed_plan()
    registry = SkillRegistry()
    handoff = TaskHandoff.from_plan(
        plan,
        world_state_snapshot=world.snapshot(),
        skill_registry=registry,
    )
    client = _FakeDogSafetyClient(configured=False)

    assessment = SafetyPreflightService(dog_safety_client=client).assess(
        handoff,
        skill_registry=registry,
    )

    assert assessment.passed is True
    assert "dog_safety_unconfigured" in assessment.warnings


def test_strict_dog_safety_blocks_when_client_missing() -> None:
    world, plan = _confirmed_plan()
    registry = SkillRegistry()
    handoff = TaskHandoff.from_plan(
        plan,
        world_state_snapshot=world.snapshot(),
        skill_registry=registry,
    )

    assessment = SafetyPreflightService(require_dog_safety=True).assess(
        handoff,
        skill_registry=registry,
    )

    assert assessment.passed is False
    assert "dog_safety_unavailable" in assessment.failed_checks
    assert "Connect dog-safety-service" in assessment.recommended_fix


def test_runtime_handoff_can_pause_resume_and_cancel_non_autocomplete_run() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, auto_complete=False)
    result = service.submit_plan_payload(plan)
    run_id = result["run"]["run_id"]

    pause = service.pause_payload(
        run_id,
        operator_id="operator-1",
        reason="visitor entered path",
        risk_acknowledgement=True,
    )
    resume = service.resume_payload(
        run_id,
        operator_id="operator-1",
        reason="path clear",
        risk_acknowledgement=True,
    )
    cancel = service.cancel_payload(run_id, operator_id="operator-1", reason="demo complete")

    assert pause["handled"] is True
    assert pause["run"]["current_state"] == "paused"
    assert pause["run"]["operator_actions"][-1]["reason"] == "visitor entered path"
    assert pause["run"]["operator_actions"][-1]["risk_acknowledgement"] is True
    assert resume["handled"] is True
    assert resume["run"]["current_state"] == "executing"
    assert resume["run"]["operator_actions"][-1]["reason"] == "path clear"
    assert cancel["handled"] is True
    assert cancel["run"]["current_state"] == "cancelled"
    assert cancel["run"]["operator_actions"][-1]["reason"] == "demo complete"
    assert cancel["run"]["report"]["status"] == "cancelled"


def test_shadow_profile_generates_would_execute_plan_without_step_execution() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="shadow")

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["status"] == "shadowed"
    assert result["run"]["profile"] == "shadow"
    assert result["run"]["terminal"] is True
    assert result["shadow_plan"]["hardware_dispatch"] is False
    assert [item["skill_name"] for item in result["shadow_plan"]["would_execute"]][:2] == [
        "go_to_area",
        "follow_patrol_route",
    ]
    assert not any(
        event["event_type"] == "step_completed"
        for event in result["run"]["runtime_events"]
    )


def test_sim_profile_advances_steps_and_exposes_sim_state() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    result = service.submit_plan_payload(plan)
    run_id = result["run"]["run_id"]

    first = service.advance_payload(run_id, operator_id="operator-1")
    second = service.advance_payload(run_id, operator_id="operator-1")

    assert result["run"]["current_state"] == "queued"
    assert first["handled"] is True
    assert first["run"]["profile"] == "sim"
    assert first["run"]["current_state"] == "executing"
    assert first["run"]["current_step_index"] == 1
    assert first["run"]["skill_results"][0]["skill_name"] == "go_to_area"
    assert first["run"]["skill_results"][0]["status"] == "completed"
    assert first["run"]["sim_state"]["remaining_steps"] == 4
    assert second["run"]["current_step_index"] == 2


def test_runtime_events_payload_returns_cursor_and_active_run() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    result = service.submit_plan_payload(plan)
    run_id = result["run"]["run_id"]
    service.advance_payload(run_id, operator_id="operator-1")

    payload = service.events_payload(limit=50)
    events = payload["events"]
    cursor = payload["cursor"]
    after = service.events_payload(after=cursor, limit=50)

    assert payload["profile"] == "sim"
    assert payload["hardware_dispatch"] is False
    assert payload["active_run"]["run_id"] == run_id
    assert any(event["event_type"] == "step_completed" for event in events)
    assert cursor >= max(event["created_at"] for event in events)
    assert after["events"] == []


def test_task_run_store_recovers_completed_run_after_restart(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "task-runs.json"
    service = RuntimeHandoffService(
        world_state=world,
        store_config={"enabled": True, "path": str(store_path)},
    )
    submitted = service.submit_plan_payload(plan)
    run_id = submitted["run"]["run_id"]

    restarted = RuntimeHandoffService(
        world_state=world,
        store_config={"enabled": True, "path": str(store_path)},
    )
    restored = restarted.get_payload(run_id)

    assert store_path.exists()
    assert restored["run"]["run_id"] == run_id
    assert restored["run"]["current_state"] == "completed"
    assert restored["run"]["report"]["status"] == "completed"
    assert any(
        event["event_type"] == "task_completed"
        for event in restored["run"]["runtime_events"]
    )


def test_task_run_store_recovers_operator_actions_and_sim_state(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "task-runs.json"
    service = RuntimeHandoffService(
        world_state=world,
        profile="sim",
        auto_complete=False,
        store_config={"enabled": True, "path": str(store_path)},
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(run_id, operator_id="operator-1")
    service.pause_payload(run_id, operator_id="operator-1", reason="visitor in path")

    restarted = RuntimeHandoffService(
        world_state=world,
        profile="sim",
        auto_complete=False,
        store_config={"enabled": True, "path": str(store_path)},
    )
    restored = restarted.get_payload(run_id)["run"]

    assert restored["current_state"] == "paused"
    assert restored["current_step_index"] == 1
    assert restored["operator_actions"][-1]["action"] == "pause"
    assert restored["operator_actions"][-1]["reason"] == "visitor in path"
    assert restored["sim_state"]["remaining_steps"] == 4


def test_sim_profile_rejects_advance_while_paused() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(run_id)
    paused = service.pause_payload(run_id)

    advanced = service.advance_payload(run_id)

    assert paused["run"]["current_state"] == "paused"
    assert advanced["handled"] is False
    assert advanced["reason"] == "run_paused"


def test_voice_turn_controls_active_runtime_without_bypassing_state_machine() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(run_id)

    paused = service.voice_turn_payload(
        "先停一下",
        speak=True,
        transcript_id="voice-1",
        confidence=0.91,
    )
    resumed = service.voice_turn_payload("继续")

    assert paused["handled"] is True
    assert paused["runtime"]["run"]["current_state"] == "paused"
    assert paused["voice_turn"]["recognized_text"] == "先停一下"
    assert paused["voice_turn"]["runtime_control_intent"] == "pause"
    assert paused["voice_turn"]["safety_bypass_allowed"] is False
    assert paused["spoken"] is False
    assert resumed["runtime"]["run"]["current_state"] == "executing"


def test_voice_turn_does_not_confirm_or_submit_new_runtime_plan() -> None:
    world, _plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)

    result = service.voice_turn_payload("确认", transcript_id="voice-confirm")

    assert result["handled"] is False
    assert result["reason"] == "no_runtime_control_intent"
    assert result["runtime"]["active_run"] is None
    assert result["voice_turn"]["safety_bypass_allowed"] is False


def test_runtime_profiles_payload_is_safe_and_explicit() -> None:
    world, _plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="shadow")

    payload = service.profiles_payload()

    assert payload["current_profile"] == "shadow"
    assert payload["hardware_dispatch"] is False
    assert {item["name"] for item in payload["profiles"]} == {
        "fake",
        "shadow",
        "sim",
        "external",
        "lab",
    }
    assert all(item["hardware_dispatch"] is False for item in payload["profiles"])


def test_external_runtime_profile_blocks_submission_until_explicitly_enabled() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        external_runtime_config={"endpoint": "http://runtime.local/submit"},
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert result["reason"] == "external_runtime_disabled"
    assert result["error"]["code"] == "external_runtime_disabled"
    assert result["error"]["endpoint_configured"] is True
    assert result["run"]["profile"] == "external"
    assert result["run"]["current_state"] == "blocked"
    assert result["hardware_dispatch"] is False
    assert "external_runtime_disabled" in result["preflight"]["failed_checks"]


def test_lab_runtime_profile_requires_endpoint_even_when_enabled() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(
        world_state=world,
        profile="lab",
        external_runtime_config={"enable_external_runtime": True},
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert result["reason"] == "external_runtime_endpoint_required"
    assert result["error"]["enable_external_runtime"] is True
    assert result["error"]["endpoint_configured"] is False
    assert result["run"]["profile"] == "lab"
    assert result["hardware_dispatch"] is False


def test_external_runtime_profile_enabled_builds_contract_only_submission() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["status"] == "queued"
    assert result["run"]["profile"] == "external"
    assert result["runtime_client"]["dispatch_mode"] == "contract_only"
    assert result["runtime_client"]["endpoint"] == "http://runtime.local/submit"
    assert result["hardware_dispatch"] is False


def test_chinese_runtime_control_and_area_id_are_stable() -> None:
    world, plan = _confirmed_plan()
    plan["goal"] = "巡检 A 区"
    mission = plan["mission"]["mission"]
    mission["goal"] = "巡检 A 区"
    mission["steps"][0]["target"] = "A区"
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]

    paused = service.handle_chat_control("先停一下")
    status = service.handle_chat_control("现在执行到哪了")
    handoff = service.get_payload(run_id)["run"]["handoff"]

    assert handoff["target_area"] == "area-a"
    assert handoff["steps"][0]["parameters"]["area_id"] == "area-a"
    assert paused is not None
    assert paused["runtime"]["run"]["current_state"] == "paused"
    assert status is not None
    assert "TaskRun" in status["reply"]
