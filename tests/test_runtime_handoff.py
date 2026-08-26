from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from askme.runtime.handoff import (
    RuntimeHandoffService,
    SafetyPreflightService,
    SkillRegistry,
    TaskHandoff,
    TaskStep,
)
from askme.runtime.mission import MissionService

from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService
from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorSubmitResult,
    RuntimeExecutorTransportError,
    RuntimeExecutorUpdate,
)
from askme.runtime.task.handoff import TaskRunStore


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


class _FakeRuntimeExecutorTransport:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.result is None:
            return None
        return replace(
            self.result,
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
        )


def _external_transport(remote_task_id: str, *, status: str = "queued"):
    return _FakeRuntimeExecutorTransport(
        RuntimeExecutorSubmitResult(
            remote_task_id=remote_task_id,
            status=status,
            correlation_id="domain-test",
            idempotency_key="domain-test",
        )
    )


def _operator_context(
    permission: str,
    *,
    operator_id: str = "operator-1",
    roles: list[str] | None = None,
    conversation_session_id: str = "runtime-test-thread",
) -> dict:
    return {
        "operator_id": operator_id,
        "roles": roles or ["operator"],
        "authenticated": True,
        "source": "test",
        "permission": permission,
        "conversation_session_id": conversation_session_id,
    }


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
    assert any(item["type"] == "image_ref" for item in result["run"]["report"]["artifacts"])
    assert any(event["kind"] == "runtime.task_completed" for event in world.snapshot()["events"])


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
        event["event_type"] == "perception_requested" for event in result["run"]["runtime_events"]
    )
    assert any(
        event["event_type"] == "replan_proposed" for event in result["run"]["runtime_events"]
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
    assert (
        result["run"]["report"]["replan_proposals"][0]["recommended_action"]
        == "clear_estop_then_retry"
    )
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
        operator_context=_operator_context("runtime:pause"),
    )
    resume = service.resume_payload(
        run_id,
        operator_id="operator-1",
        reason="path clear",
        risk_acknowledgement=True,
        operator_context=_operator_context("runtime:resume"),
    )
    cancel = service.cancel_payload(
        run_id,
        operator_id="operator-1",
        reason="demo complete",
        operator_context=_operator_context("runtime:cancel"),
    )

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
        event["event_type"] == "step_completed" for event in result["run"]["runtime_events"]
    )


def test_sim_profile_advances_steps_and_exposes_sim_state() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    result = service.submit_plan_payload(plan)
    run_id = result["run"]["run_id"]

    first = service.advance_payload(
        run_id, operator_id="operator-1", operator_context=_operator_context("runtime:advance")
    )
    second = service.advance_payload(
        run_id, operator_id="operator-1", operator_context=_operator_context("runtime:advance")
    )

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
    service.advance_payload(
        run_id, operator_id="operator-1", operator_context=_operator_context("runtime:advance")
    )

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
        event["event_type"] == "task_completed" for event in restored["run"]["runtime_events"]
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
    service.advance_payload(
        run_id, operator_id="operator-1", operator_context=_operator_context("runtime:advance")
    )
    service.pause_payload(
        run_id,
        operator_id="operator-1",
        reason="visitor in path",
        operator_context=_operator_context("runtime:pause"),
    )

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
    service.advance_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:advance"),
    )
    paused = service.pause_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:pause"),
    )

    advanced = service.advance_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:advance"),
    )

    assert paused["run"]["current_state"] == "paused"
    assert advanced["handled"] is False
    assert advanced["reason"] == "run_paused"


def test_voice_turn_controls_active_runtime_without_bypassing_state_machine() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:advance"),
    )

    paused = service.voice_turn_payload(
        "先停一下",
        speak=True,
        transcript_id="voice-1",
        confidence=0.91,
        conversation_session_id="voice-control-thread",
        operator_id="operator-1",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="test",
        runtime_permission="runtime:pause",
    )
    resumed = service.voice_turn_payload(
        "继续",
        conversation_session_id="voice-control-thread",
        operator_id="operator-1",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="test",
        runtime_permission="runtime:resume",
    )

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


def test_external_runtime_profile_fails_closed_without_transport() -> None:
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

    assert result["accepted"] is False
    assert result["status"] == "blocked"
    assert result["reason"] == "external_transport_unavailable"
    assert result["run"]["profile"] == "external"
    assert result["runtime_client"]["dispatch_mode"] == "transport_managed"
    assert result["runtime_client"]["endpoint"] == "http://runtime.local/submit"
    assert result["hardware_dispatch"] is False


def test_external_runtime_submits_once_after_preflight_with_stable_correlation() -> None:
    world, plan = _confirmed_plan()
    plan["voice_context"] = {
        "submission_id": "voice-submit-42",
        "conversation_session_id": "voice-session-7",
        "originating_turn_id": "turn-9",
    }
    transport = _FakeRuntimeExecutorTransport(
        RuntimeExecutorSubmitResult(
            remote_task_id="remote-live-1",
            status="executing",
            correlation_id="ignored-by-domain",
            idempotency_key="voice-submit-42",
            cursor="cursor-2",
            result_summary="",
            updates=(
                RuntimeExecutorUpdate(
                    event_id="remote-event-1",
                    status="queued",
                    cursor="cursor-1",
                    observed_at=110.0,
                ),
            ),
            observed_at=111.0,
        )
    )
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.idempotency_key == "voice-submit-42"
    assert request.correlation_id == result["run"]["run_id"]
    assert request.thread_id == "voice-session-7"
    assert request.turn_id == "turn-9"
    assert request.handoff["plan_id"] == plan["plan_id"]
    assert result["run"]["remote_task_id"] == "remote-live-1"
    assert result["run"]["remote_status"] == "executing"
    assert result["run"]["remote_status_cursor"] == "cursor-2"
    assert result["remote"] == {
        "remote_task_id": "remote-live-1",
        "status": "executing",
        "cursor": "cursor-2",
        "observed_at": 111.0,
        "result_summary": "",
    }


def test_external_runtime_submit_terminal_update_preserves_evidence() -> None:
    world, plan = _confirmed_plan()
    observation = {
        "type": "inspection_result",
        "summary": "No anomaly detected.",
    }
    artifact = {
        "artifact_id": "inspection-image-1",
        "type": "image_ref",
        "uri": "s3://evidence/inspection-image-1.jpg",
    }
    transport = _FakeRuntimeExecutorTransport(
        RuntimeExecutorSubmitResult(
            remote_task_id="remote-completed-with-evidence",
            status="completed",
            correlation_id="ignored-by-domain",
            idempotency_key="ignored-by-domain",
            cursor="cursor-completed",
            result_summary="Inspection completed.",
            updates=(
                RuntimeExecutorUpdate(
                    event_id="remote-completed-event",
                    status="completed",
                    cursor="cursor-completed",
                    observed_at=112.0,
                    payload={
                        "observation": observation,
                        "artifact": artifact,
                    },
                ),
            ),
            observed_at=112.0,
        )
    )
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is True
    assert result["run"]["current_state"] == "completed"
    assert len(result["run"]["skill_results"]) == 1
    assert result["run"]["skill_results"][0]["observations"] == [observation]
    assert result["run"]["skill_results"][0]["artifacts"] == [artifact]
    assert result["run"]["report"]["observations"] == [observation]
    assert result["run"]["report"]["artifacts"] == [artifact]


def test_external_runtime_does_not_submit_before_local_preflight_passes() -> None:
    world, plan = _confirmed_plan()
    plan["missing_inputs"] = ["operator confirmation"]
    transport = _FakeRuntimeExecutorTransport()
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert transport.requests == []
    assert result["preflight"]["failed_checks"]


@pytest.mark.parametrize(
    ("error", "reason", "state"),
    [
        (
            RuntimeExecutorTransportError(
                "network_error",
                "secret bearer token must not escape",
                retryable=True,
            ),
            "external_submission_failed",
            "failed",
        ),
        (
            AmbiguousRuntimeSubmissionError("secret request body must not escape"),
            "external_submission_unknown",
            "submission_unknown",
        ),
        (
            RuntimeError("secret provider exception must not escape"),
            "external_submission_failed",
            "failed",
        ),
    ],
)
def test_external_runtime_transport_errors_are_sanitized_and_not_retried(
    error, reason, state
) -> None:
    world, plan = _confirmed_plan()
    transport = _FakeRuntimeExecutorTransport(error=error)
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    serialized = json.dumps(result)
    assert result["accepted"] is False
    assert result["reason"] == reason
    assert result["run"]["current_state"] == state
    assert len(transport.requests) == 1
    assert transport.requests[0].idempotency_key == plan["plan_id"]
    assert "secret" not in serialized


def test_external_runtime_remote_rejection_is_factual_and_sanitized() -> None:
    world, plan = _confirmed_plan()
    transport = _external_transport("remote-rejected", status="rejected")
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    result = service.submit_plan_payload(plan)

    assert result["accepted"] is False
    assert result["reason"] == "external_submission_rejected"
    assert result["run"]["current_state"] == "blocked"
    assert result["remote"]["remote_task_id"] == "remote-rejected"
    assert "endpoint" not in result["remote"]


def test_remote_accept_then_local_projection_failure_remains_reconcilable(
    monkeypatch,
    tmp_path,
) -> None:
    world, plan = _confirmed_plan()
    transport = _external_transport("remote-projection-unknown")
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "projection-unknown.json"),
        "swallow_errors": False,
    }
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=store_config,
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    def fail_bind(*_args, **_kwargs):
        raise OSError("simulated local projection write failure")

    monkeypatch.setattr(service.run_service, "bind_external_submission", fail_bind)

    result = service.submit_plan_payload(plan)

    assert len(transport.requests) == 1
    assert result["accepted"] is False
    assert result["reason"] == "external_projection_commit_unknown"
    assert result["remote_may_be_running"] is True
    assert result["run"]["current_state"] == "submission_unknown"
    assert result["run"]["remote_task_id"] == "remote-projection-unknown"
    assert result["run"]["external_idempotency_key"] == transport.requests[0].idempotency_key
    assert result["run"]["terminal"] is False

    restored = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=store_config,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    ).run_service.require(result["run"]["run_id"])
    assert restored.current_state == "submission_unknown"
    assert restored.remote_task_id == "remote-projection-unknown"
    assert restored.external_idempotency_key == transport.requests[0].idempotency_key


def test_task_run_store_serializes_concurrent_writers_with_unique_temp_files(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "concurrent-task-runs.json"
    config = {"enabled": True, "path": str(store_path), "swallow_errors": False}
    service = RuntimeHandoffService(world_state=world, store_config=config)
    run = service.run_service.require(service.submit_plan_payload(plan)["run"]["run_id"])
    stores = [TaskRunStore(config) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(store.save_runs, [run]) for store in stores]
        for future in futures:
            future.result(timeout=2.0)

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert payload["runs"][0]["run_id"] == run.run_id
    assert not list(tmp_path.glob(".concurrent-task-runs.json.*.tmp"))


def test_task_run_service_keeps_concurrent_events_durable(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "concurrent-events.json"
    config = {"enabled": True, "path": str(store_path), "swallow_errors": False}
    service = RuntimeHandoffService(world_state=world, store_config=config)
    run = service.run_service.require(service.submit_plan_payload(plan)["run"]["run_id"])
    baseline = len(run.runtime_events)

    def record(index: int) -> str:
        return service.run_service.emit(
            run,
            "concurrency_probe",
            run.current_state,
            f"concurrent event {index}",
            {"index": index},
        ).event_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        event_ids = list(executor.map(record, range(32)))

    restored = RuntimeHandoffService(world_state=world, store_config=config)
    restored_run = restored.run_service.require(run.run_id)
    restored_ids = {event.event_id for event in restored_run.runtime_events}
    assert len(event_ids) == len(set(event_ids)) == 32
    assert len(restored_run.runtime_events) == baseline + 32
    assert set(event_ids) <= restored_ids


def test_report_build_cannot_overwrite_concurrent_completed_state(
    tmp_path, monkeypatch
) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "concurrent-report.json"
    config = {"enabled": True, "path": str(store_path), "swallow_errors": False}
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        executor_transport=_external_transport("remote-report-race"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    report_snapshot_ready = threading.Event()
    release_report_build = threading.Event()
    update_started = threading.Event()
    update_finished = threading.Event()
    original_build_report = service.report_service.build_report

    def blocking_build_report(run):
        report = original_build_report(run)
        if run.current_state == "queued":
            report_snapshot_ready.set()
            assert release_report_build.wait(timeout=2.0)
        return report

    def complete_run():
        update_started.set()
        try:
            return service.apply_external_update(
                run_id,
                remote_task_id="remote-report-race",
                remote_status="completed",
                update_id="report-race-completed",
                cursor=1,
                result_summary="completed during report generation",
            )
        finally:
            update_finished.set()

    monkeypatch.setattr(service.report_service, "build_report", blocking_build_report)

    with ThreadPoolExecutor(max_workers=2) as executor:
        report_future = executor.submit(service.report_payload, run_id)
        assert report_snapshot_ready.wait(timeout=2.0)
        update_future = executor.submit(complete_run)
        assert update_started.wait(timeout=2.0)
        try:
            assert not update_finished.wait(timeout=0.2)
        finally:
            release_report_build.set()
        report_future.result(timeout=2.0)
        update_future.result(timeout=2.0)

    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    persisted_run = next(item for item in persisted["runs"] if item["run_id"] == run_id)
    assert persisted_run["current_state"] == "completed"
    assert persisted_run["report"]["status"] == "completed"


def test_external_task_run_projection_persists_and_loads_legacy_store(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "external-task-runs.json"
    config = {"enabled": True, "path": str(store_path)}
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        executor_transport=_external_transport("remote-17"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]

    bound = service.bind_external_submission(
        run_id,
        remote_task_id="remote-17",
        observed_at=100.0,
    )
    service.apply_external_update(
        run_id,
        remote_task_id="remote-17",
        remote_status="working",
        update_id="update-1",
        cursor=1,
        observed_at=101.0,
    )

    assert bound["run"]["remote_task_id"] == "remote-17"
    restarted = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    restored = restarted.get_payload(run_id)["run"]
    assert restored["remote_status"] == "executing"
    assert restored["remote_status_cursor"] == "1"
    assert restored["external_idempotency_key"] == plan["plan_id"]
    assert restored["processed_remote_update_ids"] == ["update-1"]

    raw = json.loads(store_path.read_text(encoding="utf-8"))
    legacy_run = raw["runs"][0]
    for field in (
        "remote_task_id",
        "remote_status",
        "remote_status_cursor",
        "external_idempotency_key",
        "remote_observed_at",
        "last_poll_error_code",
        "processed_remote_update_ids",
        "approval_request",
        "deferred_cancel_request",
    ):
        legacy_run.pop(field, None)
    store_path.write_text(json.dumps(raw), encoding="utf-8")

    legacy = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    ).get_payload(run_id)["run"]
    assert legacy["remote_task_id"] is None
    assert legacy["remote_status"] == ""
    assert legacy["processed_remote_update_ids"] == []
    assert legacy["approval_request"] == {}
    assert legacy["deferred_cancel_request"] == {}


def test_notification_delivery_receipt_roundtrips_and_legacy_store_defaults(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "notification-receipts.json"
    config = {"enabled": True, "path": str(store_path)}
    service = RuntimeHandoffService(world_state=world, store_config=config)
    submitted = service.submit_plan_payload(plan)
    run_id = submitted["run"]["run_id"]
    event_id = submitted["run"]["runtime_events"][-1]["event_id"]

    recorded = service.record_notification_delivery_receipt(
        run_id,
        event_id=event_id,
        status="delivered",
    )

    assert recorded == {
        "run_id": run_id,
        "event_id": event_id,
        "status": "delivered",
        "recorded": True,
    }
    restarted = RuntimeHandoffService(world_state=world, store_config=config)
    assert restarted.notification_delivery_receipt(run_id, event_id=event_id) == "delivered"
    assert restarted.notification_delivery_receipts(run_id) == {event_id: "delivered"}

    raw = json.loads(store_path.read_text(encoding="utf-8"))
    raw["runs"][0].pop("notification_delivery_receipts")
    store_path.write_text(json.dumps(raw), encoding="utf-8")

    legacy = RuntimeHandoffService(world_state=world, store_config=config)
    assert legacy.notification_delivery_receipts(run_id) == {}
    assert legacy.get_payload(run_id)["run"]["notification_delivery_receipts"] == {}


def test_prepared_external_run_survives_restart_while_waiting_for_approval(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "prepared-runs.json"
    config = {"enabled": True, "path": str(store_path), "swallow_errors": False}
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )

    prepared = service.prepare_plan_payload(
        plan,
        approval_request={"prompt": "Confirm status report", "request_id": "approval-1"},
    )
    run_id = prepared["run"]["run_id"]

    restored = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config=config,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    ).get_payload(run_id)["run"]
    assert restored["current_state"] == "waiting_user"
    assert restored["approval_request"] == {
        "prompt": "Confirm status report",
        "request_id": "approval-1",
        "status": "waiting_user",
    }
    assert restored["deferred_cancel_request"] == {}


def test_confirmed_prepared_plan_submits_same_run_exactly_once() -> None:
    world, plan = _confirmed_plan()
    transport = _external_transport("remote-prepared")
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    prepared = service.prepare_plan_payload(plan, approval_request={"request_id": "approval-2"})
    run_id = prepared["run"]["run_id"]
    handoff_id = prepared["handoff"]["handoff_id"]

    confirmed = service.confirm_prepared_plan(
        run_id,
        confirmed_plan=plan,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:submit"),
    )
    run = service.run_service.require(run_id)
    stale_snapshot = dict(run.handoff.world_state_snapshot)
    stale_snapshot["updated_at"] = time.time() - 120.0
    run.handoff = replace(
        run.handoff,
        world_state_snapshot_id="world-stale",
        world_state_snapshot=stale_snapshot,
    )
    submitted = service.submit_prepared_run(run_id)

    assert confirmed["run"]["run_id"] == run_id
    assert confirmed["handoff"]["handoff_id"] == handoff_id
    assert confirmed["run"]["current_state"] == "confirmed"
    assert submitted["run"]["run_id"] == run_id
    assert submitted["handoff"]["handoff_id"] == handoff_id
    assert submitted["handoff"]["world_state_snapshot"]["updated_at"] > (
        stale_snapshot["updated_at"]
    )
    assert submitted["accepted"] is True
    assert len(transport.requests) == 1
    assert transport.requests[0].correlation_id == run_id
    assert len(service.run_service.runs()) == 1


def test_cancel_waiting_prepared_run_is_local_and_never_submits() -> None:
    world, plan = _confirmed_plan()
    transport = _external_transport("must-not-submit")
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=transport,
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    prepared = service.prepare_plan_payload(plan, approval_request={"request_id": "approval-3"})

    cancelled = service.cancel_prepared_run(
        prepared["run"]["run_id"],
        operator_id="operator-1",
        operator_context=_operator_context("runtime:cancel"),
        reason="operator declined",
    )

    assert cancelled["handled"] is True
    assert cancelled["run"]["current_state"] == "cancelled"
    assert cancelled["run"]["approval_request"]["status"] == "cancelled"
    assert transport.requests == []


@pytest.mark.parametrize("status", ["delivered", "interrupted", "suppressed", "expired"])
def test_notification_delivery_receipt_accepts_only_terminal_states(status) -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world)
    submitted = service.submit_plan_payload(plan)
    run_id = submitted["run"]["run_id"]
    event_id = submitted["run"]["runtime_events"][-1]["event_id"]

    first = service.record_notification_delivery_receipt(
        run_id,
        event_id=event_id,
        status=status,
    )
    repeated = service.record_notification_delivery_receipt(
        run_id,
        event_id=event_id,
        status=status,
    )

    assert first["recorded"] is True
    assert repeated["recorded"] is False
    assert service.notification_delivery_receipt(run_id, event_id=event_id) == status


def test_notification_delivery_receipt_rejects_invalid_or_foreign_event() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world)
    first = service.submit_plan_payload(plan)["run"]
    second = service.submit_plan_payload(plan)["run"]
    first_event_id = first["runtime_events"][-1]["event_id"]
    second_event_id = second["runtime_events"][-1]["event_id"]

    with pytest.raises(ValueError, match="delivery status"):
        service.record_notification_delivery_receipt(
            first["run_id"],
            event_id=first_event_id,
            status="pending",
        )
    with pytest.raises(ValueError, match="does not belong"):
        service.record_notification_delivery_receipt(
            first["run_id"],
            event_id=second_event_id,
            status="delivered",
        )


def test_notification_delivery_receipt_does_not_rewrite_terminal_state_and_is_bounded() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world)
    submitted = service.submit_plan_payload(plan)["run"]
    run_id = submitted["run_id"]
    run = service.run_service.get(run_id)
    assert run is not None
    event_ids = [submitted["runtime_events"][-1]["event_id"]]
    for index in range(70):
        event_ids.append(
            service.run_service.emit(
                run,
                "notification_test",
                run.current_state,
                f"notification event {index}",
            ).event_id
        )
    for event_id in event_ids:
        service.record_notification_delivery_receipt(
            run_id,
            event_id=event_id,
            status="delivered",
        )

    rewritten = service.record_notification_delivery_receipt(
        run_id,
        event_id=event_ids[-1],
        status="expired",
    )
    receipts = service.notification_delivery_receipts(run_id)

    assert rewritten["recorded"] is False
    assert rewritten["status"] == "delivered"
    assert len(receipts) == 64
    assert event_ids[0] not in receipts
    assert receipts[event_ids[-1]] == "delivered"


def test_task_run_event_observers_run_after_authoritative_sinks_and_are_isolated(
    tmp_path,
) -> None:
    world, plan = _confirmed_plan()
    store_path = tmp_path / "observer-task-runs.json"
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        store_config={"enabled": True, "path": str(store_path)},
        executor_transport=_external_transport("remote-observer"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    observed: list[str] = []

    def failing_observer(_event) -> None:
        raise RuntimeError("observer unavailable")

    def committed_observer(event) -> None:
        stored = json.loads(store_path.read_text(encoding="utf-8"))
        stored_events = stored["runs"][0]["runtime_events"]
        world_events = world.snapshot()["events"]
        assert any(item["event_id"] == event.event_id for item in stored_events)
        assert any(
            item.get("payload", {}).get("event_id") == event.event_id for item in world_events
        )
        observed.append(event.event_id)

    service.subscribe_events(failing_observer)
    unsubscribe = service.subscribe_events(committed_observer)
    service.bind_external_submission(run_id, remote_task_id="remote-observer")

    assert len(observed) == 2
    unsubscribe()
    service.apply_external_update(
        run_id,
        remote_task_id="remote-observer",
        remote_status="working",
        update_id="observer-update",
        cursor=1,
    )
    assert len(observed) == 2


def test_external_updates_are_deduplicated_ordered_and_terminal_monotonic() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=_external_transport("remote-ordered"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.bind_external_submission(run_id, remote_task_id="remote-ordered")

    mismatch = service.apply_external_update(
        run_id,
        remote_task_id="remote-other",
        remote_status="working",
        update_id="wrong-remote",
        cursor=1,
    )
    working = service.apply_external_update(
        run_id,
        remote_task_id="remote-ordered",
        remote_status="running",
        update_id="update-2",
        cursor=2,
    )
    duplicate = service.apply_external_update(
        run_id,
        remote_task_id="remote-ordered",
        remote_status="working",
        update_id="update-2",
        cursor=3,
    )
    out_of_order = service.apply_external_update(
        run_id,
        remote_task_id="remote-ordered",
        remote_status="working",
        update_id="update-1",
        cursor=1,
    )
    completed = service.apply_external_update(
        run_id,
        remote_task_id="remote-ordered",
        remote_status="succeeded",
        update_id="update-3",
        cursor=3,
        result_summary="Inspection finished with no anomaly.",
    )
    regression = service.apply_external_update(
        run_id,
        remote_task_id="remote-ordered",
        remote_status="working",
        update_id="update-4",
        cursor=4,
    )

    assert mismatch["reason"] == "remote_task_id_mismatch"
    assert working["run"]["current_state"] == "executing"
    assert duplicate["reason"] == "remote_update_duplicate"
    assert out_of_order["reason"] == "remote_update_out_of_order"
    assert completed["run"]["current_state"] == "completed"
    assert completed["run"]["result_summary"] == "Inspection finished with no anomaly."
    assert regression["reason"] in {"remote_status_transition_invalid", "run_already_terminal"}
    assert regression["run"]["current_state"] == "completed"


def test_external_evidence_is_deduplicated_and_recovers_from_task_run_store(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_config = {"enabled": True, "path": str(tmp_path / "task-runs.json")}
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=_external_transport("remote-evidence"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
        store_config=store_config,
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.bind_external_submission(run_id, remote_task_id="remote-evidence")
    observation = {
        "type": "thermal_reading",
        "value": 36.5,
        "context": {"sensor": "thermal-camera", "labels": ["nominal"]},
    }

    service.apply_external_update(
        run_id,
        remote_task_id="remote-evidence",
        remote_status="working",
        update_id="evidence-1",
        cursor=1,
        payload={"observation": observation},
    )
    service.apply_external_update(
        run_id,
        remote_task_id="remote-evidence",
        remote_status="working",
        update_id="evidence-2",
        cursor=2,
        payload={
            "observations": [observation],
            "artifact": {"type": "thermal_image", "uri": "s3://evidence/thermal-1.jpg"},
        },
    )

    restored = RuntimeHandoffService(world_state=world, store_config=store_config)
    restored_run = restored.run_service.require(run_id)
    report = restored.report_payload(run_id)["report"]

    assert len(restored_run.skill_results) == 2
    assert report["observations"] == [observation]
    assert report["artifacts"] == [
        {"type": "thermal_image", "uri": "s3://evidence/thermal-1.jpg"}
    ]


def test_external_evidence_stable_id_upserts_latest_metadata_across_restart(tmp_path) -> None:
    world, plan = _confirmed_plan()
    store_config = {"enabled": True, "path": str(tmp_path / "stable-evidence.json")}
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=_external_transport("remote-stable-evidence"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
        store_config=store_config,
    )
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.bind_external_submission(run_id, remote_task_id="remote-stable-evidence")

    service.apply_external_update(
        run_id,
        remote_task_id="remote-stable-evidence",
        remote_status="working",
        update_id="stable-evidence-1",
        cursor=1,
        payload={
            "artifact": {
                "artifact_id": "photo-1",
                "mime_type": "image/jpeg",
                "uri": "https://store/photo.jpg?sig=old",
                "observed_at": 1,
            }
        },
    )
    service.apply_external_update(
        run_id,
        remote_task_id="remote-stable-evidence",
        remote_status="working",
        update_id="stable-evidence-2",
        cursor=2,
        payload={
            "artifact": {
                "artifact_id": "photo-1",
                "mime_type": "image/jpeg",
                "uri": "https://store/photo.jpg?sig=new",
                "observed_at": 2,
            }
        },
    )

    restored = RuntimeHandoffService(world_state=world, store_config=store_config)
    report = restored.report_payload(run_id)["report"]

    assert report["artifacts"] == [
        {
            "artifact_id": "photo-1",
            "mime_type": "image/jpeg",
            "uri": "https://store/photo.jpg?sig=new",
            "observed_at": 2,
        }
    ]


def test_external_cancel_is_requested_until_remote_confirmation_and_completion_can_win() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(
        world_state=world,
        profile="external",
        executor_transport=_external_transport("remote-cancel"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    first_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.bind_external_submission(first_id, remote_task_id="remote-cancel")
    requested = service.cancel_payload(
        first_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:cancel"),
    )

    assert requested["run"]["current_state"] == "cancel_requested"
    assert requested["run"]["terminal"] is False
    confirmed = service.apply_external_update(
        first_id,
        remote_task_id="remote-cancel",
        remote_status="cancelled",
        update_id="cancel-confirmed",
        cursor=1,
    )
    assert confirmed["run"]["current_state"] == "cancelled"

    world2, plan2 = _confirmed_plan()
    service2 = RuntimeHandoffService(
        world_state=world2,
        profile="external",
        executor_transport=_external_transport("remote-race"),
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.local/submit",
        },
    )
    second_id = service2.submit_plan_payload(plan2)["run"]["run_id"]
    service2.bind_external_submission(second_id, remote_task_id="remote-race")
    service2.cancel_payload(
        second_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:cancel"),
    )
    completed = service2.apply_external_update(
        second_id,
        remote_task_id="remote-race",
        remote_status="completed",
        update_id="completed-after-cancel",
        cursor=1,
    )
    assert completed["run"]["current_state"] == "completed"


def test_chinese_runtime_control_and_area_id_are_stable() -> None:
    world, plan = _confirmed_plan()
    plan["goal"] = "巡检 A 区"
    mission = plan["mission"]["mission"]
    mission["goal"] = "巡检 A 区"
    mission["steps"][0]["target"] = "A区"
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]

    paused = service.handle_chat_control(
        "先停一下",
        conversation_session_id="chat-control-thread",
        operator_id="operator-1",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="test",
        runtime_permission="runtime:pause",
    )
    status = service.handle_chat_control(
        "现在执行到哪了",
        conversation_session_id="chat-control-thread",
        operator_id="operator-1",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="test",
        runtime_permission="runtime:read",
    )
    handoff = service.get_payload(run_id)["run"]["handoff"]

    assert handoff["target_area"] == "area-a"
    assert handoff["steps"][0]["parameters"]["area_id"] == "area-a"
    assert paused is not None
    assert paused["runtime"]["run"]["current_state"] == "paused"
    assert status is not None
    assert "TaskRun" in status["reply"]


def test_runtime_mutation_owner_rejects_missing_operator_context() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]

    result = service.pause_payload(run_id, operator_id="operator-1")
    voice = service.voice_turn_payload(
        "pause current task",
        runtime_permission="runtime:pause",
    )

    assert result["handled"] is False
    assert result["reason"] == "runtime_operator_context_required"
    assert voice["handled"] is False
    assert voice["reason"] == "runtime_operator_authentication_required"
    assert result["run"]["operator_actions"] == []


def test_voice_runtime_control_persists_sanitized_operator_provenance() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:advance"),
    )

    paused = service.voice_turn_payload(
        "pause current task",
        conversation_session_id="voice-thread-1",
        operator_id="security-1",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="oidc",
        runtime_permission="runtime:pause",
        reason="visitor entered the path",
        risk_acknowledgement=True,
    )

    action = paused["runtime"]["run"]["operator_actions"][-1]
    assert action["action"] == "pause"
    assert action["operator_id"] == "security-1"
    assert action["reason"] == "visitor entered the path"
    assert action["risk_acknowledgement"] is True
    assert action["operator_context"] == {
        "operator_id": "security-1",
        "roles": ["operator"],
        "authenticated": True,
        "source": "oidc",
        "permission": "runtime:pause",
        "thread_id": "voice-thread-1",
    }
    assert paused["voice_turn"]["operator"] == action["operator_context"]
    assert paused["voice_turn"]["runtime_permission"] == "runtime:pause"


def test_runtime_operator_actions_store_only_sanitized_provenance() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    context = _operator_context(
        "runtime:pause",
        roles=["operator", "operator", " "],
    )
    context["api_key"] = "secret"
    context["headers"] = {"authorization": "bearer secret"}

    paused = service.pause_payload(
        run_id,
        operator_id="operator-1",
        operator_context=context,
    )

    action_context = paused["run"]["operator_actions"][-1]["operator_context"]
    assert action_context == {
        "operator_id": "operator-1",
        "roles": ["operator"],
        "authenticated": True,
        "source": "test",
        "permission": "runtime:pause",
        "thread_id": "runtime-test-thread",
    }


def test_voice_runtime_control_rejects_keyword_prose_and_permission_mismatch() -> None:
    world, plan = _confirmed_plan()
    service = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    run_id = service.submit_plan_payload(plan)["run"]["run_id"]
    service.advance_payload(
        run_id,
        operator_id="operator-1",
        operator_context=_operator_context("runtime:advance"),
    )

    prose = service.voice_turn_payload(
        "Please explain how to cancel task safely",
        runtime_permission="runtime:submit",
    )
    mismatch = service.voice_turn_payload(
        "cancel current task",
        runtime_permission="runtime:submit",
    )

    assert prose["handled"] is False
    assert prose["reason"] == "no_runtime_control_intent"
    assert mismatch["handled"] is False
    assert mismatch["reason"] == "runtime_control_permission_mismatch"
    run = service.get_payload(run_id)["run"]
    assert run["current_state"] == "executing"
    assert all(action["action"] != "cancel" for action in run["operator_actions"])
