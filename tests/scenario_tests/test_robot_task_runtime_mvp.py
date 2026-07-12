from __future__ import annotations

import time

from askme.runtime.handoff import RuntimeHandoffService
from askme.runtime.mission import MissionService

from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService


def _planner_stack() -> tuple[WorldStateService, CognitivePlanner, RuntimeHandoffService]:
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 88,
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
    runtime = RuntimeHandoffService(world_state=world)
    return world, planner, runtime


def test_scenario_inspection_plan_confirm_handoff_and_report() -> None:
    world, planner, runtime = _planner_stack()

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    assert draft.interaction_state == "awaiting_confirmation"
    assert confirmed.handoff_ready is True
    assert handoff["accepted"] is True
    assert handoff["run"]["current_state"] == "completed"
    assert handoff["run"]["report"]["summary"].startswith("Completed inspection_patrol")
    assert len(handoff["run"]["report"]["skill_results"]) == 5
    assert any(
        observation["type"] == "inspection"
        for observation in handoff["run"]["report"]["observations"]
    )
    assert any(
        event["kind"] == "runtime.task_completed"
        for event in world.snapshot()["events"]
    )


def test_scenario_confirmed_plan_blocked_when_estop_active() -> None:
    world, planner, runtime = _planner_stack()
    world.update_robot_state({"estop_active": True}, stale_after_s=60.0)

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    assert handoff["accepted"] is False
    assert handoff["run"]["current_state"] == "blocked"
    assert "estop_active" in handoff["preflight"]["failed_checks"]


def test_scenario_confirmed_plan_blocked_when_area_catalog_denies_target() -> None:
    world, planner, runtime = _planner_stack()
    world.update_area_catalog(
        [
            {
                "area_id": "area-a",
                "allowed": False,
                "status": "restricted",
            }
        ],
        stale_after_s=60.0,
    )

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    assert handoff["accepted"] is False
    assert handoff["run"]["current_state"] == "blocked"
    assert "target_area_blocked" in handoff["preflight"]["failed_checks"]
    assert handoff["replan_proposal"]["recommended_action"] == "choose_allowed_area"


def test_scenario_missing_area_catalog_creates_perception_request() -> None:
    _world, planner, runtime = _planner_stack()

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    reasons = {
        item["reason"]
        for item in handoff["preflight"]["perception_requests"]
    }
    assert handoff["accepted"] is True
    assert "area_catalog_unavailable" in handoff["preflight"]["warnings"]
    assert "load_area_catalog" in reasons
    assert any(
        event["event_type"] == "perception_requested"
        for event in handoff["run"]["runtime_events"]
    )


def test_scenario_confirmed_plan_blocked_when_world_state_is_stale() -> None:
    world, planner, runtime = _planner_stack()

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    original_snapshot = world.snapshot

    def stale_snapshot(*, include_stale: bool = True):
        payload = original_snapshot(include_stale=include_stale)
        payload["updated_at"] = time.time() - 120
        return payload

    world.snapshot = stale_snapshot  # type: ignore[method-assign]
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    assert handoff["accepted"] is False
    assert "world_state_stale" in handoff["preflight"]["failed_checks"]
    assert handoff["replan_proposal"]["recommended_action"] == "refresh_world_state_then_reconfirm"


def test_scenario_chinese_inspection_confirm_shadow_preview() -> None:
    world, planner, _runtime = _planner_stack()
    runtime = RuntimeHandoffService(world_state=world, profile="shadow")

    draft = planner.plan_from_text("巡检 A 区", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    handoff = runtime.submit_plan_payload(confirmed.to_dict())

    assert handoff["accepted"] is True
    assert handoff["run"]["current_state"] == "shadowed"
    assert handoff["run"]["profile"] == "shadow"
    assert handoff["run"]["handoff"]["target_area"] == "area-a"
    assert handoff["shadow_plan"]["hardware_dispatch"] is False
    assert handoff["shadow_plan"]["would_execute"][0]["skill_name"] == "go_to_area"


def test_scenario_sim_runtime_pause_resume_advance_and_cancel() -> None:
    _world, planner, runtime = _planner_stack()
    runtime = RuntimeHandoffService(world_state=_world, profile="sim", auto_complete=False)

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    submitted = runtime.submit_plan_payload(confirmed.to_dict())
    run_id = submitted["run"]["run_id"]
    advanced = runtime.advance_payload(run_id)
    paused = runtime.pause_payload(run_id)
    resumed = runtime.resume_payload(run_id)
    cancelled = runtime.cancel_payload(run_id)

    assert submitted["run"]["current_state"] == "queued"
    assert advanced["run"]["current_step_index"] == 1
    assert paused["run"]["current_state"] == "paused"
    assert resumed["run"]["current_state"] == "executing"
    assert cancelled["run"]["current_state"] == "cancelled"


def test_scenario_voice_style_runtime_controls_share_task_state() -> None:
    _world, planner, _runtime = _planner_stack()
    runtime = RuntimeHandoffService(world_state=_world, profile="sim", auto_complete=False)

    draft = planner.plan_from_text("inspect area-a", robot_id="dog-1")
    confirmed = planner.plan_from_text(
        "",
        planning_session_id=draft.planning_session_id,
        operator_confirmation=True,
        robot_id="dog-1",
    )
    submitted = runtime.submit_plan_payload(confirmed.to_dict())
    run_id = submitted["run"]["run_id"]
    runtime.advance_payload(run_id)

    paused = runtime.voice_turn_payload("先停一下", transcript_id="voice-pause")
    status = runtime.voice_turn_payload("现在执行到哪了", transcript_id="voice-status")
    resumed = runtime.voice_turn_payload("继续", transcript_id="voice-resume")
    confirm = runtime.voice_turn_payload("确认", transcript_id="voice-confirm")

    assert paused["runtime"]["run"]["current_state"] == "paused"
    assert status["voice_turn"]["runtime_control_intent"] == "status"
    assert status["runtime"]["run"]["current_state"] == "paused"
    assert resumed["runtime"]["run"]["current_state"] == "executing"
    assert confirm["handled"] is False
    assert confirm["reason"] == "no_runtime_control_intent"
    assert confirm["voice_turn"]["safety_bypass_allowed"] is False
