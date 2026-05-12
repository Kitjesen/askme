# ruff: noqa: I001
"""Evaluate askme robot-task scenarios and write an auditable verdict artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService  # noqa: E402
from askme.runtime.handoff import (  # noqa: E402
    RuntimeHandoffService,
    SkillRegistry,
    TaskHandoff,
    TaskStep,
)
from askme.runtime.mission import MissionService  # noqa: E402


DEFAULT_REPORT_PATH = Path("artifacts/runtime_handoff/scenario-evaluation.json")


def evaluate_scenarios() -> dict[str, Any]:
    """Run deterministic robot-task scenarios without hardware dispatch."""
    scenarios = [
        _scenario_happy_path_completed(),
        _scenario_estop_blocks_and_replans(),
        _scenario_unauthorized_viewer_blocked(),
        _scenario_localization_blocked_with_perception_request(),
        _scenario_voice_runtime_controls_share_state(),
        _scenario_voice_confirm_cannot_submit_runtime(),
        _scenario_direct_motor_skill_rejected(),
    ]
    passed = sum(1 for item in scenarios if item["passed"])
    return {
        "suite": "askme-robot-task-runtime",
        "hardware_dispatch": False,
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": len(scenarios) - passed,
        "status": "passed" if passed == len(scenarios) else "failed",
        "scenarios": scenarios,
        "generated_at": time.time(),
    }


def write_report(payload: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _scenario_happy_path_completed() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    _seed_area_map(world)
    runtime = RuntimeHandoffService(world_state=world)

    result = runtime.submit_plan_payload(plan)

    return _verdict(
        "happy_path_completed",
        result["accepted"] is True
        and result["run"]["current_state"] == "completed"
        and len(result["run"]["report"]["skill_results"]) == 5,
        observed={
            "accepted": result["accepted"],
            "state": result["run"]["current_state"],
            "skill_results": len(result["run"]["report"]["skill_results"]),
        },
    )


def _scenario_estop_blocks_and_replans() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    world.update_robot_state({"estop_active": True}, stale_after_s=60.0)
    runtime = RuntimeHandoffService(world_state=world)

    result = runtime.submit_plan_payload(plan)

    return _verdict(
        "estop_blocks_and_replans",
        result["accepted"] is False
        and "estop_active" in result["preflight"]["failed_checks"]
        and result["replan_proposal"]["recommended_action"] == "clear_estop_then_retry",
        observed={
            "failed_checks": result["preflight"]["failed_checks"],
            "replan": result["replan_proposal"]["recommended_action"],
        },
    )


def _scenario_unauthorized_viewer_blocked() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    plan["operator_roles"] = ["viewer"]
    runtime = RuntimeHandoffService(world_state=world)

    result = runtime.submit_plan_payload(plan)

    return _verdict(
        "unauthorized_viewer_blocked",
        result["accepted"] is False
        and "operator_not_authorized" in result["preflight"]["failed_checks"],
        observed={
            "failed_checks": result["preflight"]["failed_checks"],
            "roles": result["handoff"]["operator_roles"],
        },
    )


def _scenario_localization_blocked_with_perception_request() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    _seed_area_map(world, localized=False, localization_quality=0.1)
    runtime = RuntimeHandoffService(world_state=world)

    result = runtime.submit_plan_payload(plan)
    reasons = {
        item["reason"]
        for item in result["preflight"]["perception_requests"]
    }

    return _verdict(
        "localization_blocked_with_perception_request",
        result["accepted"] is False
        and "map_localization_unavailable" in result["preflight"]["failed_checks"]
        and "refresh_localization" in reasons
        and result["replan_proposal"]["recommended_action"] == "refresh_localization_then_replan",
        observed={
            "failed_checks": result["preflight"]["failed_checks"],
            "perception_reasons": sorted(reasons),
            "replan": result["replan_proposal"]["recommended_action"],
        },
    )


def _scenario_voice_runtime_controls_share_state() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    runtime = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)
    submitted = runtime.submit_plan_payload(plan)
    run_id = submitted["run"]["run_id"]
    runtime.advance_payload(run_id)

    paused = runtime.voice_turn_payload("先停一下", transcript_id="scenario-voice-pause")
    status = runtime.voice_turn_payload("现在执行到哪了", transcript_id="scenario-voice-status")
    resumed = runtime.voice_turn_payload("继续", transcript_id="scenario-voice-resume")

    return _verdict(
        "voice_runtime_controls_share_state",
        paused["runtime"]["run"]["current_state"] == "paused"
        and status["runtime"]["run"]["current_state"] == "paused"
        and resumed["runtime"]["run"]["current_state"] == "executing"
        and paused["voice_turn"]["safety_bypass_allowed"] is False,
        observed={
            "pause_state": paused["runtime"]["run"]["current_state"],
            "status_state": status["runtime"]["run"]["current_state"],
            "resume_state": resumed["runtime"]["run"]["current_state"],
            "safety_bypass_allowed": paused["voice_turn"]["safety_bypass_allowed"],
        },
    )


def _scenario_voice_confirm_cannot_submit_runtime() -> dict[str, Any]:
    world, _plan = _confirmed_plan()
    runtime = RuntimeHandoffService(world_state=world, profile="sim", auto_complete=False)

    result = runtime.voice_turn_payload("确认", transcript_id="scenario-voice-confirm")

    return _verdict(
        "voice_confirm_cannot_submit_runtime",
        result["handled"] is False
        and result["reason"] == "no_runtime_control_intent"
        and result["runtime"]["active_run"] is None,
        observed={
            "handled": result["handled"],
            "reason": result["reason"],
            "active_run": result["runtime"]["active_run"],
        },
    )


def _scenario_direct_motor_skill_rejected() -> dict[str, Any]:
    world, plan = _confirmed_plan()
    registry = SkillRegistry()
    runtime = RuntimeHandoffService(world_state=world)
    now = time.time()
    handoff = TaskHandoff(
        handoff_id="handoff-direct-motor",
        plan_id=plan["plan_id"],
        session_id=plan["planning_session_id"],
        operator_id="operator-1",
        intent="direct_motor",
        task_type="navigate_to",
        target_area="area-a",
        target_object=None,
        constraints=[],
        steps=[
            TaskStep(
                step_id="step-direct-motor",
                sequence=1,
                skill_name="drive_motor_direct",
                parameters={"meters": 10},
            )
        ],
        risk_level="high",
        required_capabilities=[],
        missing_info=[],
        confirmation_status="confirmed",
        world_state_snapshot_id="world-direct-motor",
        safety_notes=[],
        created_at=now,
        expires_at=now + 60,
        planner_version="scenario",
        source_plan=plan,
        world_state_snapshot=world.snapshot(),
        operator_roles=["operator"],
    )

    result = runtime.arbiter.submit(handoff)

    return _verdict(
        "direct_motor_skill_rejected",
        result["accepted"] is False
        and any(
            item.startswith("unregistered_skill:drive_motor_direct")
            for item in result["preflight"]["failed_checks"]
        ),
        observed={
            "failed_checks": result["preflight"]["failed_checks"],
            "registered_skill_count": registry.snapshot()["count"],
        },
    )


def _confirmed_plan() -> tuple[WorldStateService, dict[str, Any]]:
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


def _seed_area_map(
    world: WorldStateService,
    *,
    localized: bool = True,
    localization_quality: float = 0.92,
) -> None:
    world.update_area_catalog(
        [{"area_id": "area-a", "allowed": True, "map_id": "map-main", "map_version": "v1"}],
        map_id="map-main",
        map_version="v1",
        stale_after_s=120.0,
    )
    world.update_map_state(
        map_id="map-main",
        map_version="v1",
        localized=localized,
        localization_quality=localization_quality,
        stale_after_s=60.0,
    )


def _verdict(name: str, passed: bool, *, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)

    payload = evaluate_scenarios()
    report_path = write_report(payload, Path(args.report_path))
    print(json.dumps({**payload, "report_path": str(report_path)}, ensure_ascii=False, indent=2))  # noqa: T201
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
