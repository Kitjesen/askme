from __future__ import annotations

import json
import time

from askme.runtime.audit import RuntimeAuditConfig, RuntimeAuditLog
from askme.runtime.handoff import TaskHandoff, TaskRunService, TaskStep


def _handoff() -> TaskHandoff:
    return TaskHandoff(
        handoff_id="handoff-audit",
        plan_id="plan-audit",
        session_id="session-audit",
        operator_id="operator-audit",
        intent="inspection",
        task_type="inspection_patrol",
        target_area="area-a",
        target_object=None,
        constraints=[],
        steps=[
            TaskStep(
                step_id="step-1",
                sequence=1,
                skill_name="go_to_area",
                parameters={"area_id": "area-a"},
            )
        ],
        risk_level="medium",
        required_capabilities=[],
        missing_info=[],
        confirmation_status="confirmed",
        world_state_snapshot_id="world-audit",
        safety_notes=[],
        created_at=time.time(),
        expires_at=time.time() + 60,
        planner_version="test",
        source_plan={},
        world_state_snapshot={"updated_at": time.time()},
    )


def test_runtime_audit_log_is_disabled_by_default(tmp_path) -> None:
    path = tmp_path / "runtime-audit.jsonl"
    audit = RuntimeAuditLog({"path": path})

    run_service = TaskRunService(audit_log=audit)
    run = run_service.create(_handoff())
    run_service.transition(run, "completed", "task_completed", "done")

    assert path.exists() is False


def test_runtime_audit_log_appends_events_actions_and_terminal_snapshot(tmp_path) -> None:
    path = tmp_path / "runtime-audit.jsonl"
    audit = RuntimeAuditLog(RuntimeAuditConfig(enabled=True, path=path))
    run_service = TaskRunService(audit_log=audit)

    run = run_service.create(_handoff())
    run_service.transition(run, "queued", "task_queued", "queued")
    pause = run_service.pause(run.run_id, operator_id="operator-audit")
    cancel = run_service.cancel(run.run_id, operator_id="operator-audit")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    kinds = [record["kind"] for record in records]

    assert pause["handled"] is True
    assert cancel["run"]["current_state"] == "cancelled"
    assert "runtime_event" in kinds
    assert "operator_action" in kinds
    assert "task_run_terminal_snapshot" in kinds
    assert records[-1]["kind"] == "task_run_terminal_snapshot"
    assert records[-1]["run"]["current_state"] == "cancelled"
    assert records[-1]["report"]["status"] == "cancelled"
    assert [
        record["action"]["action"]
        for record in records
        if record["kind"] == "operator_action"
    ] == ["pause", "cancel"]

