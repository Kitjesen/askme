from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

import pytest

from askme.cognition import WorldStateService
from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorCancelResult,
    RuntimeExecutorStatusUpdate,
    RuntimeExecutorSubmitResult,
    RuntimeExecutorUpdate,
)
from askme.runtime.task.executor_supervisor import ExternalTaskSupervisor
from askme.runtime.task.handoff import RuntimeHandoffService


class _Transport:
    def __init__(self) -> None:
        self.statuses: deque[RuntimeExecutorStatusUpdate] = deque()
        self.submit_errors: deque[Exception] = deque()
        self.submit_calls = []
        self.cancel_result: RuntimeExecutorCancelResult | None = None
        self.status_calls = []
        self.cancel_calls = []
        self.status_entered = threading.Event()
        self.status_release = threading.Event()
        self.block_status = False
        self.closed = False

    def submit(self, request):
        self.submit_calls.append(request)
        if self.submit_errors:
            raise self.submit_errors.popleft()
        return RuntimeExecutorSubmitResult(
            remote_task_id=f"remote:{request.correlation_id}",
            status="queued",
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            cursor="1",
            observed_at=time.time(),
        )

    def get_status(self, request):
        self.status_calls.append(request)
        self.status_entered.set()
        if self.block_status:
            self.status_release.wait(timeout=1.0)
        if self.statuses:
            return self.statuses.popleft()
        return RuntimeExecutorStatusUpdate(
            remote_task_id=request.remote_task_id,
            status="executing",
            correlation_id=request.correlation_id,
            cursor="2",
            observed_at=time.time(),
        )

    def cancel(self, request):
        self.cancel_calls.append(request)
        assert self.cancel_result is not None
        return self.cancel_result

    def close(self) -> None:
        self.closed = True


def _world() -> WorldStateService:
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 90,
            "estop_active": False,
            "localized": True,
        },
        stale_after_s=60.0,
    )
    return world


def _service(transport: _Transport) -> RuntimeHandoffService:
    return RuntimeHandoffService(
        world_state=_world(),
        profile="external",
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "https://runtime.example",
        },
        executor_transport=transport,
    )


def _submit(service: RuntimeHandoffService, suffix: str = "1") -> str:
    payload = service.submit_plan_payload(
        {
            "plan_id": f"plan-{suffix}",
            "planning_session_id": f"session-{suffix}",
            "intent": "status_report",
            "handoff_ready": True,
            "confirmation_status": "confirmed",
            "mission": {"mission": {"mission_type": "status_report"}},
        }
    )
    return payload["run"]["run_id"]


async def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


@pytest.mark.asyncio
async def test_start_recovers_persisted_run_and_keeps_exactly_one_poller() -> None:
    transport = _Transport()
    transport.block_status = True
    service = _service(transport)
    run_id = _submit(service)
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.01,
        poll_jitter_ratio=0.0,
    )

    await supervisor.start()
    try:
        await _wait_for(transport.status_entered.is_set)
        await asyncio.gather(*(supervisor.ensure_tracked(run_id) for _ in range(5)))
        assert supervisor.tracked_run_ids == (run_id,)
        assert len(transport.status_calls) == 1
    finally:
        transport.status_release.set()
        await supervisor.close()


@pytest.mark.asyncio
async def test_terminal_update_stops_poller_and_preserves_authoritative_summary() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service)
    remote_id = service.run_service.require(run_id).remote_task_id or ""
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id=remote_id,
            status="completed",
            correlation_id=run_id,
            cursor="3",
            result_summary="Battery 90%; systems nominal.",
            updates=(
                RuntimeExecutorUpdate(
                    event_id="completed-1",
                    status="completed",
                    cursor="3",
                    observed_at=time.time(),
                ),
            ),
            observed_at=time.time(),
        )
    )
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.01,
        poll_jitter_ratio=0.0,
    )

    await supervisor.start()
    try:
        await _wait_for(lambda: service.run_service.require(run_id).terminal)
        await _wait_for(lambda: run_id not in supervisor.tracked_run_ids)
        run = service.run_service.require(run_id)
        assert run.current_state == "completed"
        assert run.result_summary == "Battery 90%; systems nominal."
        assert run.processed_remote_update_ids == ["completed-1"]
        assert len(transport.status_calls) == 1
    finally:
        await supervisor.close()


@pytest.mark.asyncio
async def test_refresh_projects_structured_executor_evidence_into_task_report() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service)
    run = service.run_service.require(run_id)
    observation = {"type": "inspection", "finding": "no anomaly", "area": "A"}
    artifact = {"type": "image_ref", "uri": "s3://evidence/area-a.jpg"}
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id=run.remote_task_id or "",
            status="completed",
            correlation_id=run_id,
            cursor="3",
            result_summary="Inspection completed.",
            updates=(
                RuntimeExecutorUpdate(
                    event_id="evidence-1",
                    status="completed",
                    cursor="3",
                    observed_at=time.time(),
                    payload={
                        "skill_name": "inspect_equipment",
                        "observation": observation,
                        "observations": [observation, {"type": "temperature", "value": 36.5}],
                        "artifact": artifact,
                        "artifacts": [artifact],
                    },
                ),
            ),
            observed_at=time.time(),
        )
    )
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)

    result = await supervisor.refresh(run_id)

    assert result["handled"] is True
    projected = service.run_service.require(run_id)
    assert len(projected.skill_results) == 1
    assert projected.skill_results[0].skill_name == "inspect_equipment"
    report = service.report_payload(run_id)["report"]
    assert report["observations"] == [
        observation,
        {"type": "temperature", "value": 36.5},
    ]
    assert report["artifacts"] == [artifact]


@pytest.mark.asyncio
async def test_refresh_does_not_erase_existing_summary_with_blank_remote_value() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service)
    run = service.run_service.require(run_id)
    run.result_summary = "Stable summary"
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id=run.remote_task_id or "",
            status="executing",
            correlation_id=run_id,
            cursor="2",
            result_summary="",
        )
    )
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)

    result = await supervisor.refresh(run_id)

    assert result["handled"] is True
    assert result["run"]["result_summary"] == "Stable summary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_status", "acknowledged", "state", "error_code", "records_cancel"),
    [
        ("cancelling", True, "cancel_requested", "", True),
        ("cancelled", True, "cancelled", "", False),
        ("rejected", False, "blocked", "cancel_rejected", False),
        ("completed", False, "completed", "run_already_completed", False),
    ],
)
async def test_cancel_projects_remote_truth_before_local_intent(
    remote_status: str,
    acknowledged: bool,
    state: str,
    error_code: str,
    records_cancel: bool,
) -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service, remote_status)
    run = service.run_service.require(run_id)
    transport.cancel_result = RuntimeExecutorCancelResult(
        remote_task_id=run.remote_task_id or "",
        status=remote_status,
        correlation_id=run_id,
        idempotency_key=f"cancel:{run_id}",
        cursor="9",
        result_summary="remote final" if remote_status in {"completed", "cancelled"} else "",
        observed_at=time.time(),
    )
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)

    outcome = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="operator requested",
        operator_context={
            "operator_id": "operator-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "api",
            "permission": "runtime:cancel",
        },
    )

    assert outcome.remote_acknowledged is acknowledged
    assert outcome.state == state
    assert outcome.error_code == error_code
    assert outcome.snapshot["current_state"] == state
    assert bool(outcome.run["operator_actions"]) is records_cancel
    assert len(transport.cancel_calls) == 1


@pytest.mark.asyncio
async def test_cancel_terminal_update_preserves_summary_and_wins_pending_top_level() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service, "cancel-race")
    run = service.run_service.require(run_id)
    transport.cancel_result = RuntimeExecutorCancelResult(
        remote_task_id=run.remote_task_id or "",
        status="cancelling",
        correlation_id=run_id,
        idempotency_key=f"cancel:{run_id}",
        cursor="9",
        result_summary="Task completed before cancellation arrived.",
        updates=(
            RuntimeExecutorUpdate(
                event_id="completed-before-cancel",
                status="completed",
                cursor="8",
                observed_at=time.time(),
            ),
        ),
    )
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)

    outcome = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="too late",
        operator_context={
            "operator_id": "operator-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "api",
            "permission": "runtime:cancel",
        },
    )

    assert outcome.remote_acknowledged is False
    assert outcome.state == "completed"
    assert outcome.error_code == "run_already_completed"
    assert outcome.run["result_summary"] == "Task completed before cancellation arrived."
    assert outcome.run["operator_actions"] == []


@pytest.mark.asyncio
async def test_cancel_rejects_missing_operator_context_before_transport() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service, "unauthorized-cancel")
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)

    outcome = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="missing provenance",
        operator_context=None,
    )

    assert outcome.remote_acknowledged is False
    assert outcome.error_code == "runtime_operator_context_required"
    assert transport.cancel_calls == []


@pytest.mark.asyncio
async def test_close_cancels_pollers_and_is_idempotent() -> None:
    transport = _Transport()
    transport.block_status = True
    service = _service(transport)
    run_id = _submit(service)
    supervisor = ExternalTaskSupervisor(handoff_service=service, transport=transport)
    await supervisor.start()
    await _wait_for(transport.status_entered.is_set)

    transport.status_release.set()
    await supervisor.close()
    await supervisor.close()

    assert supervisor.tracked_run_ids == ()
    assert run_id not in supervisor.tracked_run_ids


@pytest.mark.asyncio
async def test_poll_deadline_degrades_frequency_without_abandoning_nonterminal_run() -> None:
    transport = _Transport()
    service = _service(transport)
    run_id = _submit(service, "long-running")
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.005,
        poll_max_s=0.01,
        poll_deadline_s=0.02,
        poll_jitter_ratio=0.0,
    )
    await supervisor.start()
    try:
        await _wait_for(
            lambda: service.run_service.require(run_id).last_poll_error_code
            == "poll_deadline_exceeded"
        )
        calls_at_deadline = len(transport.status_calls)
        await _wait_for(lambda: len(transport.status_calls) > calls_at_deadline)
        assert service.run_service.require(run_id).terminal is False
        assert run_id in supervisor.tracked_run_ids
    finally:
        await supervisor.close()


@pytest.mark.asyncio
async def test_unknown_submission_defers_cancel_then_cancels_after_reconciliation() -> None:
    transport = _Transport()
    transport.submit_errors.append(AmbiguousRuntimeSubmissionError("timeout after send"))
    service = _service(transport)
    run_id = _submit(service, "unknown-cancel")
    run = service.run_service.require(run_id)
    assert run.current_state == "submission_unknown"
    assert run.remote_task_id is None
    transport.cancel_result = RuntimeExecutorCancelResult(
        remote_task_id=f"remote:{run_id}",
        status="cancelled",
        correlation_id=run_id,
        idempotency_key=f"cancel:{run_id}",
        cursor="3",
        observed_at=time.time(),
    )
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.01,
        poll_jitter_ratio=0.0,
    )

    deferred = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="operator requested",
        operator_context={
            "operator_id": "operator-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "voice",
            "permission": "runtime:cancel",
            "conversation_session_id": "voice-session-1",
        },
    )

    assert deferred.remote_acknowledged is False
    assert deferred.error_code == "cancel_deferred_until_reconciled"
    assert deferred.run["deferred_cancel_request"]["reason"] == "operator requested"
    assert transport.cancel_calls == []

    await supervisor.start()
    try:
        await _wait_for(lambda: service.run_service.require(run_id).current_state == "cancelled")
        reconciled = service.run_service.require(run_id)
        assert len(transport.submit_calls) == 2
        assert transport.submit_calls[0].idempotency_key == transport.submit_calls[1].idempotency_key
        assert len(transport.cancel_calls) == 1
        assert reconciled.deferred_cancel_request == {}
    finally:
        await supervisor.close()


@pytest.mark.asyncio
async def test_deferred_cancel_acknowledged_as_cancelling_resumes_status_polling() -> None:
    transport = _Transport()
    transport.submit_errors.append(AmbiguousRuntimeSubmissionError("timeout after send"))
    service = _service(transport)
    run_id = _submit(service, "deferred-cancelling")
    remote_id = f"remote:{run_id}"
    transport.cancel_result = RuntimeExecutorCancelResult(
        remote_task_id=remote_id,
        status="cancelling",
        correlation_id=run_id,
        idempotency_key=f"cancel:{run_id}",
        cursor="3",
        observed_at=time.time(),
    )
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id=remote_id,
            status="cancelled",
            correlation_id=run_id,
            cursor="4",
            observed_at=time.time(),
        )
    )
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.01,
        poll_jitter_ratio=0.0,
    )

    deferred = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="operator requested",
        operator_context={
            "operator_id": "operator-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "voice",
            "permission": "runtime:cancel",
            "conversation_session_id": "voice-session-1",
        },
    )

    assert deferred.error_code == "cancel_deferred_until_reconciled"
    await supervisor.start()
    try:
        await _wait_for(lambda: service.run_service.require(run_id).terminal)
        run = service.run_service.require(run_id)
        assert run.current_state == "cancelled"
        assert run.deferred_cancel_request == {}
        assert len(transport.cancel_calls) == 1
        assert len(transport.status_calls) == 1
    finally:
        await supervisor.close()


@pytest.mark.asyncio
async def test_deferred_cancel_not_accepted_cannot_starve_terminal_status() -> None:
    transport = _Transport()
    transport.submit_errors.append(AmbiguousRuntimeSubmissionError("timeout after send"))
    service = _service(transport)
    run_id = _submit(service, "deferred-not-accepted")
    remote_id = f"remote:{run_id}"
    transport.cancel_result = RuntimeExecutorCancelResult(
        remote_task_id=remote_id,
        status="queued",
        correlation_id=run_id,
        idempotency_key=f"cancel:{run_id}",
        cursor="3",
        observed_at=time.time(),
    )
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id=remote_id,
            status="completed",
            correlation_id=run_id,
            cursor="4",
            result_summary="remote completed before cancel was accepted",
            observed_at=time.time(),
        )
    )
    supervisor = ExternalTaskSupervisor(
        handoff_service=service,
        transport=transport,
        poll_initial_s=0.01,
        poll_jitter_ratio=0.0,
    )

    deferred = await supervisor.request_cancel(
        run_id,
        operator_id="operator-1",
        reason="operator requested",
        operator_context={
            "operator_id": "operator-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "voice",
            "permission": "runtime:cancel",
            "conversation_session_id": "voice-session-1",
        },
    )

    assert deferred.error_code == "cancel_deferred_until_reconciled"
    await supervisor.start()
    try:
        await _wait_for(lambda: service.run_service.require(run_id).terminal)
        run = service.run_service.require(run_id)
        assert run.current_state == "completed"
        assert run.result_summary == "remote completed before cancel was accepted"
        assert run.deferred_cancel_request == {}
        assert len(transport.cancel_calls) == 1
        assert len(transport.status_calls) == 1
    finally:
        await supervisor.close()
