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
    RuntimeExecutorTransportError,
    RuntimeExecutorUpdate,
)
from askme.runtime.task.executor_supervisor import ExternalTaskSupervisor
from askme.runtime.task.handoff import RuntimeHandoffService
from askme.runtime.task.mission import MissionService
from askme.runtime.task.voice_lifecycle import (
    VoiceTaskLifecycleService,
    VoiceTaskOperatorContext,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.submit_calls = []
        self.status_calls = []
        self.cancel_calls = []
        self.call_threads: list[int] = []
        self.statuses: deque[RuntimeExecutorStatusUpdate] = deque()
        self.default_status = "executing"
        self.default_cursor = "cursor-2"
        self.cancel_confirmed = False
        self.closed = False

    def submit(self, request):
        self.call_threads.append(threading.get_ident())
        self.submit_calls.append(request)
        return RuntimeExecutorSubmitResult(
            remote_task_id="remote-status-1",
            status="queued",
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            cursor="cursor-1",
            observed_at=time.time(),
        )

    def get_status(self, request):
        self.call_threads.append(threading.get_ident())
        self.status_calls.append(request)
        if self.statuses:
            return self.statuses.popleft()
        if self.cancel_confirmed:
            return RuntimeExecutorStatusUpdate(
                remote_task_id=request.remote_task_id,
                status="cancelled",
                correlation_id=request.correlation_id,
                cursor="cursor-9",
                observed_at=time.time(),
            )
        return RuntimeExecutorStatusUpdate(
            remote_task_id=request.remote_task_id,
            status=self.default_status,
            correlation_id=request.correlation_id,
            cursor=self.default_cursor,
            observed_at=time.time(),
        )

    def cancel(self, request):
        self.call_threads.append(threading.get_ident())
        self.cancel_calls.append(request)
        return RuntimeExecutorCancelResult(
            remote_task_id=request.remote_task_id,
            status="cancelling",
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            cursor="cursor-8",
            observed_at=time.time(),
        )


class _CrashAfterRemoteAcceptTransport(_FakeTransport):
    def submit(self, request):
        self.call_threads.append(threading.get_ident())
        self.submit_calls.append(request)
        raise SystemExit("simulated process crash after remote acceptance")


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


_TRUSTED_OPERATOR = VoiceTaskOperatorContext(
    operator_id="voice-device-1",
    roles=("operator",),
    authenticated=True,
    source="speaker_verification",
    person_id="operator-person-1",
    permissions=("runtime:read", "runtime:submit", "runtime:cancel"),
)


def _service(
    transport: _FakeTransport,
    *,
    operator_context: VoiceTaskOperatorContext | None = _TRUSTED_OPERATOR,
    store_config: dict | None = None,
    max_delivery_attempts: int = 3,
    mission_service: MissionService | None = None,
    clarification_ttl_s: float = 45.0,
) -> tuple[RuntimeHandoffService, ExternalTaskSupervisor, VoiceTaskLifecycleService]:
    handoff = RuntimeHandoffService(
        world_state=_world(),
        profile="external",
        external_runtime_config={
            "enable_external_runtime": True,
            "endpoint": "http://runtime.invalid/v1",
        },
        executor_transport=transport,
        store_config=store_config,
    )
    supervisor = ExternalTaskSupervisor(
        handoff_service=handoff,
        transport=transport,
        poll_initial_s=0.01,
        poll_max_s=0.02,
        poll_deadline_s=1.0,
        poll_jitter_ratio=0.0,
    )
    lifecycle = VoiceTaskLifecycleService(
        handoff_service=handoff,
        supervisor=supervisor,
        mission_service=mission_service,
        operator_context=operator_context,
        clarification_ttl_s=clarification_ttl_s,
        delivery_retry_delay_s=0.0,
        max_delivery_attempts=max_delivery_attempts,
    )
    return handoff, supervisor, lifecycle


def test_missing_target_can_be_completed_on_the_next_owned_voice_turn() -> None:
    transport = _FakeTransport()
    _handoff, _supervisor, lifecycle = _service(
        transport,
        mission_service=MissionService(),
    )

    with pytest.raises(ValueError, match="task_target_required"):
        lifecycle.reserve_task(
            "导航到",
            "session-clarification",
            "turn-missing-target",
        )

    pending = lifecycle.pending_clarification("session-clarification")
    assert pending is not None
    assert pending.task_type == "navigate_to"
    assert pending.missing_parameter == "target"
    assert lifecycle.can_continue_pending_task(
        "北门",
        "session-clarification",
    )
    assert not lifecycle.can_continue_pending_task(
        "你会导航吗？",
        "session-clarification",
    )
    assert not lifecycle.can_continue_pending_task(
        "今天天气不错",
        "session-clarification",
    )

    reservation = lifecycle.continue_pending_task(
        "北门",
        "session-clarification",
        "turn-target-answer",
    )

    assert reservation.task_type == "navigate_to"
    assert reservation.target == "北门"
    assert reservation.state == "waiting_user"
    assert reservation.turn_id == "turn-target-answer"
    assert lifecycle.pending_clarification("session-clarification") is None
    assert transport.submit_calls == []


def test_pending_clarification_is_person_session_bound_and_expires() -> None:
    transport = _FakeTransport()
    _handoff, _supervisor, lifecycle = _service(
        transport,
        mission_service=MissionService(),
        clarification_ttl_s=0.01,
    )
    with pytest.raises(ValueError, match="task_target_required"):
        lifecycle.reserve_task("巡检", "session-owner-a", "turn-missing-area")

    other_person = VoiceTaskOperatorContext(
        operator_id=_TRUSTED_OPERATOR.operator_id,
        roles=_TRUSTED_OPERATOR.roles,
        authenticated=True,
        source="speaker_verification",
        person_id="operator-person-2",
        permissions=_TRUSTED_OPERATOR.permissions,
    )
    assert (
        lifecycle.pending_clarification(
            "session-owner-a",
            operator_context=other_person,
        )
        is None
    )
    assert lifecycle.pending_clarification("session-owner-b") is None

    time.sleep(0.02)
    assert lifecycle.pending_clarification("session-owner-a") is None


def test_pending_inspection_can_be_revised_before_confirmation() -> None:
    transport = _FakeTransport()
    handoff, _supervisor, lifecycle = _service(
        transport,
        mission_service=MissionService(),
    )
    original = lifecycle.reserve_task(
        "巡检A区并拍照",
        "session-revise-inspection",
        "turn-original-inspection",
    )

    assert lifecycle.can_revise_pending_task(
        "改成B区，拍两张",
        "session-revise-inspection",
    )
    revised = lifecycle.revise_pending_task(
        "改成B区，拍两张",
        "session-revise-inspection",
        "turn-revised-inspection",
    )

    assert handoff.run_service.require(original.run_id).current_state == "cancelled"
    assert revised.state == "waiting_user"
    assert revised.target == "B区"
    assert revised.revision == 2
    assert revised.supersedes_reservation_id == original.reservation_id
    assert "2张照片" in revised.confirmation_prompt
    assert revised.plan["mission"]["mission"]["metadata"]["task_parameters"] == {
        "capture_evidence": True,
        "photo_count": 2,
    }
    replayed = lifecycle.revise_pending_task(
        "改成B区，拍两张",
        "session-revise-inspection",
        "turn-revised-inspection",
    )
    assert replayed is revised
    assert handoff.run_service.require(revised.run_id).current_state == "waiting_user"
    with pytest.raises(RuntimeError, match="task_revision_turn_conflict"):
        lifecycle.revise_pending_task(
            "改成C区",
            "session-revise-inspection",
            "turn-revised-inspection",
        )
    assert handoff.run_service.require(revised.run_id).current_state == "waiting_user"
    assert transport.submit_calls == []


async def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _drain(lifecycle: VoiceTaskLifecycleService, thread_id: str) -> list:
    events = []
    while event := lifecycle.claim_next(thread_id):
        events.append(event)
        lifecycle.settle_delivery(event.event_id, "delivered")
    return events


@pytest.mark.asyncio
async def test_ack_barrier_is_deterministic_and_submits_exactly_once_off_loop() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    main_thread = threading.get_ident()
    try:
        first = lifecycle.reserve_status_report("汇报机器人状态", "session-a", "turn-1")
        second = lifecycle.reserve_status_report("different replay text", "session-a", "turn-1")

        assert first is second
        assert first.plan["intent"] == "status_report"
        assert first.plan["mission"]["mission"]["mission_type"] == "status_report"
        assert transport.submit_calls == []
        assert transport.status_calls == []
        assert transport.cancel_calls == []

        handle, replay = await asyncio.gather(
            lifecycle.commit_ack_and_submit(first.reservation_id),
            lifecycle.commit_ack_and_submit(first.reservation_id),
        )

        assert handle == replay
        assert handle.remote_task_id == "remote-status-1"
        assert handle.accepted is True
        assert handle.state == "queued"
        assert len(transport.submit_calls) == 1
        assert transport.submit_calls[0].thread_id == "session-a"
        assert transport.submit_calls[0].turn_id == "turn-1"
        assert transport.submit_calls[0].idempotency_key == first.reservation_id
        assert transport.call_threads[0] != main_thread
    finally:
        await lifecycle.close()
        await supervisor.close()


def test_reservation_replay_rejects_a_different_operator_or_person() -> None:
    transport = _FakeTransport()
    _handoff, _supervisor, lifecycle = _service(transport)
    reservation = lifecycle.reserve_status_report(
        "汇报机器人状态",
        "session-owner",
        "turn-owner",
    )
    different_person = VoiceTaskOperatorContext(
        operator_id=_TRUSTED_OPERATOR.operator_id,
        roles=_TRUSTED_OPERATOR.roles,
        authenticated=True,
        source=_TRUSTED_OPERATOR.source,
        person_id="operator-person-2",
        permissions=_TRUSTED_OPERATOR.permissions,
    )

    with pytest.raises(PermissionError, match="different operator"):
        lifecycle.reserve_status_report(
            "重放同一轮",
            "session-owner",
            "turn-owner",
            operator_context=different_person,
        )

    assert lifecycle.reserve_status_report(
        "合法重放",
        "session-owner",
        "turn-owner",
    ) is reservation


@pytest.mark.asyncio
async def test_submit_revalidates_reservation_owner() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    foreign_operator = VoiceTaskOperatorContext(
        operator_id="voice-device-foreign",
        roles=("operator",),
        authenticated=True,
        source="speaker_verification",
        person_id="operator-person-foreign",
        permissions=("runtime:submit",),
    )
    try:
        reservation = lifecycle.reserve_status_report(
            "汇报机器人状态",
            "session-submit-owner",
            "turn-submit-owner",
        )

        with pytest.raises(PermissionError, match="different operator"):
            await lifecycle.commit_ack_and_submit(
                reservation.reservation_id,
                operator_context=foreign_operator,
            )

        assert reservation.submit_attempted is False
        assert reservation.state == "reserved"
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_cancelled_submit_waits_for_persisted_outcome_before_propagating() -> None:
    submit_started = threading.Event()
    release_submit = threading.Event()

    class _BlockingTransport(_FakeTransport):
        def submit(self, request):
            submit_started.set()
            assert release_submit.wait(timeout=1.0)
            return super().submit(request)

    transport = _BlockingTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report(
            "汇报机器人状态",
            "session-cancel-submit",
            "turn-cancel-submit",
        )
        commit = asyncio.create_task(
            lifecycle.commit_ack_and_submit(reservation.reservation_id)
        )
        assert await asyncio.to_thread(submit_started.wait, 1.0)

        commit.cancel()
        await asyncio.sleep(0)
        assert not commit.done()
        release_submit.set()

        with pytest.raises(asyncio.CancelledError):
            await commit
        assert reservation.state != "submitting"
        replay = await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        assert replay.state == "queued"
        assert replay.remote_task_id == "remote-status-1"
        assert len(transport.submit_calls) == 1
    finally:
        release_submit.set()
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_unknown_submission_is_reconciled_with_the_same_idempotency_key() -> None:
    class _AmbiguousOnceTransport(_FakeTransport):
        def submit(self, request):
            self.call_threads.append(threading.get_ident())
            self.submit_calls.append(request)
            if len(self.submit_calls) == 1:
                raise AmbiguousRuntimeSubmissionError("request outcome unknown")
            return RuntimeExecutorSubmitResult(
                remote_task_id="remote-reconciled-1",
                status="queued",
                correlation_id=request.correlation_id,
                idempotency_key=request.idempotency_key,
                cursor="cursor-reconciled",
                observed_at=time.time(),
            )

    transport = _AmbiguousOnceTransport()
    handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report(
            "汇报状态", "session-unknown", "turn-unknown"
        )
        handle = await lifecycle.commit_ack_and_submit(reservation.reservation_id)

        assert handle.accepted is False
        assert handle.state == "submission_unknown"
        assert handle.remote_task_id == ""
        await _wait_for(
            lambda: handoff.run_service.require(handle.run_id).remote_task_id
            == "remote-reconciled-1"
        )
        assert len(transport.submit_calls) == 2
        assert {
            call.idempotency_key for call in transport.submit_calls
        } == {handle.idempotency_key}
        assert handoff.run_service.require(handle.run_id).current_state in {
            "queued",
            "executing",
        }
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_unknown_submission_recovers_from_store_and_reconciles_after_restart(
    tmp_path,
) -> None:
    class _AlwaysAmbiguousTransport(_FakeTransport):
        def submit(self, request):
            self.submit_calls.append(request)
            raise AmbiguousRuntimeSubmissionError("request outcome unknown")

    store_config = {
        "enabled": True,
        "path": str(tmp_path / "voice-task-runs.json"),
        "swallow_errors": False,
    }
    first_transport = _AlwaysAmbiguousTransport()
    first_handoff, first_supervisor, first_lifecycle = _service(
        first_transport,
        store_config=store_config,
    )
    await first_lifecycle.start()
    reservation = first_lifecycle.reserve_status_report(
        "汇报状态", "session-before-restart", "turn-before-restart"
    )
    handle = await first_lifecycle.commit_ack_and_submit(reservation.reservation_id)
    assert handle.state == "submission_unknown"
    assert first_handoff.run_service.require(handle.run_id).external_idempotency_key
    await first_lifecycle.close()
    await first_supervisor.close()

    recovered_transport = _FakeTransport()
    recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        await _wait_for(
            lambda: bool(recovered_handoff.run_service.require(handle.run_id).remote_task_id)
        )
        snapshot = recovered_lifecycle.status_snapshot("session-after-restart")
        assert snapshot.run_id == handle.run_id
        assert snapshot.active is True
        assert len(recovered_transport.submit_calls) == 1
        assert recovered_transport.submit_calls[0].idempotency_key == handle.idempotency_key
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_navigation_is_durably_confirmed_then_submitted_with_the_same_run(
    tmp_path,
) -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "navigation-task-runs.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(
            {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
        ),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_task(
            "请前往 A 区",
            "session-navigation",
            "turn-navigation-prompt",
        )
        prepared_run_id = reservation.run_id

        assert reservation.task_type == "navigate_to"
        assert reservation.target
        assert reservation.state == "waiting_user"
        assert prepared_run_id
        assert handoff.run_service.require(prepared_run_id).current_state == "waiting_user"
        assert transport.submit_calls == []
        with pytest.raises(PermissionError, match="scope mismatch"):
            lifecycle.confirm_pending(
                "session-navigation",
                "turn-navigation-prompt",
            )

        confirmed = lifecycle.confirm_pending(
            "session-navigation",
            "turn-navigation-confirm",
        )
        assert confirmed.run_id == prepared_run_id
        assert confirmed.state == "confirmed"
        assert transport.submit_calls == []

        handle = await lifecycle.commit_ack_and_submit(confirmed.reservation_id)
        assert handle.run_id == prepared_run_id
        assert handle.accepted is True
        assert len(transport.submit_calls) == 1
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_navigation_confirmation_rejects_changed_payload(tmp_path) -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "tampered-navigation.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(
            {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
        ),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_task(
            "导航到北门",
            "session-tampered-navigation",
            "turn-tampered-prompt",
        )
        mission = reservation.plan["mission"]["mission"]
        mission["target"] = "南门"

        with pytest.raises(RuntimeError, match="payload changed"):
            lifecycle.confirm_pending(
                "session-tampered-navigation",
                "turn-tampered-confirm",
            )

        assert handoff.run_service.require(reservation.run_id).current_state == "waiting_user"
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_navigation_confirmation_expires_without_submit(tmp_path) -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "expired-navigation.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(
            {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
        ),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_task(
            "导航到北门",
            "session-expired-navigation",
            "turn-expired-prompt",
        )
        run = handoff.run_service.require(reservation.run_id)
        run.approval_request["expires_at"] = time.time() - 1.0
        handoff.run_service.persist()

        with pytest.raises(TimeoutError, match="expired"):
            lifecycle.confirm_pending(
                "session-expired-navigation",
                "turn-expired-confirm",
            )

        assert run.current_state == "cancelled"
        assert run.approval_request["status"] == "expired"
        replacement = lifecycle.reserve_task(
            "生成状态报告",
            "session-expired-navigation",
            "turn-after-expiry",
        )
        assert replacement.state == "reserved"
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_waiting_navigation_recovers_and_cancels_locally_without_transport(
    tmp_path,
) -> None:
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "pending-navigation.json"),
        "swallow_errors": False,
    }
    mission_service = MissionService(
        {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
    )
    first_transport = _FakeTransport()
    first_handoff, first_supervisor, first_lifecycle = _service(
        first_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await first_lifecycle.start()
    reservation = first_lifecycle.reserve_task(
        "导航到北门",
        "session-pending-navigation",
        "turn-pending-navigation",
    )
    run_id = reservation.run_id
    await first_lifecycle.close()
    await first_supervisor.close()

    recovered_transport = _FakeTransport()
    recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        snapshot = recovered_lifecycle.status_snapshot("session-pending-navigation")
        assert snapshot.run_id == run_id
        assert snapshot.state == "waiting_user"
        cancelled = await recovered_lifecycle.cancel_active(
            "session-pending-navigation",
            reason="operator declined movement",
        )
        assert cancelled.error_code == "pending_task_cancelled"
        assert cancelled.snapshot.state == "cancelled"
        assert recovered_handoff.run_service.require(run_id).current_state == "cancelled"
        assert recovered_transport.submit_calls == []
        assert recovered_transport.cancel_calls == []
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_confirmed_navigation_recovers_before_submit_and_resumes_same_run(
    tmp_path,
) -> None:
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "confirmed-navigation.json"),
        "swallow_errors": False,
    }
    mission_service = MissionService(
        {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
    )
    first_transport = _FakeTransport()
    _first_handoff, first_supervisor, first_lifecycle = _service(
        first_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await first_lifecycle.start()
    reservation = first_lifecycle.reserve_task(
        "导航到北门",
        "session-confirmed-navigation",
        "turn-confirmed-prompt",
    )
    prepared_run = _first_handoff.run_service.require(reservation.run_id)
    prepared_handoff_id = prepared_run.handoff.handoff_id
    assert prepared_run.handoff.target_area == "北门"
    confirmed = first_lifecycle.confirm_pending(
        "session-confirmed-navigation",
        "turn-confirmed-response",
    )
    run_id = confirmed.run_id
    confirmed_run = _first_handoff.run_service.require(run_id)
    assert confirmed.state == "confirmed"
    assert confirmed_run.handoff.handoff_id == prepared_handoff_id
    assert confirmed_run.handoff.target_area == "北门"
    assert first_transport.submit_calls == []
    await first_lifecycle.close()
    await first_supervisor.close()

    recovered_transport = _FakeTransport()
    _recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        recovered = recovered_lifecycle.confirm_pending(
            "session-confirmed-navigation",
            "turn-confirmed-retry",
        )
        handle = await recovered_lifecycle.commit_ack_and_submit(
            recovered.reservation_id
        )
        assert handle.run_id == run_id
        recovered_run = _recovered_handoff.run_service.require(handle.run_id)
        assert handle.accepted is True, recovered_run.safety_assessments[-1].to_dict()
        assert recovered_run.handoff.handoff_id == prepared_handoff_id
        assert recovered_run.handoff.target_area == "北门"
        assert len(recovered_transport.submit_calls) == 1
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_waiting_navigation_recovers_then_confirms_and_submits_same_run(
    tmp_path,
) -> None:
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "waiting-navigation-confirm.json"),
        "swallow_errors": False,
    }
    mission_service = MissionService(
        {"runtime": {"mission": {"confirmation_threshold": "medium"}}}
    )
    first_transport = _FakeTransport()
    _first_handoff, first_supervisor, first_lifecycle = _service(
        first_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await first_lifecycle.start()
    reservation = first_lifecycle.reserve_task(
        "导航到北门",
        "session-waiting-confirm",
        "turn-waiting-prompt",
    )
    run_id = reservation.run_id
    await first_lifecycle.close()
    await first_supervisor.close()

    recovered_transport = _FakeTransport()
    recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        confirmed = recovered_lifecycle.confirm_pending(
            "session-waiting-confirm",
            "turn-waiting-confirm",
        )
        handle = await recovered_lifecycle.commit_ack_and_submit(
            confirmed.reservation_id
        )

        assert handle.run_id == run_id
        assert handle.accepted is True
        assert recovered_handoff.run_service.require(run_id).handoff.target_area == "北门"
        assert len(recovered_transport.submit_calls) == 1
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_physical_voice_tasks_always_require_confirmation(tmp_path) -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "forced-physical-confirmation.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(
            {"runtime": {"mission": {"confirmation_threshold": "critical"}}}
        ),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_task(
            "导航到北门",
            "session-forced-confirmation",
            "turn-forced-confirmation",
        )

        assert reservation.requires_confirmation is True
        assert reservation.state == "waiting_user"
        assert handoff.run_service.require(reservation.run_id).current_state == "waiting_user"
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_physical_voice_task_requires_trusted_speaker_identity(tmp_path) -> None:
    operator_without_speaker = VoiceTaskOperatorContext(
        operator_id="voice-device-1",
        roles=("operator",),
        authenticated=True,
        source="device_certificate",
        permissions=("runtime:read", "runtime:submit", "runtime:cancel"),
    )
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(
        transport,
        operator_context=operator_without_speaker,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "missing-speaker.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        with pytest.raises(PermissionError, match="speaker_identity_required"):
            lifecycle.reserve_task(
                "导航到北门",
                "session-missing-speaker",
                "turn-missing-speaker",
            )
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_pending_physical_confirmation_rejects_different_speaker(tmp_path) -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "different-speaker.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        lifecycle.reserve_task(
            "导航到北门",
            "session-different-speaker",
            "turn-speaker-prompt",
        )
        different_speaker = VoiceTaskOperatorContext(
            operator_id=_TRUSTED_OPERATOR.operator_id,
            roles=_TRUSTED_OPERATOR.roles,
            authenticated=True,
            source="speaker_verification",
            person_id="operator-person-2",
            permissions=_TRUSTED_OPERATOR.permissions,
        )

        with pytest.raises(PermissionError, match="different speaker"):
            lifecycle.confirm_pending(
                "session-different-speaker",
                "turn-speaker-confirm",
                operator_context=different_speaker,
            )
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_same_device_different_person_cannot_observe_cancel_or_claim_task() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    other_person = VoiceTaskOperatorContext(
        operator_id=_TRUSTED_OPERATOR.operator_id,
        roles=_TRUSTED_OPERATOR.roles,
        authenticated=True,
        source="speaker_verification",
        person_id="operator-person-2",
        permissions=_TRUSTED_OPERATOR.permissions,
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report(
            "汇报状态",
            "session-owned-task",
            "turn-owned-task",
            operator_context=_TRUSTED_OPERATOR,
        )
        await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(
            lambda: lifecycle.status_snapshot(
                "session-owned-task",
                operator_context=_TRUSTED_OPERATOR,
            ).state
            == "executing"
        )

        hidden = lifecycle.status_snapshot(
            "session-owned-task",
            operator_context=other_person,
        )
        denied_cancel = await lifecycle.cancel_active(
            "session-owned-task",
            operator_context=other_person,
        )

        assert hidden.state == "idle"
        assert hidden.reservation_id == ""
        assert denied_cancel.error_code == "no_active_external_task"
        assert lifecycle.claim_next(
            "session-owned-task",
            operator_context=other_person,
        ) is None
        assert transport.cancel_calls == []

        visible = lifecycle.status_snapshot(
            "rotated-owner-session",
            operator_context=_TRUSTED_OPERATOR,
        )
        assert visible.reservation_id == reservation.reservation_id
        assert lifecycle.claim_next(
            "rotated-owner-session",
            operator_context=_TRUSTED_OPERATOR,
        ) is not None
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_session_rejects_second_active_voice_task(tmp_path) -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(
        transport,
        store_config={
            "enabled": True,
            "path": str(tmp_path / "single-active-task.json"),
            "swallow_errors": False,
        },
        mission_service=MissionService(),
    )
    await supervisor.start()
    await lifecycle.start()
    try:
        lifecycle.reserve_task(
            "导航到北门",
            "session-single-active",
            "turn-first-task",
        )
        with pytest.raises(RuntimeError, match="voice_task_already_active"):
            lifecycle.reserve_task(
                "生成状态报告",
                "session-single-active",
                "turn-second-task",
            )
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


def test_operator_can_reserve_independent_tasks_in_distinct_sessions() -> None:
    transport = _FakeTransport()
    _handoff, _supervisor, lifecycle = _service(
        transport,
        mission_service=MissionService(),
    )

    first = lifecycle.reserve_task(
        "导航到北门",
        "session-independent-a",
        "turn-independent-a",
    )
    second = lifecycle.reserve_status_report(
        "生成状态报告",
        "session-independent-b",
        "turn-independent-b",
    )

    assert first.thread_id == "session-independent-a"
    assert second.thread_id == "session-independent-b"
    assert first.reservation_id != second.reservation_id


def test_physical_task_confirmation_names_target_and_risk() -> None:
    transport = _FakeTransport()
    _handoff, _supervisor, lifecycle = _service(
        transport,
        mission_service=MissionService(),
    )

    reservation = lifecycle.reserve_task(
        "导航到北门",
        "session-risk-prompt",
        "turn-risk-prompt",
    )

    assert "北门" in reservation.confirmation_prompt
    assert "高风险" in reservation.confirmation_prompt
    assert "实际移动" in reservation.confirmation_prompt


@pytest.mark.asyncio
async def test_recovery_keeps_newest_task_as_session_control_target(tmp_path) -> None:
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "newest-session-task.json"),
        "swallow_errors": False,
    }
    mission_service = MissionService()
    first_transport = _FakeTransport()
    first_handoff, first_supervisor, first_creator = _service(
        first_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    second_creator = VoiceTaskLifecycleService(
        handoff_service=first_handoff,
        supervisor=first_supervisor,
        mission_service=mission_service,
        operator_context=_TRUSTED_OPERATOR,
    )
    first_creator.reserve_task(
        "导航到北门",
        "session-legacy-multiple",
        "turn-old",
    )
    time.sleep(0.002)
    second_creator.reserve_task(
        "巡检 A 区",
        "session-legacy-multiple",
        "turn-new",
    )

    recovered_transport = _FakeTransport()
    _recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
        mission_service=mission_service,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        snapshot = recovered_lifecycle.status_snapshot("session-legacy-multiple")
        assert snapshot.turn_id == "turn-new"
        assert snapshot.state == "waiting_user"
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_queued_submission_crash_window_reconciles_after_restart(tmp_path) -> None:
    store_config = {
        "enabled": True,
        "path": str(tmp_path / "queued-crash-window.json"),
        "swallow_errors": False,
    }
    crashing_transport = _CrashAfterRemoteAcceptTransport()
    first_handoff, first_supervisor, first_lifecycle = _service(
        crashing_transport,
        store_config=store_config,
    )
    await first_lifecycle.start()
    reservation = first_lifecycle.reserve_status_report(
        "生成状态报告",
        "session-crash-window",
        "turn-crash-window",
    )
    with pytest.raises(SystemExit, match="simulated process crash"):
        await first_lifecycle.commit_ack_and_submit(reservation.reservation_id)
    crashed_run = first_handoff.run_service.runs()[0]
    run_id = crashed_run.run_id
    idempotency_key = crashed_run.external_idempotency_key
    assert crashed_run.current_state == "queued"
    assert idempotency_key
    assert crashed_run.remote_task_id is None
    await first_lifecycle.close()
    await first_supervisor.close()

    recovered_transport = _FakeTransport()
    recovered_handoff, recovered_supervisor, recovered_lifecycle = _service(
        recovered_transport,
        store_config=store_config,
    )
    await recovered_supervisor.start()
    await recovered_lifecycle.start()
    try:
        await _wait_for(
            lambda: bool(recovered_handoff.run_service.require(run_id).remote_task_id)
        )
        recovered_run = recovered_handoff.run_service.require(run_id)
        snapshot = recovered_lifecycle.status_snapshot("session-crash-window")

        assert recovered_run.remote_task_id == "remote-status-1"
        assert snapshot.run_id == run_id
        assert snapshot.state != "idle"
        assert {call.idempotency_key for call in recovered_transport.submit_calls} == {
            idempotency_key
        }
    finally:
        await recovered_lifecycle.close()
        await recovered_supervisor.close()


@pytest.mark.asyncio
async def test_abandon_keeps_ack_barrier_closed() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-a", "turn-abandon")
        assert lifecycle.abandon(reservation.reservation_id) is True
        assert transport.submit_calls == []
        with pytest.raises(RuntimeError, match="abandoned"):
            await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        assert transport.submit_calls == []
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_poller_projects_progress_completion_and_session_scoped_receipts() -> None:
    transport = _FakeTransport()
    transport.statuses.extend(
        [
            RuntimeExecutorStatusUpdate(
                remote_task_id="remote-status-1",
                status="executing",
                correlation_id="ignored",
                cursor="cursor-2",
                updates=(
                    RuntimeExecutorUpdate(
                        event_id="remote-progress-1",
                        status="executing",
                        message="Reading status panel.",
                        cursor="cursor-2",
                        observed_at=time.time(),
                    ),
                ),
                observed_at=time.time(),
            ),
            RuntimeExecutorStatusUpdate(
                remote_task_id="remote-status-1",
                status="completed",
                correlation_id="ignored",
                cursor="cursor-3",
                result_summary="Battery 90%; systems nominal.",
                observed_at=time.time(),
            ),
        ]
    )
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-a", "turn-2")
        await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(lambda: lifecycle.status_snapshot("session-a").state == "completed")

        snapshot = lifecycle.status_snapshot("session-a")
        assert snapshot.active is False
        assert snapshot.result_summary == "Battery 90%; systems nominal."
        assert lifecycle.claim_next("session-b") is None
        assert await lifecycle.wait_ready("session-a", timeout=0.1) is True

        events = _drain(lifecycle, "session-a")
        kinds = {event.kind for event in events}
        assert {"reserved", "started", "progress", "completed"} <= kinds
        assert events[-1].result_summary == "Battery 90%; systems nominal."
        assert all(
            lifecycle.delivery_receipt(event.event_id).state == "delivered" for event in events
        )
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_cancel_remains_requested_until_remote_poll_confirms_truth() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-c", "turn-3")
        await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(lambda: lifecycle.status_snapshot("session-c").state == "executing")

        result = await lifecycle.cancel_active("session-c", reason="user interrupted task")

        assert result.remote_acknowledged is True
        assert result.snapshot.state == "cancel_requested"
        assert result.snapshot.active is True
        assert len(transport.cancel_calls) == 1
        assert transport.cancel_calls[0].reason == "user interrupted task"

        transport.cancel_confirmed = True
        await _wait_for(lambda: lifecycle.status_snapshot("session-c").state == "cancelled")
        assert lifecycle.status_snapshot("session-c").active is False
        assert any(event.kind == "cancelled" for event in _drain(lifecycle, "session-c"))
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_cancel_transport_failure_does_not_claim_cancellation_was_requested() -> None:
    class _CancelFailTransport(_FakeTransport):
        def cancel(self, request):
            self.cancel_calls.append(request)
            raise RuntimeExecutorTransportError("network_error", "remote unavailable")

    transport = _CancelFailTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-c-fail", "turn-3b")
        await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(lambda: lifecycle.status_snapshot("session-c-fail").state == "executing")

        result = await lifecycle.cancel_active("session-c-fail")

        assert result.remote_acknowledged is False
        assert result.error_code == "network_error"
        assert result.snapshot.state == "executing"
        assert len(transport.cancel_calls) == 1
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_duplicate_and_out_of_order_remote_updates_do_not_duplicate_delivery() -> None:
    transport = _FakeTransport()
    duplicate = RuntimeExecutorUpdate(
        event_id="remote-dup",
        status="executing",
        cursor="cursor-2",
        observed_at=time.time(),
    )
    transport.statuses.append(
        RuntimeExecutorStatusUpdate(
            remote_task_id="remote-status-1",
            status="executing",
            correlation_id="ignored",
            cursor="cursor-2",
            updates=(duplicate, duplicate),
            observed_at=time.time(),
        )
    )
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-d", "turn-4")
        handle = await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(
            lambda: (
                "remote-dup"
                in _handoff.run_service.require(handle.run_id).processed_remote_update_ids
            )
        )
        rejected = _handoff.apply_external_update(
            handle.run_id,
            remote_task_id=handle.remote_task_id,
            remote_status="queued",
            update_id="remote-old",
            cursor="cursor-1",
        )
        await asyncio.sleep(0)

        assert rejected["handled"] is False
        assert rejected["reason"] == "remote_update_out_of_order"
        events = _drain(lifecycle, "session-d")
        assert sum(event.event_id == "remote-dup" for event in events) == 0
        assert sum(event.kind == "progress" and event.state == "executing" for event in events) == 1
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_missing_trusted_operator_context_fails_closed_before_ack_or_submit() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport, operator_context=None)
    await supervisor.start()
    await lifecycle.start()
    try:
        with pytest.raises(PermissionError, match="runtime:submit"):
            lifecycle.reserve_status_report("汇报状态", "session-auth", "turn-auth")
        assert transport.submit_calls == []
        with pytest.raises(PermissionError, match="runtime:read"):
            lifecycle.status_snapshot("session-auth")
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_cancel_requires_separate_trusted_permission() -> None:
    transport = _FakeTransport()
    submit_only = VoiceTaskOperatorContext(
        operator_id="voice-device-2",
        roles=("operator",),
        authenticated=True,
        source="device_certificate",
        permissions=("runtime:read", "runtime:submit"),
    )
    _handoff, supervisor, lifecycle = _service(transport, operator_context=submit_only)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-authz", "turn-authz")
        await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(lambda: lifecycle.status_snapshot("session-authz").state == "executing")

        result = await lifecycle.cancel_active("session-authz")

        assert result.remote_acknowledged is False
        assert result.error_code == "operator_not_authorized"
        assert transport.cancel_calls == []
        assert result.snapshot.state == "idle"
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_internal_poll_errors_are_not_spoken_and_event_state_stays_historical() -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report("汇报状态", "session-events", "turn-events")
        handle = await lifecycle.commit_ack_and_submit(reservation.reservation_id)
        await _wait_for(lambda: lifecycle.status_snapshot("session-events").state == "executing")
        before = _drain(lifecycle, "session-events")
        progress = next(event for event in before if event.kind == "progress")

        handoff.record_external_poll_error(handle.run_id, error_code="network_error")
        assert lifecycle.claim_next("session-events") is None

        handoff.apply_external_update(
            handle.run_id,
            remote_task_id=handle.remote_task_id,
            remote_status="completed",
            cursor="cursor-99",
            result_summary="status is nominal",
        )
        assert progress.state == "executing"
        terminal = lifecycle.claim_next("session-events")
        assert terminal is not None
        assert terminal.kind == "completed"
        assert terminal.state == "completed"
        assert terminal.remote_task_id == handle.remote_task_id
        assert terminal.correlation_id == handle.run_id
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_transient_delivery_failure_is_redelivered_with_bounded_attempts() -> None:
    transport = _FakeTransport()
    _handoff, supervisor, lifecycle = _service(transport)
    await supervisor.start()
    await lifecycle.start()
    try:
        reservation = lifecycle.reserve_status_report(
            "汇报状态", "session-delivery", "turn-delivery"
        )
        first = lifecycle.claim_next("session-delivery")
        assert first is not None and first.event_id.endswith(":reserved")
        assert lifecycle.retry_delivery(first.event_id, error_code="tts_unavailable") is True

        replay = lifecycle.claim_next("session-delivery")
        assert replay == first
        receipt = lifecycle.settle_delivery(replay.event_id, "delivered")
        assert receipt.attempt_count == 2
        assert receipt.last_error_code == "tts_unavailable"
        assert lifecycle.settle_delivery(replay.event_id, "delivered") is receipt
        assert reservation.reservation_id == replay.reservation_id
    finally:
        await lifecycle.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_delivery_retry_exhaustion_is_persisted_and_not_replayed() -> None:
    transport = _FakeTransport()
    handoff, supervisor, lifecycle = _service(
        transport,
        max_delivery_attempts=1,
    )
    await supervisor.start()
    await lifecycle.start()
    reservation = lifecycle.reserve_status_report(
        "汇报状态", "session-exhausted", "turn-exhausted"
    )
    handle = await lifecycle.commit_ack_and_submit(reservation.reservation_id)
    await _wait_for(
        lambda: lifecycle.status_snapshot("session-exhausted").state == "executing"
    )
    _drain(lifecycle, "session-exhausted")
    completed = handoff.apply_external_update(
        handle.run_id,
        remote_task_id=handle.remote_task_id,
        remote_status="completed",
        cursor="cursor-exhausted",
        result_summary="状态正常。",
    )
    terminal_event_id = completed["event"]["event_id"]
    terminal = lifecycle.claim_next("session-exhausted")
    assert terminal is not None and terminal.event_id == terminal_event_id
    assert lifecycle.retry_delivery(
        terminal.event_id,
        error_code="tts_unavailable",
    ) is False
    assert lifecycle.delivery_receipt(terminal.event_id).state == "interrupted"
    assert (
        handoff.notification_delivery_receipt(
            handle.run_id,
            event_id=terminal.event_id,
        )
        == "interrupted"
    )
    await lifecycle.close()

    restarted = VoiceTaskLifecycleService(
        handoff_service=handoff,
        supervisor=supervisor,
        operator_context=_TRUSTED_OPERATOR,
    )
    await restarted.start()
    try:
        assert restarted.status_snapshot("session-after-exhaustion").state == "completed"
        assert restarted.claim_next("session-after-exhaustion") is None
    finally:
        await restarted.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_terminal_notification_replays_after_voice_restart_exactly_until_settled() -> None:
    transport = _FakeTransport()
    handoff, supervisor, first = _service(transport)
    await supervisor.start()
    await first.start()
    reservation = first.reserve_status_report("汇报状态", "session-replay", "turn-replay")
    handle = await first.commit_ack_and_submit(reservation.reservation_id)
    await _wait_for(lambda: first.status_snapshot("session-replay").state == "executing")
    _drain(first, "session-replay")
    await first.close()

    completed = handoff.apply_external_update(
        handle.run_id,
        remote_task_id=handle.remote_task_id,
        remote_status="completed",
        cursor="cursor-terminal",
        result_summary="巡检状态正常。",
    )
    terminal_event_id = completed["event"]["event_id"]

    recovered = VoiceTaskLifecycleService(
        handoff_service=handoff,
        supervisor=supervisor,
        operator_context=_TRUSTED_OPERATOR,
    )
    await recovered.start()
    try:
        assert await recovered.wait_ready("session-after-restart", timeout=0.1) is True
        terminal = recovered.claim_next("session-after-restart")
        assert terminal is not None
        assert terminal.event_id == terminal_event_id
        assert terminal.kind == "completed"
        assert terminal.result_summary == "巡检状态正常。"
        assert terminal.thread_id == "session-after-restart"
        assert terminal.originating_thread_id == "session-replay"
        recovered.settle_delivery(terminal.event_id, "delivered")
        assert (
            handoff.notification_delivery_receipt(
                handle.run_id,
                event_id=terminal_event_id,
            )
            == "delivered"
        )
    finally:
        await recovered.close()

    restarted = VoiceTaskLifecycleService(
        handoff_service=handoff,
        supervisor=supervisor,
        operator_context=_TRUSTED_OPERATOR,
    )
    await restarted.start()
    try:
        assert restarted.status_snapshot("another-new-session").state == "completed"
        assert restarted.claim_next("another-new-session") is None
    finally:
        await restarted.close()
        await supervisor.close()


@pytest.mark.asyncio
async def test_start_recovers_existing_nonterminal_run_and_close_stops_single_poller() -> None:
    transport = _FakeTransport()
    handoff, first_supervisor, first = _service(transport)
    await first_supervisor.start()
    await first.start()
    reservation = first.reserve_status_report("汇报状态", "session-r", "turn-r")
    handle = await first.commit_ack_and_submit(reservation.reservation_id)
    await _wait_for(lambda: len(transport.status_calls) >= 1)
    await first.close()
    await first_supervisor.close()
    calls_after_first_close = len(transport.status_calls)

    recovered_supervisor = ExternalTaskSupervisor(
        handoff_service=handoff,
        transport=transport,
        poll_initial_s=0.01,
        poll_max_s=0.02,
        poll_deadline_s=1.0,
        poll_jitter_ratio=0.0,
    )
    recovered = VoiceTaskLifecycleService(
        handoff_service=handoff,
        supervisor=recovered_supervisor,
        operator_context=_TRUSTED_OPERATOR,
    )
    await recovered_supervisor.start()
    await recovered.start()
    try:
        snapshot = recovered.status_snapshot("session-r")
        assert snapshot.run_id == handle.run_id
        assert snapshot.active is True
        await _wait_for(lambda: len(transport.status_calls) > calls_after_first_close)
        assert recovered_supervisor.tracked_run_ids == (handle.run_id,)
    finally:
        await recovered.close()
        await recovered_supervisor.close()
    calls_after_close = len(transport.status_calls)
    await asyncio.sleep(0.04)
    assert len(transport.status_calls) == calls_after_close
