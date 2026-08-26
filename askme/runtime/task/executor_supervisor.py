"""Runtime-owned supervision for externally executed TaskRuns."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorCancelRequest,
    RuntimeExecutorStatusRequest,
    RuntimeExecutorStatusUpdate,
    RuntimeExecutorSubmitRequest,
    RuntimeExecutorSubmitResult,
    RuntimeExecutorTransport,
    RuntimeExecutorTransportError,
)
from askme.runtime.task.handoff import RuntimeEvent, RuntimeHandoffService, TaskRun

logger = logging.getLogger(__name__)

_EXTERNAL_PROFILES = frozenset({"external", "lab"})
_CANCEL_PENDING_STATUSES = frozenset({"cancelling"})
_CANCEL_CONFIRMED_STATUSES = frozenset({"cancelled"})
_REMOTE_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "rejected", "blocked", "cancelled", "shadowed"}
)


@dataclass(frozen=True)
class ExternalCancelOutcome:
    """Truthful result of asking an external executor to cancel one run."""

    remote_acknowledged: bool
    state: str
    error_code: str
    run: dict[str, Any]

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.run

    def to_dict(self) -> dict[str, Any]:
        return {
            "handled": self.remote_acknowledged,
            "remote_acknowledged": self.remote_acknowledged,
            "state": self.state,
            "error_code": self.error_code,
            "run": dict(self.run),
        }


class ExternalTaskSupervisor:
    """Sole post-submit owner of external status, reconciliation, and cancel."""

    def __init__(
        self,
        *,
        handoff_service: RuntimeHandoffService,
        transport: RuntimeExecutorTransport,
        poll_initial_s: float = 0.25,
        poll_max_s: float = 4.0,
        poll_deadline_s: float = 300.0,
        poll_jitter_ratio: float = 0.1,
        random_source: Any = random.random,
    ) -> None:
        self._handoff = handoff_service
        self._transport = transport
        self._poll_initial_s = max(0.01, float(poll_initial_s))
        self._poll_max_s = max(self._poll_initial_s, float(poll_max_s))
        self._poll_deadline_s = max(self._poll_initial_s, float(poll_deadline_s))
        self._poll_jitter_ratio = min(max(float(poll_jitter_ratio), 0.0), 1.0)
        self._random = random_source
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe: Any = None
        self._pollers: dict[str, asyncio.Task[None]] = {}
        self._poller_lock: asyncio.Lock | None = None
        self._deferred_cancel_refresh_due: set[str] = set()
        self._closing = False

    @property
    def transport(self) -> RuntimeExecutorTransport:
        """Expose transport identity for diagnostics, never for shared ownership."""
        return self._transport

    @property
    def tracked_run_ids(self) -> tuple[str, ...]:
        return tuple(sorted(run_id for run_id, task in self._pollers.items() if not task.done()))

    async def start(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._poller_lock = asyncio.Lock()
        self._closing = False
        self._unsubscribe = self._handoff.subscribe_events(self._observe_runtime_event)
        for run in self._handoff.run_service.runs():
            if self._should_track(run):
                await self.ensure_tracked(run.run_id)

    async def close(self) -> None:
        self._closing = True
        unsubscribe = self._unsubscribe
        self._unsubscribe = None
        if callable(unsubscribe):
            unsubscribe()
        tasks = list(self._pollers.values())
        self._pollers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._deferred_cancel_refresh_due.clear()
        self._poller_lock = None
        self._loop = None

    async def ensure_tracked(self, run_id: str) -> None:
        """Ensure exactly one off-loop status poller owns a persisted external run."""
        if self._closing or self._loop is None or self._poller_lock is None:
            return
        run = self._handoff.run_service.get(run_id)
        if run is None or not self._should_track(run):
            return
        async with self._poller_lock:
            current = self._pollers.get(run.run_id)
            if current is not None and not current.done():
                return
            task = self._loop.create_task(
                self._poll_run(run.run_id), name=f"external-task-poll:{run.run_id}"
            )
            self._pollers[run.run_id] = task
            task.add_done_callback(partial(self._poller_done, run.run_id))

    async def refresh(self, run_id: str) -> dict[str, Any]:
        """Fetch and project one authoritative external status observation."""
        run = self._handoff.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        if run.profile not in _EXTERNAL_PROFILES:
            return self._failure(run, "external_profile_required")
        if run.terminal:
            return {"handled": True, "run": run.to_dict()}
        if not run.remote_task_id:
            return self._failure(run, "remote_task_id_required")
        try:
            status = await asyncio.to_thread(
                self._transport.get_status,
                RuntimeExecutorStatusRequest(
                    remote_task_id=run.remote_task_id,
                    correlation_id=run.run_id,
                    cursor=run.remote_status_cursor,
                ),
            )
        except RuntimeExecutorTransportError as exc:
            self._handoff.record_external_poll_error(run.run_id, error_code=exc.kind)
            return self._failure(run, exc.kind)
        except Exception:
            logger.exception("External TaskRun poll failed for %s", run.run_id)
            self._handoff.record_external_poll_error(run.run_id, error_code="external_poll_failed")
            return self._failure(run, "external_poll_failed")

        self._project_status(run.run_id, status)
        current = self._handoff.run_service.require(run.run_id)
        if current.terminal and current.deferred_cancel_request:
            self._handoff.run_service.clear_deferred_cancel_request(current.run_id)
            self._deferred_cancel_refresh_due.discard(current.run_id)
            current = self._handoff.run_service.require(run.run_id)
        return {"handled": True, "run": current.to_dict()}

    async def request_cancel(
        self,
        run_id: str,
        *,
        operator_id: str,
        reason: str,
        operator_context: dict[str, Any] | None,
        risk_acknowledgement: bool = False,
    ) -> ExternalCancelOutcome:
        """Request remote cancellation without inventing local cancellation truth."""
        run = self._handoff.run_service.get(run_id)
        if run is None:
            return ExternalCancelOutcome(False, "missing", "run_not_found", {})
        if run.profile not in _EXTERNAL_PROFILES:
            return self._cancel_failure(run, "external_profile_required")
        if run.terminal:
            return self._cancel_failure(run, "run_already_terminal")
        operator_error = _operator_context_error(operator_id, operator_context)
        if operator_error:
            return self._cancel_failure(run, operator_error)
        assert isinstance(operator_context, dict)
        if _needs_submission_reconciliation(run):
            self._handoff.run_service.set_deferred_cancel_request(
                run.run_id,
                {
                    "operator_id": str(operator_id),
                    "reason": str(reason or ""),
                    "risk_acknowledgement": bool(risk_acknowledgement),
                    "operator_context": _stored_operator_context(operator_context),
                    "requested_at": time.time(),
                },
            )
            await self.ensure_tracked(run.run_id)
            return self._cancel_failure(run, "cancel_deferred_until_reconciled")
        if not run.remote_task_id:
            return self._cancel_failure(run, "remote_task_id_required")

        try:
            result = await asyncio.to_thread(
                self._transport.cancel,
                RuntimeExecutorCancelRequest(
                    remote_task_id=run.remote_task_id,
                    idempotency_key=f"cancel:{run.run_id}",
                    correlation_id=run.run_id,
                    reason=reason,
                ),
            )
        except RuntimeExecutorTransportError as exc:
            self._handoff.record_external_poll_error(run.run_id, error_code=exc.kind)
            return self._cancel_failure(run, exc.kind)
        except Exception:
            logger.exception("External TaskRun cancel failed for %s", run.run_id)
            self._handoff.record_external_poll_error(
                run.run_id, error_code="external_cancel_failed"
            )
            return self._cancel_failure(run, "external_cancel_failed")

        normalized_status = str(result.status or "").strip().lower()
        self._project_updates(
            run.run_id,
            remote_task_id=result.remote_task_id,
            updates=result.updates,
            terminal_summary=result.result_summary,
        )
        current = self._handoff.run_service.require(run.run_id)
        if normalized_status in _CANCEL_PENDING_STATUSES and not current.terminal:
            self._handoff.run_service.request_external_cancel(
                current.run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        self._project_top_level(
            run.run_id,
            remote_task_id=result.remote_task_id,
            status=result.status,
            cursor=result.cursor,
            result_summary=result.result_summary,
            observed_at=result.observed_at,
        )
        current = self._handoff.run_service.require(run.run_id)
        acknowledged = (
            normalized_status in (_CANCEL_PENDING_STATUSES | _CANCEL_CONFIRMED_STATUSES)
            and current.current_state != "completed"
        )
        if normalized_status in (_CANCEL_PENDING_STATUSES | _CANCEL_CONFIRMED_STATUSES):
            self._handoff.run_service.clear_deferred_cancel_request(current.run_id)
            self._deferred_cancel_refresh_due.discard(current.run_id)
            current = self._handoff.run_service.require(run.run_id)
        error_code = ""
        if not acknowledged:
            if current.current_state == "completed" or normalized_status == "completed":
                error_code = "run_already_completed"
            elif normalized_status in {"rejected", "failed", "blocked"}:
                error_code = "cancel_rejected"
            else:
                error_code = "cancel_not_accepted"
        if not current.terminal:
            await self.ensure_tracked(current.run_id)
        return ExternalCancelOutcome(
            acknowledged,
            current.current_state,
            error_code,
            current.to_dict(),
        )

    def _observe_runtime_event(self, event: RuntimeEvent) -> None:
        if event.event_type != "external_submission_committed" or self._closing:
            return
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._schedule_tracking, event.run_id)

    def _schedule_tracking(self, run_id: str) -> None:
        if self._closing or self._loop is None:
            return
        self._loop.create_task(self.ensure_tracked(run_id))

    async def _poll_run(self, run_id: str) -> None:
        deadline = time.monotonic() + self._poll_deadline_s
        delay = self._poll_initial_s
        while not self._closing:
            run = self._handoff.run_service.get(run_id)
            if run is None or not self._should_track(run):
                return
            if time.monotonic() >= deadline:
                self._handoff.record_external_poll_error(
                    run_id, error_code="poll_deadline_exceeded"
                )
                deadline = time.monotonic() + self._poll_deadline_s
                delay = self._poll_max_s
                await asyncio.sleep(delay)
                continue
            if _needs_submission_reconciliation(run):
                result = await self._reconcile_submission(run_id)
            elif run_id in self._deferred_cancel_refresh_due:
                self._deferred_cancel_refresh_due.discard(run_id)
                result = await self.refresh(run_id)
            elif run.deferred_cancel_request and run.remote_task_id:
                result = await self._attempt_deferred_cancel(run_id)
            else:
                result = await self.refresh(run_id)
            current = self._handoff.run_service.get(run_id)
            if current is None or current.terminal:
                return
            if result.get("handled", False) and not current.deferred_cancel_request:
                delay = self._poll_initial_s
            else:
                delay = min(self._poll_max_s, delay * 2.0)
            jitter = delay * self._poll_jitter_ratio * float(self._random())
            await asyncio.sleep(min(self._poll_max_s, delay + jitter))

    async def _reconcile_submission(self, run_id: str) -> dict[str, Any]:
        """Safely replay one unknown submission with its persisted idempotency key."""

        run = self._handoff.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error_code": "run_not_found", "run_id": run_id}
        idempotency_key = str(run.external_idempotency_key or "").strip()
        if not idempotency_key:
            return self._failure(run, "external_idempotency_key_required")
        voice_context = run.handoff.source_plan.get("voice_context", {})
        if not isinstance(voice_context, dict):
            voice_context = {}
        request = RuntimeExecutorSubmitRequest(
            handoff=run.handoff.to_dict(),
            idempotency_key=idempotency_key,
            correlation_id=run.run_id,
            thread_id=str(
                voice_context.get("thread_id")
                or voice_context.get("conversation_session_id")
                or run.handoff.session_id
            ),
            turn_id=str(
                voice_context.get("turn_id") or voice_context.get("originating_turn_id") or ""
            ),
        )
        try:
            remote = await asyncio.to_thread(self._transport.submit, request)
        except AmbiguousRuntimeSubmissionError:
            self._handoff.record_external_poll_error(
                run.run_id, error_code="external_submission_unknown"
            )
            return self._failure(run, "external_submission_unknown")
        except RuntimeExecutorTransportError as exc:
            self._handoff.run_service.transition(
                run,
                "failed",
                "external_submission_failed",
                "External submission reconciliation failed definitively.",
                {"error_code": exc.kind},
            )
            return self._failure(run, exc.kind)
        except Exception:
            logger.exception("External submission reconciliation failed for %s", run.run_id)
            self._handoff.record_external_poll_error(
                run.run_id, error_code="external_submission_reconcile_failed"
            )
            return self._failure(run, "external_submission_reconcile_failed")
        if (
            not isinstance(remote, RuntimeExecutorSubmitResult)
            or not str(remote.remote_task_id or "").strip()
            or remote.correlation_id != run.run_id
            or remote.idempotency_key != idempotency_key
        ):
            self._handoff.record_external_poll_error(
                run.run_id, error_code="external_submission_unknown"
            )
            return self._failure(run, "external_submission_unknown")
        bound = self._handoff.run_service.bind_external_submission(
            run.run_id,
            remote_task_id=remote.remote_task_id,
            external_idempotency_key=idempotency_key,
            remote_status="submitted",
            observed_at=remote.observed_at,
        )
        if not bound.get("handled", False):
            return self._failure(run, str(bound.get("reason") or "external_bind_failed"))
        self._project_updates(
            run.run_id,
            remote_task_id=remote.remote_task_id,
            updates=remote.updates,
            terminal_summary=remote.result_summary,
        )
        self._project_top_level(
            run.run_id,
            remote_task_id=remote.remote_task_id,
            status=remote.status,
            cursor=remote.cursor,
            result_summary=remote.result_summary,
            observed_at=remote.observed_at,
        )
        current = self._handoff.run_service.require(run.run_id)
        self._handoff.run_service.emit(
            current,
            "external_submission_reconciled",
            current.current_state,
            "External submission identity reconciled.",
            {
                "remote_task_id": current.remote_task_id,
                "external_idempotency_key": current.external_idempotency_key,
            },
        )
        if current.deferred_cancel_request and current.remote_task_id:
            return await self._attempt_deferred_cancel(current.run_id)
        return {"handled": True, "run": current.to_dict()}

    async def _attempt_deferred_cancel(self, run_id: str) -> dict[str, Any]:
        run = self._handoff.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error_code": "run_not_found", "run_id": run_id}
        request = dict(run.deferred_cancel_request)
        if not request:
            return {"handled": True, "run": run.to_dict()}
        operator_context = request.get("operator_context")
        outcome = await self.request_cancel(
            run_id,
            operator_id=str(request.get("operator_id") or ""),
            reason=str(request.get("reason") or ""),
            risk_acknowledgement=bool(request.get("risk_acknowledgement", False)),
            operator_context=(operator_context if isinstance(operator_context, dict) else None),
        )
        current = self._handoff.run_service.get(run_id)
        if current is not None and current.deferred_cancel_request and not current.terminal:
            # The next poll must consume authoritative status before another
            # cancel attempt.  This prevents an unaccepted cancel response from
            # starving completion/failure updates indefinitely.
            self._deferred_cancel_refresh_due.add(run_id)
        else:
            self._deferred_cancel_refresh_due.discard(run_id)
        return outcome.to_dict()

    def _poller_done(self, run_id: str, task: asyncio.Task[None]) -> None:
        if self._pollers.get(run_id) is task:
            self._pollers.pop(run_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "External TaskRun poller crashed for %s",
                run_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _project_status(self, run_id: str, status: RuntimeExecutorStatusUpdate) -> None:
        self._project_updates(
            run_id,
            remote_task_id=status.remote_task_id,
            updates=status.updates,
            terminal_summary=status.result_summary,
        )
        self._project_top_level(
            run_id,
            remote_task_id=status.remote_task_id,
            status=status.status,
            cursor=status.cursor,
            result_summary=status.result_summary,
            observed_at=status.observed_at,
        )

    def _project_updates(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        updates: tuple[Any, ...],
        terminal_summary: str,
    ) -> None:
        for update in updates:
            current = self._handoff.run_service.require(run_id)
            if current.terminal:
                return
            normalized_status = str(update.status or "").strip().lower()
            self._handoff.apply_external_update(
                run_id,
                remote_task_id=remote_task_id,
                remote_status=normalized_status,
                update_id=update.event_id,
                cursor=update.cursor,
                payload=dict(update.payload),
                result_summary=(
                    terminal_summary if normalized_status in _REMOTE_TERMINAL_STATUSES else ""
                ),
                observed_at=update.observed_at,
            )

    def _project_top_level(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        status: str,
        cursor: str,
        result_summary: str,
        observed_at: float | None,
    ) -> None:
        run = self._handoff.run_service.require(run_id)
        if run.terminal:
            return
        normalized_status = str(status or "").strip().lower()
        normalized_cursor = str(cursor or "").strip()
        unchanged = (
            normalized_status == run.remote_status
            and normalized_cursor == run.remote_status_cursor
            and not result_summary
        )
        if unchanged:
            return
        self._handoff.apply_external_update(
            run_id,
            remote_task_id=remote_task_id,
            remote_status=normalized_status,
            cursor=normalized_cursor,
            result_summary=result_summary,
            observed_at=observed_at,
        )

    @staticmethod
    def _should_track(run: TaskRun) -> bool:
        return bool(
            run.profile in _EXTERNAL_PROFILES
            and not run.terminal
            and (
                run.remote_task_id
                or _needs_submission_reconciliation(run)
            )
        )

    @staticmethod
    def _failure(run: TaskRun, error_code: str) -> dict[str, Any]:
        return {
            "handled": False,
            "error_code": error_code,
            "run": run.to_dict(),
        }
    @staticmethod
    def _cancel_failure(run: TaskRun, error_code: str) -> ExternalCancelOutcome:
        return ExternalCancelOutcome(False, run.current_state, error_code, run.to_dict())


def _needs_submission_reconciliation(run: TaskRun) -> bool:
    return bool(
        not run.remote_task_id
        and run.external_idempotency_key
        and run.current_state in {"queued", "submission_unknown"}
    )


def _operator_context_error(
    operator_id: str,
    operator_context: dict[str, Any] | None,
) -> str:
    if not isinstance(operator_context, dict):
        return "runtime_operator_context_required"
    if operator_context.get("authenticated") is not True:
        return "runtime_operator_authentication_required"
    context_operator_id = str(operator_context.get("operator_id") or "").strip()
    if not context_operator_id or context_operator_id != str(operator_id or "").strip():
        return "runtime_operator_context_mismatch"
    roles = operator_context.get("roles")
    normalized_roles = {
        str(role).strip().lower() for role in roles or [] if str(role).strip()
    }
    if not isinstance(roles, list) or not normalized_roles.intersection({"operator", "admin"}):
        return "runtime_operator_context_incomplete"
    if not str(operator_context.get("source") or "").strip():
        return "runtime_operator_context_incomplete"
    if str(operator_context.get("permission") or "").strip() != "runtime:cancel":
        return "runtime_control_permission_mismatch"
    return ""


def _stored_operator_context(operator_context: dict[str, Any]) -> dict[str, Any]:
    """Persist only the authenticated cancellation provenance needed for replay."""

    return {
        "operator_id": str(operator_context.get("operator_id") or ""),
        "roles": [str(role) for role in operator_context.get("roles", [])],
        "authenticated": operator_context.get("authenticated") is True,
        "source": str(operator_context.get("source") or ""),
        "permission": str(operator_context.get("permission") or ""),
        **(
            {
                "thread_id": str(
                    operator_context.get("thread_id")
                    or operator_context.get("conversation_session_id")
                    or ""
                )
            }
            if operator_context.get("thread_id")
            or operator_context.get("conversation_session_id")
            else {}
        ),
    }
