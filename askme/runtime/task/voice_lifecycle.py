"""Voice-facing lifecycle for bounded external runtime tasks.

The service deliberately owns no audio output.  It bridges an acknowledged
voice turn to the existing runtime handoff, projects remote task updates, and
offers a thread-scoped delivery inbox for a separate conversation layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from askme.conversation import ApprovalScope, ConfirmationKind, InteractionTurnContext
from askme.runtime.task.executor_supervisor import ExternalTaskSupervisor
from askme.runtime.task.handoff import RuntimeEvent, RuntimeHandoffService, TaskRun

TaskEventKind = Literal["reserved", "started", "progress", "completed", "failed", "cancelled"]
DeliveryState = Literal[
    "pending", "delivering", "delivered", "interrupted", "suppressed", "expired"
]

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "blocked", "shadowed"})
_DELIVERY_FINAL_STATES = frozenset({"delivered", "interrupted", "suppressed", "expired"})
_VOICE_OPERATOR_ROLES = frozenset({"operator", "admin"})


@dataclass(frozen=True)
class VoiceTaskOperatorContext:
    """Deployment-trusted identity used for voice-originated runtime actions.

    This context is injected by the runtime composition root.  It must never be
    populated from recognized speech or model output.
    """

    operator_id: str
    roles: tuple[str, ...]
    authenticated: bool
    source: str
    person_id: str = ""
    permissions: tuple[str, ...] = ("runtime:read", "runtime:submit", "runtime:cancel")

    @classmethod
    def from_mapping(cls, payload: Any) -> VoiceTaskOperatorContext | None:
        """Normalize identity produced by a trusted per-turn verifier adapter."""

        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return None
        raw_roles = payload.get("roles", ())
        if isinstance(raw_roles, str):
            raw_roles = (raw_roles,)
        if not isinstance(raw_roles, (list, tuple, set)):
            raw_roles = ()
        raw_permissions = payload.get("permissions", ())
        if isinstance(raw_permissions, str):
            raw_permissions = (raw_permissions,)
        if not isinstance(raw_permissions, (list, tuple, set)):
            raw_permissions = ()
        return cls(
            operator_id=str(payload.get("operator_id") or "").strip(),
            roles=tuple(
                dict.fromkeys(
                    str(role).strip().lower() for role in raw_roles if str(role).strip()
                )
            ),
            authenticated=payload.get("authenticated") is True,
            source=str(payload.get("source") or "").strip(),
            person_id=str(payload.get("person_id") or "").strip(),
            permissions=tuple(
                dict.fromkeys(
                    str(permission).strip().lower()
                    for permission in raw_permissions
                    if str(permission).strip()
                )
            ),
        )

    def allows(self, permission: str) -> bool:
        roles = {role.strip().lower() for role in self.roles if role}
        permissions = {value.strip().lower() for value in self.permissions if value}
        return bool(
            self.authenticated
            and self.operator_id.strip()
            and self.source.strip()
            and roles.intersection(_VOICE_OPERATOR_ROLES)
            and permission.strip().lower() in permissions
        )

    def to_runtime_context(self, *, permission: str, thread_id: str) -> dict[str, Any]:
        context = {
            "operator_id": self.operator_id,
            "roles": list(self.roles),
            "authenticated": self.authenticated,
            "source": self.source,
            "permission": permission,
            "permissions": list(self.permissions),
            "thread_id": thread_id,
        }
        if self.person_id.strip():
            context["person_id"] = self.person_id.strip()
        return context


@dataclass
class TaskReservation:
    reservation_id: str
    thread_id: str
    turn_id: str
    user_text: str
    operator_id: str
    person_id: str
    operator_roles: tuple[str, ...]
    operator_authenticated: bool
    operator_source: str
    runtime_permission: str
    plan: dict[str, Any]
    task_type: str = "status_report"
    target: str = ""
    requires_confirmation: bool = False
    confirmation_prompt: str = ""
    approval_id: str = ""
    approval_payload_digest: str = ""
    approval_expires_at: float = 0.0
    state: str = "reserved"
    run_id: str = ""
    remote_task_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    submit_attempted: bool = False
    recovered: bool = False
    revision: int = 1
    supersedes_reservation_id: str = ""
    revision_request_digest: str = ""


@dataclass(frozen=True)
class PendingTaskClarification:
    """Owner-bound parameter collection state before a TaskRun exists."""

    clarification_id: str
    thread_id: str
    operator_id: str
    person_id: str
    turn_id: str
    task_type: str
    original_text: str
    missing_parameter: str
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class TaskHandle:
    reservation_id: str
    run_id: str
    remote_task_id: str
    correlation_id: str
    idempotency_key: str
    thread_id: str
    turn_id: str
    state: str
    accepted: bool


@dataclass(frozen=True)
class TaskLifecycleEvent:
    event_id: str
    reservation_id: str
    run_id: str
    thread_id: str
    turn_id: str
    kind: TaskEventKind
    state: str
    message: str
    remote_task_id: str = ""
    correlation_id: str = ""
    originating_thread_id: str = ""
    result_summary: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class DeliveryReceipt:
    event_id: str
    thread_id: str
    state: DeliveryState = "pending"
    claimed_at: float | None = None
    settled_at: float | None = None
    attempt_count: int = 0
    next_attempt_at: float = 0.0
    last_error_code: str = ""


@dataclass(frozen=True)
class TaskStatusSnapshot:
    thread_id: str
    reservation_id: str = ""
    run_id: str = ""
    remote_task_id: str = ""
    turn_id: str = ""
    state: str = "idle"
    result_summary: str = ""
    active: bool = False
    updated_at: float = 0.0


@dataclass(frozen=True)
class CancelRequestResult:
    remote_acknowledged: bool
    snapshot: TaskStatusSnapshot
    error_code: str = ""


class VoiceTaskLifecycleService:
    """ACK-gated voice projection over the runtime-owned task supervisor."""

    def __init__(
        self,
        *,
        handoff_service: RuntimeHandoffService,
        supervisor: ExternalTaskSupervisor,
        mission_service: Any | None = None,
        operator_context: VoiceTaskOperatorContext | None = None,
        approval_ttl_s: float = 60.0,
        clarification_ttl_s: float = 45.0,
        delivery_ttl_s: float = 120.0,
        delivery_retry_delay_s: float = 0.25,
        max_delivery_attempts: int = 3,
    ) -> None:
        self._handoff = handoff_service
        self._supervisor = supervisor
        self._mission_service = mission_service
        self._operator_context = operator_context
        self._approval_ttl_s = max(5.0, float(approval_ttl_s))
        self._clarification_ttl_s = max(0.01, float(clarification_ttl_s))
        self._delivery_ttl_s = max(0.01, float(delivery_ttl_s))
        self._delivery_retry_delay_s = max(0.0, float(delivery_retry_delay_s))
        self._max_delivery_attempts = max(1, int(max_delivery_attempts))
        self._reservations: dict[str, TaskReservation] = {}
        self._pending_clarifications: dict[
            tuple[str, str, str], PendingTaskClarification
        ] = {}
        self._reservation_keys: dict[tuple[str, str], str] = {}
        self._reservation_by_run: dict[str, str] = {}
        self._handles: dict[str, TaskHandle] = {}
        self._latest_by_thread: dict[str, str] = {}
        self._events: dict[str, TaskLifecycleEvent] = {}
        self._receipts: dict[str, DeliveryReceipt] = {}
        self._event_order: dict[str, list[str]] = defaultdict(list)
        self._last_announced_signature: dict[str, tuple[str, str, str]] = {}
        self._replayable_event_ids: set[str] = set()
        self._seen_runtime_event_ids: set[str] = set()
        self._waiters: dict[str, list[asyncio.Event]] = defaultdict(list)
        self._commit_lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe: Any = None
        self._closing = False
        self._lock = threading.RLock()

    @property
    def default_operator_context(self) -> VoiceTaskOperatorContext | None:
        """Deployment fallback; VoiceLoop actions should pass a per-turn context."""

        return self._operator_context

    async def start(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._commit_lock = asyncio.Lock()
        self._closing = False
        self._unsubscribe = self._handoff.subscribe_events(self.observe_runtime_event)
        for run in self._handoff.run_service.runs():
            if (
                run.current_state == "waiting_user"
                and float(run.approval_request.get("expires_at") or 0.0) <= time.time()
            ):
                self._handoff.expire_prepared_run(run.run_id)
                self._recover_run(run)
                continue
            recoverable_submission = bool(
                run.remote_task_id
                or _needs_submission_reconciliation(run)
                or run.current_state in {"waiting_user", "confirmed"}
            )
            if run.profile in {"external", "lab"} and recoverable_submission:
                recovered = self._recover_run(run)
                if recovered and run.terminal:
                    self._replay_terminal_notification(run)

    async def close(self) -> None:
        self._closing = True
        unsubscribe = self._unsubscribe
        self._unsubscribe = None
        if callable(unsubscribe):
            unsubscribe()
        with self._lock:
            waiters = [waiter for values in self._waiters.values() for waiter in values]
            self._waiters.clear()
        for waiter in waiters:
            waiter.set()
        self._loop = None

    def reserve_status_report(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        return self._reserve_task(
            user_text,
            thread_id,
            turn_id,
            forced_task_type="status_report",
            operator_context=operator_context,
        )

    def reserve_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        """Draft one supported runtime task without bypassing confirmation."""

        return self._reserve_task(
            user_text,
            thread_id,
            turn_id,
            operator_context=operator_context,
        )

    def pending_clarification(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> PendingTaskClarification | None:
        """Return the unexpired parameter request owned by this principal/thread."""

        session = _required(thread_id, "thread_id")
        operator = self._require_operator("runtime:submit", operator_context)
        key = self._clarification_key(session, operator)
        with self._lock:
            pending = self._pending_clarifications.get(key)
            if pending is None:
                return None
            if pending.expires_at <= time.time():
                self._pending_clarifications.pop(key, None)
                return None
            return pending

    def can_continue_pending_task(
        self,
        user_text: str,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> bool:
        """Return whether this turn can safely fill the pending target slot."""

        if not _plausible_task_target_reply(user_text):
            return False
        return (
            self.pending_clarification(
                thread_id,
                operator_context=operator_context,
            )
            is not None
        )

    def continue_pending_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        """Complete a missing target without trusting another thread or speaker."""

        target = _task_target_reply(user_text)
        if not _plausible_task_target_reply(target):
            raise ValueError("task_target_reply_invalid")
        session = _required(thread_id, "thread_id")
        turn = _required(turn_id, "turn_id")
        operator = self._require_operator("runtime:submit", operator_context)
        key = self._clarification_key(session, operator)
        with self._lock:
            pending = self._pending_clarifications.get(key)
            if pending is None or pending.expires_at <= time.time():
                self._pending_clarifications.pop(key, None)
                raise LookupError("no_pending_task_clarification")
            combined_text = f"{pending.original_text.rstrip('，,。.!！?？')} {target}".strip()
            reservation = self._reserve_task(
                combined_text,
                session,
                turn,
                forced_task_type=pending.task_type,
                target_override=target,
                clarification=pending,
                operator_context=operator,
            )
            self._pending_clarifications.pop(key, None)
            return reservation

    def cancel_pending_clarification(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> bool:
        """Discard only the current principal's incomplete task draft."""

        session = _required(thread_id, "thread_id")
        operator = self._require_operator("runtime:cancel", operator_context)
        key = self._clarification_key(session, operator)
        with self._lock:
            pending = self._pending_clarifications.get(key)
            if pending is None:
                return False
            self._pending_clarifications.pop(key, None)
            return pending.expires_at > time.time()

    def can_revise_pending_task(
        self,
        user_text: str,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> bool:
        """Return whether this turn can revise an unsubmitted approval draft."""

        if not _parse_task_revision(user_text):
            return False
        session = _required(thread_id, "thread_id")
        operator = self._require_operator("runtime:submit", operator_context)
        with self._lock:
            return self._pending_confirmation_for_operator(session, operator) is not None

    def revise_pending_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        """Cancel the old approval version and create a newly bound one."""

        revision_request = _parse_task_revision(user_text)
        if not revision_request:
            raise ValueError("task_revision_not_understood")
        session = _required(thread_id, "thread_id")
        turn = _required(turn_id, "turn_id")
        operator = self._require_operator("runtime:submit", operator_context)
        if not operator.allows("runtime:cancel"):
            raise PermissionError("voice operator is not authorized for runtime:cancel")
        request_digest = _plan_digest(revision_request)
        turn_key = (session, turn)
        with self._lock:
            existing_id = self._reservation_keys.get(turn_key)
            if existing_id:
                existing = self._reservations[existing_id]
                if not self._operator_owns_reservation(operator, existing):
                    raise PermissionError("voice task revision belongs to a different operator")
                if existing.revision_request_digest == request_digest:
                    return existing
                raise RuntimeError("task_revision_turn_conflict")
            current = self._pending_confirmation_for_operator(session, operator)
            if current is None:
                raise LookupError("no_pending_task_revision")
            target = str(revision_request.get("target") or current.target).strip()
            if not target:
                raise ValueError("task_target_required")
            parameters = _reservation_task_parameters(current)
            if "photo_count" in revision_request:
                parameters["photo_count"] = int(revision_request["photo_count"])
                parameters["capture_evidence"] = True
            revised_text = _revised_task_text(current.task_type, target, parameters)
            cancelled = self._handoff.cancel_prepared_run(
                current.run_id,
                operator_id=operator.operator_id,
                operator_context=operator.to_runtime_context(
                    permission="runtime:cancel",
                    thread_id=session,
                ),
                reason="voice_task_revised_before_confirmation",
            )
            if not cancelled.get("handled", False):
                raise RuntimeError("pending_task_revision_cancel_failed")
            current.state = str(
                dict(cancelled.get("run") or {}).get("current_state") or "cancelled"
            )
            current.updated_at = time.time()
            return self._reserve_task(
                revised_text,
                session,
                turn,
                forced_task_type=current.task_type,
                target_override=target,
                revision=current.revision + 1,
                supersedes_reservation_id=current.reservation_id,
                revision_request_digest=request_digest,
                operator_context=operator,
            )

    def _reserve_task(
        self,
        user_text: str,
        thread_id: str,
        turn_id: str,
        *,
        forced_task_type: str = "",
        target_override: str = "",
        clarification: PendingTaskClarification | None = None,
        revision: int = 1,
        supersedes_reservation_id: str = "",
        revision_request_digest: str = "",
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        text = _required(user_text, "user_text")
        session = _required(thread_id, "thread_id")
        turn = _required(turn_id, "turn_id")
        operator = self._require_operator("runtime:submit", operator_context)
        key = (session, turn)
        clarification_key = self._clarification_key(session, operator)
        with self._lock:
            existing_id = self._reservation_keys.get(key)
            if existing_id:
                existing = self._reservations[existing_id]
                if not self._operator_owns_reservation(operator, existing):
                    raise PermissionError(
                        "voice task reservation belongs to a different operator"
                    )
                return existing
            active = self._active_reservation_for_thread(session)
            if active is not None:
                self._latest_by_thread[session] = active.reservation_id
                raise RuntimeError("voice_task_already_active")
            reservation_id = _stable_id("voice-task", session, turn)
            roles = tuple(
                dict.fromkeys(str(role).strip().lower() for role in operator.roles if role)
            )
            mission = self._draft_mission(
                text,
                operator_id=operator.operator_id,
                thread_id=session,
                turn_id=turn,
                forced_task_type=forced_task_type,
            )
            if forced_task_type:
                mission["mission_type"] = forced_task_type
            if target_override:
                mission["target"] = target_override
            task_type = str(mission.get("mission_type") or "").strip()
            if task_type not in {"status_report", "inspection_patrol", "navigate_to"}:
                raise ValueError(f"unsupported_voice_task:{task_type or 'unknown'}")
            target = _mission_target(mission)
            physical_task = task_type in {"inspection_patrol", "navigate_to"}
            if physical_task and not target:
                if not operator.person_id.strip():
                    raise PermissionError("physical_task_speaker_identity_required")
                now = time.time()
                self._pending_clarifications[clarification_key] = PendingTaskClarification(
                    clarification_id=_stable_id(
                        "voice-clarification",
                        session,
                        turn,
                        operator.operator_id,
                        operator.person_id,
                    ),
                    thread_id=session,
                    operator_id=operator.operator_id.strip(),
                    person_id=operator.person_id.strip(),
                    turn_id=turn,
                    task_type=task_type,
                    original_text=text,
                    missing_parameter="target",
                    created_at=now,
                    expires_at=now + self._clarification_ttl_s,
                )
                raise ValueError("task_target_required")
            if physical_task and not operator.person_id.strip():
                raise PermissionError("physical_task_speaker_identity_required")
            requires_confirmation = physical_task or bool(
                mission.get("requires_confirmation", False)
            )
            confirmed = not requires_confirmation
            confirmation_prompt = _voice_confirmation_prompt(
                task_type,
                target,
                _mission_task_parameters(mission),
                risk_tier=str(mission.get("risk_tier") or "medium"),
            )
            plan = _voice_task_plan(
                reservation_id=reservation_id,
                user_text=text,
                thread_id=session,
                turn_id=turn,
                operator_id=operator.operator_id,
                person_id=operator.person_id,
                operator_roles=roles,
                operator_authenticated=operator.authenticated,
                operator_source=operator.source,
                runtime_permission="runtime:submit",
                mission=mission,
                confirmed=confirmed,
            )
            if clarification is not None:
                voice_context = plan.get("voice_context")
                if isinstance(voice_context, dict):
                    voice_context.update(
                        {
                            "clarification_id": clarification.clarification_id,
                            "clarification_turn_id": (
                                clarification.turn_id
                            ),
                            "clarification_answer_turn_id": turn,
                        }
                    )
            if supersedes_reservation_id:
                voice_context = plan.get("voice_context")
                if isinstance(voice_context, dict):
                    voice_context.update(
                        {
                            "task_revision": max(1, int(revision)),
                            "supersedes_reservation_id": supersedes_reservation_id,
                            "revision_request_digest": revision_request_digest,
                        }
                    )
            approval_id = ""
            approval_payload_digest = ""
            approval_expires_at = 0.0
            run_id = ""
            reservation_state = "reserved"
            if requires_confirmation:
                approval_id = _stable_id("voice-approval", reservation_id, task_type, target)
                voice_context = plan.get("voice_context")
                if isinstance(voice_context, dict):
                    voice_context["approval_id"] = approval_id
                approval_payload_digest = _plan_digest(plan)
                approval_expires_at = time.time() + self._approval_ttl_s
                approval_request = {
                    "kind": ConfirmationKind.RUNTIME_HANDOFF.value,
                    "thread_id": session,
                    "prompt_turn_id": turn,
                    "operator_id": operator.operator_id,
                    "person_id": operator.person_id.strip(),
                    "expires_at": approval_expires_at,
                    "allows_short_reply": False,
                    "approval_id": approval_id,
                    "subject": f"{task_type}:{target}",
                    "risk_level": str(mission.get("risk_tier") or "high"),
                    "payload_digest": approval_payload_digest,
                }
                prepared = self._handoff.prepare_plan_payload(
                    plan,
                    approval_request=approval_request,
                )
                prepared_run = dict(prepared.get("run") or {})
                run_id = _required(prepared_run.get("run_id"), "prepared run_id")
                reservation_state = str(
                    prepared_run.get("current_state") or "waiting_user"
                )
            reservation = TaskReservation(
                reservation_id=reservation_id,
                thread_id=session,
                turn_id=turn,
                user_text=text,
                operator_id=_required(operator.operator_id, "operator_id"),
                person_id=operator.person_id.strip(),
                operator_roles=roles,
                operator_authenticated=operator.authenticated,
                operator_source=operator.source,
                runtime_permission="runtime:submit",
                plan=plan,
                task_type=task_type,
                target=target,
                requires_confirmation=requires_confirmation,
                confirmation_prompt=confirmation_prompt,
                approval_id=approval_id,
                approval_payload_digest=approval_payload_digest,
                approval_expires_at=approval_expires_at,
                state=reservation_state,
                run_id=run_id,
                revision=max(1, int(revision)),
                supersedes_reservation_id=supersedes_reservation_id,
                revision_request_digest=revision_request_digest,
            )
            self._reservations[reservation_id] = reservation
            self._pending_clarifications.pop(clarification_key, None)
            self._reservation_keys[key] = reservation_id
            self._latest_by_thread[session] = reservation_id
            if run_id:
                self._reservation_by_run[run_id] = reservation_id
        self._publish(
            TaskLifecycleEvent(
                event_id=f"{reservation_id}:reserved",
                reservation_id=reservation_id,
                run_id="",
                thread_id=session,
                turn_id=turn,
                kind="reserved",
                state=reservation.state,
                message=(
                    "Task reserved; waiting for operator confirmation."
                    if requires_confirmation
                    else "Task reserved; waiting for acknowledgement."
                ),
            )
        )
        return reservation

    def confirm_pending(
        self,
        thread_id: str,
        turn_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskReservation:
        """Confirm the latest side-effecting task for this conversation thread."""

        session = _required(thread_id, "thread_id")
        response_turn = _required(turn_id, "turn_id")
        operator = self._require_operator("runtime:submit", operator_context)
        with self._lock:
            reservation_id = self._latest_by_thread.get(session)
            reservation = self._reservations.get(reservation_id or "")
            if reservation is None:
                raise LookupError("no_pending_task_confirmation")
            if reservation.operator_id != operator.operator_id:
                raise PermissionError("pending task belongs to a different operator")
            if reservation.person_id != operator.person_id.strip():
                raise PermissionError("pending task belongs to a different speaker")
            if reservation.state == "confirmed":
                return reservation
            if reservation.state != "waiting_user":
                raise LookupError("no_pending_task_confirmation")
            if not reservation.run_id:
                raise RuntimeError("pending task is not durably prepared")
            run = self._handoff.run_service.require(reservation.run_id)
            approval = dict(run.approval_request)
            if not approval or str(approval.get("approval_id") or "") != reservation.approval_id:
                raise RuntimeError("pending task approval identity is unavailable")
            if _plan_digest(reservation.plan) != str(
                approval.get("payload_digest") or ""
            ):
                raise RuntimeError("pending task payload changed after approval prompt")
            expires_at = float(approval.get("expires_at") or 0.0)
            remaining = expires_at - time.time()
            if remaining <= 0.0:
                self._handoff.expire_prepared_run(reservation.run_id)
                reservation.state = "cancelled"
                reservation.updated_at = time.time()
                raise TimeoutError("pending task confirmation expired")
            now_monotonic = time.monotonic()
            scope = ApprovalScope(
                kind=ConfirmationKind.RUNTIME_HANDOFF,
                thread_id=str(approval.get("thread_id") or ""),
                prompt_turn_id=str(approval.get("prompt_turn_id") or ""),
                person_id=str(approval.get("person_id") or "") or None,
                operator_id=str(approval.get("operator_id") or "") or None,
                expires_at_monotonic=now_monotonic + max(0.0, remaining),
                allows_short_reply=False,
                approval_id=reservation.approval_id,
                subject=str(approval.get("subject") or reservation.task_type),
                risk_level=str(approval.get("risk_level") or "high"),
                payload_digest=reservation.approval_payload_digest,
            )
            response_context = InteractionTurnContext(
                thread_id=session,
                turn_id=response_turn,
                channel="voice",
                source="voice",
                user_text="确认执行",
                person_id=operator.person_id.strip() or None,
                operator_id=operator.operator_id,
            )
            if not scope.matches(
                response_context,
                approval_id=reservation.approval_id,
                now_monotonic=now_monotonic,
            ):
                raise PermissionError("pending task confirmation scope mismatch")
            reservation.plan["handoff_ready"] = True
            task_session = reservation.plan.get("session")
            if isinstance(task_session, dict):
                task_session["confirmation_status"] = "confirmed"
            mission_container = reservation.plan.get("mission")
            mission = (
                mission_container.get("mission")
                if isinstance(mission_container, dict)
                else None
            )
            if isinstance(mission, dict):
                mission["status"] = "confirmed"
            runtime_context = operator.to_runtime_context(
                permission="runtime:submit",
                thread_id=session,
            )
            runtime_context["approval_id"] = reservation.approval_id
            confirmed = self._handoff.confirm_prepared_plan(
                reservation.run_id,
                confirmed_plan=dict(reservation.plan),
                operator_id=operator.operator_id,
                operator_context=runtime_context,
            )
            if not confirmed.get("handled", False):
                reason = str(confirmed.get("reason") or "approval_confirmation_failed")
                if reason in {
                    "approval_operator_mismatch",
                    "approval_person_mismatch",
                    "approval_id_mismatch",
                    "runtime_operator_context_required",
                    "runtime_operator_authentication_required",
                }:
                    raise PermissionError(reason)
                if reason == "approval_expired":
                    raise TimeoutError(reason)
                raise RuntimeError(reason)
            reservation.state = str(
                dict(confirmed.get("run") or {}).get("current_state") or "confirmed"
            )
            reservation.updated_at = time.time()
            return reservation

    async def commit_ack_and_submit(
        self,
        reservation_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskHandle:
        if self._loop is None or self._commit_lock is None:
            raise RuntimeError("voice task lifecycle is not started")
        async with self._commit_lock:
            reservation = self._require_reservation(reservation_id)
            operator = self._require_operator("runtime:submit", operator_context)
            if not self._operator_owns_reservation(operator, reservation):
                raise PermissionError(
                    "voice task reservation belongs to a different operator"
                )
            return await self._commit_reserved_task(reservation)

    async def _commit_reserved_task(self, reservation: TaskReservation) -> TaskHandle:
        existing = self._handles.get(reservation.reservation_id)
        if existing is not None:
            return existing
        if reservation.state == "abandoned":
            raise RuntimeError("reservation was abandoned")
        if reservation.state == "waiting_user":
            raise RuntimeError("reservation requires operator confirmation")
        if reservation.submit_attempted:
            raise RuntimeError("reservation submission already attempted")
        reservation.submit_attempted = True
        reservation.state = "submitting"
        reservation.updated_at = time.time()
        cancelled: asyncio.CancelledError | None = None
        try:
            loop = asyncio.get_running_loop()
            if reservation.requires_confirmation:
                submission = loop.run_in_executor(
                    None,
                    self._handoff.submit_prepared_run,
                    reservation.run_id,
                )
            else:
                submission = loop.run_in_executor(
                    None,
                    self._handoff.submit_plan_payload,
                    dict(reservation.plan),
                )
            try:
                result = await asyncio.shield(submission)
            except asyncio.CancelledError as error:
                cancelled = error
                result = await submission
        except Exception:
            reservation.state = "failed"
            reservation.updated_at = time.time()
            if cancelled is not None:
                raise cancelled from None
            raise
        run_payload = dict(result.get("run") or {})
        run_id = _required(run_payload.get("run_id"), "submitted run_id")
        if reservation.run_id and run_id != reservation.run_id:
            reservation.state = "failed"
            reservation.updated_at = time.time()
            raise RuntimeError("prepared task submission changed run identity")
        remote_task_id = str(run_payload.get("remote_task_id") or "")
        state = str(
            run_payload.get("current_state") or result.get("status") or "queued"
        ).strip()
        accepted = result.get("accepted") is True and bool(remote_task_id)
        handle = TaskHandle(
            reservation_id=reservation.reservation_id,
            run_id=run_id,
            remote_task_id=remote_task_id,
            correlation_id=run_id,
            idempotency_key=str(
                run_payload.get("external_idempotency_key") or reservation.reservation_id
            ),
            thread_id=reservation.thread_id,
            turn_id=reservation.turn_id,
            state=state,
            accepted=accepted,
        )
        reservation.run_id = run_id
        reservation.remote_task_id = remote_task_id
        reservation.state = state
        reservation.updated_at = time.time()
        self._handles[reservation.reservation_id] = handle
        self._reservation_by_run[run_id] = reservation.reservation_id
        current_run = self._handoff.run_service.require(run_id)
        if reservation.state not in _TERMINAL_STATES and (
            remote_task_id or _needs_submission_reconciliation(current_run)
        ):
            await self._supervisor.ensure_tracked(run_id)
        if cancelled is not None:
            raise cancelled
        return handle

    def abandon(self, reservation_id: str) -> bool:
        reservation = self._require_reservation(reservation_id)
        if reservation.submit_attempted or reservation.run_id:
            return False
        reservation.state = "abandoned"
        reservation.updated_at = time.time()
        return True

    def status_snapshot(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskStatusSnapshot:
        """Return the active task for a thread, falling back to its latest task."""
        operator = self._require_operator("runtime:read", operator_context)
        return self._status_snapshot_unchecked(thread_id, operator_context=operator)

    def task_report(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> dict[str, Any]:
        """Return the latest owned TaskRun report, including evidence artifacts."""

        operator = self._require_operator("runtime:read", operator_context)
        snapshot = self._status_snapshot_unchecked(
            thread_id,
            operator_context=operator,
        )
        if not snapshot.run_id:
            return {}
        payload = self._handoff.report_payload(snapshot.run_id)
        report = payload.get("report")
        return dict(report) if isinstance(report, dict) else {}

    def _status_snapshot_unchecked(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext,
    ) -> TaskStatusSnapshot:
        session = _required(thread_id, "thread_id")
        self._adopt_recovered_task_alias(session, operator_context)
        return self.active_status_snapshot(
            thread_id,
            operator_context=operator_context,
        ) or self.latest_status_snapshot(
            thread_id,
            operator_context=operator_context,
        )

    def active_status_snapshot(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext,
    ) -> TaskStatusSnapshot | None:
        session = _required(thread_id, "thread_id")
        with self._lock:
            candidates = sorted(
                (
                    reservation
                    for reservation in self._reservations.values()
                    if reservation.thread_id == session
                    and self._operator_owns_reservation(operator_context, reservation)
                ),
                key=lambda item: item.updated_at,
                reverse=True,
            )
        for reservation in candidates:
            snapshot = self._snapshot_for_reservation(reservation)
            if snapshot.active:
                return snapshot
        return None

    def latest_status_snapshot(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext,
    ) -> TaskStatusSnapshot:
        session = _required(thread_id, "thread_id")
        with self._lock:
            reservation = self._reservations.get(self._latest_by_thread.get(session, ""))
        if reservation is None or not self._operator_owns_reservation(
            operator_context,
            reservation,
        ):
            return TaskStatusSnapshot(thread_id=session)
        return self._snapshot_for_reservation(reservation, thread_id=session)

    def _snapshot_for_reservation(
        self,
        reservation: TaskReservation,
        *,
        thread_id: str | None = None,
    ) -> TaskStatusSnapshot:
        self._expire_reservation_if_needed(reservation)
        run = self._handoff.run_service.get(reservation.run_id) if reservation.run_id else None
        state = run.current_state if run is not None else reservation.state
        summary = run.result_summary if run is not None else ""
        remote_task_id = (
            str(run.remote_task_id or "") if run is not None else reservation.remote_task_id
        )
        updated_at = (
            float(
                run.remote_observed_at or run.ended_at or run.started_at or reservation.updated_at
            )
            if run is not None
            else reservation.updated_at
        )
        return TaskStatusSnapshot(
            thread_id=thread_id or reservation.thread_id,
            reservation_id=reservation.reservation_id,
            run_id=reservation.run_id,
            remote_task_id=remote_task_id,
            turn_id=reservation.turn_id,
            state=state,
            result_summary=summary,
            active=state not in _TERMINAL_STATES and state != "abandoned",
            updated_at=updated_at,
        )

    def _adopt_recovered_task_alias(
        self,
        thread_id: str,
        operator_context: VoiceTaskOperatorContext,
    ) -> None:
        """Expose the latest task owned by the current principal after transport rotation."""

        operator = operator_context
        if not operator.allows("runtime:read"):
            return
        with self._lock:
            current = self._reservations.get(self._latest_by_thread.get(thread_id, ""))
            if current is not None and self._operator_owns_reservation(operator, current):
                return
            candidates = sorted(
                (
                    reservation
                    for reservation in self._reservations.values()
                    if self._operator_owns_reservation(operator, reservation)
                ),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            if candidates:
                self._latest_by_thread[thread_id] = candidates[0].reservation_id

    async def cancel_active(
        self,
        thread_id: str,
        *,
        reason: str = "voice_operator_cancelled",
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> CancelRequestResult:
        try:
            operator = self._require_operator("runtime:cancel", operator_context)
        except PermissionError:
            return CancelRequestResult(
                False,
                TaskStatusSnapshot(thread_id=_required(thread_id, "thread_id")),
                "operator_not_authorized",
            )
        snapshot = self._status_snapshot_unchecked(
            thread_id,
            operator_context=operator,
        )
        if (
            snapshot.active
            and snapshot.state in {"waiting_user", "confirmed"}
            and snapshot.reservation_id
            and snapshot.run_id
            and not snapshot.remote_task_id
        ):
            cancelled = self._handoff.cancel_prepared_run(
                snapshot.run_id,
                operator_id=operator.operator_id,
                operator_context=operator.to_runtime_context(
                    permission="runtime:cancel",
                    thread_id=snapshot.thread_id,
                ),
                reason=reason,
            )
            reservation = self._require_reservation(snapshot.reservation_id)
            reservation.state = str(
                dict(cancelled.get("run") or {}).get("current_state") or "cancelled"
            )
            reservation.updated_at = time.time()
            return CancelRequestResult(
                False,
                self._status_snapshot_unchecked(
                    thread_id,
                    operator_context=operator,
                ),
                "pending_task_cancelled",
            )
        if (
            not snapshot.run_id
            or not snapshot.active
            or (not snapshot.remote_task_id and snapshot.state != "submission_unknown")
        ):
            return CancelRequestResult(False, snapshot, "no_active_external_task")
        outcome = await self._supervisor.request_cancel(
            snapshot.run_id,
            operator_id=operator.operator_id,
            reason=reason,
            operator_context=operator.to_runtime_context(
                permission="runtime:cancel",
                thread_id=snapshot.thread_id,
            ),
        )
        return CancelRequestResult(
            outcome.remote_acknowledged,
            self._status_snapshot_unchecked(
                thread_id,
                operator_context=operator,
            ),
            outcome.error_code,
        )

    def observe_runtime_event(self, event: RuntimeEvent) -> None:
        """Thread-safe observer suitable for ``RuntimeHandoffService.subscribe_events``."""
        loop = self._loop
        if loop is None or self._closing:
            return
        try:
            if asyncio.get_running_loop() is loop:
                self._ingest_runtime_event(event)
                return
        except RuntimeError:
            pass
        loop.call_soon_threadsafe(self._ingest_runtime_event, event)

    async def wait_ready(
        self,
        thread_id: str,
        timeout: float | None = None,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> bool:
        session = _required(thread_id, "thread_id")
        try:
            operator = self._require_operator("runtime:read", operator_context)
        except PermissionError:
            return False
        self._adopt_replayable_notification(
            session,
            operator,
            include_live=operator_context is not None,
        )
        with self._lock:
            if self._next_pending_id(session, operator) is not None:
                return True
            if self._closing:
                return False
            waiter = asyncio.Event()
            self._waiters[session].append(waiter)
        try:
            if timeout is None:
                await waiter.wait()
            else:
                await asyncio.wait_for(waiter.wait(), max(0.0, float(timeout)))
        except TimeoutError:
            return False
        finally:
            with self._lock:
                try:
                    self._waiters[session].remove(waiter)
                except ValueError:
                    pass
        with self._lock:
            return self._next_pending_id(session, operator) is not None

    def claim_next(
        self,
        thread_id: str,
        *,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> TaskLifecycleEvent | None:
        session = _required(thread_id, "thread_id")
        try:
            operator = self._require_operator("runtime:read", operator_context)
        except PermissionError:
            return None
        self._adopt_replayable_notification(
            session,
            operator,
            include_live=operator_context is not None,
        )
        with self._lock:
            event_id = self._next_pending_id(session, operator)
            if event_id is None:
                return None
            receipt = self._receipts[event_id]
            receipt.state = "delivering"
            receipt.claimed_at = time.time()
            receipt.attempt_count += 1
            return self._events[event_id]

    def settle_delivery(self, event_id: str, state: DeliveryState) -> DeliveryReceipt:
        if state not in _DELIVERY_FINAL_STATES:
            raise ValueError("delivery state must be final")
        normalized_event_id = _required(event_id, "event_id")
        with self._lock:
            receipt = self._receipts.get(normalized_event_id)
            if receipt is None:
                raise KeyError(event_id)
            if receipt.state == state:
                return receipt
            if receipt.state in _DELIVERY_FINAL_STATES:
                return receipt
            if receipt.state != "delivering":
                raise RuntimeError(f"cannot settle delivery from {receipt.state}")
            event = self._events[normalized_event_id]
        persisted_state = state
        if event.run_id:
            result = self._handoff.record_notification_delivery_receipt(
                event.run_id,
                event_id=event.event_id,
                status=state,
            )
            raw_persisted_state = str(result.get("status") or state)
            if raw_persisted_state not in _DELIVERY_FINAL_STATES:
                raise RuntimeError("invalid persisted delivery state")
            persisted_state = cast(DeliveryState, raw_persisted_state)
        with self._lock:
            receipt = self._receipts[normalized_event_id]
            if receipt.state in _DELIVERY_FINAL_STATES:
                return receipt
            if receipt.state != "delivering":
                raise RuntimeError(f"cannot settle delivery from {receipt.state}")
            receipt.state = persisted_state
            receipt.settled_at = time.time()
            self._replayable_event_ids.discard(normalized_event_id)
            return receipt

    def retry_delivery(self, event_id: str, *, error_code: str = "delivery_failed") -> bool:
        """Release a transiently failed claim for bounded redelivery."""

        normalized_event_id = _required(event_id, "event_id")
        with self._lock:
            receipt = self._receipts.get(normalized_event_id)
            if receipt is None or receipt.state != "delivering":
                return False
            receipt.last_error_code = str(error_code or "delivery_failed").strip()
            if receipt.attempt_count >= self._max_delivery_attempts:
                exhausted = True
            else:
                exhausted = False
                receipt.state = "pending"
                receipt.claimed_at = None
                delay = self._delivery_retry_delay_s
                receipt.next_attempt_at = time.time() + delay
                thread_id = receipt.thread_id
        if exhausted:
            self.settle_delivery(normalized_event_id, "interrupted")
            return False
        loop = self._loop
        if delay > 0.0 and loop is not None:
            loop.call_later(delay, self._wake_thread_waiters, thread_id)
        else:
            self._wake_thread_waiters(thread_id)
        return True

    def _wake_thread_waiters(self, thread_id: str) -> None:
        with self._lock:
            waiters = list(self._waiters.pop(thread_id, []))
        for waiter in waiters:
            waiter.set()

    def _adopt_replayable_notification(
        self,
        thread_id: str,
        operator_context: VoiceTaskOperatorContext,
        *,
        include_live: bool = False,
    ) -> bool:
        """Move one recovered event to the matching operator's current thread."""

        operator = operator_context
        if not operator.allows("runtime:read"):
            return False
        with self._lock:
            candidate_ids = self._events if include_live else self._replayable_event_ids
            candidates = sorted(
                (
                    self._events[event_id]
                    for event_id in candidate_ids
                    if event_id in self._events
                    and self._receipts[event_id].state == "pending"
                    and self._events[event_id].kind != "reserved"
                ),
                key=lambda item: item.created_at,
            )
            for event in candidates:
                reservation = self._reservations.get(event.reservation_id)
                if reservation is None or not self._operator_owns_reservation(
                    operator,
                    reservation,
                ):
                    continue
                if event.thread_id == thread_id:
                    return True
                old_order = self._event_order.get(event.thread_id, [])
                try:
                    old_order.remove(event.event_id)
                except ValueError:
                    pass
                self._events[event.event_id] = replace(
                    event,
                    thread_id=thread_id,
                    originating_thread_id=(
                        event.originating_thread_id
                        or event.thread_id
                    ),
                )
                self._receipts[event.event_id].thread_id = thread_id
                self._event_order[thread_id].append(event.event_id)
                self._latest_by_thread[thread_id] = reservation.reservation_id
                return True
        return False

    def delivery_receipt(self, event_id: str) -> DeliveryReceipt | None:
        with self._lock:
            return self._receipts.get(str(event_id or "").strip())

    def _require_reservation(self, reservation_id: str) -> TaskReservation:
        with self._lock:
            reservation = self._reservations.get(_required(reservation_id, "reservation_id"))
        if reservation is None:
            raise KeyError(reservation_id)
        return reservation

    def _require_operator(
        self,
        permission: str,
        operator_context: VoiceTaskOperatorContext | None = None,
    ) -> VoiceTaskOperatorContext:
        operator = operator_context or self._operator_context
        if operator is None or not operator.allows(permission):
            raise PermissionError(f"voice operator is not authorized for {permission}")
        return operator

    @staticmethod
    def _operator_owns_reservation(
        operator: VoiceTaskOperatorContext,
        reservation: TaskReservation,
    ) -> bool:
        return bool(
            reservation.operator_id == operator.operator_id.strip()
            and reservation.person_id == operator.person_id.strip()
        )

    @staticmethod
    def _clarification_key(
        thread_id: str,
        operator: VoiceTaskOperatorContext,
    ) -> tuple[str, str, str]:
        return (
            thread_id.strip(),
            operator.operator_id.strip(),
            operator.person_id.strip(),
        )

    def _draft_mission(
        self,
        user_text: str,
        *,
        operator_id: str,
        thread_id: str,
        turn_id: str,
        forced_task_type: str,
    ) -> dict[str, Any]:
        if forced_task_type == "status_report":
            return _fallback_status_mission(user_text, operator_id=operator_id)
        service = self._mission_service
        draft = getattr(service, "draft", None)
        if not callable(draft):
            raise RuntimeError("mission_service_unavailable")
        mission = draft(
            user_text,
            operator_id=operator_id,
            channel="voice",
            metadata={
                "thread_id": thread_id,
                "turn_id": turn_id,
            },
        )
        to_dict = getattr(mission, "to_dict", None)
        payload = to_dict() if callable(to_dict) else mission
        if not isinstance(payload, dict):
            raise RuntimeError("mission_service_returned_invalid_plan")
        return dict(payload)

    def _active_reservation_for_thread(self, thread_id: str) -> TaskReservation | None:
        for reservation in self._reservations.values():
            if reservation.thread_id != thread_id:
                continue
            self._expire_reservation_if_needed(reservation)
            state = reservation.state
            if reservation.run_id:
                run = self._handoff.run_service.get(reservation.run_id)
                if run is not None:
                    state = run.current_state
            if state not in _TERMINAL_STATES:
                return reservation
        return None

    def _pending_confirmation_for_operator(
        self,
        thread_id: str,
        operator: VoiceTaskOperatorContext,
    ) -> TaskReservation | None:
        reservation = self._active_reservation_for_thread(thread_id)
        if reservation is None or not self._operator_owns_reservation(operator, reservation):
            return None
        self._expire_reservation_if_needed(reservation)
        if reservation.state != "waiting_user" or not reservation.run_id:
            return None
        run = self._handoff.run_service.get(reservation.run_id)
        if (
            run is None
            or run.current_state != "waiting_user"
            or bool(run.remote_task_id)
        ):
            return None
        return reservation

    def _expire_reservation_if_needed(self, reservation: TaskReservation) -> None:
        if reservation.state != "waiting_user" or not reservation.run_id:
            return
        run = self._handoff.run_service.get(reservation.run_id)
        if run is None or run.current_state != "waiting_user":
            return
        if float(run.approval_request.get("expires_at") or 0.0) > time.time():
            return
        expired = self._handoff.expire_prepared_run(run.run_id)
        if expired.get("handled", False):
            reservation.state = "cancelled"
            reservation.updated_at = time.time()

    def _recover_run(self, run: TaskRun) -> bool:
        voice = run.handoff.source_plan.get("voice_context", {})
        if not isinstance(voice, dict):
            return False
        session = str(
            voice.get("thread_id") or voice.get("conversation_session_id") or ""
        ).strip()
        turn = str(voice.get("turn_id") or voice.get("originating_turn_id") or "").strip()
        if not session or not turn:
            return False
        reservation_id = str(
            voice.get("reservation_id") or voice.get("submission_id") or ""
        ).strip()
        reservation_id = reservation_id or _stable_id("voice-status", session, turn)
        reservation = TaskReservation(
            reservation_id=reservation_id,
            thread_id=session,
            turn_id=turn,
            user_text=str(run.handoff.source_plan.get("goal") or "status report"),
            operator_id=run.handoff.operator_id,
            person_id=str(
                voice.get("person_id")
                or run.approval_request.get("person_id")
                or ""
            ),
            operator_roles=tuple(run.handoff.operator_roles),
            operator_authenticated=voice.get("operator_authenticated") is True,
            operator_source=str(voice.get("operator_source") or "recovered"),
            runtime_permission=str(voice.get("runtime_permission") or "runtime:submit"),
            plan=deepcopy(run.handoff.source_plan),
            task_type=run.handoff.task_type,
            target=str(run.handoff.target_area or run.handoff.target_object or ""),
            requires_confirmation=run.handoff.risk_level in {"medium", "high", "critical"},
            confirmation_prompt=_voice_confirmation_prompt(
                run.handoff.task_type,
                str(run.handoff.target_area or run.handoff.target_object or ""),
                _plan_task_parameters(run.handoff.source_plan),
                risk_tier=run.handoff.risk_level,
            ),
            approval_id=str(run.approval_request.get("approval_id") or ""),
            approval_payload_digest=str(
                run.approval_request.get("payload_digest") or ""
            ),
            approval_expires_at=float(run.approval_request.get("expires_at") or 0.0),
            state=run.current_state,
            run_id=run.run_id,
            remote_task_id=str(run.remote_task_id or ""),
            submit_attempted=bool(
                run.remote_task_id or _needs_submission_reconciliation(run)
            ),
            created_at=run.handoff.created_at,
            updated_at=(
                run.runtime_events[-1].created_at
                if run.runtime_events
                else run.handoff.created_at
            ),
            recovered=True,
            revision=max(1, int(voice.get("task_revision") or 1)),
            supersedes_reservation_id=str(
                voice.get("supersedes_reservation_id") or ""
            ),
            revision_request_digest=str(
                voice.get("revision_request_digest") or ""
            ),
        )
        handle = None
        if reservation.submit_attempted:
            handle = TaskHandle(
                reservation_id=reservation_id,
                run_id=run.run_id,
                remote_task_id=str(run.remote_task_id or ""),
                correlation_id=run.run_id,
                idempotency_key=run.external_idempotency_key or reservation_id,
                thread_id=session,
                turn_id=turn,
                state=run.current_state,
                accepted=bool(run.remote_task_id) and run.remote_status != "rejected",
            )
        with self._lock:
            self._reservations[reservation_id] = reservation
            self._reservation_keys[(session, turn)] = reservation_id
            self._reservation_by_run[run.run_id] = reservation_id
            if handle is not None:
                self._handles[reservation_id] = handle
            current_id = self._latest_by_thread.get(session)
            current = self._reservations.get(current_id or "")
            if current is None or (
                reservation.created_at,
                reservation.updated_at,
                reservation.reservation_id,
            ) > (
                current.created_at,
                current.updated_at,
                current.reservation_id,
            ):
                self._latest_by_thread[session] = reservation_id
        return True

    def _replay_terminal_notification(self, run: TaskRun) -> None:
        for event in reversed(run.runtime_events):
            kind = _event_kind(event)
            if kind not in {"completed", "failed", "cancelled"}:
                continue
            if time.time() - event.created_at > self._delivery_ttl_s:
                if (
                    self._handoff.notification_delivery_receipt(
                        run.run_id,
                        event_id=event.event_id,
                    )
                    is None
                ):
                    self._handoff.record_notification_delivery_receipt(
                        run.run_id,
                        event_id=event.event_id,
                        status="expired",
                    )
                return
            self._replayable_event_ids.add(event.event_id)
            self._ingest_runtime_event(event)
            if event.event_id not in self._events:
                self._replayable_event_ids.discard(event.event_id)
            return

    def _ingest_runtime_event(self, event: RuntimeEvent) -> None:
        if event.event_id in self._seen_runtime_event_ids:
            return
        self._seen_runtime_event_ids.add(event.event_id)
        run = self._handoff.run_service.get(event.run_id)
        if run is None:
            return
        reservation_id = self._reservation_by_run.get(run.run_id)
        if not reservation_id:
            voice = run.handoff.source_plan.get("voice_context", {})
            if not isinstance(voice, dict):
                return
            reservation_id = str(
                voice.get("reservation_id") or voice.get("submission_id") or ""
            ).strip()
        reservation = self._reservations.get(reservation_id)
        if reservation is None:
            return
        reservation.run_id = run.run_id
        reservation.remote_task_id = str(run.remote_task_id or reservation.remote_task_id)
        reservation.state = run.current_state
        reservation.updated_at = event.created_at
        self._reservation_by_run[run.run_id] = reservation.reservation_id
        kind = _event_kind(event)
        if kind is None:
            return
        event_state = str(event.state or run.current_state).strip().lower()
        summary = run.result_summary if kind in {"completed", "failed", "cancelled"} else ""
        signature = (kind, event_state, summary)
        if self._last_announced_signature.get(run.run_id) == signature:
            return
        self._last_announced_signature[run.run_id] = signature
        self._publish(
            TaskLifecycleEvent(
                event_id=event.event_id,
                reservation_id=reservation.reservation_id,
                run_id=run.run_id,
                thread_id=reservation.thread_id,
                turn_id=reservation.turn_id,
                kind=kind,
                state=event_state,
                message=_voice_event_message(kind, event.message),
                remote_task_id=str(run.remote_task_id or ""),
                correlation_id=str(event.payload.get("correlation_id") or run.run_id),
                originating_thread_id=reservation.thread_id,
                result_summary=summary,
                created_at=event.created_at,
            )
        )

    def _publish(self, event: TaskLifecycleEvent) -> bool:
        if (
            event.run_id
            and self._handoff.notification_delivery_receipt(
                event.run_id,
                event_id=event.event_id,
            )
            is not None
        ):
            self._replayable_event_ids.discard(event.event_id)
            return False
        with self._lock:
            if event.event_id in self._events:
                return False
            self._events[event.event_id] = event
            self._receipts[event.event_id] = DeliveryReceipt(
                event_id=event.event_id,
                thread_id=event.thread_id,
            )
            self._event_order[event.thread_id].append(event.event_id)
            waiters = list(self._waiters.pop(event.thread_id, []))
        for waiter in waiters:
            waiter.set()
        return True

    def _next_pending_id(
        self,
        thread_id: str,
        operator_context: VoiceTaskOperatorContext,
    ) -> str | None:
        now = time.time()
        for event_id in self._event_order.get(thread_id, []):
            receipt = self._receipts[event_id]
            if receipt.state != "pending":
                continue
            if receipt.next_attempt_at > now:
                continue
            event = self._events[event_id]
            reservation = self._reservations.get(event.reservation_id)
            if reservation is None or not self._operator_owns_reservation(
                operator_context,
                reservation,
            ):
                continue
            if now - event.created_at > self._delivery_ttl_s:
                receipt.state = "expired"
                receipt.settled_at = now
                if event.run_id:
                    self._handoff.record_notification_delivery_receipt(
                        event.run_id,
                        event_id=event.event_id,
                        status="expired",
                    )
                continue
            return event_id
        return None


def _event_kind(event: RuntimeEvent) -> TaskEventKind | None:
    """Map only user-relevant external lifecycle facts into voice events."""

    event_type = str(event.event_type or "").strip().lower()
    state = str(event.state or "").strip().lower()
    if event_type == "external_submission_bound" or event_type in {
        "external_submitted",
        "external_accepted",
        "external_queued",
    }:
        return "started"
    if event_type == "external_executing":
        return "progress"
    if event_type == "external_completed" or state == "completed":
        return "completed"
    if event_type == "external_cancelled" or state == "cancelled":
        return "cancelled"
    if event_type in {"external_failed", "external_rejected"} or state in {
        "failed",
        "blocked",
        "shadowed",
    }:
        return "failed"
    return None


def _voice_event_message(kind: TaskEventKind, runtime_message: str) -> str:
    if kind == "started":
        return "任务已提交，正在处理中。"
    if kind == "progress":
        return "任务正在处理中。"
    if kind == "completed":
        return "任务已完成。"
    if kind == "cancelled":
        return "任务已取消。"
    if kind == "failed":
        return "任务执行失败。"
    return str(runtime_message or "").strip()


def _voice_task_plan(
    *,
    reservation_id: str,
    user_text: str,
    thread_id: str,
    turn_id: str,
    operator_id: str,
    person_id: str,
    operator_roles: tuple[str, ...],
    operator_authenticated: bool,
    operator_source: str,
    runtime_permission: str,
    mission: dict[str, Any],
    confirmed: bool,
) -> dict[str, Any]:
    mission_payload = dict(mission)
    mission_payload["operator_id"] = operator_id
    mission_payload["operator_roles"] = list(operator_roles)
    mission_payload["status"] = "confirmed" if confirmed else "pending_confirmation"
    task_type = str(mission_payload.get("mission_type") or "status_report")
    return {
        "plan_id": reservation_id,
        "planning_session_id": reservation_id,
        "intent": task_type,
        "goal": user_text,
        "handoff_ready": confirmed,
        "operator_id": operator_id,
        "operator_roles": list(operator_roles),
        "session": {
            "operator_id": operator_id,
            "person_id": person_id,
            "operator_roles": list(operator_roles),
            "confirmation_status": "confirmed" if confirmed else "pending_confirmation",
        },
        "mission": {"mission": mission_payload},
        "voice_context": {
            "reservation_id": reservation_id,
            "submission_id": reservation_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "operator_authenticated": bool(operator_authenticated),
            "operator_source": str(operator_source or "voice"),
            "person_id": str(person_id or ""),
            "runtime_permission": str(runtime_permission or "runtime:submit"),
        },
    }


def _fallback_status_mission(user_text: str, *, operator_id: str) -> dict[str, Any]:
    return {
        "mission_id": _stable_id("voice-mission", operator_id, user_text),
        "mission_type": "status_report",
        "goal": user_text,
        "risk_tier": "low",
        "requires_confirmation": False,
        "operator_id": operator_id,
        "safety_notes": [],
    }


def _mission_target(mission: dict[str, Any]) -> str:
    direct = str(
        mission.get("target")
        or mission.get("target_area")
        or mission.get("destination")
        or ""
    ).strip()
    if direct:
        return direct
    for step in mission.get("steps", []):
        if isinstance(step, dict):
            target = str(step.get("target") or "").strip()
            if target:
                return target
    return ""


def _task_target_reply(user_text: str) -> str:
    target = str(user_text or "").strip().strip("，,。.!！?？")
    for prefix in (
        "目标是",
        "地点是",
        "目的地是",
        "改成",
        "改到",
        "改为",
        "换成",
        "换到",
        "就是",
        "就在",
    ):
        if target.startswith(prefix):
            target = target[len(prefix) :].strip()
            break
    return target


def _plausible_task_target_reply(user_text: str) -> bool:
    target = _task_target_reply(user_text)
    if not target or len(target) > 64:
        return False
    compact = "".join(target.split())
    if compact in {
        "嗯",
        "啊",
        "好",
        "好的",
        "可以",
        "行",
        "是",
        "不是",
        "确认",
        "确认执行",
        "取消",
        "取消任务",
    }:
        return False
    if target.rstrip().endswith(("吗", "么", "呢", "嘛", "?", "？")):
        return False
    if any(
        marker in compact
        for marker in (
            "如何",
            "怎么",
            "怎样",
            "为什么",
            "是什么",
            "你会",
            "你能",
            "能否",
            "可不可以",
            "然后",
            "顺便",
            "再去",
        )
    ):
        return False
    location_markers = (
        "区",
        "门",
        "厅",
        "站",
        "点",
        "线",
        "楼",
        "层",
        "室",
        "房",
        "库",
        "仓",
        "柜",
        "车间",
        "走廊",
        "大厅",
        "大堂",
        "前台",
        "办公室",
        "餐厅",
        "咖啡",
        "酒店",
        "停车场",
        "充电桩",
    )
    return any(marker in compact for marker in location_markers) or bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,15}", compact)
    )


def _parse_task_revision(user_text: str) -> dict[str, Any]:
    text = str(user_text or "").strip().strip("。.!！?？")
    if not text:
        return {}
    revision: dict[str, Any] = {}
    for marker in ("改成", "改到", "改为", "换成", "换到"):
        if marker not in text:
            continue
        target = text.split(marker, 1)[1].strip()
        for stop in ("，", ",", "并且", "然后", "再拍", "拍摄", "拍照", "拍"):
            target = target.split(stop, 1)[0].strip()
        if target:
            revision["target"] = target
        break
    count_match = re.search(r"([0-9一二两三四五六七八九十]+)\s*张", text)
    if count_match is not None:
        count = _parse_voice_small_count(count_match.group(1))
        if 1 <= count <= 20:
            revision["photo_count"] = count
    return revision


def _parse_voice_small_count(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        high, low = value.split("十", 1)
        return digits.get(high, 1) * 10 + digits.get(low, 0)
    return digits.get(value, 0)


def _reservation_task_parameters(reservation: TaskReservation) -> dict[str, Any]:
    return _plan_task_parameters(reservation.plan)


def _plan_task_parameters(plan: dict[str, Any]) -> dict[str, Any]:
    mission_wrapper = plan.get("mission")
    if not isinstance(mission_wrapper, dict):
        return {}
    mission = mission_wrapper.get("mission")
    if not isinstance(mission, dict):
        return {}
    metadata = mission.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    parameters = metadata.get("task_parameters")
    return dict(parameters) if isinstance(parameters, dict) else {}


def _revised_task_text(task_type: str, target: str, parameters: dict[str, Any]) -> str:
    if task_type == "inspection_patrol":
        text = f"巡检{target}"
        if parameters.get("capture_evidence") is True:
            photo_count = int(parameters.get("photo_count") or 0)
            text += f"并拍{photo_count}张照片" if photo_count > 0 else "并拍照"
        if parameters.get("generate_report") is True:
            text += "，最后生成报告"
        return text
    if task_type == "navigate_to":
        return f"导航到{target}"
    raise ValueError(f"unsupported_task_revision:{task_type}")


def _mission_task_parameters(mission: dict[str, Any]) -> dict[str, Any]:
    metadata = mission.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    parameters = metadata.get("task_parameters")
    return dict(parameters) if isinstance(parameters, dict) else {}


def _voice_confirmation_prompt(
    task_type: str,
    target: str,
    task_parameters: dict[str, Any] | None = None,
    *,
    risk_tier: str = "medium",
) -> str:
    target_text = target or "指定区域"
    normalized_risk = str(risk_tier or "medium").strip().lower()
    risk_label = {
        "low": "低风险",
        "medium": "中风险",
        "high": "高风险",
        "critical": "关键安全风险",
    }.get(normalized_risk, "中风险")
    movement_warning = f"这是{risk_label}任务，机器人将发生实际移动，请确认路径和周边人员安全。"
    if task_type == "inspection_patrol":
        parameters = dict(task_parameters or {})
        actions: list[str] = []
        photo_count = int(parameters.get("photo_count") or 0)
        if photo_count > 0:
            actions.append(f"拍摄{photo_count}张照片")
        elif parameters.get("capture_evidence") is True:
            actions.append("拍照")
        if parameters.get("generate_report") is True:
            actions.append("生成报告")
        detail = f"，并{'、'.join(actions)}" if actions else ""
        return (
            f"将前往{target_text}执行巡检{detail}。"
            f"{movement_warning}请说确认执行或取消任务。"
        )
    if task_type == "navigate_to":
        return (
            f"将移动机器人前往{target_text}。"
            f"{movement_warning}请说确认执行或取消任务。"
        )
    return ""


def _plan_digest(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _needs_submission_reconciliation(run: TaskRun) -> bool:
    return bool(
        not run.remote_task_id
        and run.external_idempotency_key
        and run.current_state in {"queued", "submission_unknown"}
    )


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _required(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text
