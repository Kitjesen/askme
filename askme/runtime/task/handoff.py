"""Auditable fake runtime handoff for confirmed cognitive plans.

This layer starts where cognition stops. It converts a confirmed, high-level
plan into a structured handoff, runs local safety preflight, and drives an
in-memory fake arbiter. It never calls hardware, gait, motor, or control
service APIs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorSubmitRequest,
    RuntimeExecutorSubmitResult,
    RuntimeExecutorTransport,
    RuntimeExecutorTransportError,
)
from askme.runtime.control_intent import (
    runtime_control_intent,
    runtime_control_permission,
)
from askme.runtime.task.arbiter_client import EXTERNAL_RUNTIME_PROFILES, RuntimeArbiterClient
from askme.runtime.task.audit import RuntimeAuditConfig, RuntimeAuditLog

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "blocked", "shadowed"})
_SUPPORTED_RUNTIME_PROFILES = ("fake", "shadow", "sim", "external", "lab")
_REMOTE_UPDATE_ID_LIMIT = 64
_NOTIFICATION_DELIVERY_RECEIPT_LIMIT = 64
_NOTIFICATION_DELIVERY_STATES = frozenset({"delivered", "interrupted", "suppressed", "expired"})
_REMOTE_STATUS_ALIASES = {
    "accepted": "submitted",
    "pending": "queued",
    "running": "executing",
    "working": "executing",
    "in_progress": "executing",
    "succeeded": "completed",
    "success": "completed",
    "error": "failed",
    "canceled": "cancelled",
}
_REMOTE_STATUS_TRANSITIONS = {
    "": frozenset(
        {
            "submitted",
            "created",
            "validating",
            "preflight",
            "queued",
            "executing",
            "paused",
            "resuming",
            "input_required",
            "auth_required",
            "blocked",
            "cancelling",
            "cancelled",
            "completed",
            "failed",
            "rejected",
            "shadowed",
        }
    ),
    "submitted": frozenset(
        {
            "submitted",
            "created",
            "validating",
            "preflight",
            "queued",
            "executing",
            "paused",
            "input_required",
            "auth_required",
            "blocked",
            "cancelling",
            "cancelled",
            "completed",
            "failed",
            "rejected",
        }
    ),
    "created": frozenset(
        {
            "created",
            "validating",
            "preflight",
            "queued",
            "executing",
            "blocked",
            "cancelled",
            "completed",
            "failed",
            "rejected",
        }
    ),
    "validating": frozenset(
        {
            "validating",
            "preflight",
            "queued",
            "executing",
            "blocked",
            "cancelled",
            "completed",
            "failed",
            "rejected",
        }
    ),
    "preflight": frozenset(
        {
            "preflight",
            "queued",
            "executing",
            "blocked",
            "cancelled",
            "completed",
            "failed",
            "rejected",
        }
    ),
    "queued": frozenset(
        {
            "queued",
            "executing",
            "paused",
            "input_required",
            "auth_required",
            "blocked",
            "cancelling",
            "cancelled",
            "completed",
            "failed",
            "rejected",
        }
    ),
    "executing": frozenset(
        {
            "executing",
            "paused",
            "resuming",
            "input_required",
            "auth_required",
            "blocked",
            "cancelling",
            "cancelled",
            "completed",
            "failed",
        }
    ),
    "paused": frozenset(
        {"paused", "resuming", "executing", "cancelling", "cancelled", "completed", "failed"}
    ),
    "input_required": frozenset(
        {"input_required", "resuming", "executing", "cancelling", "cancelled", "failed"}
    ),
    "auth_required": frozenset(
        {"auth_required", "resuming", "executing", "cancelling", "cancelled", "failed"}
    ),
    "resuming": frozenset(
        {"resuming", "executing", "paused", "cancelling", "cancelled", "completed", "failed"}
    ),
    "cancelling": frozenset({"cancelling", "cancelled", "completed", "failed"}),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed"}),
    "cancelled": frozenset({"cancelled"}),
    "blocked": frozenset({"blocked"}),
    "rejected": frozenset({"rejected"}),
    "shadowed": frozenset({"shadowed"}),
}

_P = ParamSpec("_P")
_R = TypeVar("_R")
_TASK_RUN_STORE_LOCKS_GUARD = threading.Lock()
_TASK_RUN_STORE_LOCKS: dict[str, threading.RLock] = {}


def _task_run_service_locked(
    method: Callable[_P, _R],
) -> Callable[_P, _R]:
    @wraps(method)
    def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        service = args[0]
        with service._lock:  # type: ignore[attr-defined]
            return method(*args, **kwargs)

    return _wrapped


def _task_run_store_lock(path: Path | None) -> threading.RLock:
    if path is None:
        return threading.RLock()
    key = str(path.resolve())
    with _TASK_RUN_STORE_LOCKS_GUARD:
        return _TASK_RUN_STORE_LOCKS.setdefault(key, threading.RLock())


def _runtime_handoff_task_run_transaction(
    method: Callable[_P, _R],
) -> Callable[_P, _R]:
    @wraps(method)
    def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        service = args[0]
        with service.run_service.transaction():  # type: ignore[attr-defined]
            return method(*args, **kwargs)

    return _wrapped
_REMOTE_TO_LOCAL_STATE = {
    "submitted": "queued",
    "created": "queued",
    "validating": "queued",
    "preflight": "queued",
    "queued": "queued",
    "executing": "executing",
    "paused": "paused",
    "resuming": "executing",
    "input_required": "paused",
    "auth_required": "paused",
    "blocked": "blocked",
    "cancelling": "cancel_requested",
    "completed": "completed",
    "failed": "failed",
    "rejected": "blocked",
    "cancelled": "cancelled",
    "shadowed": "shadowed",
}
_UTC_TS_SCALE = 1000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillDefinition:
    """One high-level skill the runtime arbiter is allowed to schedule."""

    name: str
    required_parameters: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    abort_conditions: tuple[str, ...] = ()
    timeout_ms: int = 30000
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required_parameters": list(self.required_parameters),
            "required_capabilities": list(self.required_capabilities),
            "preconditions": list(self.preconditions),
            "success_criteria": list(self.success_criteria),
            "abort_conditions": list(self.abort_conditions),
            "timeout_ms": self.timeout_ms,
            "requires_confirmation": self.requires_confirmation,
        }


class SkillRegistry:
    """Allowlist for machine-readable TaskHandoff steps."""

    def __init__(self, skills: list[SkillDefinition] | None = None) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills or default_skill_definitions():
            self.register(skill)

    def register(self, skill: SkillDefinition) -> None:
        name = str(skill.name).strip()
        if not name:
            raise ValueError("skill name is required")
        self._skills[name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(str(name or "").strip())

    def validate_step(self, step: TaskStep) -> list[str]:
        skill = self.get(step.skill_name)
        if skill is None:
            return [f"unregistered_skill:{step.skill_name}"]
        missing = [
            name for name in skill.required_parameters if step.parameters.get(name) in (None, "")
        ]
        return [f"missing_parameter:{step.skill_name}.{name}" for name in missing]

    def capabilities_for(self, steps: list[TaskStep]) -> list[str]:
        capabilities: list[str] = []
        for step in steps:
            skill = self.get(step.skill_name)
            if skill is None:
                continue
            capabilities.extend(skill.required_capabilities)
        return _unique(capabilities)

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": len(self._skills),
            "skills": [skill.to_dict() for skill in self._skills.values()],
        }


class OperatorPolicyService:
    """Small local RBAC gate for runtime handoff."""

    runtime_submit_roles = frozenset({"operator", "supervisor", "admin"})
    supervisor_roles = frozenset({"supervisor", "admin"})

    def __init__(self, *, require_supervisor_for_high_risk: bool = False) -> None:
        self.require_supervisor_for_high_risk = bool(require_supervisor_for_high_risk)

    def validate_handoff(self, handoff: TaskHandoff) -> list[str]:
        roles = set(_normalize_roles(handoff.operator_roles))
        failed: list[str] = []
        if roles.isdisjoint(self.runtime_submit_roles):
            failed.append("operator_not_authorized")
        if handoff.risk_level == "critical" and roles.isdisjoint(self.supervisor_roles):
            failed.append("supervisor_confirmation_required")
        if (
            self.require_supervisor_for_high_risk
            and handoff.risk_level == "high"
            and roles.isdisjoint(self.supervisor_roles)
        ):
            failed.append("supervisor_confirmation_required")
        return failed

    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime_submit_roles": sorted(self.runtime_submit_roles),
            "supervisor_roles": sorted(self.supervisor_roles),
            "require_supervisor_for_high_risk": self.require_supervisor_for_high_risk,
        }


@dataclass(frozen=True)
class TaskStep:
    """One structured, high-level runtime step."""

    step_id: str
    sequence: int
    skill_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    abort_conditions: list[str] = field(default_factory=list)
    timeout_ms: int = 30000
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskHandoff:
    """Stable handoff contract between askme and the runtime arbiter."""

    handoff_id: str
    plan_id: str
    session_id: str
    operator_id: str
    intent: str
    task_type: str
    target_area: str | None
    target_object: str | None
    constraints: list[str]
    steps: list[TaskStep]
    risk_level: str
    required_capabilities: list[str]
    missing_info: list[str]
    confirmation_status: str
    world_state_snapshot_id: str
    safety_notes: list[str]
    created_at: float
    expires_at: float
    planner_version: str
    source_plan: dict[str, Any] = field(default_factory=dict)
    world_state_snapshot: dict[str, Any] = field(default_factory=dict)
    operator_roles: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(
        cls,
        plan: dict[str, Any],
        *,
        world_state_snapshot: dict[str, Any],
        skill_registry: SkillRegistry,
        default_operator_id: str = "askme.operator",
        planner_version: str = "askme-cognition-v1",
        ttl_s: float = 300.0,
        now: float | None = None,
    ) -> TaskHandoff:
        current = now if now is not None else time.time()
        mission = _mission_from_plan(plan)
        task_type = str(mission.get("mission_type") or plan.get("intent") or "operator_assist")
        intent = str(plan.get("intent") or task_type)
        steps = _task_steps_for_plan(plan, task_type=task_type)
        operator_id = (
            _first_text(
                plan.get("operator_id"),
                plan.get("session", {}).get("operator_id")
                if isinstance(plan.get("session"), dict)
                else None,
                mission.get("operator_id"),
                default_operator_id,
            )
            or default_operator_id
        )
        snapshot_id = _world_snapshot_id(world_state_snapshot)
        confirmation_status = _confirmation_status(plan)
        return cls(
            handoff_id=f"handoff-{uuid4().hex[:12]}",
            plan_id=str(plan.get("plan_id") or ""),
            session_id=str(plan.get("planning_session_id") or ""),
            operator_id=operator_id,
            intent=intent,
            task_type=task_type,
            target_area=_target_area(plan, mission),
            target_object=_target_object(plan, mission),
            constraints=[
                str(item) for item in plan.get("safety_constraints", []) if str(item).strip()
            ],
            steps=steps,
            risk_level=_risk_for(task_type, mission),
            required_capabilities=skill_registry.capabilities_for(steps),
            missing_info=[
                str(item) for item in plan.get("missing_inputs", []) if str(item).strip()
            ],
            confirmation_status=confirmation_status,
            world_state_snapshot_id=snapshot_id,
            safety_notes=[
                str(item) for item in mission.get("safety_notes", []) if str(item).strip()
            ],
            created_at=current,
            expires_at=current + max(1.0, float(ttl_s)),
            planner_version=planner_version,
            source_plan=deepcopy(plan),
            world_state_snapshot=deepcopy(world_state_snapshot),
            operator_roles=_operator_roles(plan),
        )

    @property
    def handoff_ready(self) -> bool:
        return (
            self.confirmation_status == "confirmed"
            and not self.missing_info
            and bool(self.plan_id)
            and bool(self.session_id)
        )

    def validate(self, skill_registry: SkillRegistry, *, now: float | None = None) -> list[str]:
        current = now if now is not None else time.time()
        errors: list[str] = []
        if not self.plan_id:
            errors.append("missing_plan_id")
        if not self.session_id:
            errors.append("missing_session_id")
        if not self.operator_id:
            errors.append("missing_operator_id")
        if self.confirmation_status != "confirmed":
            errors.append("operator_confirmation_required")
        if self.missing_info:
            errors.append("missing_info:" + ",".join(self.missing_info))
        if current > self.expires_at:
            errors.append("plan_expired")
        if not self.steps:
            errors.append("no_steps")
        for step in self.steps:
            errors.extend(skill_registry.validate_step(step))
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        payload["handoff_ready"] = self.handoff_ready
        return payload


@dataclass(frozen=True)
class ActivePerceptionRequest:
    """Structured request for fresh runtime/perception facts.

    askme produces this request when preflight cannot safely trust current
    world facts. It is an observation request, not a hardware command.
    """

    request_id: str
    reason: str
    required_facts: list[str]
    suggested_sources: list[str] = field(default_factory=list)
    target_area: str | None = None
    target_object: str | None = None
    priority: str = "normal"
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyAssessment:
    """Structured preflight result for one TaskHandoff."""

    assessment_id: str
    handoff_id: str
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    perception_requests: list[ActivePerceptionRequest] = field(default_factory=list)
    required_operator_confirmation: bool = False
    recommended_fix: str = ""
    profile_decision: str = "fake"
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeEvent:
    """One auditable runtime timeline event."""

    event_id: str
    run_id: str
    event_type: str
    state: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillResult:
    """Structured output from one high-level skill execution."""

    result_id: str
    run_id: str
    step_id: str
    sequence: int
    skill_name: str
    status: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplanProposal:
    """Operator-confirmed recommendation for recovering a blocked TaskRun."""

    proposal_id: str
    run_id: str
    source: str
    reason: str
    recommended_action: str
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    perception_requests: list[dict[str, Any]] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    operator_confirmation_required: bool = True
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRun:
    """Execution record produced by the local runtime arbiter."""

    run_id: str
    handoff: TaskHandoff
    profile: str = "fake"
    current_state: str = "created"
    current_step_index: int = 0
    started_at: float | None = None
    ended_at: float | None = None
    assigned_robot_id: str | None = None
    runtime_events: list[RuntimeEvent] = field(default_factory=list)
    safety_assessments: list[SafetyAssessment] = field(default_factory=list)
    skill_results: list[SkillResult] = field(default_factory=list)
    replan_proposals: list[ReplanProposal] = field(default_factory=list)
    operator_actions: list[dict[str, Any]] = field(default_factory=list)
    result_summary: str = ""
    report: dict[str, Any] | None = None
    shadow_plan: dict[str, Any] | None = None
    sim_state: dict[str, Any] | None = None
    remote_task_id: str | None = None
    remote_status: str = ""
    remote_status_cursor: str = ""
    external_idempotency_key: str = ""
    remote_observed_at: float | None = None
    last_poll_error_code: str = ""
    processed_remote_update_ids: list[str] = field(default_factory=list)
    notification_delivery_receipts: dict[str, str] = field(default_factory=dict)
    approval_request: dict[str, Any] = field(default_factory=dict)
    deferred_cancel_request: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.current_state in _TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "handoff_id": self.handoff.handoff_id,
            "plan_id": self.handoff.plan_id,
            "session_id": self.handoff.session_id,
            "profile": self.profile,
            "task_type": self.handoff.task_type,
            "target_area": self.handoff.target_area,
            "target_object": self.handoff.target_object,
            "risk_level": self.handoff.risk_level,
            "current_state": self.current_state,
            "current_step_index": self.current_step_index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "assigned_robot_id": self.assigned_robot_id,
            "handoff": self.handoff.to_dict(),
            "runtime_events": [event.to_dict() for event in self.runtime_events],
            "safety_assessments": [item.to_dict() for item in self.safety_assessments],
            "skill_results": [item.to_dict() for item in self.skill_results],
            "replan_proposals": [item.to_dict() for item in self.replan_proposals],
            "operator_actions": [dict(item) for item in self.operator_actions],
            "result_summary": self.result_summary,
            "report": self.report,
            "shadow_plan": self.shadow_plan,
            "sim_state": self.sim_state,
            "remote_task_id": self.remote_task_id,
            "remote_status": self.remote_status,
            "remote_status_cursor": self.remote_status_cursor,
            "external_idempotency_key": self.external_idempotency_key,
            "remote_observed_at": self.remote_observed_at,
            "last_poll_error_code": self.last_poll_error_code,
            "processed_remote_update_ids": list(self.processed_remote_update_ids),
            "notification_delivery_receipts": dict(self.notification_delivery_receipts),
            "approval_request": dict(self.approval_request),
            "deferred_cancel_request": dict(self.deferred_cancel_request),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class TaskRunStoreConfig:
    """Opt-in persistent TaskRun state store.

    Audit JSONL is good for forensics; this store is the product-facing
    recoverable state used by Dashboard/API after a service restart.
    """

    enabled: bool = False
    path: str | Path | None = None
    swallow_errors: bool = True

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> TaskRunStoreConfig:
        if not isinstance(config, dict):
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            path=config.get("path") or config.get("json_path"),
            swallow_errors=bool(config.get("swallow_errors", True)),
        )


class TaskRunStore:
    """Recoverable JSON state for TaskRun records."""

    def __init__(self, config: TaskRunStoreConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, TaskRunStoreConfig):
            self.config = config
        else:
            self.config = TaskRunStoreConfig.from_mapping(config)
        self.path = Path(self.config.path).expanduser() if self.config.path else None
        self._lock = _task_run_store_lock(self.path)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.path is not None)

    def load_runs(self) -> list[TaskRun]:
        if not self.enabled or self.path is None or not self.path.exists():
            return []
        try:
            with self._lock:
                with self.path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            raw_runs = payload.get("runs", payload) if isinstance(payload, dict) else payload
            if not isinstance(raw_runs, list):
                return []
            runs: list[TaskRun] = []
            for item in raw_runs:
                if isinstance(item, dict):
                    runs.append(_task_run_from_dict(item))
            return runs
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            if not self.config.swallow_errors:
                raise
            logger.exception("TaskRun store load failed for %s", self.path)
            return []

    def save_runs(self, runs: list[TaskRun]) -> None:
        if not self.enabled or self.path is None:
            return
        tmp_path: Path | None = None
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 1,
                    "updated_at": time.time(),
                    "runs": [run.to_dict() for run in runs],
                }
                tmp_path = self.path.with_name(
                    f".{self.path.name}.{uuid4().hex}.tmp"
                )
                with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self.path)
        except OSError:
            if not self.config.swallow_errors:
                raise
            logger.exception("TaskRun store save failed for %s", self.path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.warning("TaskRun temp cleanup failed for %s", tmp_path)


def _task_run_from_dict(payload: dict[str, Any]) -> TaskRun:
    handoff_payload = payload.get("handoff", {})
    if not isinstance(handoff_payload, dict):
        handoff_payload = {}
    handoff = _task_handoff_from_dict(handoff_payload)
    run = TaskRun(
        run_id=str(payload.get("run_id") or f"run-{uuid4().hex[:12]}"),
        handoff=handoff,
        profile=_normalize_runtime_profile(str(payload.get("profile") or "fake")),
        current_state=str(payload.get("current_state") or "created"),
        current_step_index=int(payload.get("current_step_index") or 0),
        started_at=_optional_float(payload.get("started_at")),
        ended_at=_optional_float(payload.get("ended_at")),
        assigned_robot_id=(
            str(payload.get("assigned_robot_id"))
            if payload.get("assigned_robot_id") is not None
            else None
        ),
        runtime_events=[
            _runtime_event_from_dict(item) for item in _dict_items(payload.get("runtime_events"))
        ],
        safety_assessments=[
            _safety_assessment_from_dict(item)
            for item in _dict_items(payload.get("safety_assessments"))
        ],
        skill_results=[
            _skill_result_from_dict(item) for item in _dict_items(payload.get("skill_results"))
        ],
        replan_proposals=[
            _replan_proposal_from_dict(item)
            for item in _dict_items(payload.get("replan_proposals"))
        ],
        operator_actions=[dict(item) for item in _dict_items(payload.get("operator_actions"))],
        result_summary=str(payload.get("result_summary") or ""),
        report=dict(payload["report"]) if isinstance(payload.get("report"), dict) else None,
        shadow_plan=(
            dict(payload["shadow_plan"]) if isinstance(payload.get("shadow_plan"), dict) else None
        ),
        sim_state=dict(payload["sim_state"])
        if isinstance(payload.get("sim_state"), dict)
        else None,
        remote_task_id=(
            str(payload.get("remote_task_id"))
            if payload.get("remote_task_id") is not None
            else None
        ),
        remote_status=str(payload.get("remote_status") or ""),
        remote_status_cursor=str(payload.get("remote_status_cursor") or ""),
        external_idempotency_key=str(payload.get("external_idempotency_key") or ""),
        remote_observed_at=_optional_float(payload.get("remote_observed_at")),
        last_poll_error_code=str(payload.get("last_poll_error_code") or ""),
        processed_remote_update_ids=[
            str(item)
            for item in payload.get("processed_remote_update_ids", [])
            if str(item).strip()
        ][-_REMOTE_UPDATE_ID_LIMIT:],
        notification_delivery_receipts=_notification_delivery_receipts_from_dict(
            payload.get("notification_delivery_receipts")
        ),
        approval_request=(
            dict(payload["approval_request"])
            if isinstance(payload.get("approval_request"), dict)
            else {}
        ),
        deferred_cancel_request=(
            dict(payload["deferred_cancel_request"])
            if isinstance(payload.get("deferred_cancel_request"), dict)
            else {}
        ),
    )
    return run


def _notification_delivery_receipts_from_dict(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    receipts: dict[str, str] = {}
    for event_id, status in payload.items():
        normalized_event_id = str(event_id or "").strip()
        normalized_status = str(status or "").strip().lower()
        if normalized_event_id and normalized_status in _NOTIFICATION_DELIVERY_STATES:
            receipts[normalized_event_id] = normalized_status
    if len(receipts) <= _NOTIFICATION_DELIVERY_RECEIPT_LIMIT:
        return receipts
    return dict(list(receipts.items())[-_NOTIFICATION_DELIVERY_RECEIPT_LIMIT:])


def _task_handoff_from_dict(payload: dict[str, Any]) -> TaskHandoff:
    return TaskHandoff(
        handoff_id=str(payload.get("handoff_id") or f"handoff-{uuid4().hex[:12]}"),
        plan_id=str(payload.get("plan_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        operator_id=str(payload.get("operator_id") or "askme.operator"),
        intent=str(payload.get("intent") or ""),
        task_type=str(payload.get("task_type") or "operator_assist"),
        target_area=(
            str(payload.get("target_area")) if payload.get("target_area") is not None else None
        ),
        target_object=(
            str(payload.get("target_object")) if payload.get("target_object") is not None else None
        ),
        constraints=[str(item) for item in payload.get("constraints", [])],
        steps=[_task_step_from_dict(item) for item in _dict_items(payload.get("steps"))],
        risk_level=str(payload.get("risk_level") or "low"),
        required_capabilities=[str(item) for item in payload.get("required_capabilities", [])],
        missing_info=[str(item) for item in payload.get("missing_info", [])],
        confirmation_status=str(payload.get("confirmation_status") or "unknown"),
        world_state_snapshot_id=str(payload.get("world_state_snapshot_id") or ""),
        safety_notes=[str(item) for item in payload.get("safety_notes", [])],
        created_at=float(payload.get("created_at") or time.time()),
        expires_at=float(payload.get("expires_at") or (time.time() + 60.0)),
        planner_version=str(payload.get("planner_version") or "restored"),
        source_plan=dict(payload.get("source_plan") or {}),
        world_state_snapshot=dict(payload.get("world_state_snapshot") or {}),
        operator_roles=[str(item) for item in payload.get("operator_roles", [])],
    )


def _task_step_from_dict(payload: dict[str, Any]) -> TaskStep:
    return TaskStep(
        step_id=str(payload.get("step_id") or f"step-{uuid4().hex[:8]}"),
        sequence=int(payload.get("sequence") or 0),
        skill_name=str(payload.get("skill_name") or "unknown"),
        parameters=dict(payload.get("parameters") or {}),
        preconditions=[str(item) for item in payload.get("preconditions", [])],
        success_criteria=[str(item) for item in payload.get("success_criteria", [])],
        abort_conditions=[str(item) for item in payload.get("abort_conditions", [])],
        timeout_ms=int(payload.get("timeout_ms") or 30000),
        requires_confirmation=bool(payload.get("requires_confirmation", False)),
    )


def _safety_assessment_from_dict(payload: dict[str, Any]) -> SafetyAssessment:
    return SafetyAssessment(
        assessment_id=str(payload.get("assessment_id") or f"safety-{uuid4().hex[:12]}"),
        handoff_id=str(payload.get("handoff_id") or ""),
        passed=bool(payload.get("passed", False)),
        failed_checks=[str(item) for item in payload.get("failed_checks", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
        perception_requests=[
            _active_perception_request_from_dict(item)
            for item in _dict_items(payload.get("perception_requests"))
        ],
        required_operator_confirmation=bool(payload.get("required_operator_confirmation", False)),
        recommended_fix=str(payload.get("recommended_fix") or ""),
        profile_decision=str(payload.get("profile_decision") or "fake"),
        checked_at=float(payload.get("checked_at") or time.time()),
    )


def _active_perception_request_from_dict(payload: dict[str, Any]) -> ActivePerceptionRequest:
    return ActivePerceptionRequest(
        request_id=str(payload.get("request_id") or f"perception-{uuid4().hex[:12]}"),
        reason=str(payload.get("reason") or "refresh_world_state"),
        required_facts=[str(item) for item in payload.get("required_facts", [])],
        suggested_sources=[str(item) for item in payload.get("suggested_sources", [])],
        target_area=(
            str(payload.get("target_area")) if payload.get("target_area") is not None else None
        ),
        target_object=(
            str(payload.get("target_object")) if payload.get("target_object") is not None else None
        ),
        priority=str(payload.get("priority") or "normal"),
        created_at=float(payload.get("created_at") or time.time()),
        expires_at=_optional_float(payload.get("expires_at")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _runtime_event_from_dict(payload: dict[str, Any]) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=str(payload.get("event_id") or f"evt-{uuid4().hex[:12]}"),
        run_id=str(payload.get("run_id") or ""),
        event_type=str(payload.get("event_type") or "restored"),
        state=str(payload.get("state") or ""),
        message=str(payload.get("message") or ""),
        payload=dict(payload.get("payload") or {}),
        created_at=float(payload.get("created_at") or time.time()),
    )


def _skill_result_from_dict(payload: dict[str, Any]) -> SkillResult:
    return SkillResult(
        result_id=str(payload.get("result_id") or f"skill-result-{uuid4().hex[:12]}"),
        run_id=str(payload.get("run_id") or ""),
        step_id=str(payload.get("step_id") or ""),
        sequence=int(payload.get("sequence") or 0),
        skill_name=str(payload.get("skill_name") or ""),
        status=str(payload.get("status") or "unknown"),
        observations=[dict(item) for item in _dict_items(payload.get("observations"))],
        artifacts=[dict(item) for item in _dict_items(payload.get("artifacts"))],
        metrics=dict(payload.get("metrics") or {}),
        confidence=float(payload.get("confidence") or 0.0),
        started_at=float(payload.get("started_at") or time.time()),
        ended_at=_optional_float(payload.get("ended_at")),
        error=str(payload.get("error") or ""),
    )


def _replan_proposal_from_dict(payload: dict[str, Any]) -> ReplanProposal:
    return ReplanProposal(
        proposal_id=str(payload.get("proposal_id") or f"replan-{uuid4().hex[:12]}"),
        run_id=str(payload.get("run_id") or ""),
        source=str(payload.get("source") or "restored"),
        reason=str(payload.get("reason") or ""),
        recommended_action=str(payload.get("recommended_action") or ""),
        proposed_actions=[dict(item) for item in _dict_items(payload.get("proposed_actions"))],
        perception_requests=[
            dict(item) for item in _dict_items(payload.get("perception_requests"))
        ],
        safety_notes=[str(item) for item in payload.get("safety_notes", [])],
        operator_confirmation_required=bool(payload.get("operator_confirmation_required", True)),
        status=str(payload.get("status") or "proposed"),
        created_at=float(payload.get("created_at") or time.time()),
        expires_at=_optional_float(payload.get("expires_at")),
    )


def _dict_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _external_evidence_result(
    run: TaskRun,
    payload: dict[str, Any] | None,
    *,
    update_id: str,
    cursor: str,
    remote_status: str,
    observed_at: float,
) -> tuple[SkillResult | None, bool]:
    """Normalize one executor evidence payload into a durable SkillResult."""
    if not isinstance(payload, dict):
        return None, False

    observations = _external_evidence_items(payload, "observation", "observations")
    artifacts = _external_evidence_items(payload, "artifact", "artifacts")
    if not observations and not artifacts:
        return None, False

    observations, observations_updated = _upsert_existing_evidence(
        run,
        observations,
        field_name="observations",
    )
    artifacts, artifacts_updated = _upsert_existing_evidence(
        run,
        artifacts,
        field_name="artifacts",
    )
    evidence_changed = observations_updated or artifacts_updated
    if not observations and not artifacts:
        return None, evidence_changed

    evidence_source = update_id or cursor or uuid4().hex[:12]
    evidence_key = hashlib.sha256(f"{run.run_id}:{evidence_source}".encode()).hexdigest()[:16]
    ended_at = (
        observed_at if _REMOTE_TO_LOCAL_STATE.get(remote_status) in _TERMINAL_STATES else None
    )
    return (
        SkillResult(
            result_id=f"external-result-{evidence_key}",
            run_id=run.run_id,
            step_id=str(payload.get("step_id") or f"external:{evidence_key}"),
            sequence=_external_evidence_sequence(payload.get("sequence"), run),
            skill_name=str(payload.get("skill_name") or "external_executor"),
            status=remote_status,
            observations=observations,
            artifacts=artifacts,
            started_at=observed_at,
            ended_at=ended_at,
        ),
        True,
    )


def _external_evidence_items(
    payload: dict[str, Any], singular_key: str, plural_key: str
) -> list[dict[str, Any]]:
    candidates: list[Any] = [payload.get(singular_key)]
    plural = payload.get(plural_key)
    if isinstance(plural, Sequence) and not isinstance(plural, (str, bytes, bytearray)):
        candidates.extend(plural)
    elif plural is not None:
        candidates.append(plural)

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        item = _external_evidence_copy(candidate)
        identity = _evidence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        items.append(item)
    return items


def _evidence_identity(item: dict[str, Any]) -> str:
    for key in ("artifact_id", "evidence_id", "observation_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return "content:" + json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _upsert_existing_evidence(
    run: TaskRun,
    items: list[dict[str, Any]],
    *,
    field_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    new_items: list[dict[str, Any]] = []
    changed = False
    for item in items:
        identity = _evidence_identity(item)
        matched = False
        for result_index, result in enumerate(run.skill_results):
            existing_items = (
                result.observations if field_name == "observations" else result.artifacts
            )
            for item_index, existing in enumerate(existing_items):
                if _evidence_identity(existing) != identity:
                    continue
                matched = True
                if existing != item:
                    updated_items = list(existing_items)
                    updated_items[item_index] = item
                    run.skill_results[result_index] = (
                        replace(result, observations=updated_items)
                        if field_name == "observations"
                        else replace(result, artifacts=updated_items)
                    )
                    changed = True
                break
            if matched:
                break
        if not matched:
            new_items.append(item)
    return new_items, changed


def _external_evidence_copy(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _external_evidence_value(item) for key, item in value.items()}


def _external_evidence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _external_evidence_copy(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_external_evidence_value(item) for item in value]
    return deepcopy(value)


def _external_evidence_sequence(value: Any, run: TaskRun) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return len(run.skill_results) + 1


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SafetyPreflightService:
    """Local preflight checks before fake runtime execution."""

    def __init__(
        self,
        *,
        max_world_state_age_s: float = 30.0,
        min_battery_percent: float = 20.0,
        dog_safety_client: Any | None = None,
        require_dog_safety: bool = False,
        operator_policy: OperatorPolicyService | None = None,
    ) -> None:
        self.max_world_state_age_s = max(1.0, float(max_world_state_age_s))
        self.min_battery_percent = max(0.0, float(min_battery_percent))
        self.dog_safety_client = dog_safety_client
        self.require_dog_safety = bool(require_dog_safety)
        self.operator_policy = operator_policy or OperatorPolicyService()

    def assess(
        self,
        handoff: TaskHandoff,
        *,
        skill_registry: SkillRegistry,
        profile: str = "fake",
        now: float | None = None,
    ) -> SafetyAssessment:
        current = now if now is not None else time.time()
        failed = handoff.validate(skill_registry, now=current)
        failed.extend(self.operator_policy.validate_handoff(handoff))
        warnings: list[str] = []
        perception_requests: list[ActivePerceptionRequest] = []
        snapshot = handoff.world_state_snapshot
        snapshot_age = current - float(snapshot.get("updated_at", current) or current)
        if snapshot_age > self.max_world_state_age_s:
            failed.append("world_state_stale")
            perception_requests.append(
                _perception_request(
                    "refresh_world_state",
                    ["robot", "environment", "map", "scene"],
                    suggested_sources=["runtime", "nav", "perception"],
                    handoff=handoff,
                    priority="high",
                )
            )
        stale_keys = [str(item) for item in snapshot.get("stale_keys", [])]
        if stale_keys:
            warnings.append("stale_facts:" + ",".join(stale_keys[:5]))
            perception_requests.append(
                _perception_request(
                    "refresh_stale_facts",
                    stale_keys[:8],
                    suggested_sources=["runtime", "nav", "perception"],
                    handoff=handoff,
                    priority="normal",
                )
            )

        robot = snapshot.get("robot", {}) if isinstance(snapshot.get("robot"), dict) else {}
        if robot.get("estop_active") is True:
            failed.append("estop_active")
        if robot.get("online") is False:
            failed.append("robot_offline")
        if "online" not in robot:
            warnings.append("robot_online_unknown_fake_profile")
            perception_requests.append(
                _perception_request(
                    "refresh_robot_status",
                    ["robot.online"],
                    suggested_sources=["runtime"],
                    handoff=handoff,
                )
            )
        battery = _float_or_none(robot.get("battery_percent") or robot.get("battery"))
        if battery is not None and battery < self.min_battery_percent:
            failed.append("battery_below_threshold")
        if battery is None:
            warnings.append("battery_unknown_fake_profile")
            perception_requests.append(
                _perception_request(
                    "refresh_robot_status",
                    ["robot.battery_percent"],
                    suggested_sources=["runtime"],
                    handoff=handoff,
                )
            )
        if robot.get("localized") is False:
            failed.append("localization_unavailable")
            perception_requests.append(
                _perception_request(
                    "refresh_localization",
                    ["robot.localized", "map.localized", "map.localization_quality"],
                    suggested_sources=["nav"],
                    handoff=handoff,
                    priority="high",
                )
            )

        if handoff.task_type in {"inspection_patrol", "navigate_to"} and not handoff.target_area:
            failed.append("target_area_required")
        if handoff.risk_level in {"high", "critical"} and not handoff.operator_id:
            failed.append("operator_identity_required")

        self._assess_world_model(handoff, snapshot, failed, warnings, perception_requests)
        self._assess_dog_safety(failed, warnings)

        return SafetyAssessment(
            assessment_id=f"preflight-{uuid4().hex[:12]}",
            handoff_id=handoff.handoff_id,
            passed=not failed,
            failed_checks=_unique(failed),
            warnings=_unique(warnings),
            perception_requests=_unique_perception_requests(perception_requests),
            required_operator_confirmation=handoff.confirmation_status != "confirmed",
            recommended_fix=_recommended_fix(failed) if failed else "",
            profile_decision=_normalize_runtime_profile(profile),
            checked_at=current,
        )

    def _assess_dog_safety(self, failed: list[str], warnings: list[str]) -> None:
        client = self.dog_safety_client
        if client is None:
            if self.require_dog_safety:
                failed.append("dog_safety_unavailable")
            return

        configured = _dog_safety_is_configured(client)
        if configured is False:
            if self.require_dog_safety:
                failed.append("dog_safety_unconfigured")
            else:
                warnings.append("dog_safety_unconfigured")
            return

        active = _dog_safety_estop_active(client)
        if active is True:
            failed.append("dog_safety_estop_active")
        elif active is None:
            if self.require_dog_safety:
                failed.append("dog_safety_unavailable")
            else:
                warnings.append("dog_safety_estop_unknown")

    def _assess_world_model(
        self,
        handoff: TaskHandoff,
        snapshot: dict[str, Any],
        failed: list[str],
        warnings: list[str],
        perception_requests: list[ActivePerceptionRequest],
    ) -> None:
        environment = snapshot.get("environment", {})
        if not isinstance(environment, dict):
            environment = {}
        areas = environment.get("areas")
        if handoff.target_area and isinstance(areas, list):
            area = _find_world_item(areas, "area_id", handoff.target_area)
            if area is None:
                failed.append("target_area_unknown")
                perception_requests.append(
                    _perception_request(
                        "observe_or_register_area",
                        ["environment.areas"],
                        suggested_sources=["site_catalog", "perception"],
                        handoff=handoff,
                        priority="high",
                    )
                )
            else:
                if _area_blocked(area):
                    failed.append("target_area_blocked")
                _assess_area_map(area, snapshot, failed, warnings, perception_requests, handoff)
        elif handoff.target_area and handoff.task_type in {"inspection_patrol", "navigate_to"}:
            warnings.append("area_catalog_unavailable")
            perception_requests.append(
                _perception_request(
                    "load_area_catalog",
                    ["environment.areas"],
                    suggested_sources=["site_catalog"],
                    handoff=handoff,
                    priority="normal",
                )
            )

        devices = environment.get("devices")
        if handoff.target_object and isinstance(devices, list):
            device = _find_world_item(devices, "device_id", handoff.target_object)
            if device is None:
                failed.append("target_device_unknown")
                perception_requests.append(
                    _perception_request(
                        "observe_or_register_device",
                        ["environment.devices", "scene.objects"],
                        suggested_sources=["site_catalog", "perception"],
                        handoff=handoff,
                        priority="high",
                    )
                )

        map_state = snapshot.get("map", {})
        if isinstance(map_state, dict):
            if map_state.get("localized") is False:
                failed.append("map_localization_unavailable")
                perception_requests.append(
                    _perception_request(
                        "refresh_localization",
                        ["map.localized", "map.localization_quality"],
                        suggested_sources=["nav"],
                        handoff=handoff,
                        priority="high",
                    )
                )
            quality = _float_or_none(map_state.get("localization_quality"))
            if quality is not None and quality < 0.5:
                failed.append("localization_quality_low")
                perception_requests.append(
                    _perception_request(
                        "refresh_localization",
                        ["map.localization_quality"],
                        suggested_sources=["nav"],
                        handoff=handoff,
                        priority="high",
                    )
                )
        else:
            perception_requests.append(
                _perception_request(
                    "refresh_map_state",
                    ["map.current_id", "map.current_version", "map.localized"],
                    suggested_sources=["nav"],
                    handoff=handoff,
                    priority="normal",
                )
            )


class TaskReportService:
    """Build a small report from a TaskRun timeline."""

    def build_report(self, run: TaskRun) -> dict[str, Any]:
        completed_steps = [
            event.payload.get("skill_name")
            for event in run.runtime_events
            if event.event_type == "step_completed"
        ]
        failed = [
            event.to_dict()
            for event in run.runtime_events
            if event.event_type in {"task_failed", "task_blocked", "task_cancelled"}
        ]
        skill_results = [item.to_dict() for item in run.skill_results]
        return {
            "report_id": f"report-{run.run_id}",
            "run_id": run.run_id,
            "plan_id": run.handoff.plan_id,
            "profile": run.profile,
            "task_type": run.handoff.task_type,
            "target_area": run.handoff.target_area,
            "status": run.current_state,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "completed_steps": [str(item) for item in completed_steps if item],
            "skill_results": skill_results,
            "observations": [
                observation
                for result in skill_results
                for observation in result.get("observations", [])
            ],
            "artifacts": [
                artifact for result in skill_results for artifact in result.get("artifacts", [])
            ],
            "replan_proposals": [item.to_dict() for item in run.replan_proposals],
            "event_count": len(run.runtime_events),
            "issues": failed,
            "shadow_plan": run.shadow_plan,
            "sim_state": run.sim_state,
            "summary": run.result_summary or _run_summary(run),
        }


class TaskRunService:
    """Task run state machine with optional recoverable product state."""

    def __init__(
        self,
        *,
        report_service: TaskReportService | None = None,
        event_sink: Callable[[RuntimeEvent], None] | None = None,
        audit_log: RuntimeAuditLog | None = None,
        store: TaskRunStore | TaskRunStoreConfig | dict[str, Any] | None = None,
        max_runs: int = 50,
    ) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, TaskRun] = {}
        self._report_service = report_service or TaskReportService()
        self._event_sink = event_sink
        self._event_observers: list[Callable[[RuntimeEvent], None]] = []
        self._audit_log = audit_log or RuntimeAuditLog()
        self._store = store if isinstance(store, TaskRunStore) else TaskRunStore(store)
        self.max_runs = max(1, int(max_runs))
        for run in self._store.load_runs():
            self._runs[run.run_id] = run
        self._trim_runs(persist=False)

    @property
    def durable_store_ready(self) -> bool:
        """Return whether TaskRun writes are configured to fail closed."""

        return bool(self._store.enabled and not self._store.config.swallow_errors)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize one TaskRun read-modify-persist sequence in this process."""

        with self._lock:
            yield

    @_task_run_service_locked
    def get_or_build_report(self, run_id: str) -> dict[str, Any] | None:
        """Return one report without allowing a newer run state to be overwritten."""

        run = self._runs.get(run_id)
        if run is None:
            return None
        if run.report is None:
            run.report = self._report_service.build_report(run)
            self._persist_runs()
        return run.report

    @_task_run_service_locked
    def create(self, handoff: TaskHandoff, *, profile: str = "fake") -> TaskRun:
        run = TaskRun(
            run_id=f"run-{uuid4().hex[:12]}",
            handoff=handoff,
            profile=_normalize_runtime_profile(profile),
            assigned_robot_id=_robot_id_from_handoff(handoff),
        )
        self._runs[run.run_id] = run
        self._trim_runs()
        self.emit(
            run,
            "handoff_received",
            "created",
            f"Task handoff received by {run.profile} runtime arbiter.",
            {"handoff_id": handoff.handoff_id, "plan_id": handoff.plan_id},
        )
        return run

    @_task_run_service_locked
    def subscribe_events(self, observer: Callable[[RuntimeEvent], None]) -> Callable[[], None]:
        """Observe committed events without joining the authoritative write path."""
        if not callable(observer):
            raise TypeError("observer must be callable")
        self._event_observers.append(observer)
        subscribed = True

        def unsubscribe() -> None:
            nonlocal subscribed
            with self._lock:
                if not subscribed:
                    return
                subscribed = False
                try:
                    self._event_observers.remove(observer)
                except ValueError:
                    pass

        return unsubscribe

    @_task_run_service_locked
    def add_safety_assessment(self, run: TaskRun, assessment: SafetyAssessment) -> None:
        run.safety_assessments.append(assessment)
        self.emit(
            run,
            "preflight_passed" if assessment.passed else "preflight_failed",
            run.current_state,
            "Safety preflight passed." if assessment.passed else "Safety preflight failed.",
            {"assessment": assessment.to_dict()},
        )
        for request in assessment.perception_requests:
            self.emit(
                run,
                "perception_requested",
                run.current_state,
                f"Perception requested: {request.reason}.",
                {"perception_request": request.to_dict()},
            )

    @_task_run_service_locked
    def record_skill_result(
        self,
        run: TaskRun,
        step: TaskStep,
        *,
        status: str = "completed",
        observations: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
        confidence: float = 1.0,
        error: str = "",
    ) -> SkillResult:
        ended_at = time.time()
        result = SkillResult(
            result_id=f"skill-result-{uuid4().hex[:12]}",
            run_id=run.run_id,
            step_id=step.step_id,
            sequence=step.sequence,
            skill_name=step.skill_name,
            status=str(status or "completed"),
            observations=[dict(item) for item in observations or [] if isinstance(item, dict)],
            artifacts=[dict(item) for item in artifacts or [] if isinstance(item, dict)],
            metrics=dict(metrics or {}),
            confidence=min(max(float(confidence), 0.0), 1.0),
            started_at=ended_at,
            ended_at=ended_at,
            error=str(error or ""),
        )
        run.skill_results.append(result)
        self.emit(
            run,
            "skill_result_recorded",
            run.current_state,
            f"Skill result recorded for {step.skill_name}.",
            {"skill_result": result.to_dict()},
        )
        return result

    @_task_run_service_locked
    def record_replan_proposal(
        self,
        run: TaskRun,
        assessment: SafetyAssessment,
        *,
        source: str = "preflight",
    ) -> ReplanProposal:
        proposal = _replan_proposal_for(run, assessment, source=source)
        run.replan_proposals.append(proposal)
        self.emit(
            run,
            "replan_proposed",
            run.current_state,
            f"Replan proposed: {proposal.recommended_action}.",
            {"replan_proposal": proposal.to_dict()},
        )
        return proposal

    @_task_run_service_locked
    def transition(
        self,
        run: TaskRun,
        state: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        if run.terminal and state not in _TERMINAL_STATES:
            raise RuntimeError(f"cannot transition terminal run {run.run_id}")
        run.current_state = state
        if state == "executing" and run.started_at is None:
            run.started_at = time.time()
        if state in _TERMINAL_STATES:
            run.ended_at = time.time()
        terminal_transition = state in _TERMINAL_STATES
        event = self.emit(
            run,
            event_type,
            state,
            message,
            payload or {},
            _notify_observers=not terminal_transition,
        )
        if state in _TERMINAL_STATES:
            run.result_summary = run.result_summary or _run_summary(run)
            run.report = self._report_service.build_report(run)
            self._audit_log.append_terminal_snapshot(run)
            self._persist_runs()
            self._notify_event_observers(event)
        return event

    @_task_run_service_locked
    def emit(
        self,
        run: TaskRun,
        event_type: str,
        state: str,
        message: str,
        payload: dict[str, Any] | None = None,
        *,
        _notify_observers: bool = True,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_id=f"evt-{uuid4().hex[:12]}",
            run_id=run.run_id,
            event_type=event_type,
            state=state,
            message=message,
            payload=dict(payload or {}),
        )
        run.runtime_events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)
        self._audit_log.append_event(event)
        self._persist_runs()
        if _notify_observers:
            self._notify_event_observers(event)
        return event

    def _notify_event_observers(self, event: RuntimeEvent) -> None:
        for observer in tuple(self._event_observers):
            try:
                observer(event)
            except Exception:
                logger.exception("TaskRun event observer failed for %s", event.event_id)

    @_task_run_service_locked
    def bind_external_submission(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        external_idempotency_key: str = "",
        remote_status: str = "submitted",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        """Bind one accepted external submission to its local TaskRun."""
        run = self.require(run_id)
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return self._remote_update_rejected(run, "external_profile_required")
        if run.terminal:
            return self._remote_update_rejected(run, "run_already_terminal")
        normalized_remote_id = str(remote_task_id or "").strip()
        if not normalized_remote_id:
            return self._remote_update_rejected(run, "remote_task_id_required")
        if run.remote_task_id and run.remote_task_id != normalized_remote_id:
            return self._remote_update_rejected(run, "remote_task_id_mismatch")
        normalized_idempotency_key = str(
            external_idempotency_key or run.external_idempotency_key or run.run_id
        ).strip()
        if (
            run.external_idempotency_key
            and run.external_idempotency_key != normalized_idempotency_key
        ):
            return self._remote_update_rejected(run, "external_idempotency_key_mismatch")
        normalized_status = _normalize_remote_status(remote_status)
        if normalized_status not in _REMOTE_STATUS_TRANSITIONS[""]:
            return self._remote_update_rejected(run, "remote_status_invalid")
        run.remote_task_id = normalized_remote_id
        run.external_idempotency_key = normalized_idempotency_key
        run.remote_status = normalized_status
        run.remote_observed_at = float(observed_at if observed_at is not None else time.time())
        run.last_poll_error_code = ""
        local_state = "queued" if run.current_state == "submission_unknown" else run.current_state
        event = self.transition(
            run,
            local_state,
            "external_submission_bound",
            "External runtime submission accepted.",
            {
                "remote_task_id": run.remote_task_id,
                "remote_status": run.remote_status,
                "external_idempotency_key": run.external_idempotency_key,
            },
        )
        return {"handled": True, "event": event.to_dict(), "run": run.to_dict()}

    @_task_run_service_locked
    def prepare_external_submission(
        self,
        run_id: str,
        *,
        external_idempotency_key: str,
    ) -> dict[str, Any]:
        """Persist the replay key before the first external side effect."""

        run = self.require(run_id)
        normalized = str(external_idempotency_key or "").strip()
        if not normalized:
            raise ValueError("external_idempotency_key is required")
        if run.external_idempotency_key and run.external_idempotency_key != normalized:
            return self._remote_update_rejected(run, "external_idempotency_key_mismatch")
        run.external_idempotency_key = normalized
        self._persist_runs()
        return {"handled": True, "run": run.to_dict()}

    @_task_run_service_locked
    def refresh_prepared_handoff_snapshot(
        self,
        run_id: str,
        *,
        world_state_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.current_state != "confirmed":
            return self._remote_update_rejected(run, "prepared_run_not_confirmed")
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return self._remote_update_rejected(run, "external_profile_required")
        snapshot = deepcopy(world_state_snapshot)
        run.handoff = replace(
            run.handoff,
            world_state_snapshot_id=_world_snapshot_id(snapshot),
            world_state_snapshot=snapshot,
        )
        self._persist_runs()
        return {"handled": True, "run": run.to_dict()}

    @_task_run_service_locked
    def set_deferred_cancel_request(
        self,
        run_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        run = self.require(run_id)
        run.deferred_cancel_request = deepcopy(request)
        event = self.emit(
            run,
            "external_cancel_deferred",
            run.current_state,
            "External cancellation deferred until submission identity is reconciled.",
        )
        return {"handled": True, "event": event.to_dict(), "run": run.to_dict()}

    @_task_run_service_locked
    def clear_deferred_cancel_request(self, run_id: str) -> dict[str, Any]:
        run = self.require(run_id)
        if run.deferred_cancel_request:
            run.deferred_cancel_request = {}
            self._persist_runs()
        return {"handled": True, "run": run.to_dict()}

    @_task_run_service_locked
    def mark_external_projection_unknown(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        external_idempotency_key: str,
        error_code: str,
    ) -> dict[str, Any]:
        """Keep a remotely accepted task reconcilable after local commit failure."""

        run = self.require(run_id)
        if run.terminal:
            return {"handled": True, "run": run.to_dict()}
        remote_id = str(remote_task_id or "").strip()
        if remote_id and (not run.remote_task_id or run.remote_task_id == remote_id):
            run.remote_task_id = remote_id
        run.external_idempotency_key = str(
            external_idempotency_key or run.external_idempotency_key or run.run_id
        ).strip()
        if not run.remote_status:
            run.remote_status = "submitted"
        run.current_state = "submission_unknown"
        run.last_poll_error_code = str(error_code or "external_projection_commit_unknown")
        try:
            event = self.emit(
                run,
                "external_projection_commit_unknown",
                run.current_state,
                "External submission was accepted but local projection commit is uncertain.",
                {
                    "error_code": run.last_poll_error_code,
                    "remote_task_id": run.remote_task_id,
                    "external_idempotency_key": run.external_idempotency_key,
                },
            )
        except OSError:
            logger.exception(
                "TaskRun projection uncertainty could not be persisted for %s",
                run.run_id,
            )
            return {"handled": False, "run": run.to_dict()}
        return {"handled": True, "event": event.to_dict(), "run": run.to_dict()}

    @_task_run_service_locked
    def apply_external_update(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        remote_status: str,
        update_id: str = "",
        cursor: str | int = "",
        payload: dict[str, Any] | None = None,
        result_summary: str = "",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        """Project one deduplicated, monotonic external status update locally."""
        run = self.require(run_id)
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return self._remote_update_rejected(run, "external_profile_required")
        normalized_remote_id = str(remote_task_id or "").strip()
        if not run.remote_task_id or normalized_remote_id != run.remote_task_id:
            return self._remote_update_rejected(run, "remote_task_id_mismatch")
        normalized_update_id = str(update_id or "").strip()
        if normalized_update_id and normalized_update_id in run.processed_remote_update_ids:
            return self._remote_update_rejected(run, "remote_update_duplicate")
        normalized_cursor = str(cursor or "").strip()
        if _remote_cursor_is_out_of_order(normalized_cursor, run.remote_status_cursor):
            return self._remote_update_rejected(run, "remote_update_out_of_order")
        normalized_status = _normalize_remote_status(remote_status)
        allowed = _REMOTE_STATUS_TRANSITIONS.get(run.remote_status, frozenset())
        if normalized_status not in allowed:
            return self._remote_update_rejected(run, "remote_status_transition_invalid")
        if run.terminal:
            return self._remote_update_rejected(run, "run_already_terminal")

        run.remote_status = normalized_status
        if normalized_cursor:
            run.remote_status_cursor = normalized_cursor
        run.remote_observed_at = float(observed_at if observed_at is not None else time.time())
        run.last_poll_error_code = ""
        if normalized_update_id:
            run.processed_remote_update_ids.append(normalized_update_id)
            del run.processed_remote_update_ids[:-_REMOTE_UPDATE_ID_LIMIT]
        if result_summary:
            run.result_summary = str(result_summary).strip()

        skill_result, evidence_changed = _external_evidence_result(
            run,
            payload,
            update_id=normalized_update_id,
            cursor=normalized_cursor,
            remote_status=normalized_status,
            observed_at=run.remote_observed_at,
        )
        if skill_result is not None:
            run.skill_results.append(skill_result)
        if evidence_changed:
            run.report = None

        local_state = _REMOTE_TO_LOCAL_STATE[normalized_status]
        if run.current_state == "cancel_requested" and local_state not in _TERMINAL_STATES:
            local_state = "cancel_requested"
        event = self.transition(
            run,
            local_state,
            f"external_{normalized_status}",
            f"External TaskRun is {normalized_status}.",
            {
                "remote_task_id": run.remote_task_id,
                "remote_status": normalized_status,
                "update_id": normalized_update_id,
                "cursor": normalized_cursor,
                "skill_result_id": skill_result.result_id if skill_result is not None else "",
            },
        )
        return {"handled": True, "event": event.to_dict(), "run": run.to_dict()}

    @_task_run_service_locked
    def record_external_poll_error(self, run_id: str, *, error_code: str) -> dict[str, Any]:
        """Persist a transport observation failure without changing task state."""
        run = self.require(run_id)
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return self._remote_update_rejected(run, "external_profile_required")
        run.last_poll_error_code = str(error_code or "external_poll_failed").strip()
        event = self.emit(
            run,
            "external_poll_failed",
            run.current_state,
            "External TaskRun status poll failed.",
            {"error_code": run.last_poll_error_code},
        )
        return {"handled": True, "event": event.to_dict(), "run": run.to_dict()}

    @_task_run_service_locked
    def request_external_cancel(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record cancel intent; only a later remote update may confirm termination."""
        run = self.require(run_id)
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return self._control_rejected(run, "cancel", "external_profile_required")
        if run.terminal:
            return self._control_rejected(run, "cancel", "run_already_terminal")
        if run.current_state == "cancel_requested":
            return {"handled": True, "run": run.to_dict(), "reply": "TaskRun cancel pending."}
        self._record_operator_action(
            run,
            "cancel",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        self.transition(run, "cancel_requested", "cancel_requested", "TaskRun cancel requested.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun cancel requested."}

    @_task_run_service_locked
    def pause(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.current_state not in {"queued", "executing"}:
            return self._control_rejected(run, "pause", "pause_not_allowed")
        self._record_operator_action(
            run,
            "pause",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        self.transition(run, "paused", "task_paused", "TaskRun paused by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun paused."}

    @_task_run_service_locked
    def resume(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.current_state != "paused":
            return self._control_rejected(run, "resume", "resume_not_allowed")
        self._record_operator_action(
            run,
            "resume",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        self.transition(run, "executing", "execution_resumed", "TaskRun resumed by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun resumed."}

    @_task_run_service_locked
    def cancel(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.profile in EXTERNAL_RUNTIME_PROFILES:
            return self.request_external_cancel(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        if run.terminal:
            return self._control_rejected(run, "cancel", "run_already_terminal")
        self._record_operator_action(
            run,
            "cancel",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        self.transition(run, "cancel_requested", "cancel_requested", "TaskRun cancel requested.")
        self.transition(run, "cancelled", "task_cancelled", "TaskRun cancelled by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun cancelled."}

    def _remote_update_rejected(self, run: TaskRun, reason: str) -> dict[str, Any]:
        return {
            "handled": False,
            "reason": reason,
            "run": run.to_dict(),
            "reply": f"External TaskRun update rejected: {reason}.",
        }

    @_task_run_service_locked
    def advance(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.profile != "sim":
            return self._control_rejected(run, "advance", "advance_only_allowed_in_sim")
        if run.terminal:
            return self._control_rejected(run, "advance", "run_already_terminal")
        if run.current_state == "paused":
            return self._control_rejected(run, "advance", "run_paused")
        if run.current_state == "queued":
            self.transition(run, "executing", "execution_started", "Sim TaskRun execution started.")
        if run.current_state != "executing":
            return self._control_rejected(run, "advance", "advance_not_allowed")

        next_index = int(run.current_step_index or 0) + 1
        steps = list(run.handoff.steps)
        if next_index > len(steps):
            self.transition(run, "completed", "task_completed", "Sim TaskRun completed.")
            return {"handled": True, "run": run.to_dict(), "reply": "Sim TaskRun completed."}

        self._record_operator_action(
            run,
            "advance",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        step = steps[next_index - 1]
        run.current_step_index = next_index
        run.sim_state = _sim_state(run)
        self.emit(
            run,
            "step_started",
            run.current_state,
            f"Started {step.skill_name}.",
            {"step_id": step.step_id, "skill_name": step.skill_name, "sequence": step.sequence},
        )
        result_payload = _simulated_skill_result_payload(step)
        skill_result = self.record_skill_result(run, step, **result_payload)
        self.emit(
            run,
            "step_completed",
            run.current_state,
            f"Completed {step.skill_name}.",
            {
                "step_id": step.step_id,
                "skill_name": step.skill_name,
                "sequence": step.sequence,
                "skill_result_id": skill_result.result_id,
                "skill_result_status": skill_result.status,
            },
        )
        run.sim_state = _sim_state(run)
        if next_index >= len(steps):
            self.transition(run, "completed", "task_completed", "Sim TaskRun completed.")
            return {"handled": True, "run": run.to_dict(), "reply": "Sim TaskRun completed."}
        return {"handled": True, "run": run.to_dict(), "reply": f"Advanced to step {next_index}."}

    @_task_run_service_locked
    def get(self, run_id: str) -> TaskRun | None:
        return self._runs.get(str(run_id or "").strip())

    @_task_run_service_locked
    def require(self, run_id: str) -> TaskRun:
        run = self.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    @_task_run_service_locked
    def runs(self) -> list[TaskRun]:
        return sorted(
            self._runs.values(),
            key=lambda run: run.runtime_events[0].created_at if run.runtime_events else 0.0,
            reverse=True,
        )

    def active_run(self) -> TaskRun | None:
        for run in self.runs():
            if not run.terminal:
                return run
        return None

    @_task_run_service_locked
    def recent_events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        events: list[RuntimeEvent] = []
        for run in self._runs.values():
            events.extend(run.runtime_events)
        events.sort(key=lambda event: event.created_at, reverse=True)
        return [event.to_dict() for event in events[: max(1, int(limit))]]

    def _control_rejected(self, run: TaskRun, action: str, reason: str) -> dict[str, Any]:
        event = self.emit(
            run,
            "operator_action_rejected",
            run.current_state,
            f"{action} rejected: {reason}",
            {"action": action, "reason": reason},
        )
        return {
            "handled": False,
            "reason": reason,
            "event": event.to_dict(),
            "run": run.to_dict(),
            "reply": f"TaskRun {action} rejected: {reason}.",
        }

    @_task_run_service_locked
    def _record_operator_action(
        self,
        run: TaskRun,
        action: str,
        operator_id: str,
        *,
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> None:
        record = _operator_action(
            action,
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
            operator_context=operator_context,
        )
        run.operator_actions.append(record)
        self._audit_log.append_operator_action(run, record)
        self._persist_runs()

    @_task_run_service_locked
    def persist(self) -> None:
        self._persist_runs()

    @_task_run_service_locked
    def _persist_runs(self) -> None:
        self._store.save_runs(self.runs())

    @_task_run_service_locked
    def _trim_runs(self, *, persist: bool = True) -> None:
        if len(self._runs) <= self.max_runs:
            return
        old = self.runs()[self.max_runs :]
        for run in old:
            self._runs.pop(run.run_id, None)
        if persist:
            self._persist_runs()


class RuntimeArbiter:
    """Common interface for local, non-hardware runtime profiles."""

    profile = "fake"

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        raise NotImplementedError


class FakeRuntimeArbiter(RuntimeArbiter):
    """Deterministic local arbiter for product flow and tests."""

    profile = "fake"

    def __init__(
        self,
        *,
        run_service: TaskRunService,
        safety_preflight: SafetyPreflightService,
        skill_registry: SkillRegistry,
        auto_complete: bool = True,
    ) -> None:
        self.run_service = run_service
        self.safety_preflight = safety_preflight
        self.skill_registry = skill_registry
        self.auto_complete = auto_complete

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        run = self.run_service.create(handoff, profile=self.profile)
        self.run_service.transition(run, "submitted", "plan_submitted", "TaskHandoff submitted.")
        self.run_service.transition(
            run, "validating", "plan_validated", "TaskHandoff schema validated."
        )
        self.run_service.transition(
            run, "preflight", "preflight_started", "Safety preflight started."
        )
        assessment = self.safety_preflight.assess(
            handoff,
            skill_registry=self.skill_registry,
            profile=self.profile,
        )
        self.run_service.add_safety_assessment(run, assessment)
        if not assessment.passed:
            return _blocked_submission_payload(self.run_service, run, handoff, assessment)

        self.run_service.transition(run, "queued", "task_queued", "TaskRun queued.")
        if self.auto_complete:
            self._complete_run(run)
        return {
            "accepted": True,
            "status": run.current_state,
            "preflight": assessment.to_dict(),
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
        }

    def _complete_run(self, run: TaskRun) -> None:
        self.run_service.transition(
            run, "executing", "execution_started", "TaskRun execution started."
        )
        for index, step in enumerate(run.handoff.steps, start=1):
            run.current_step_index = index
            self.run_service.emit(
                run,
                "step_started",
                run.current_state,
                f"Started {step.skill_name}.",
                {"step_id": step.step_id, "skill_name": step.skill_name, "sequence": step.sequence},
            )
            result_payload = _simulated_skill_result_payload(step)
            skill_result = self.run_service.record_skill_result(run, step, **result_payload)
            self.run_service.emit(
                run,
                "step_completed",
                run.current_state,
                f"Completed {step.skill_name}.",
                {
                    "step_id": step.step_id,
                    "skill_name": step.skill_name,
                    "sequence": step.sequence,
                    "skill_result_id": skill_result.result_id,
                    "skill_result_status": skill_result.status,
                },
            )
        self.run_service.transition(
            run, "completed", "task_completed", "TaskRun completed in fake runtime."
        )


class ShadowRuntimeArbiter(RuntimeArbiter):
    """Validate a handoff and expose the would-execute plan without execution."""

    profile = "shadow"

    def __init__(
        self,
        *,
        run_service: TaskRunService,
        safety_preflight: SafetyPreflightService,
        skill_registry: SkillRegistry,
    ) -> None:
        self.run_service = run_service
        self.safety_preflight = safety_preflight
        self.skill_registry = skill_registry

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        run = self.run_service.create(handoff, profile=self.profile)
        self.run_service.transition(run, "submitted", "plan_submitted", "TaskHandoff submitted.")
        self.run_service.transition(
            run, "validating", "plan_validated", "TaskHandoff schema validated."
        )
        self.run_service.transition(
            run, "preflight", "preflight_started", "Safety preflight started."
        )
        assessment = self.safety_preflight.assess(
            handoff,
            skill_registry=self.skill_registry,
            profile=self.profile,
        )
        self.run_service.add_safety_assessment(run, assessment)
        if not assessment.passed:
            return _blocked_submission_payload(self.run_service, run, handoff, assessment)

        shadow_plan = _shadow_plan_for(handoff, self.skill_registry, assessment)
        run.shadow_plan = shadow_plan
        self.run_service.transition(
            run,
            "shadowed",
            "shadow_plan_ready",
            "Shadow runtime produced an auditable would-execute plan.",
            {"shadow_plan": shadow_plan},
        )
        return {
            "accepted": True,
            "status": run.current_state,
            "preflight": assessment.to_dict(),
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
            "shadow_plan": shadow_plan,
        }


class SimRuntimeArbiter(RuntimeArbiter):
    """Step-advance simulator for TaskRun lifecycle tests and demos."""

    profile = "sim"

    def __init__(
        self,
        *,
        run_service: TaskRunService,
        safety_preflight: SafetyPreflightService,
        skill_registry: SkillRegistry,
        auto_complete: bool = False,
    ) -> None:
        self.run_service = run_service
        self.safety_preflight = safety_preflight
        self.skill_registry = skill_registry
        self.auto_complete = auto_complete

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        run = self.run_service.create(handoff, profile=self.profile)
        self.run_service.transition(run, "submitted", "plan_submitted", "TaskHandoff submitted.")
        self.run_service.transition(
            run, "validating", "plan_validated", "TaskHandoff schema validated."
        )
        self.run_service.transition(
            run, "preflight", "preflight_started", "Safety preflight started."
        )
        assessment = self.safety_preflight.assess(
            handoff,
            skill_registry=self.skill_registry,
            profile=self.profile,
        )
        self.run_service.add_safety_assessment(run, assessment)
        if not assessment.passed:
            return _blocked_submission_payload(self.run_service, run, handoff, assessment)

        run.sim_state = _sim_state(run)
        self.run_service.transition(run, "queued", "task_queued", "Sim TaskRun queued.")
        if self.auto_complete:
            while not run.terminal:
                self.run_service.advance(run.run_id)
        return {
            "accepted": True,
            "status": run.current_state,
            "preflight": assessment.to_dict(),
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
        }


class ExternalRuntimeArbiter(RuntimeArbiter):
    """Lab/external client contract with explicit local enablement gates."""

    def __init__(
        self,
        *,
        run_service: TaskRunService,
        safety_preflight: SafetyPreflightService,
        skill_registry: SkillRegistry,
        client: RuntimeArbiterClient,
        transport: RuntimeExecutorTransport | None = None,
    ) -> None:
        self.profile = client.profile
        self.run_service = run_service
        self.safety_preflight = safety_preflight
        self.skill_registry = skill_registry
        self.client = client
        self.transport = transport

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        run = self.run_service.create(handoff, profile=self.profile)
        return self.submit_existing(run)

    def submit_existing(self, run: TaskRun) -> dict[str, Any]:
        """Submit a persisted external run without changing its identity."""

        if run.profile != self.profile:
            raise ValueError("prepared TaskRun profile does not match external arbiter")
        if run.terminal:
            return {
                "accepted": False,
                "status": run.current_state,
                "reason": "run_already_terminal",
                "run": run.to_dict(),
                "handoff": run.handoff.to_dict(),
                "hardware_dispatch": False,
            }
        handoff = run.handoff
        self.run_service.transition(run, "submitted", "plan_submitted", "TaskHandoff submitted.")
        self.run_service.transition(
            run, "validating", "plan_validated", "TaskHandoff schema validated."
        )

        client_error = self.client.validate_submit_ready()
        if client_error is not None:
            assessment = _runtime_client_assessment(handoff, self.profile, client_error.to_dict())
            self.run_service.add_safety_assessment(run, assessment)
            self.run_service.transition(
                run,
                "blocked",
                "external_runtime_rejected",
                "External runtime submission rejected by local client contract.",
                {"error": client_error.to_dict()},
            )
            return {
                "accepted": False,
                "status": run.current_state,
                "reason": client_error.code,
                "error": client_error.to_dict(),
                "preflight": assessment.to_dict(),
                "run": run.to_dict(),
                "handoff": handoff.to_dict(),
                "hardware_dispatch": False,
            }

        self.run_service.transition(
            run, "preflight", "preflight_started", "Safety preflight started."
        )
        assessment = self.safety_preflight.assess(
            handoff,
            skill_registry=self.skill_registry,
            profile=self.profile,
        )
        self.run_service.add_safety_assessment(run, assessment)
        if not assessment.passed:
            return _blocked_submission_payload(self.run_service, run, handoff, assessment)

        envelope = self.client.submission_envelope(handoff.to_dict())
        if self.transport is None:
            return self._submission_failure(
                run,
                handoff,
                assessment,
                code="external_transport_unavailable",
                event_type="external_transport_unavailable",
                message="External runtime transport is unavailable.",
                state="blocked",
                envelope=envelope,
            )

        self.run_service.transition(
            run,
            "queued",
            "external_runtime_contract_ready",
            "External runtime contract envelope is ready.",
            {"runtime_client": envelope},
        )
        idempotency_key = _external_submission_idempotency_key(handoff)
        prepared = self.run_service.prepare_external_submission(
            run.run_id,
            external_idempotency_key=idempotency_key,
        )
        if not prepared.get("handled", False):
            return self._submission_failure(
                run,
                handoff,
                assessment,
                code=str(prepared.get("reason") or "external_idempotency_checkpoint_failed"),
                event_type="external_idempotency_checkpoint_failed",
                message="External submission replay checkpoint failed.",
                state="failed",
                envelope=envelope,
            )
        voice_context = _voice_context_from_handoff(handoff)
        request = RuntimeExecutorSubmitRequest(
            handoff=handoff.to_dict(),
            idempotency_key=idempotency_key,
            correlation_id=run.run_id,
            thread_id=str(
                voice_context.get("thread_id")
                or voice_context.get("conversation_session_id")
                or handoff.session_id
            ),
            turn_id=str(
                voice_context.get("turn_id") or voice_context.get("originating_turn_id") or ""
            ),
        )
        try:
            remote = self.transport.submit(request)
        except AmbiguousRuntimeSubmissionError as exc:
            result = self._submission_failure(
                run,
                handoff,
                assessment,
                code="external_submission_unknown",
                event_type="external_submission_unknown",
                message="External runtime submission outcome is unknown.",
                state="submission_unknown",
                envelope=envelope,
                transport_error=exc,
            )
            self._commit_supervision(run)
            return result
        except RuntimeExecutorTransportError as exc:
            return self._submission_failure(
                run,
                handoff,
                assessment,
                code="external_submission_failed",
                event_type="external_submission_failed",
                message="External runtime submission failed.",
                state="failed",
                envelope=envelope,
                transport_error=exc,
            )
        except Exception:
            return self._submission_failure(
                run,
                handoff,
                assessment,
                code="external_submission_failed",
                event_type="external_submission_failed",
                message="External runtime submission failed.",
                state="failed",
                envelope=envelope,
            )

        if (
            not isinstance(remote, RuntimeExecutorSubmitResult)
            or not str(remote.remote_task_id or "").strip()
            or remote.correlation_id != run.run_id
            or remote.idempotency_key != idempotency_key
        ):
            result = self._submission_failure(
                run,
                handoff,
                assessment,
                code="external_submission_unknown",
                event_type="external_submission_unknown",
                message="External runtime response could not prove submission identity.",
                state="submission_unknown",
                envelope=envelope,
            )
            self._commit_supervision(run)
            return result

        try:
            bound = self.run_service.bind_external_submission(
                run.run_id,
                remote_task_id=remote.remote_task_id,
                external_idempotency_key=idempotency_key,
                remote_status="submitted",
                observed_at=remote.observed_at,
            )
            if not bound.get("handled", False):
                raise RuntimeError(
                    str(bound.get("reason") or "external_submission_bind_rejected")
                )
            for update in remote.updates:
                applied = self.run_service.apply_external_update(
                    run.run_id,
                    remote_task_id=remote.remote_task_id,
                    remote_status=update.status,
                    update_id=update.event_id,
                    cursor=update.cursor,
                    payload=dict(update.payload),
                    observed_at=update.observed_at,
                )
                if not applied.get("handled", False):
                    logger.warning(
                        "External runtime update %s rejected: %s",
                        update.event_id,
                        applied.get("reason", "unknown"),
                    )
            if not (
                remote.cursor
                and remote.cursor == run.remote_status_cursor
                and _normalize_remote_status(remote.status) == run.remote_status
            ):
                projection = self.run_service.apply_external_update(
                    run.run_id,
                    remote_task_id=remote.remote_task_id,
                    remote_status=remote.status,
                    cursor=remote.cursor,
                    result_summary=remote.result_summary,
                    observed_at=remote.observed_at,
                )
                if not projection.get("handled", False) and not run.terminal:
                    raise RuntimeError(
                        str(projection.get("reason") or "external_update_rejected")
                    )
            elif remote.result_summary:
                with self.run_service.transaction():
                    run.result_summary = str(remote.result_summary).strip()
                    if run.terminal:
                        run.report = self.run_service._report_service.build_report(run)
                    self.run_service.persist()
            self._commit_supervision(run)
        except Exception as exc:
            logger.error(
                "External submission accepted but local projection commit failed for %s: %s",
                run.run_id,
                type(exc).__name__,
            )
            return self._projection_commit_unknown(
                run,
                handoff,
                assessment,
                envelope=envelope,
                remote=remote,
                idempotency_key=idempotency_key,
            )
        remote_metadata = {
            "remote_task_id": run.remote_task_id,
            "status": run.remote_status,
            "cursor": run.remote_status_cursor,
            "observed_at": run.remote_observed_at,
            "result_summary": run.result_summary,
        }
        if run.remote_status == "rejected":
            return {
                "accepted": False,
                "status": run.current_state,
                "reason": "external_submission_rejected",
                "error": {"code": "external_submission_rejected"},
                "preflight": assessment.to_dict(),
                "run": run.to_dict(),
                "handoff": handoff.to_dict(),
                "runtime_client": envelope,
                "remote": remote_metadata,
                "hardware_dispatch": False,
            }
        return {
            "accepted": True,
            "status": run.current_state,
            "preflight": assessment.to_dict(),
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
            "runtime_client": envelope,
            "remote": remote_metadata,
            "hardware_dispatch": False,
        }

    def _projection_commit_unknown(
        self,
        run: TaskRun,
        handoff: TaskHandoff,
        assessment: SafetyAssessment,
        *,
        envelope: dict[str, Any],
        remote: RuntimeExecutorSubmitResult,
        idempotency_key: str,
    ) -> dict[str, Any]:
        error_code = "external_projection_commit_unknown"
        try:
            self.run_service.mark_external_projection_unknown(
                run.run_id,
                remote_task_id=remote.remote_task_id,
                external_idempotency_key=idempotency_key,
                error_code=error_code,
            )
        except Exception:
            logger.exception(
                "TaskRun projection uncertainty recording failed for %s",
                run.run_id,
            )
            with self.run_service.transaction():
                current = self.run_service.require(run.run_id)
                if not current.terminal:
                    current.remote_task_id = str(remote.remote_task_id or "").strip() or None
                    current.external_idempotency_key = idempotency_key
                    current.remote_status = current.remote_status or "submitted"
                    current.current_state = "submission_unknown"
                    current.last_poll_error_code = error_code
        current = self.run_service.require(run.run_id)
        return {
            "accepted": False,
            "status": current.current_state,
            "reason": error_code,
            "error": {"code": error_code, "ambiguous": True, "retryable": True},
            "preflight": assessment.to_dict(),
            "run": current.to_dict(),
            "handoff": handoff.to_dict(),
            "runtime_client": envelope,
            "remote": {
                "remote_task_id": remote.remote_task_id,
                "status": remote.status,
                "cursor": remote.cursor,
                "observed_at": remote.observed_at,
                "result_summary": remote.result_summary,
            },
            "remote_may_be_running": True,
            "hardware_dispatch": False,
        }

    def _commit_supervision(self, run: TaskRun) -> None:
        self.run_service.emit(
            run,
            "external_submission_committed",
            run.current_state,
            "Initial external submission projection committed.",
            {
                "remote_task_id": run.remote_task_id,
                "external_idempotency_key": run.external_idempotency_key,
                "submission_unknown": run.current_state == "submission_unknown",
            },
        )

    def _submission_failure(
        self,
        run: TaskRun,
        handoff: TaskHandoff,
        assessment: SafetyAssessment,
        *,
        code: str,
        event_type: str,
        message: str,
        state: str,
        envelope: dict[str, Any],
        transport_error: RuntimeExecutorTransportError | None = None,
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code}
        if transport_error is not None:
            error.update(
                {
                    "kind": transport_error.kind,
                    "status_code": transport_error.status_code,
                    "retryable": transport_error.retryable,
                    "ambiguous": transport_error.ambiguous,
                }
            )
        self.run_service.transition(run, state, event_type, message, {"error": error})
        return {
            "accepted": False,
            "status": run.current_state,
            "reason": code,
            "error": error,
            "preflight": assessment.to_dict(),
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
            "runtime_client": envelope,
            "hardware_dispatch": False,
        }


class RuntimeHandoffService:
    """Facade used by HTTP, Dashboard, and chat control paths."""

    def __init__(
        self,
        *,
        world_state: Any,
        default_operator_id: str = "askme.operator",
        planner_version: str = "askme-cognition-v1",
        profile: str = "fake",
        auto_complete: bool = True,
        max_world_state_age_s: float = 30.0,
        max_runs: int = 50,
        audit_config: RuntimeAuditConfig | dict[str, Any] | None = None,
        store_config: TaskRunStoreConfig | dict[str, Any] | None = None,
        dog_safety_client: Any | None = None,
        require_dog_safety: bool = False,
        require_supervisor_for_high_risk: bool = False,
        external_runtime_config: dict[str, Any] | None = None,
        executor_transport: RuntimeExecutorTransport | None = None,
    ) -> None:
        self.world_state = world_state
        self.default_operator_id = default_operator_id
        self.planner_version = planner_version
        self.profile = _normalize_runtime_profile(profile)
        self.skill_registry = SkillRegistry()
        self.operator_policy = OperatorPolicyService(
            require_supervisor_for_high_risk=require_supervisor_for_high_risk,
        )
        self.report_service = TaskReportService()
        self.run_service = TaskRunService(
            report_service=self.report_service,
            event_sink=self._record_world_event,
            audit_log=RuntimeAuditLog(audit_config),
            store=store_config,
            max_runs=max_runs,
        )
        self.safety_preflight = SafetyPreflightService(
            max_world_state_age_s=max_world_state_age_s,
            dog_safety_client=dog_safety_client,
            require_dog_safety=require_dog_safety,
            operator_policy=self.operator_policy,
        )
        self.runtime_arbiter_client = RuntimeArbiterClient.from_config(
            self.profile,
            external_runtime_config,
        )
        self.executor_transport = executor_transport
        self.arbiter = self._build_arbiter(auto_complete=auto_complete)

    def subscribe_events(self, observer: Callable[[RuntimeEvent], None]) -> Callable[[], None]:
        return self.run_service.subscribe_events(observer)

    def bind_external_submission(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        external_idempotency_key: str = "",
        remote_status: str = "submitted",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        result = self.run_service.bind_external_submission(
            run_id,
            remote_task_id=remote_task_id,
            external_idempotency_key=external_idempotency_key,
            remote_status=remote_status,
            observed_at=observed_at,
        )
        if result.get("handled", False):
            run = self.run_service.require(run_id)
            self.run_service.emit(
                run,
                "external_submission_committed",
                run.current_state,
                "External submission binding committed.",
                {
                    "remote_task_id": run.remote_task_id,
                    "external_idempotency_key": run.external_idempotency_key,
                },
            )
            result = {**result, "run": run.to_dict()}
        self._update_runtime_facts(result["run"])
        return result

    def apply_external_update(
        self,
        run_id: str,
        *,
        remote_task_id: str,
        remote_status: str,
        update_id: str = "",
        cursor: str | int = "",
        payload: dict[str, Any] | None = None,
        result_summary: str = "",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        result = self.run_service.apply_external_update(
            run_id,
            remote_task_id=remote_task_id,
            remote_status=remote_status,
            update_id=update_id,
            cursor=cursor,
            payload=payload,
            result_summary=result_summary,
            observed_at=observed_at,
        )
        self._update_runtime_facts(result["run"])
        return result

    def record_external_poll_error(self, run_id: str, *, error_code: str) -> dict[str, Any]:
        result = self.run_service.record_external_poll_error(run_id, error_code=error_code)
        self._update_runtime_facts(result["run"])
        return result

    @_runtime_handoff_task_run_transaction
    def record_notification_delivery_receipt(
        self,
        run_id: str,
        *,
        event_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Persist the first terminal delivery outcome for a committed run event."""
        run = self.run_service.require(run_id)
        normalized_event_id = str(event_id or "").strip()
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in _NOTIFICATION_DELIVERY_STATES:
            allowed = ", ".join(sorted(_NOTIFICATION_DELIVERY_STATES))
            raise ValueError(f"delivery status must be one of: {allowed}")
        if not any(event.event_id == normalized_event_id for event in run.runtime_events):
            raise ValueError(
                f"event_id {normalized_event_id!r} does not belong to TaskRun {run.run_id}"
            )

        existing = run.notification_delivery_receipts.get(normalized_event_id)
        if existing is not None:
            return {
                "run_id": run.run_id,
                "event_id": normalized_event_id,
                "status": existing,
                "recorded": False,
            }

        run.notification_delivery_receipts[normalized_event_id] = normalized_status
        while len(run.notification_delivery_receipts) > _NOTIFICATION_DELIVERY_RECEIPT_LIMIT:
            oldest_event_id = next(iter(run.notification_delivery_receipts))
            del run.notification_delivery_receipts[oldest_event_id]
        self.run_service.persist()
        return {
            "run_id": run.run_id,
            "event_id": normalized_event_id,
            "status": normalized_status,
            "recorded": True,
        }

    @_runtime_handoff_task_run_transaction
    def notification_delivery_receipt(self, run_id: str, *, event_id: str) -> str | None:
        run = self.run_service.require(run_id)
        normalized_event_id = str(event_id or "").strip()
        if not any(event.event_id == normalized_event_id for event in run.runtime_events):
            raise ValueError(
                f"event_id {normalized_event_id!r} does not belong to TaskRun {run.run_id}"
            )
        return run.notification_delivery_receipts.get(normalized_event_id)

    @_runtime_handoff_task_run_transaction
    def notification_delivery_receipts(self, run_id: str) -> dict[str, str]:
        run = self.run_service.require(run_id)
        return dict(run.notification_delivery_receipts)

    def submit_plan_payload(self, plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("plan must be an object")
        snapshot = self.world_state.snapshot() if self.world_state is not None else {}
        handoff = TaskHandoff.from_plan(
            plan,
            world_state_snapshot=snapshot,
            skill_registry=self.skill_registry,
            default_operator_id=self.default_operator_id,
            planner_version=self.planner_version,
        )
        result = self.arbiter.submit(handoff)
        self._update_runtime_facts(result["run"])
        return result

    @_runtime_handoff_task_run_transaction
    def prepare_plan_payload(
        self,
        plan: dict[str, Any],
        *,
        approval_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist an external TaskRun before asking the operator to confirm it."""

        if not isinstance(plan, dict):
            raise ValueError("plan must be an object")
        if not isinstance(approval_request, dict) or not approval_request:
            raise ValueError("approval_request must be a non-empty object")
        if str(approval_request.get("kind") or "") == "runtime_handoff":
            required = {
                "approval_id",
                "thread_id",
                "prompt_turn_id",
                "operator_id",
                "person_id",
                "expires_at",
                "payload_digest",
            }
            if any(not approval_request.get(field) for field in required):
                raise ValueError("runtime_handoff approval_request is incomplete")
            if str(approval_request.get("payload_digest")) != _approval_plan_digest(plan):
                raise ValueError("runtime_handoff approval payload digest mismatch")
        if self.profile not in EXTERNAL_RUNTIME_PROFILES:
            raise ValueError("prepared plans require an external or lab runtime profile")
        snapshot = self.world_state.snapshot() if self.world_state is not None else {}
        handoff = TaskHandoff.from_plan(
            plan,
            world_state_snapshot=snapshot,
            skill_registry=self.skill_registry,
            default_operator_id=self.default_operator_id,
            planner_version=self.planner_version,
        )
        run = self.run_service.create(handoff, profile=self.profile)
        run.approval_request = {**dict(approval_request), "status": "waiting_user"}
        self.run_service.transition(
            run,
            "waiting_user",
            "approval_requested",
            "TaskRun is waiting for operator confirmation.",
            {"approval_request": dict(run.approval_request)},
        )
        result: dict[str, Any] = {
            "accepted": False,
            "status": run.current_state,
            "reason": "approval_required",
            "run": run.to_dict(),
            "handoff": handoff.to_dict(),
            "hardware_dispatch": False,
        }
        self._update_runtime_facts(result["run"])
        return result

    @_runtime_handoff_task_run_transaction
    def confirm_prepared_plan(
        self,
        run_id: str,
        *,
        confirmed_plan: dict[str, Any],
        operator_id: str,
        operator_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist confirmation and the final handoff without contacting the executor."""

        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("submit", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "submit", failure)
        if run.profile not in EXTERNAL_RUNTIME_PROFILES:
            return _runtime_operator_rejected_payload(run, "submit", "external_profile_required")
        if run.current_state != "waiting_user":
            return _runtime_operator_rejected_payload(run, "submit", "approval_not_pending")
        if not isinstance(confirmed_plan, dict):
            raise ValueError("confirmed_plan must be an object")
        approval = run.approval_request
        if str(approval.get("kind") or "") == "runtime_handoff":
            if str(approval.get("status") or "") != "waiting_user":
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_not_pending"
                )
            if float(approval.get("expires_at") or 0.0) <= time.time():
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_expired"
                )
            if str(approval.get("operator_id") or "") != str(operator_id):
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_operator_mismatch"
                )
            if str(approval.get("person_id") or "") != str(
                operator_context.get("person_id") or ""
            ):
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_person_mismatch"
                )
            if str(operator_context.get("approval_id") or "") != str(
                approval.get("approval_id") or ""
            ):
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_id_mismatch"
                )
            approval_digest = str(approval.get("payload_digest") or "")
            if _approval_plan_digest(run.handoff.source_plan) != approval_digest:
                return _runtime_operator_rejected_payload(
                    run, "submit", "approval_payload_mismatch"
                )
            if _approval_plan_digest(
                _unconfirmed_approval_plan(confirmed_plan)
            ) != approval_digest:
                return _runtime_operator_rejected_payload(
                    run, "submit", "confirmed_payload_mismatch"
                )
        snapshot = self.world_state.snapshot() if self.world_state is not None else {}
        confirmed_handoff = TaskHandoff.from_plan(
            confirmed_plan,
            world_state_snapshot=snapshot,
            skill_registry=self.skill_registry,
            default_operator_id=self.default_operator_id,
            planner_version=self.planner_version,
        )
        run.handoff = replace(
            confirmed_handoff,
            handoff_id=run.handoff.handoff_id,
        )
        run.approval_request = {
            **run.approval_request,
            "status": "confirmed",
            "confirmed_at": time.time(),
            "confirmed_by": str(operator_id),
        }
        self.run_service._record_operator_action(
            run,
            "submit",
            operator_id,
            operator_context=operator_context,
        )
        self.run_service.transition(
            run,
            "confirmed",
            "approval_confirmed",
            "TaskRun was confirmed by the operator.",
            {"approval_request": dict(run.approval_request)},
        )
        result: dict[str, Any] = {
            "handled": True,
            "run": run.to_dict(),
            "handoff": run.handoff.to_dict(),
        }
        self._update_runtime_facts(result["run"])
        return result

    def submit_prepared_run(self, run_id: str) -> dict[str, Any]:
        """Submit a previously confirmed TaskRun using the same durable run id."""

        if self.run_service.get(run_id) is None:
            return {"accepted": False, "error": "run not found", "run_id": run_id}
        if not isinstance(self.arbiter, ExternalRuntimeArbiter):
            run = self.run_service.get(run_id)
            if run is None:
                return {"accepted": False, "error": "run not found", "run_id": run_id}
            return {
                "accepted": False,
                "status": run.current_state,
                "reason": "external_profile_required",
                "run": run.to_dict(),
            }
        snapshot = self.world_state.snapshot() if self.world_state is not None else {}
        refreshed = self.run_service.refresh_prepared_handoff_snapshot(
            run_id,
            world_state_snapshot=snapshot,
        )
        if not refreshed.get("handled", False):
            run_payload = dict(refreshed.get("run") or {})
            return {
                "accepted": False,
                "status": str(run_payload.get("current_state") or "unknown"),
                "reason": str(refreshed.get("reason") or "prepared_run_not_confirmed"),
                "run": run_payload,
            }
        run = self.run_service.require(run_id)
        result = self.arbiter.submit_existing(run)
        self._update_runtime_facts(result["run"])
        return result

    def submit_prepared_plan(
        self,
        run_id: str,
        *,
        confirmed_plan: dict[str, Any],
        operator_id: str,
        operator_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility wrapper for non-voice callers that can submit immediately."""

        confirmed = self.confirm_prepared_plan(
            run_id,
            confirmed_plan=confirmed_plan,
            operator_id=operator_id,
            operator_context=operator_context,
        )
        if not confirmed.get("handled", False):
            return {"accepted": False, **confirmed}
        return self.submit_prepared_run(run_id)

    @_runtime_handoff_task_run_transaction
    def cancel_prepared_run(
        self,
        run_id: str,
        *,
        operator_id: str,
        operator_context: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Cancel a local pre-submit TaskRun without contacting the executor."""

        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("cancel", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "cancel", failure)
        if run.current_state not in {"waiting_user", "confirmed"} or run.remote_task_id:
            return _runtime_operator_rejected_payload(run, "cancel", "prepared_cancel_not_allowed")
        self.run_service._record_operator_action(
            run,
            "cancel",
            operator_id,
            reason=reason,
            operator_context=operator_context,
        )
        run.approval_request = {**run.approval_request, "status": "cancelled"}
        self.run_service.transition(
            run,
            "cancelled",
            "task_cancelled",
            "Prepared TaskRun cancelled by operator.",
        )
        result: dict[str, Any] = {
            "handled": True,
            "run": run.to_dict(),
            "reply": "TaskRun cancelled.",
        }
        self._update_runtime_facts(result["run"])
        return result

    @_runtime_handoff_task_run_transaction
    def expire_prepared_run(
        self,
        run_id: str,
        *,
        reason: str = "approval_expired",
    ) -> dict[str, Any]:
        """Expire an unsubmitted approval challenge without executor contact."""

        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        if run.current_state != "waiting_user" or run.remote_task_id:
            return _runtime_operator_rejected_payload(
                run,
                "expire",
                "prepared_expiry_not_allowed",
            )
        run.approval_request = {
            **run.approval_request,
            "status": "expired",
            "expired_at": time.time(),
        }
        self.run_service.transition(
            run,
            "cancelled",
            "approval_expired",
            "Prepared TaskRun approval expired before submission.",
            {"reason": str(reason or "approval_expired")},
        )
        result: dict[str, Any] = {
            "handled": True,
            "run": run.to_dict(),
            "reply": "TaskRun approval expired.",
        }
        self._update_runtime_facts(result["run"])
        return result

    def context_payload(self) -> dict[str, Any]:
        runs = [run.to_dict() for run in self.run_service.runs()]
        active = self.run_service.active_run()
        return {
            "profile": self.profile,
            "supported_profiles": list(_SUPPORTED_RUNTIME_PROFILES),
            "hardware_dispatch": False,
            "runtime_client": self.runtime_arbiter_client.safe_config(),
            "skill_registry": self.skill_registry.snapshot(),
            "operator_policy": self.operator_policy.snapshot(),
            "runs": runs,
            "active_run": active.to_dict() if active else None,
            "recent_events": self.run_service.recent_events(limit=20),
        }

    def events_payload(
        self,
        *,
        after: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        cursor = float(after or 0.0)
        recent = self.run_service.recent_events(limit=max(1, min(int(limit), 100)))
        events = [
            event
            for event in sorted(recent, key=lambda item: float(item.get("created_at", 0.0)))
            if float(event.get("created_at", 0.0)) > cursor
        ]
        next_cursor = cursor
        for event in events:
            next_cursor = max(next_cursor, float(event.get("created_at", 0.0) or 0.0))
        active = self.run_service.active_run()
        return {
            "profile": self.profile,
            "hardware_dispatch": False,
            "cursor": next_cursor,
            "events": events,
            "event_count": len(events),
            "active_run": active.to_dict() if active else None,
        }

    def profiles_payload(self) -> dict[str, Any]:
        return {
            "current_profile": self.profile,
            "hardware_dispatch": False,
            "profiles": [
                {
                    "name": "fake",
                    "hardware_dispatch": False,
                    "description": "Deterministic local fake runtime that can auto-complete runs.",
                },
                {
                    "name": "shadow",
                    "hardware_dispatch": False,
                    "description": "Preflight plus would-execute preview; no step execution.",
                },
                {
                    "name": "sim",
                    "hardware_dispatch": False,
                    "description": "Local step simulator with pause, resume, cancel, and advance.",
                },
                {
                    "name": "external",
                    "hardware_dispatch": False,
                    "description": "Disabled-by-default external runtime client contract.",
                    "requires": ["endpoint", "enable_external_runtime=true"],
                },
                {
                    "name": "lab",
                    "hardware_dispatch": False,
                    "description": "Disabled-by-default lab runtime client contract.",
                    "requires": ["endpoint", "enable_external_runtime=true"],
                },
            ],
        }

    def list_payload(self) -> dict[str, Any]:
        runs = [run.to_dict() for run in self.run_service.runs()]
        return {"runs": runs, "count": len(runs)}

    def get_payload(self, run_id: str) -> dict[str, Any]:
        run = self.run_service.get(run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        return {"run": run.to_dict()}

    def report_payload(self, run_id: str) -> dict[str, Any]:
        report = self.run_service.get_or_build_report(run_id)
        if report is None:
            return {"error": "run not found", "run_id": run_id}
        return {"report": report}

    def pause_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("pause", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "pause", failure)
        try:
            result = self.run_service.pause(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        except KeyError:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        self._update_runtime_facts(result["run"])
        return result

    def resume_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("resume", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "resume", failure)
        try:
            result = self.run_service.resume(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        except KeyError:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        self._update_runtime_facts(result["run"])
        return result

    def cancel_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("cancel", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "cancel", failure)
        try:
            result = self.run_service.cancel(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        except KeyError:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        self._update_runtime_facts(result["run"])
        return result

    def advance_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
        operator_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.run_service.get(run_id)
        if run is None:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        failure = _runtime_operator_context_failure("advance", operator_id, operator_context)
        if failure:
            return _runtime_operator_rejected_payload(run, "advance", failure)
        try:
            result = self.run_service.advance(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context,
            )
        except KeyError:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        self._update_runtime_facts(result["run"])
        return result

    def handle_chat_control(
        self,
        text: str,
        *,
        operator_id: str | None = None,
        operator_roles: list[str] | tuple[str, ...] | None = None,
        operator_authenticated: bool | None = None,
        operator_source: str = "",
        runtime_permission: str = "",
        conversation_session_id: str | None = None,
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any] | None:
        intent = runtime_control_intent(text)
        if intent is None:
            return None
        active = self.run_service.active_run()
        latest = active or (self.run_service.runs()[0] if self.run_service.runs() else None)
        if latest is None:
            if intent != "status":
                return None
            resolved_operator_id = str(operator_id or "").strip()
            operator_context = _runtime_operator_provenance(
                operator_id=(resolved_operator_id if operator_id is not None else None),
                operator_roles=operator_roles,
                operator_authenticated=operator_authenticated,
                operator_source=operator_source,
                runtime_permission=runtime_permission,
                conversation_session_id=conversation_session_id,
            )
            failure = _runtime_operator_context_failure(
                "read",
                resolved_operator_id,
                operator_context or None,
            )
            if failure:
                return {
                    "handled": False,
                    "reason": failure,
                    "reply": f"TaskRun status rejected: {failure}.",
                    "runtime": self.context_payload(),
                }
            return {
                "handled": True,
                "reply": "No TaskRun is active yet.",
                "runtime": self.context_payload(),
            }
        resolved_operator_id = str(operator_id or "").strip()
        operator_context = _runtime_operator_provenance(
            operator_id=(resolved_operator_id if operator_id is not None else None),
            operator_roles=operator_roles,
            operator_authenticated=operator_authenticated,
            operator_source=operator_source,
            runtime_permission=runtime_permission,
            conversation_session_id=conversation_session_id,
        )
        if intent == "status":
            failure = _runtime_operator_context_failure(
                "read",
                resolved_operator_id,
                operator_context or None,
            )
            if failure:
                return _runtime_operator_rejected_payload(latest, "status", failure)
            return {
                "handled": True,
                "reply": f"TaskRun {latest.run_id} is {latest.current_state}.",
                "runtime": {
                    "run": latest.to_dict(),
                    "active_run": active.to_dict() if active else None,
                },
            }
        if active is None:
            return None
        if intent == "pause":
            result = self.pause_payload(
                latest.run_id,
                operator_id=resolved_operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context or None,
            )
        elif intent == "resume":
            result = self.resume_payload(
                latest.run_id,
                operator_id=resolved_operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context or None,
            )
        else:
            result = self.cancel_payload(
                latest.run_id,
                operator_id=resolved_operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
                operator_context=operator_context or None,
            )
        payload = {
            "handled": bool(result.get("handled", False)),
            "reply": result.get("reply", ""),
            "runtime": result,
        }
        if result.get("reason"):
            payload["reason"] = result["reason"]
        return payload

    def voice_turn_payload(
        self,
        text: str,
        *,
        speak: bool = False,
        transcript_id: str = "",
        confidence: float | None = None,
        is_final: bool = True,
        channel: str = "voice",
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        operator_id: str | None = None,
        operator_roles: list[str] | tuple[str, ...] | None = None,
        operator_authenticated: bool | None = None,
        operator_source: str = "",
        runtime_permission: str = "",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        recognized = str(text or "").strip()
        voice_turn = _voice_turn_metadata(
            recognized,
            transcript_id=transcript_id,
            confidence=confidence,
            is_final=is_final,
            channel=channel,
            conversation_session_id=conversation_session_id,
            planning_session_id=planning_session_id,
        )
        control_intent = runtime_control_intent(recognized)
        expected_permission = runtime_control_permission(recognized)
        operator_provenance = _runtime_operator_provenance(
            operator_id=operator_id,
            operator_roles=operator_roles,
            operator_authenticated=operator_authenticated,
            operator_source=operator_source,
            runtime_permission=runtime_permission,
            conversation_session_id=conversation_session_id,
        )
        if operator_provenance:
            voice_turn["operator"] = operator_provenance
        if runtime_permission:
            voice_turn["runtime_permission"] = runtime_permission
        if not recognized:
            return {
                "handled": False,
                "reason": "empty_transcript",
                "reply": "",
                "spoken": False if speak else None,
                "voice_turn": voice_turn,
            }
        if (
            control_intent is not None
            and runtime_permission
            and runtime_permission != expected_permission
        ):
            return {
                "handled": False,
                "reason": "runtime_control_permission_mismatch",
                "reply": "",
                "spoken": False if speak else None,
                "voice_turn": voice_turn,
            }
        control = self.handle_chat_control(
            recognized,
            operator_id=operator_id,
            operator_roles=operator_roles,
            operator_authenticated=operator_authenticated,
            operator_source=operator_source,
            runtime_permission=runtime_permission,
            conversation_session_id=conversation_session_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )
        if control is None:
            return {
                "handled": False,
                "reason": "no_runtime_control_intent",
                "reply": "",
                "spoken": False if speak else None,
                "voice_turn": voice_turn,
                "runtime": self.context_payload(),
            }
        payload = dict(control)
        payload["voice_turn"] = {
            **voice_turn,
            "runtime_control_intent": control_intent,
            "handled_by": "runtime_handoff",
        }
        if speak:
            payload.setdefault("spoken", False)
        return payload

    def health(self) -> dict[str, Any]:
        runs = self.run_service.runs()
        return {
            "status": "ok",
            "profile": self.profile,
            "runs": len(runs),
            "active_run": bool(self.run_service.active_run()),
            "task_run_store": self.run_service._store.enabled,
            "hardware_dispatch": False,
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "task_handoff": True,
            "fake_arbiter": True,
            "shadow_arbiter": True,
            "sim_arbiter": True,
            "external_runtime_client": True,
            "safety_preflight": True,
            "active_perception_requests": True,
            "operator_policy": True,
            "skill_registry": True,
            "task_reports": True,
            "task_run_store": self.run_service._store.enabled,
            "runtime_profiles": list(_SUPPORTED_RUNTIME_PROFILES),
            "hardware_dispatch": False,
            "http_paths": [
                "GET /api/runtime/context",
                "GET /api/runtime/events",
                "GET /api/runtime/profiles",
                "GET /api/runtime/runs",
                "GET /api/runtime/runs/{run_id}",
                "POST /api/runtime/voice-turn",
                "POST /api/runtime/runs/{run_id}/pause",
                "POST /api/runtime/runs/{run_id}/resume",
                "POST /api/runtime/runs/{run_id}/cancel",
                "POST /api/runtime/runs/{run_id}/advance",
            ],
        }

    def _build_arbiter(self, *, auto_complete: bool) -> RuntimeArbiter:
        if self.profile in EXTERNAL_RUNTIME_PROFILES:
            return ExternalRuntimeArbiter(
                run_service=self.run_service,
                safety_preflight=self.safety_preflight,
                skill_registry=self.skill_registry,
                client=self.runtime_arbiter_client,
                transport=self.executor_transport,
            )
        if self.profile == "shadow":
            return ShadowRuntimeArbiter(
                run_service=self.run_service,
                safety_preflight=self.safety_preflight,
                skill_registry=self.skill_registry,
            )
        if self.profile == "sim":
            return SimRuntimeArbiter(
                run_service=self.run_service,
                safety_preflight=self.safety_preflight,
                skill_registry=self.skill_registry,
                auto_complete=auto_complete,
            )
        return FakeRuntimeArbiter(
            run_service=self.run_service,
            safety_preflight=self.safety_preflight,
            skill_registry=self.skill_registry,
            auto_complete=auto_complete,
        )

    def _record_world_event(self, event: RuntimeEvent) -> None:
        record = getattr(self.world_state, "record_event", None)
        if callable(record):
            record(
                f"runtime.{event.event_type}",
                event.to_dict(),
                source=f"{self.profile}_runtime_arbiter",
                observed_at=event.created_at,
            )

    def _update_runtime_facts(self, run_payload: dict[str, Any]) -> None:
        update = getattr(self.world_state, "update_fact", None)
        if not callable(update):
            return
        update(
            "task.last_run",
            {
                "run_id": run_payload.get("run_id"),
                "state": run_payload.get("current_state"),
                "plan_id": run_payload.get("plan_id"),
                "profile": run_payload.get("profile") or self.profile,
            },
            source=f"{self.profile}_runtime_arbiter",
            stale_after_s=300.0,
        )


def _blocked_submission_payload(
    run_service: TaskRunService,
    run: TaskRun,
    handoff: TaskHandoff,
    assessment: SafetyAssessment,
) -> dict[str, Any]:
    proposal = run_service.record_replan_proposal(run, assessment, source="preflight")
    run_service.transition(
        run,
        "blocked",
        "task_blocked",
        "TaskRun blocked by safety preflight.",
        {
            "failed_checks": assessment.failed_checks,
            "recommended_fix": assessment.recommended_fix,
            "perception_requests": [item.to_dict() for item in assessment.perception_requests],
            "replan_proposal": proposal.to_dict(),
        },
    )
    return {
        "accepted": False,
        "status": "blocked",
        "reason": "preflight_failed",
        "preflight": assessment.to_dict(),
        "replan_proposal": proposal.to_dict(),
        "run": run.to_dict(),
        "handoff": handoff.to_dict(),
    }


def _runtime_client_assessment(
    handoff: TaskHandoff,
    profile: str,
    error: dict[str, Any],
) -> SafetyAssessment:
    code = str(error.get("code") or "external_runtime_not_ready")
    return SafetyAssessment(
        assessment_id=f"assessment-{uuid4().hex[:12]}",
        handoff_id=handoff.handoff_id,
        passed=False,
        failed_checks=[code],
        warnings=[],
        perception_requests=[],
        required_operator_confirmation=True,
        recommended_fix=str(
            error.get("remediation") or "Configure external runtime before handoff."
        ),
        profile_decision=_normalize_runtime_profile(profile),
    )


def _shadow_plan_for(
    handoff: TaskHandoff,
    skill_registry: SkillRegistry,
    assessment: SafetyAssessment,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for step in handoff.steps:
        definition = skill_registry.get(step.skill_name)
        items.append(
            {
                "sequence": step.sequence,
                "step_id": step.step_id,
                "skill_name": step.skill_name,
                "parameters": dict(step.parameters),
                "required_capabilities": list(definition.required_capabilities)
                if definition
                else [],
                "requires_confirmation": bool(step.requires_confirmation),
                "timeout_ms": step.timeout_ms,
                "would_dispatch_to": "runtime_arbiter",
                "hardware_dispatch": False,
            }
        )
    return {
        "mode": "shadow",
        "handoff_id": handoff.handoff_id,
        "plan_id": handoff.plan_id,
        "task_type": handoff.task_type,
        "risk_level": handoff.risk_level,
        "target_area": handoff.target_area,
        "preflight_passed": assessment.passed,
        "would_execute": items,
        "hardware_dispatch": False,
    }


def _sim_state(run: TaskRun) -> dict[str, Any]:
    total = len(run.handoff.steps)
    completed = max(0, min(int(run.current_step_index or 0), total))
    return {
        "mode": "sim",
        "total_steps": total,
        "completed_steps": completed,
        "remaining_steps": max(0, total - completed),
        "next_step": (run.handoff.steps[completed].to_dict() if completed < total else None),
    }


def _simulated_skill_result_payload(step: TaskStep) -> dict[str, Any]:
    area_id = str(step.parameters.get("area_id") or "")
    payload: dict[str, Any] = {
        "status": "completed",
        "observations": [
            {
                "type": "skill_status",
                "skill_name": step.skill_name,
                "area_id": area_id,
                "summary": f"Simulated {step.skill_name} completed.",
            }
        ],
        "artifacts": [],
        "metrics": {"simulated": True, "timeout_ms": step.timeout_ms},
        "confidence": 0.95,
    }
    if step.skill_name == "go_to_area":
        payload["observations"] = [
            {
                "type": "arrival",
                "area_id": area_id,
                "summary": f"Arrived at {area_id or 'target area'} in simulation.",
            }
        ]
    elif step.skill_name == "follow_patrol_route":
        payload["observations"] = [
            {
                "type": "route_progress",
                "area_id": area_id,
                "summary": f"Patrol route completed for {area_id or 'target area'} in simulation.",
            }
        ]
    elif step.skill_name == "inspect_equipment":
        payload["observations"] = [
            {
                "type": "inspection",
                "area_id": area_id,
                "finding": "simulated_normal",
                "summary": "No simulated anomaly detected.",
            }
        ]
    elif step.skill_name == "capture_image":
        payload["artifacts"] = [
            {
                "type": "image_ref",
                "uri": f"sim://{step.step_id}/image",
                "area_id": area_id,
            }
        ]
    elif step.skill_name == "generate_report":
        payload["observations"] = [
            {
                "type": "report_summary",
                "summary": "Simulated report data is ready.",
            }
        ]
    return payload


def default_skill_definitions() -> list[SkillDefinition]:
    common_abort = ("estop_active", "operator_cancelled", "runtime_safety_violation")
    return [
        SkillDefinition(
            "go_to_area",
            required_parameters=("area_id",),
            required_capabilities=("nav", "control"),
            preconditions=("localized", "route_available"),
            success_criteria=("arrived_at_area",),
            abort_conditions=common_abort,
            timeout_ms=120000,
            requires_confirmation=True,
        ),
        SkillDefinition(
            "follow_patrol_route",
            required_parameters=("area_id",),
            required_capabilities=("nav", "control"),
            preconditions=("route_available",),
            success_criteria=("route_completed",),
            abort_conditions=common_abort,
            timeout_ms=300000,
            requires_confirmation=True,
        ),
        SkillDefinition(
            "inspect_equipment",
            required_parameters=("area_id",),
            required_capabilities=("sense",),
            preconditions=("target_area_visible_or_mapped",),
            success_criteria=("inspection_observation_recorded",),
            abort_conditions=("target_missing", *common_abort),
            timeout_ms=90000,
        ),
        SkillDefinition(
            "capture_image",
            required_parameters=("area_id",),
            required_capabilities=("payload", "sense"),
            success_criteria=("image_reference_recorded",),
            abort_conditions=common_abort,
            timeout_ms=30000,
        ),
        SkillDefinition(
            "read_status_panel",
            required_parameters=("area_id",),
            required_capabilities=("sense",),
            success_criteria=("status_reading_recorded",),
            abort_conditions=common_abort,
            timeout_ms=45000,
        ),
        SkillDefinition(
            "wait",
            required_parameters=("duration_ms",),
            required_capabilities=("arbiter",),
            success_criteria=("wait_elapsed",),
            abort_conditions=("operator_cancelled",),
            timeout_ms=60000,
        ),
        SkillDefinition(
            "stop_and_hold",
            required_parameters=("area_id", "reason"),
            required_capabilities=("arbiter", "control"),
            preconditions=("localized",),
            success_criteria=("robot_stationary", "operator_notified"),
            abort_conditions=("estop_active", "operator_cancelled"),
            timeout_ms=15000,
        ),
        SkillDefinition(
            "safe_pause",
            required_parameters=("area_id", "reason"),
            required_capabilities=("arbiter", "control"),
            preconditions=("localized",),
            success_criteria=("current_task_paused", "robot_stationary"),
            abort_conditions=("estop_active", "operator_cancelled"),
            timeout_ms=15000,
        ),
        SkillDefinition(
            "retreat_to_safe_distance",
            required_parameters=("area_id", "minimum_distance_m"),
            required_capabilities=("nav", "control", "sense"),
            preconditions=("localized", "obstacle_or_person_tracked"),
            success_criteria=("safe_distance_reached",),
            abort_conditions=common_abort,
            timeout_ms=45000,
            requires_confirmation=True,
        ),
        SkillDefinition(
            "keep_distance_observe",
            required_parameters=("area_id", "minimum_distance_m"),
            required_capabilities=("nav", "sense"),
            preconditions=("localized",),
            success_criteria=("observation_recorded", "safe_distance_maintained"),
            abort_conditions=common_abort,
            timeout_ms=60000,
        ),
        SkillDefinition(
            "observe_then_recheck",
            required_parameters=("area_id", "recheck_after_ms"),
            required_capabilities=("sense", "arbiter"),
            preconditions=("target_area_visible_or_mapped",),
            success_criteria=("observation_recorded", "recheck_scheduled"),
            abort_conditions=("operator_cancelled", "runtime_safety_violation"),
            timeout_ms=90000,
        ),
        SkillDefinition(
            "record_then_continue",
            required_parameters=("area_id",),
            required_capabilities=("payload", "sense"),
            preconditions=("target_area_visible_or_mapped",),
            success_criteria=("evidence_recorded", "continue_authorized"),
            abort_conditions=("operator_cancelled", "runtime_safety_violation"),
            timeout_ms=45000,
        ),
        SkillDefinition(
            "low_speed_escort",
            required_parameters=("area_id", "destination"),
            required_capabilities=("nav", "control", "voice"),
            preconditions=("localized", "route_available", "interaction_target_locked"),
            success_criteria=("escort_completed_or_handed_off",),
            abort_conditions=common_abort,
            timeout_ms=300000,
            requires_confirmation=True,
        ),
        SkillDefinition(
            "return_home",
            required_capabilities=("nav", "control"),
            preconditions=("dock_or_home_known",),
            success_criteria=("arrived_home",),
            abort_conditions=common_abort,
            timeout_ms=180000,
            requires_confirmation=True,
        ),
        SkillDefinition(
            "generate_report",
            required_capabilities=("catalog",),
            success_criteria=("report_created",),
            abort_conditions=("report_store_unavailable",),
            timeout_ms=30000,
        ),
    ]


def _task_steps_for_plan(plan: dict[str, Any], *, task_type: str) -> list[TaskStep]:
    area_id = _target_area(plan, _mission_from_plan(plan)) or "area-unspecified"
    target_object = _target_object(plan, _mission_from_plan(plan))
    sequence: list[tuple[str, dict[str, Any]]] = []
    if task_type == "inspection_patrol":
        sequence = [
            ("go_to_area", {"area_id": area_id, "speed_limit": "safe"}),
            ("follow_patrol_route", {"area_id": area_id, "route_policy": "default"}),
            ("inspect_equipment", {"area_id": area_id, "target_object": target_object or ""}),
            ("capture_image", {"area_id": area_id, "purpose": "inspection"}),
            ("generate_report", {"format": "summary"}),
        ]
    elif task_type == "field_incident_response":
        sequence = _field_incident_response_sequence(plan, area_id=area_id)
    elif task_type == "visitor_escort":
        sequence = _visitor_escort_sequence(plan, area_id=area_id)
    elif task_type == "navigate_to":
        sequence = [("go_to_area", {"area_id": area_id, "speed_limit": "safe"})]
    elif task_type == "capture_evidence":
        sequence = [
            ("capture_image", {"area_id": area_id, "purpose": "evidence"}),
            ("generate_report", {"format": "evidence_summary"}),
        ]
    elif task_type == "status_report":
        sequence = [
            ("read_status_panel", {"area_id": area_id}),
            ("generate_report", {"format": "status_summary"}),
        ]
    elif plan.get("intent") == "manipulation":
        sequence = [("manipulate_object", {"target_object": target_object or ""})]
    else:
        sequence = [("generate_report", {"format": "operator_summary"})]
    return [
        _step_from_skill(name, parameters, index)
        for index, (name, parameters) in enumerate(sequence, start=1)
    ]


def _visitor_escort_sequence(
    plan: dict[str, Any],
    *,
    area_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    mission = _mission_from_plan(plan)
    raw_event = mission.get("field_event")
    event: dict[str, Any] = raw_event if isinstance(raw_event, dict) else {}
    destination = str(event.get("destination") or event.get("destination_name") or area_id)
    escort_parameters = {
        "area_id": area_id,
        "destination": destination,
        "destination_point_id": str(event.get("destination_point_id") or ""),
        "route_id": str(event.get("route_id") or ""),
        "map_id": str(event.get("map_id") or ""),
        "service_point_id": str(event.get("service_point_id") or ""),
        "speed_limit": str(event.get("speed_limit") or "low"),
        "interaction_policy": "visitor_must_remain_tracked",
    }
    return [
        ("low_speed_escort", escort_parameters),
        ("generate_report", {"format": "escort_summary"}),
    ]


def _field_incident_response_sequence(
    plan: dict[str, Any],
    *,
    area_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    mission = _mission_from_plan(plan)
    raw_event = mission.get("field_event")
    event: dict[str, Any] = raw_event if isinstance(raw_event, dict) else {}
    policy = str(event.get("robot_motion_policy") or "").strip().lower()
    scenario_id = str(event.get("scenario_id") or "field_event")
    destination = str(
        event.get("destination") or event.get("target_location") or event.get("location") or area_id
    )
    reason = f"{scenario_id}:{policy or 'field_incident'}"
    if policy in {"stop_and_hold", "hold_position"}:
        return [
            ("stop_and_hold", {"area_id": area_id, "reason": reason}),
            ("generate_report", {"format": "incident_summary"}),
        ]
    if policy in {"safe_pause", "pause_current_and_dispatch"}:
        sequence = [("safe_pause", {"area_id": area_id, "reason": reason})]
        if policy == "pause_current_and_dispatch":
            sequence.extend(
                [
                    ("go_to_area", {"area_id": area_id, "speed_limit": "safe"}),
                    ("capture_image", {"area_id": area_id, "purpose": "urgent_dispatch"}),
                ]
            )
        sequence.append(("generate_report", {"format": "incident_summary"}))
        return sequence
    if policy == "retreat_to_safe_distance":
        return [
            (
                "retreat_to_safe_distance",
                {"area_id": area_id, "minimum_distance_m": 2.0},
            ),
            ("capture_image", {"area_id": area_id, "purpose": "safety_evidence"}),
            ("generate_report", {"format": "incident_summary"}),
        ]
    if policy == "keep_distance_observe":
        return [
            ("keep_distance_observe", {"area_id": area_id, "minimum_distance_m": 2.0}),
            ("capture_image", {"area_id": area_id, "purpose": "security_evidence"}),
            ("generate_report", {"format": "incident_summary"}),
        ]
    if policy == "observe_then_recheck":
        return [
            ("observe_then_recheck", {"area_id": area_id, "recheck_after_ms": 1800000}),
            ("capture_image", {"area_id": area_id, "purpose": "recheck_evidence"}),
            ("generate_report", {"format": "incident_summary"}),
        ]
    if policy == "low_speed_escort":
        return [
            (
                "low_speed_escort",
                {"area_id": area_id, "destination": destination, "speed_limit": "low"},
            ),
            ("generate_report", {"format": "escort_summary"}),
        ]
    return [
        ("record_then_continue", {"area_id": area_id}),
        ("generate_report", {"format": "incident_summary"}),
    ]


def _step_from_skill(name: str, parameters: dict[str, Any], sequence: int) -> TaskStep:
    registry = SkillRegistry()
    definition = registry.get(name)
    return TaskStep(
        step_id=f"step-{sequence}",
        sequence=sequence,
        skill_name=name,
        parameters=dict(parameters),
        preconditions=list(definition.preconditions) if definition else [],
        success_criteria=list(definition.success_criteria) if definition else [],
        abort_conditions=list(definition.abort_conditions) if definition else [],
        timeout_ms=definition.timeout_ms if definition else 30000,
        requires_confirmation=bool(definition.requires_confirmation) if definition else False,
    )


def _mission_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    mission_wrapper = plan.get("mission")
    if isinstance(mission_wrapper, dict):
        mission = mission_wrapper.get("mission")
        if isinstance(mission, dict):
            return dict(mission)
    return {}


def _target_area(plan: dict[str, Any], mission: dict[str, Any]) -> str | None:
    for candidate in (
        mission.get("target"),
        mission.get("target_area"),
        mission.get("destination"),
    ):
        area = _area_label(candidate)
        if area:
            return area
    for step in mission.get("steps", []):
        if isinstance(step, dict) and step.get("target"):
            area = _area_label(step["target"])
            if area:
                return area
    reference = plan.get("reference", {})
    if isinstance(reference, dict):
        resolved = reference.get("resolved")
        if isinstance(resolved, dict):
            label = resolved.get("area_id") or resolved.get("zone") or resolved.get("label")
            if label:
                area = _area_label(label)
                if area:
                    return area
    return _infer_area(str(plan.get("goal") or mission.get("goal") or ""))


def _area_label(value: Any) -> str | None:
    text = str(value or "").strip().strip(":：;；")
    if not text:
        return None
    return _normalize_area_id(text) or _infer_area(text) or text


def _target_object(plan: dict[str, Any], mission: dict[str, Any]) -> str | None:
    reference = plan.get("reference", {})
    if isinstance(reference, dict):
        resolved = reference.get("resolved")
        if isinstance(resolved, dict):
            label = resolved.get("object_id") or resolved.get("label") or resolved.get("class_id")
            if label:
                return str(label)
    return _infer_object(str(plan.get("goal") or mission.get("goal") or ""))


def _risk_for(task_type: str, mission: dict[str, Any]) -> str:
    risk = str(mission.get("risk_tier") or "").strip().lower()
    if risk:
        return risk
    if task_type in {"inspection_patrol", "navigate_to"}:
        return "high"
    if task_type == "visitor_escort":
        return "high"
    if task_type == "capture_evidence":
        return "medium"
    if task_type == "status_report":
        return "low"
    return "medium"


def _confirmation_status(plan: dict[str, Any]) -> str:
    if bool(plan.get("handoff_ready")):
        return "confirmed"
    session = plan.get("session")
    if isinstance(session, dict) and session.get("confirmation_status"):
        return str(session["confirmation_status"])
    return "unconfirmed"


def _operator_roles(plan: dict[str, Any]) -> list[str]:
    session = plan.get("session") if isinstance(plan.get("session"), dict) else {}
    mission = _mission_from_plan(plan)
    values: list[Any] = []
    for source in (
        plan.get("operator_roles"),
        plan.get("roles"),
        session.get("operator_roles") if isinstance(session, dict) else None,
        mission.get("operator_roles"),
        mission.get("roles"),
    ):
        if isinstance(source, (list, tuple, set)):
            values.extend(source)
        elif source:
            values.append(source)
    return _normalize_roles(values)


def _normalize_roles(values: list[Any]) -> list[str]:
    roles = [str(item or "").strip().lower() for item in values]
    cleaned = _unique([role for role in roles if role])
    return cleaned or ["operator"]


def _normalize_runtime_profile(value: str) -> str:
    profile = str(value or "fake").strip().lower()
    return profile if profile in _SUPPORTED_RUNTIME_PROFILES else "fake"


def _normalize_remote_status(value: str) -> str:
    status = str(value or "").strip().lower().replace("-", "_")
    return _REMOTE_STATUS_ALIASES.get(status, status)


def _voice_context_from_handoff(handoff: TaskHandoff) -> dict[str, Any]:
    context = handoff.source_plan.get("voice_context", {})
    return dict(context) if isinstance(context, dict) else {}


def _external_submission_idempotency_key(handoff: TaskHandoff) -> str:
    voice_context = _voice_context_from_handoff(handoff)
    return str(voice_context.get("submission_id") or handoff.plan_id).strip()


def _remote_cursor_is_out_of_order(candidate: str, current: str) -> bool:
    if not candidate or not current:
        return False
    if candidate == current:
        return True
    left = re.fullmatch(r"(.*?)(\d+)", candidate)
    right = re.fullmatch(r"(.*?)(\d+)", current)
    if left is None or right is None or left.group(1) != right.group(1):
        return False
    return int(left.group(2)) <= int(right.group(2))


def _world_snapshot_id(snapshot: dict[str, Any]) -> str:
    updated_at = float(snapshot.get("updated_at", time.time()) or time.time())
    count = int(snapshot.get("fact_count", 0) or 0)
    return f"world-{int(updated_at * _UTC_TS_SCALE)}-{count}"


def _infer_area(text: str) -> str | None:
    normalized = text.replace(",", " ").replace("，", " ").replace("。", " ")
    explicit = re.search(
        r"\b(?:area|zone|checkpoint|route)-[a-z0-9_-]+\b", normalized, re.IGNORECASE
    )
    if explicit:
        return explicit.group(0).lower()
    compact = re.sub(r"\s+", "", normalized)
    letter_area = re.search(r"([a-zA-Z])区", compact)
    if letter_area:
        return f"area-{letter_area.group(1).lower()}"
    number_area = re.search(r"(\d+)区", compact)
    if number_area:
        return f"area-{number_area.group(1)}"
    chinese_area = re.search(r"([一二三四五六七八九十])区", compact)
    if chinese_area:
        return f"area-{_chinese_digit(chinese_area.group(1))}"
    for token in normalized.split():
        cleaned = token.strip(":：;；")
        area_id = _normalize_area_id(cleaned)
        if area_id:
            return area_id
    return None


def _infer_object(text: str) -> str | None:
    normalized = text.replace(",", " ").replace("，", " ").replace("。", " ")
    for token in normalized.split():
        cleaned = token.strip(":：;；")
        lowered = cleaned.lower()
        if lowered.startswith(("equipment-", "device-", "asset-", "panel-")):
            return cleaned
        if any(marker in cleaned for marker in ("设备", "面板", "阀", "仪表")):
            return cleaned
    return None


def _normalize_area_id(value: str) -> str | None:
    text = str(value or "").strip().strip(":：;；")
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith(("area-", "zone-", "checkpoint-", "route-")):
        return lowered
    compact = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"([a-zA-Z])区", compact)
    if match:
        return f"area-{match.group(1).lower()}"
    match = re.fullmatch(r"(\d+)区", compact)
    if match:
        return f"area-{match.group(1)}"
    match = re.fullmatch(r"([一二三四五六七八九十])区", compact)
    if match:
        return f"area-{_chinese_digit(match.group(1))}"
    return None


def _chinese_digit(value: str) -> str:
    mapping = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
    }
    return mapping.get(value, value)


def _robot_id_from_handoff(handoff: TaskHandoff) -> str | None:
    session = handoff.source_plan.get("session", {})
    if isinstance(session, dict) and session.get("robot_id"):
        return str(session["robot_id"])
    mission = _mission_from_plan(handoff.source_plan)
    if mission.get("robot_id"):
        return str(mission["robot_id"])
    return None


def _sanitize_runtime_operator_context(
    operator_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(operator_context, dict):
        return None
    cleaned: dict[str, Any] = {}
    operator_id = str(operator_context.get("operator_id") or "").strip()
    if operator_id:
        cleaned["operator_id"] = operator_id
    person_id = str(operator_context.get("person_id") or "").strip()
    if person_id:
        cleaned["person_id"] = person_id
    roles = operator_context.get("roles")
    if isinstance(roles, (list, tuple, set)):
        cleaned["roles"] = _unique([str(role).strip() for role in roles if str(role).strip()])
    if "authenticated" in operator_context:
        cleaned["authenticated"] = operator_context.get("authenticated") is True
    source = str(operator_context.get("source") or "").strip()
    if source:
        cleaned["source"] = source
    permission = str(operator_context.get("permission") or "").strip()
    if permission:
        cleaned["permission"] = permission
    thread_id = str(
        operator_context.get("thread_id")
        or operator_context.get("conversation_session_id")
        or ""
    ).strip()
    if thread_id:
        cleaned["thread_id"] = thread_id
    return cleaned


def _runtime_operator_context_failure(
    action: str,
    operator_id: str | None,
    operator_context: dict[str, Any] | None,
) -> str:
    operator_context = _sanitize_runtime_operator_context(operator_context)
    if not isinstance(operator_context, dict):
        return "runtime_operator_context_required"
    if operator_context.get("authenticated") is not True:
        return "runtime_operator_authentication_required"
    expected_permission = f"runtime:{action}"
    context_operator_id = str(operator_context.get("operator_id") or "").strip()
    requested_operator_id = str(operator_id or "").strip()
    if not context_operator_id or context_operator_id != requested_operator_id:
        return "runtime_operator_context_mismatch"
    roles = operator_context.get("roles")
    normalized_roles = {
        str(role).strip().lower() for role in roles or [] if str(role).strip()
    }
    if not isinstance(roles, list) or not normalized_roles.intersection({"operator", "admin"}):
        return "runtime_operator_context_incomplete"
    if not str(operator_context.get("source") or "").strip():
        return "runtime_operator_context_incomplete"
    if str(operator_context.get("permission") or "").strip() != expected_permission:
        return "runtime_control_permission_mismatch"
    return ""


def _runtime_operator_rejected_payload(
    run: TaskRun,
    action: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "handled": False,
        "reason": reason,
        "run": run.to_dict(),
        "reply": f"TaskRun {action} rejected: {reason}.",
    }


def _runtime_operator_provenance(
    *,
    operator_id: str | None,
    operator_roles: list[str] | tuple[str, ...] | None,
    operator_authenticated: bool | None,
    operator_source: str,
    runtime_permission: str,
    conversation_session_id: str | None,
) -> dict[str, Any]:
    """Return the non-secret identity fields safe for runtime audit records."""
    has_provenance = any(
        (
            operator_id is not None,
            operator_roles is not None,
            operator_authenticated is not None,
            bool(str(operator_source or "").strip()),
            bool(str(runtime_permission or "").strip()),
            bool(str(conversation_session_id or "").strip()),
        )
    )
    if not has_provenance:
        return {}

    payload: dict[str, Any] = {}
    cleaned_operator_id = str(operator_id or "").strip()
    if cleaned_operator_id:
        payload["operator_id"] = cleaned_operator_id
    if operator_roles is not None:
        payload["roles"] = _unique(
            [str(role).strip() for role in operator_roles if str(role).strip()]
        )
    if operator_authenticated is not None:
        payload["authenticated"] = operator_authenticated is True
    cleaned_source = str(operator_source or "").strip()
    if cleaned_source:
        payload["source"] = cleaned_source
    cleaned_permission = str(runtime_permission or "").strip()
    if cleaned_permission:
        payload["permission"] = cleaned_permission
    cleaned_session_id = str(conversation_session_id or "").strip()
    if cleaned_session_id:
        payload["thread_id"] = cleaned_session_id
    return payload


def _operator_action(
    action: str,
    operator_id: str,
    *,
    reason: str = "",
    risk_acknowledgement: bool = False,
    operator_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "action": action,
        "operator_id": operator_id,
        "reason": str(reason or ""),
        "risk_acknowledgement": bool(risk_acknowledgement),
        "created_at": time.time(),
    }
    cleaned_context = _sanitize_runtime_operator_context(operator_context)
    if cleaned_context:
        record["operator_context"] = cleaned_context
    return record


def _voice_turn_metadata(
    text: str,
    *,
    transcript_id: str = "",
    confidence: float | None = None,
    is_final: bool = True,
    channel: str = "voice",
    conversation_session_id: str | None = None,
    planning_session_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transcript_id": str(transcript_id or f"voice-turn-{uuid4().hex[:12]}"),
        "recognized_text": str(text or "").strip(),
        "is_final": bool(is_final),
        "channel": str(channel or "voice"),
        "safety_bypass_allowed": False,
        "created_at": time.time(),
    }
    if conversation_session_id:
        payload["conversation_session_id"] = str(conversation_session_id)
    if planning_session_id:
        payload["planning_session_id"] = str(planning_session_id)
    if confidence is not None:
        payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
    return payload


def _approval_plan_digest(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unconfirmed_approval_plan(plan: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(plan, ensure_ascii=False))
    normalized["handoff_ready"] = False
    session = normalized.get("session")
    if isinstance(session, dict):
        session["confirmation_status"] = "pending_confirmation"
    mission_container = normalized.get("mission")
    mission = (
        mission_container.get("mission")
        if isinstance(mission_container, dict)
        else None
    )
    if isinstance(mission, dict):
        mission["status"] = "pending_confirmation"
    return normalized


def _run_summary(run: TaskRun) -> str:
    if run.current_state == "completed":
        return (
            f"Completed {run.handoff.task_type} in {run.profile} runtime with "
            f"{len(run.handoff.steps)} step(s)."
        )
    if run.current_state == "shadowed":
        return (
            f"Shadowed {run.handoff.task_type} with {len(run.handoff.steps)} would-execute step(s)."
        )
    if run.current_state == "blocked":
        checks = []
        if run.safety_assessments:
            checks = run.safety_assessments[-1].failed_checks
        return "Blocked by safety preflight: " + (", ".join(checks) if checks else "unknown")
    if run.current_state == "cancelled":
        return "Cancelled by operator before completion."
    return f"TaskRun ended with state {run.current_state}."


def _perception_request(
    reason: str,
    required_facts: list[str],
    *,
    suggested_sources: list[str],
    handoff: TaskHandoff,
    priority: str = "normal",
    ttl_s: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> ActivePerceptionRequest:
    now = time.time()
    return ActivePerceptionRequest(
        request_id=f"perception-{uuid4().hex[:12]}",
        reason=str(reason or "refresh_facts"),
        required_facts=[str(item) for item in required_facts if str(item).strip()],
        suggested_sources=[str(item) for item in suggested_sources if str(item).strip()],
        target_area=handoff.target_area,
        target_object=handoff.target_object,
        priority=str(priority or "normal"),
        created_at=now,
        expires_at=now + max(1.0, float(ttl_s)),
        metadata=dict(metadata or {}),
    )


def _unique_perception_requests(
    requests: list[ActivePerceptionRequest],
) -> list[ActivePerceptionRequest]:
    merged: dict[tuple[str, str, str], ActivePerceptionRequest] = {}
    order: list[tuple[str, str, str]] = []
    for request in requests:
        key = (
            request.reason,
            request.target_area or "",
            request.target_object or "",
        )
        current = merged.get(key)
        if current is None:
            merged[key] = request
            order.append(key)
            continue
        expires_at = None
        if current.expires_at is not None and request.expires_at is not None:
            expires_at = min(current.expires_at, request.expires_at)
        else:
            expires_at = current.expires_at or request.expires_at
        merged[key] = ActivePerceptionRequest(
            request_id=current.request_id,
            reason=current.reason,
            required_facts=_unique(current.required_facts + request.required_facts),
            suggested_sources=_unique(current.suggested_sources + request.suggested_sources),
            target_area=current.target_area,
            target_object=current.target_object,
            priority="high" if "high" in {current.priority, request.priority} else current.priority,
            created_at=min(current.created_at, request.created_at),
            expires_at=expires_at,
            metadata={**current.metadata, **request.metadata},
        )
    return [merged[key] for key in order]


def _replan_proposal_for(
    run: TaskRun,
    assessment: SafetyAssessment,
    *,
    source: str,
) -> ReplanProposal:
    failed = list(assessment.failed_checks)
    action = _recommended_replan_action(failed)
    perception_requests = [item.to_dict() for item in assessment.perception_requests]
    proposed_actions = _replan_actions_for(action, run, assessment)
    now = time.time()
    return ReplanProposal(
        proposal_id=f"replan-{uuid4().hex[:12]}",
        run_id=run.run_id,
        source=str(source or "preflight"),
        reason=",".join(failed) if failed else "runtime_blocked",
        recommended_action=action,
        proposed_actions=proposed_actions,
        perception_requests=perception_requests,
        safety_notes=_replan_safety_notes(action, failed),
        operator_confirmation_required=True,
        created_at=now,
        expires_at=now + 300.0,
    )


def _recommended_replan_action(failed: list[str]) -> str:
    if "estop_active" in failed or "dog_safety_estop_active" in failed:
        return "clear_estop_then_retry"
    if "world_state_stale" in failed:
        return "refresh_world_state_then_reconfirm"
    if (
        "localization_unavailable" in failed
        or "map_localization_unavailable" in failed
        or "localization_quality_low" in failed
    ):
        return "refresh_localization_then_replan"
    if "target_area_unknown" in failed:
        return "load_site_catalog_or_clarify_area"
    if "target_area_blocked" in failed:
        return "choose_allowed_area"
    if "map_id_mismatch" in failed or "map_version_mismatch" in failed:
        return "refresh_map_state_then_replan"
    if "target_device_unknown" in failed:
        return "clarify_or_register_device"
    if "operator_not_authorized" in failed or "supervisor_confirmation_required" in failed:
        return "request_authorized_operator_review"
    if any(item.startswith("unregistered_skill") for item in failed):
        return "rewrite_plan_with_registered_skills"
    return "operator_review_required"


def _replan_actions_for(
    action: str,
    run: TaskRun,
    assessment: SafetyAssessment,
) -> list[dict[str, Any]]:
    target_area = run.handoff.target_area
    if action == "clear_estop_then_retry":
        return [
            {
                "type": "operator_action",
                "action": "clear_estop",
                "confirmation_required": True,
            },
            {"type": "retry_handoff", "plan_id": run.handoff.plan_id},
        ]
    if action == "refresh_world_state_then_reconfirm":
        return [
            {"type": "request_perception", "facts": ["robot", "environment", "map", "scene"]},
            {"type": "operator_reconfirm", "plan_id": run.handoff.plan_id},
        ]
    if action == "refresh_localization_then_replan":
        return [
            {
                "type": "request_perception",
                "facts": ["map.localized", "map.localization_quality"],
                "target_area": target_area,
            },
            {"type": "replan_route", "target_area": target_area},
        ]
    if action == "load_site_catalog_or_clarify_area":
        return [
            {"type": "request_catalog", "facts": ["environment.areas"]},
            {"type": "operator_clarification", "field": "target_area", "value": target_area},
        ]
    if action == "choose_allowed_area":
        return [
            {"type": "operator_clarification", "field": "target_area", "value": target_area},
            {"type": "replan_with_allowed_area"},
        ]
    if action == "refresh_map_state_then_replan":
        return [
            {"type": "request_perception", "facts": ["map.current_id", "map.current_version"]},
            {"type": "replan_against_active_map", "target_area": target_area},
        ]
    if action == "clarify_or_register_device":
        return [
            {"type": "request_catalog", "facts": ["environment.devices"]},
            {
                "type": "operator_clarification",
                "field": "target_object",
                "value": run.handoff.target_object,
            },
        ]
    if action == "request_authorized_operator_review":
        return [
            {
                "type": "operator_authorization",
                "required_roles": ["operator", "supervisor", "admin"],
            },
            {"type": "operator_reconfirm", "plan_id": run.handoff.plan_id},
        ]
    if action == "rewrite_plan_with_registered_skills":
        return [{"type": "rewrite_handoff", "allowed_skills_only": True}]
    return [
        {
            "type": "operator_review",
            "failed_checks": list(assessment.failed_checks),
        }
    ]


def _replan_safety_notes(action: str, failed: list[str]) -> list[str]:
    notes = ["Proposal is advisory and requires operator confirmation before a new handoff."]
    if action in {"clear_estop_then_retry", "refresh_localization_then_replan"}:
        notes.append("Do not dispatch movement until safety preflight passes again.")
    if "target_area_blocked" in failed:
        notes.append("Blocked or restricted areas require supervisor review before reuse.")
    return notes


def _recommended_fix(failed: list[str]) -> str:
    if "world_state_stale" in failed:
        return "Refresh perception/world state and re-confirm the task."
    if "estop_active" in failed or "dog_safety_estop_active" in failed:
        return "Clear E-STOP through the safety service before retrying."
    if "dog_safety_unavailable" in failed or "dog_safety_unconfigured" in failed:
        return "Connect dog-safety-service or disable strict dog safety preflight."
    if "target_area_required" in failed:
        return "Provide a target area or route before handoff."
    if "target_area_unknown" in failed:
        return "Register the target area in the site catalog before handoff."
    if "target_area_blocked" in failed:
        return "Choose an allowed area or update the site catalog after supervisor review."
    if "map_id_mismatch" in failed or "map_version_mismatch" in failed:
        return "Refresh localization/map state or replan against the active map."
    if (
        "localization_unavailable" in failed
        or "map_localization_unavailable" in failed
        or "localization_quality_low" in failed
    ):
        return "Restore localization quality before movement."
    if "target_device_unknown" in failed:
        return "Register the target device or ask the operator to clarify the asset."
    if "operator_not_authorized" in failed:
        return "Use an operator, supervisor, or admin role for runtime handoff."
    if "supervisor_confirmation_required" in failed:
        return "Ask a supervisor or admin to confirm this higher-risk task."
    if any(item.startswith("unregistered_skill") for item in failed):
        return "Rewrite the plan using registered high-level skills only."
    if "operator_confirmation_required" in failed:
        return "Ask the operator to confirm before runtime handoff."
    return "Review failed checks and submit a revised plan."


def _find_world_item(items: list[Any], key: str, value: str) -> dict[str, Any] | None:
    target = str(value or "").strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get(key, "")).strip().lower() == target:
            return dict(item)
    return None


def _area_blocked(area: dict[str, Any]) -> bool:
    status = str(area.get("status") or "").strip().lower()
    if area.get("allowed") is False:
        return True
    return status in {"blocked", "disabled", "no_go", "no-go", "closed", "restricted"}


def _assess_area_map(
    area: dict[str, Any],
    snapshot: dict[str, Any],
    failed: list[str],
    warnings: list[str],
    perception_requests: list[ActivePerceptionRequest],
    handoff: TaskHandoff,
) -> None:
    map_state = snapshot.get("map", {})
    if not isinstance(map_state, dict):
        warnings.append("map_state_unavailable")
        perception_requests.append(
            _perception_request(
                "refresh_map_state",
                ["map.current_id", "map.current_version", "map.localized"],
                suggested_sources=["nav"],
                handoff=handoff,
            )
        )
        return
    area_map_id = str(area.get("map_id") or "").strip()
    current_map_id = str(map_state.get("current_id") or "").strip()
    if area_map_id and current_map_id and area_map_id != current_map_id:
        failed.append("map_id_mismatch")
        perception_requests.append(
            _perception_request(
                "refresh_map_state",
                ["map.current_id"],
                suggested_sources=["nav"],
                handoff=handoff,
                priority="high",
            )
        )
    area_map_version = str(area.get("map_version") or "").strip()
    current_map_version = str(map_state.get("current_version") or "").strip()
    if area_map_version and current_map_version and area_map_version != current_map_version:
        failed.append("map_version_mismatch")
        perception_requests.append(
            _perception_request(
                "refresh_map_state",
                ["map.current_version"],
                suggested_sources=["nav"],
                handoff=handoff,
                priority="high",
            )
        )
    if area_map_id and not current_map_id:
        warnings.append("map_id_unknown")
        perception_requests.append(
            _perception_request(
                "refresh_map_state",
                ["map.current_id"],
                suggested_sources=["nav"],
                handoff=handoff,
            )
        )
    if area_map_version and not current_map_version:
        warnings.append("map_version_unknown")
        perception_requests.append(
            _perception_request(
                "refresh_map_state",
                ["map.current_version"],
                suggested_sources=["nav"],
                handoff=handoff,
            )
        )


def _dog_safety_is_configured(client: Any) -> bool | None:
    is_configured = getattr(client, "is_configured", None)
    if not callable(is_configured):
        return None
    try:
        return bool(is_configured())
    except Exception:
        return None


def _dog_safety_estop_active(client: Any) -> bool | None:
    # Use the client's cached/non-blocking view. Runtime handoff preflight must
    # not perform network queries or dispatch any hardware/control operation.
    is_active = getattr(client, "is_estop_active", None)
    if not callable(is_active):
        return None
    try:
        return bool(is_active())
    except Exception:
        return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
