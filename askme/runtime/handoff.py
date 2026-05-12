"""Auditable fake runtime handoff for confirmed cognitive plans.

This layer starts where cognition stops. It converts a confirmed, high-level
plan into a structured handoff, runs local safety preflight, and drives an
in-memory fake arbiter. It never calls hardware, gait, motor, or control
service APIs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from askme.runtime.arbiter_client import EXTERNAL_RUNTIME_PROFILES, RuntimeArbiterClient
from askme.runtime.audit import RuntimeAuditConfig, RuntimeAuditLog

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "blocked", "shadowed"})
_SUPPORTED_RUNTIME_PROFILES = ("fake", "shadow", "sim", "external", "lab")
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
            name
            for name in skill.required_parameters
            if step.parameters.get(name) in (None, "")
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
                plan.get("session", {}).get("operator_id") if isinstance(plan.get("session"), dict) else None,
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
                str(item)
                for item in plan.get("safety_constraints", [])
                if str(item).strip()
            ],
            steps=steps,
            risk_level=_risk_for(task_type, mission),
            required_capabilities=skill_registry.capabilities_for(steps),
            missing_info=[
                str(item)
                for item in plan.get("missing_inputs", [])
                if str(item).strip()
            ],
            confirmation_status=confirmation_status,
            world_state_snapshot_id=snapshot_id,
            safety_notes=[
                str(item)
                for item in mission.get("safety_notes", [])
                if str(item).strip()
            ],
            created_at=current,
            expires_at=current + max(1.0, float(ttl_s)),
            planner_version=planner_version,
            source_plan=dict(plan),
            world_state_snapshot=dict(world_state_snapshot),
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

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.path is not None)

    def load_runs(self) -> list[TaskRun]:
        if not self.enabled or self.path is None or not self.path.exists():
            return []
        try:
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
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "runs": [run.to_dict() for run in runs],
            }
            tmp_path = self.path.with_name(f"{self.path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
            os.replace(tmp_path, self.path)
        except OSError:
            if not self.config.swallow_errors:
                raise
            logger.exception("TaskRun store save failed for %s", self.path)


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
            _runtime_event_from_dict(item)
            for item in _dict_items(payload.get("runtime_events"))
        ],
        safety_assessments=[
            _safety_assessment_from_dict(item)
            for item in _dict_items(payload.get("safety_assessments"))
        ],
        skill_results=[
            _skill_result_from_dict(item)
            for item in _dict_items(payload.get("skill_results"))
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
        sim_state=dict(payload["sim_state"]) if isinstance(payload.get("sim_state"), dict) else None,
    )
    return run


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
            str(payload.get("target_object"))
            if payload.get("target_object") is not None
            else None
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
        required_operator_confirmation=bool(
            payload.get("required_operator_confirmation", False)
        ),
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
            str(payload.get("target_object"))
            if payload.get("target_object") is not None
            else None
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
        operator_confirmation_required=bool(
            payload.get("operator_confirmation_required", True)
        ),
        status=str(payload.get("status") or "proposed"),
        created_at=float(payload.get("created_at") or time.time()),
        expires_at=_optional_float(payload.get("expires_at")),
    )


def _dict_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


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
                artifact
                for result in skill_results
                for artifact in result.get("artifacts", [])
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
        self._runs: dict[str, TaskRun] = {}
        self._report_service = report_service or TaskReportService()
        self._event_sink = event_sink
        self._audit_log = audit_log or RuntimeAuditLog()
        self._store = store if isinstance(store, TaskRunStore) else TaskRunStore(store)
        self.max_runs = max(1, int(max_runs))
        for run in self._store.load_runs():
            self._runs[run.run_id] = run
        self._trim_runs(persist=False)

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
        event = self.emit(run, event_type, state, message, payload or {})
        if state in _TERMINAL_STATES:
            run.result_summary = _run_summary(run)
            run.report = self._report_service.build_report(run)
            self._audit_log.append_terminal_snapshot(run)
            self._persist_runs()
        return event

    def emit(
        self,
        run: TaskRun,
        event_type: str,
        state: str,
        message: str,
        payload: dict[str, Any] | None = None,
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
        return event

    def pause(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
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
        )
        self.transition(run, "paused", "task_paused", "TaskRun paused by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun paused."}

    def resume(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
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
        )
        self.transition(run, "executing", "execution_resumed", "TaskRun resumed by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun resumed."}

    def cancel(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        run = self.require(run_id)
        if run.terminal:
            return self._control_rejected(run, "cancel", "run_already_terminal")
        self._record_operator_action(
            run,
            "cancel",
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )
        self.transition(run, "cancel_requested", "cancel_requested", "TaskRun cancel requested.")
        self.transition(run, "cancelled", "task_cancelled", "TaskRun cancelled by operator.")
        return {"handled": True, "run": run.to_dict(), "reply": "TaskRun cancelled."}

    def advance(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
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

    def get(self, run_id: str) -> TaskRun | None:
        return self._runs.get(str(run_id or "").strip())

    def require(self, run_id: str) -> TaskRun:
        run = self.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

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

    def _record_operator_action(
        self,
        run: TaskRun,
        action: str,
        operator_id: str,
        *,
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> None:
        record = _operator_action(
            action,
            operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )
        run.operator_actions.append(record)
        self._audit_log.append_operator_action(run, record)
        self._persist_runs()

    def persist(self) -> None:
        self._persist_runs()

    def _persist_runs(self) -> None:
        self._store.save_runs(self.runs())

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
        self.run_service.transition(run, "validating", "plan_validated", "TaskHandoff schema validated.")
        self.run_service.transition(run, "preflight", "preflight_started", "Safety preflight started.")
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
        self.run_service.transition(run, "executing", "execution_started", "TaskRun execution started.")
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
        self.run_service.transition(run, "completed", "task_completed", "TaskRun completed in fake runtime.")


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
        self.run_service.transition(run, "validating", "plan_validated", "TaskHandoff schema validated.")
        self.run_service.transition(run, "preflight", "preflight_started", "Safety preflight started.")
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
        self.run_service.transition(run, "validating", "plan_validated", "TaskHandoff schema validated.")
        self.run_service.transition(run, "preflight", "preflight_started", "Safety preflight started.")
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
    ) -> None:
        self.profile = client.profile
        self.run_service = run_service
        self.safety_preflight = safety_preflight
        self.skill_registry = skill_registry
        self.client = client

    def submit(self, handoff: TaskHandoff) -> dict[str, Any]:
        run = self.run_service.create(handoff, profile=self.profile)
        self.run_service.transition(run, "submitted", "plan_submitted", "TaskHandoff submitted.")
        self.run_service.transition(run, "validating", "plan_validated", "TaskHandoff schema validated.")

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

        self.run_service.transition(run, "preflight", "preflight_started", "Safety preflight started.")
        assessment = self.safety_preflight.assess(
            handoff,
            skill_registry=self.skill_registry,
            profile=self.profile,
        )
        self.run_service.add_safety_assessment(run, assessment)
        if not assessment.passed:
            return _blocked_submission_payload(self.run_service, run, handoff, assessment)

        envelope = self.client.submission_envelope(handoff.to_dict())
        self.run_service.transition(
            run,
            "queued",
            "external_runtime_contract_ready",
            "External runtime contract envelope is ready.",
            {"runtime_client": envelope},
        )
        return {
            "accepted": True,
            "status": run.current_state,
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
        self.arbiter = self._build_arbiter(auto_complete=auto_complete)

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
        run = self.run_service.get(run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        if run.report is None:
            run.report = self.report_service.build_report(run)
            self.run_service.persist()
        return {"report": run.report}

    def pause_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        try:
            result = self.run_service.pause(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
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
    ) -> dict[str, Any]:
        try:
            result = self.run_service.resume(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
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
    ) -> dict[str, Any]:
        try:
            result = self.run_service.cancel(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
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
    ) -> dict[str, Any]:
        try:
            result = self.run_service.advance(
                run_id,
                operator_id=operator_id,
                reason=reason,
                risk_acknowledgement=risk_acknowledgement,
            )
        except KeyError:
            return {"handled": False, "error": "run not found", "run_id": run_id}
        self._update_runtime_facts(result["run"])
        return result

    def handle_chat_control(self, text: str) -> dict[str, Any] | None:
        intent = _runtime_control_intent(text)
        if intent is None:
            return None
        active = self.run_service.active_run()
        latest = active or (self.run_service.runs()[0] if self.run_service.runs() else None)
        if latest is None:
            if intent != "status":
                return None
            return {
                "handled": True,
                "reply": "No TaskRun is active yet.",
                "runtime": self.context_payload(),
            }
        if intent == "status":
            return {
                "handled": True,
                "reply": f"TaskRun {latest.run_id} is {latest.current_state}.",
                "runtime": {"run": latest.to_dict(), "active_run": active.to_dict() if active else None},
            }
        if active is None:
            return None
        if intent == "pause":
            result = self.pause_payload(latest.run_id)
        elif intent == "resume":
            result = self.resume_payload(latest.run_id)
        else:
            result = self.cancel_payload(latest.run_id)
        return {
            "handled": True,
            "reply": result.get("reply", ""),
            "runtime": result,
        }

    def voice_turn_payload(
        self,
        text: str,
        *,
        speak: bool = False,
        transcript_id: str = "",
        confidence: float | None = None,
        is_final: bool = True,
        channel: str = "voice",
    ) -> dict[str, Any]:
        recognized = str(text or "").strip()
        voice_turn = _voice_turn_metadata(
            recognized,
            transcript_id=transcript_id,
            confidence=confidence,
            is_final=is_final,
            channel=channel,
        )
        if not recognized:
            return {
                "handled": False,
                "reason": "empty_transcript",
                "reply": "",
                "spoken": False if speak else None,
                "voice_turn": voice_turn,
            }
        control = self.handle_chat_control(recognized)
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
            "runtime_control_intent": _runtime_control_intent(recognized),
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
        recommended_fix=str(error.get("remediation") or "Configure external runtime before handoff."),
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
                "required_capabilities": list(definition.required_capabilities) if definition else [],
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
        "next_step": (
            run.handoff.steps[completed].to_dict()
            if completed < total
            else None
        ),
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


def _field_incident_response_sequence(
    plan: dict[str, Any],
    *,
    area_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    mission = _mission_from_plan(plan)
    event = mission.get("field_event") if isinstance(mission.get("field_event"), dict) else {}
    policy = str(event.get("robot_motion_policy") or "").strip().lower()
    scenario_id = str(event.get("scenario_id") or "field_event")
    destination = str(
        event.get("destination")
        or event.get("target_location")
        or event.get("location")
        or area_id
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
    for step in mission.get("steps", []):
        if isinstance(step, dict) and step.get("target"):
            area = _normalize_area_id(str(step["target"])) or _infer_area(str(step["target"]))
            if area:
                return area
    reference = plan.get("reference", {})
    if isinstance(reference, dict):
        resolved = reference.get("resolved")
        if isinstance(resolved, dict):
            label = resolved.get("area_id") or resolved.get("zone") or resolved.get("label")
            if label:
                area = _normalize_area_id(str(label)) or _infer_area(str(label))
                if area:
                    return area
    return _infer_area(str(plan.get("goal") or mission.get("goal") or ""))


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


def _runtime_control_intent(text: str) -> str | None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    if any(token in lowered for token in ("pause", "hold", "stop for now", "暂停", "停一下", "先停", "等一下")):
        return "pause"
    if any(token in lowered for token in ("resume", "continue", "go on", "继续", "恢复", "接着执行")):
        return "resume"
    if any(token in lowered for token in ("cancel task", "cancel run", "取消任务", "取消执行", "终止任务")):
        return "cancel"
    if any(token in lowered for token in ("status", "progress", "where are we", "执行到哪", "现在状态", "任务状态", "到哪了")):
        return "status"
    return None


def _normalize_runtime_profile(value: str) -> str:
    profile = str(value or "fake").strip().lower()
    return profile if profile in _SUPPORTED_RUNTIME_PROFILES else "fake"


def _world_snapshot_id(snapshot: dict[str, Any]) -> str:
    updated_at = float(snapshot.get("updated_at", time.time()) or time.time())
    count = int(snapshot.get("fact_count", 0) or 0)
    return f"world-{int(updated_at * _UTC_TS_SCALE)}-{count}"


def _infer_area(text: str) -> str | None:
    normalized = text.replace(",", " ").replace("，", " ").replace("。", " ")
    explicit = re.search(r"\b(?:area|zone|checkpoint|route)-[a-z0-9_-]+\b", normalized, re.IGNORECASE)
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


def _operator_action(
    action: str,
    operator_id: str,
    *,
    reason: str = "",
    risk_acknowledgement: bool = False,
) -> dict[str, Any]:
    return {
        "action": action,
        "operator_id": operator_id,
        "reason": str(reason or ""),
        "risk_acknowledgement": bool(risk_acknowledgement),
        "created_at": time.time(),
    }


def _voice_turn_metadata(
    text: str,
    *,
    transcript_id: str = "",
    confidence: float | None = None,
    is_final: bool = True,
    channel: str = "voice",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transcript_id": str(transcript_id or f"voice-turn-{uuid4().hex[:12]}"),
        "recognized_text": str(text or "").strip(),
        "is_final": bool(is_final),
        "channel": str(channel or "voice"),
        "safety_bypass_allowed": False,
        "created_at": time.time(),
    }
    if confidence is not None:
        payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
    return payload


def _run_summary(run: TaskRun) -> str:
    if run.current_state == "completed":
        return (
            f"Completed {run.handoff.task_type} in {run.profile} runtime with "
            f"{len(run.handoff.steps)} step(s)."
        )
    if run.current_state == "shadowed":
        return (
            f"Shadowed {run.handoff.task_type} with "
            f"{len(run.handoff.steps)} would-execute step(s)."
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
