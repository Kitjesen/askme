"""Mission drafting and safe dry-run submission for industrial inspection flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import requests

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_VALID_RISKS = frozenset(_RISK_ORDER)
_UTC = timezone.utc  # noqa: UP017 - Sunrise runs Python 3.10, where datetime.UTC is unavailable.


@dataclass
class MissionStep:
    """One auditable, high-level runtime step.

    Steps are intentionally not motor commands. They describe what the runtime
    arbiter should coordinate through safety, navigation, control, and payload
    services.
    """

    step_id: str
    sequence: int
    action: str
    capability: str
    target: str | None = None
    risk_tier: str = "medium"
    requires_confirmation: bool = False
    service: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MissionStep:
        return cls(
            step_id=str(payload.get("step_id") or payload.get("id") or uuid4().hex[:8]),
            sequence=int(payload.get("sequence", 0)),
            action=str(payload.get("action", "")),
            capability=str(payload.get("capability", "")),
            target=_clean_optional(payload.get("target")),
            risk_tier=_clean_risk(payload.get("risk_tier", payload.get("risk_level", "medium"))),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
            service=_clean_optional(payload.get("service")),
            notes=[str(item) for item in payload.get("notes", [])],
        )


@dataclass
class MissionPlan:
    """Operator-facing mission object produced by askme."""

    mission_id: str
    goal: str
    mission_type: str
    source_text: str
    risk_tier: str
    required_services: list[str]
    steps: list[MissionStep]
    requested_capability: str
    requested_by: str
    adapter_plan_id: str
    idempotency_key: str
    status: str = "draft"
    requires_confirmation: bool = False
    approval_required: bool = False
    auto_approve: bool = False
    risk_reason: str = ""
    confirmation_prompt: str = ""
    operator_id: str = "askme.operator"
    robot_id: str | None = None
    site_id: str | None = None
    channel: str = "text"
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    evidence: list[dict[str, Any]] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MissionPlan:
        steps = [
            MissionStep.from_dict(step)
            for step in payload.get("steps", [])
            if isinstance(step, dict)
        ]
        risk = _clean_risk(payload.get("risk_tier", payload.get("risk_level", "medium")))
        mission_id = str(payload.get("mission_id") or payload.get("id") or _new_mission_id())
        operator_id = str(payload.get("operator_id", "askme.operator")).strip() or "askme.operator"
        return cls(
            mission_id=mission_id,
            goal=str(payload.get("goal", "")).strip(),
            mission_type=str(payload.get("mission_type", "custom")).strip() or "custom",
            source_text=str(payload.get("source_text") or payload.get("goal", "")).strip(),
            risk_tier=risk,
            required_services=[
                str(item)
                for item in payload.get("required_services", [])
                if str(item).strip()
            ],
            steps=steps,
            requested_capability=str(
                payload.get("requested_capability") or _requested_capability(
                    str(payload.get("mission_type", "custom")).strip() or "custom"
                )
            ),
            requested_by=str(payload.get("requested_by") or operator_id),
            adapter_plan_id=str(payload.get("adapter_plan_id") or mission_id),
            idempotency_key=str(payload.get("idempotency_key") or f"mission-{mission_id}"),
            status=str(payload.get("status", "draft")).strip() or "draft",
            requires_confirmation=bool(
                payload.get("requires_confirmation", _risk_at_least(risk, "high"))
            ),
            approval_required=bool(payload.get("approval_required", _risk_at_least(risk, "medium"))),
            auto_approve=bool(payload.get("auto_approve", not _risk_at_least(risk, "medium"))),
            risk_reason=str(payload.get("risk_reason", "")),
            confirmation_prompt=str(payload.get("confirmation_prompt", "")),
            operator_id=operator_id,
            robot_id=_clean_optional(payload.get("robot_id")),
            site_id=_clean_optional(payload.get("site_id")),
            channel=str(payload.get("channel", "text")).strip() or "text",
            created_at=str(payload.get("created_at") or _utc_now()),
            updated_at=str(payload.get("updated_at") or _utc_now()),
            evidence=[
                dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
            safety_notes=[
                str(item)
                for item in payload.get("safety_notes", [])
                if str(item).strip()
            ],
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), dict)
            else {},
        )


@dataclass
class InspectionReport:
    """Report shell created from mission evidence and current status."""

    report_id: str
    mission_id: str
    site: str | None
    status: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    media: list[dict[str, Any]] = field(default_factory=list)
    operator_summary: str = ""
    machine_summary: str = ""
    created_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MissionService:
    """Draft missions and submit them only through the runtime arbiter.

    By default this service is dry-run only. Real submission requires explicit
    config and caller confirmation, so local voice/text paths cannot silently
    control hardware.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        root_cfg = config or {}
        runtime_cfg = root_cfg.get("runtime", {}) if isinstance(root_cfg, dict) else {}
        mission_cfg = runtime_cfg.get("mission", {})
        arbiter_cfg = runtime_cfg.get("arbiter", {})
        voice_cfg = runtime_cfg.get("voice_bridge", {})

        self.enabled = bool(mission_cfg.get("enabled", True))
        self.submit_enabled = bool(mission_cfg.get("submit_enabled", False))
        self.base_url = str(
            mission_cfg.get("base_url")
            or arbiter_cfg.get("base_url")
            or ""
        ).rstrip("/")
        self.operator_id = str(
            mission_cfg.get("operator_id")
            or voice_cfg.get("operator_id")
            or "askme.operator"
        ).strip()
        self.robot_id = _clean_optional(
            mission_cfg.get("robot_id") or voice_cfg.get("robot_id")
        )
        self.site_id = _clean_optional(
            mission_cfg.get("site_id") or voice_cfg.get("site_id")
        )
        self.timeout_s = float(mission_cfg.get("timeout", 3.0))
        self.confirmation_threshold = _clean_risk(
            mission_cfg.get("confirmation_threshold", "medium")
        )
        self.api_key = _clean_optional(
            mission_cfg.get("api_key")
            or arbiter_cfg.get("api_key")
            or runtime_cfg.get("api_key")
        )
        self._missions: dict[str, MissionPlan] = {}
        self._submissions: dict[str, dict[str, Any]] = {}

    def draft(
        self,
        text: str,
        *,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        channel: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> MissionPlan:
        if not self.enabled:
            raise RuntimeError("mission service is disabled")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("mission text is required")

        mission_type = _infer_mission_type(cleaned)
        target = _infer_target(cleaned)
        risk = _infer_risk(cleaned, mission_type)
        required_services = _required_services(mission_type, risk)
        requires_confirmation = _risk_at_least(risk, self.confirmation_threshold)
        approval_required = _risk_at_least(risk, "medium")
        auto_approve = not approval_required
        status = "pending_confirmation" if requires_confirmation else "draft"
        mission_id = _new_mission_id()
        steps = _build_steps(
            mission_type,
            target=target,
            risk_tier=risk,
            requires_confirmation=requires_confirmation,
        )
        plan = MissionPlan(
            mission_id=mission_id,
            goal=cleaned,
            mission_type=mission_type,
            source_text=cleaned,
            risk_tier=risk,
            required_services=required_services,
            steps=steps,
            requested_capability=_requested_capability(mission_type),
            requested_by=(operator_id or self.operator_id).strip() or "askme.operator",
            adapter_plan_id=mission_id,
            idempotency_key=f"mission-{mission_id}",
            status=status,
            requires_confirmation=requires_confirmation,
            approval_required=approval_required,
            auto_approve=auto_approve,
            risk_reason=_risk_reason(risk, mission_type),
            confirmation_prompt=_confirmation_prompt(cleaned, risk)
            if requires_confirmation else "",
            operator_id=(operator_id or self.operator_id).strip() or "askme.operator",
            robot_id=robot_id if robot_id is not None else self.robot_id,
            site_id=site_id if site_id is not None else self.site_id,
            channel=channel or "text",
            safety_notes=_safety_notes(risk, mission_type),
            metadata=metadata or {},
        )
        self._remember(plan)
        return plan

    def draft_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = self.draft(
            str(payload.get("text") or payload.get("goal") or ""),
            operator_id=_clean_optional(payload.get("operator_id")),
            robot_id=_clean_optional(payload.get("robot_id")),
            site_id=_clean_optional(payload.get("site_id")),
            channel=str(payload.get("channel", "http")).strip() or "http",
            metadata=_metadata(payload),
        )
        return {"mission": plan.to_dict(), "drafted": True}

    def submit_from_payload(
        self,
        payload: dict[str, Any],
        *,
        trusted_confirmation: bool = False,
    ) -> dict[str, Any]:
        plan = self._plan_from_payload(payload)
        dry_run = bool(payload.get("dry_run", True))
        confirmed = trusted_confirmation and bool(
            payload.get("confirmed", False) or payload.get("confirm", False)
        )
        return self.submit(plan, dry_run=dry_run, confirmed=confirmed)

    def submit(
        self,
        plan: MissionPlan,
        *,
        dry_run: bool = True,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("mission service is disabled")

        if plan.risk_tier == "critical":
            return self._blocked(
                plan,
                reason="critical_action_requires_safety_service",
                next_action="Use the dedicated safety runtime or physical E-STOP path.",
            )

        if dry_run:
            plan.status = "dry_run"
            plan.updated_at = _utc_now()
            self._remember(plan)
            submission = {
                "submitted": False,
                "dry_run": True,
                "reason": "dry_run",
                "endpoint": self._mission_endpoint(),
                "confirmation_required": plan.requires_confirmation,
                "next_action": "Review the plan and submit with confirmation through the runtime arbiter.",
            }
            self._submissions[plan.mission_id] = submission
            return {"mission": plan.to_dict(), "submission": submission}

        if plan.requires_confirmation and not confirmed:
            plan.status = "pending_confirmation"
            plan.updated_at = _utc_now()
            self._remember(plan)
            submission = {
                "submitted": False,
                "dry_run": False,
                "reason": "confirmation_required",
                "confirmation_required": True,
                "endpoint": self._mission_endpoint(),
            }
            self._submissions[plan.mission_id] = submission
            return {"mission": plan.to_dict(), "submission": submission}

        if not self.submit_enabled or not self.base_url:
            return self._blocked(
                plan,
                reason="runtime_arbiter_not_configured",
                next_action="Set runtime.mission.submit_enabled=true and runtime.arbiter.base_url.",
            )

        payload = self._runtime_payload(plan)
        request_id = uuid4().hex[:16]
        headers = {
            "Content-Type": "application/json",
            "X-Operator-Id": plan.operator_id,
            "X-Service-Name": "askme",
            "X-Request-Id": request_id,
            "X-Correlation-Id": request_id,
            "Idempotency-Key": f"mission-{plan.mission_id}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self._mission_endpoint(),
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            runtime_payload = response.json()
        except Exception as exc:
            plan.status = "error"
            plan.updated_at = _utc_now()
            self._remember(plan)
            submission = {
                "submitted": False,
                "dry_run": False,
                "reason": "runtime_submit_failed",
                "error": str(exc),
                "endpoint": self._mission_endpoint(),
            }
            self._submissions[plan.mission_id] = submission
            return {"mission": plan.to_dict(), "submission": submission}

        plan.status = "submitted"
        plan.updated_at = _utc_now()
        self._remember(plan)
        submission = {
            "submitted": True,
            "dry_run": False,
            "endpoint": self._mission_endpoint(),
            "runtime": runtime_payload,
        }
        self._submissions[plan.mission_id] = submission
        return {"mission": plan.to_dict(), "submission": submission}

    def list_payload(self) -> dict[str, Any]:
        missions = [mission.to_dict() for mission in self._missions.values()]
        missions.sort(key=lambda item: item.get("created_at", ""))
        return {"missions": missions, "count": len(missions)}

    def get_payload(self, mission_id: str) -> dict[str, Any]:
        mission = self._missions.get(mission_id)
        if mission is None:
            return {"error": "mission not found", "mission_id": mission_id}
        payload = {"mission": mission.to_dict()}
        if mission_id in self._submissions:
            payload["submission"] = self._submissions[mission_id]
        return payload

    def report_payload(self, mission_id: str) -> dict[str, Any]:
        mission = self._missions.get(mission_id)
        if mission is None:
            return {"error": "mission not found", "mission_id": mission_id}
        report = InspectionReport(
            report_id=f"report-{mission_id}",
            mission_id=mission_id,
            site=mission.site_id,
            status=mission.status,
            findings=list(mission.evidence),
            media=[
                item
                for item in mission.evidence
                if item.get("kind") in {"image", "video", "thermal", "audio"}
            ],
            operator_summary=mission.goal,
            machine_summary=_report_summary(mission),
        )
        return {"report": report.to_dict()}

    def capabilities(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dry_run_default": True,
            "submit_enabled": self.submit_enabled,
            "arbiter_configured": bool(self.base_url),
            "confirmation_threshold": self.confirmation_threshold,
            "supported_mission_types": [
                "inspection_patrol",
                "navigate_to",
                "capture_evidence",
                "status_report",
                "emergency_stop",
                "custom",
            ],
            "http_paths": [
                "POST /api/missions/draft",
                "POST /api/missions",
                "GET /api/missions",
                "GET /api/missions/{mission_id}",
                "GET /api/missions/{mission_id}/report",
            ],
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.enabled else "disabled",
            "missions": len(self._missions),
            "submit_enabled": self.submit_enabled,
            "arbiter_configured": bool(self.base_url),
        }

    def _plan_from_payload(self, payload: dict[str, Any]) -> MissionPlan:
        mission_payload = payload.get("mission")
        if isinstance(mission_payload, dict):
            plan = MissionPlan.from_dict(mission_payload)
            if not plan.required_services:
                plan.required_services = _required_services(
                    plan.mission_type,
                    plan.risk_tier,
                )
            if not plan.steps:
                plan.steps = _build_steps(
                    plan.mission_type,
                    target=_infer_target(plan.goal),
                    risk_tier=plan.risk_tier,
                    requires_confirmation=plan.requires_confirmation,
                )
            self._remember(plan)
            return plan
        return self.draft(
            str(payload.get("text") or payload.get("goal") or ""),
            operator_id=_clean_optional(payload.get("operator_id")),
            robot_id=_clean_optional(payload.get("robot_id")),
            site_id=_clean_optional(payload.get("site_id")),
            channel=str(payload.get("channel", "http")).strip() or "http",
            metadata=_metadata(payload),
        )

    def _runtime_payload(self, plan: MissionPlan) -> dict[str, Any]:
        metadata = _public_metadata(plan.metadata)
        return {
            "mission_type": plan.mission_type,
            "requested_capability": plan.requested_capability,
            "requested_by": plan.requested_by,
            "channel": plan.channel,
            "robot_id": plan.robot_id,
            "site_id": plan.site_id,
            "priority": _priority_for_risk(plan.risk_tier),
            "approval_required": plan.approval_required,
            "parameters": {
                **metadata,
                "goal": plan.goal,
                "source_text": plan.source_text,
                "risk_tier": plan.risk_tier,
                "risk_reason": plan.risk_reason,
                "required_services": plan.required_services,
                "steps": [step.to_dict() for step in plan.steps],
            },
        }

    def _mission_endpoint(self) -> str:
        base = self.base_url or "runtime-arbiter-not-configured"
        return f"{base}/api/v1/missions"

    def _blocked(
        self,
        plan: MissionPlan,
        *,
        reason: str,
        next_action: str,
    ) -> dict[str, Any]:
        plan.status = "blocked"
        plan.updated_at = _utc_now()
        self._remember(plan)
        submission = {
            "submitted": False,
            "dry_run": False,
            "reason": reason,
            "endpoint": self._mission_endpoint(),
            "next_action": next_action,
        }
        self._submissions[plan.mission_id] = submission
        return {"mission": plan.to_dict(), "submission": submission}

    def _remember(self, plan: MissionPlan) -> None:
        self._missions[plan.mission_id] = plan


def _build_steps(
    mission_type: str,
    *,
    target: str | None,
    risk_tier: str,
    requires_confirmation: bool,
) -> list[MissionStep]:
    raw_steps: list[tuple[str, str, str | None, str | None]] = [
        ("safety_precheck", "safety.evaluate", target, "safety"),
    ]
    if mission_type == "inspection_patrol":
        raw_steps.extend([
            ("plan_route", "nav.plan_route", target, "nav"),
            ("dispatch_patrol", "control.start_patrol", target, "control"),
            ("collect_evidence", "payload.collect_evidence", target, "sense"),
            ("generate_report", "catalog.generate_report", target, "catalog"),
        ])
    elif mission_type == "navigate_to":
        raw_steps.extend([
            ("plan_route", "nav.plan_route", target, "nav"),
            ("go_checkpoint", "control.go_checkpoint", target, "control"),
            ("record_arrival", "telemetry.audit_event", target, "telemetry"),
        ])
    elif mission_type == "capture_evidence":
        raw_steps.extend([
            ("capture_snapshot", "payload.capture_snapshot", target, "sense"),
            ("record_evidence", "catalog.record_evidence", target, "catalog"),
        ])
    elif mission_type == "status_report":
        raw_steps.extend([
            ("read_runtime_status", "telemetry.status_snapshot", target, "telemetry"),
            ("summarize_status", "catalog.generate_report", target, "catalog"),
        ])
    elif mission_type == "emergency_stop":
        raw_steps = [
            ("safety_estop", "safety.estop", target, "safety"),
            ("audit_estop_request", "telemetry.audit_event", target, "telemetry"),
        ]
    else:
        raw_steps.extend([
            ("classify_goal", "arbiter.classify_goal", target, "arbiter"),
            ("operator_review", "arbiter.request_review", target, "arbiter"),
        ])

    return [
        MissionStep(
            step_id=f"step-{index}",
            sequence=index,
            action=action,
            capability=capability,
            target=step_target,
            risk_tier=risk_tier if service in {"control", "safety"} else "medium",
            requires_confirmation=requires_confirmation and service in {"control", "safety"},
            service=service,
        )
        for index, (action, capability, step_target, service) in enumerate(raw_steps, start=1)
    ]


def _required_services(mission_type: str, risk_tier: str) -> list[str]:
    if mission_type == "status_report":
        return ["telemetry", "catalog"]
    if mission_type == "capture_evidence":
        return ["safety", "sense", "catalog"]
    if mission_type == "emergency_stop" or risk_tier == "critical":
        return ["safety", "telemetry"]
    if mission_type in {"inspection_patrol", "navigate_to"}:
        return ["safety", "nav", "control", "telemetry"]
    return ["arbiter", "safety", "telemetry"]


def _infer_mission_type(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("急停", "紧急停止", "e-stop", "estop", "emergency stop")):
        return "emergency_stop"
    if any(token in lowered for token in ("巡检", "巡逻", "patrol", "inspect", "inspection")):
        return "inspection_patrol"
    if any(token in lowered for token in ("去", "到", "导航", "前往", "go to", "navigate")):
        return "navigate_to"
    if any(token in lowered for token in ("拍照", "截图", "取证", "抓拍", "photo", "snapshot", "evidence")):
        return "capture_evidence"
    if any(token in lowered for token in ("状态", "报告", "status", "report")):
        return "status_report"
    return "custom"


def _infer_risk(text: str, mission_type: str) -> str:
    lowered = text.lower()
    if mission_type == "emergency_stop" or any(
        token in lowered
        for token in ("禁用安全", "关闭避障", "disable safety", "override safety")
    ):
        return "critical"
    if mission_type in {"inspection_patrol", "navigate_to"}:
        return "high"
    if mission_type == "capture_evidence":
        return "medium"
    if mission_type == "status_report":
        return "low"
    return "medium"


def _infer_target(text: str) -> str | None:
    tokens = text.replace("，", " ").replace(",", " ").replace("。", " ").split()
    for token in tokens:
        cleaned = token.strip(":：;；")
        lowered = cleaned.lower()
        if any(mark in cleaned for mark in ("区", "站", "点", "线", "号")):
            return cleaned
        if lowered.startswith(("area-", "zone-", "checkpoint-", "route-")):
            return cleaned
    return None


def _safety_notes(risk: str, mission_type: str) -> list[str]:
    notes = ["Missions are submitted to the runtime arbiter, not directly to motors."]
    if _risk_at_least(risk, "high"):
        notes.append("Operator confirmation is required before physical movement.")
    if mission_type == "emergency_stop":
        notes.append("Emergency stop requests must use the dedicated safety service or physical E-STOP.")
    return notes


def _requested_capability(mission_type: str) -> str:
    mapping = {
        "inspection_patrol": "patrol",
        "navigate_to": "go_checkpoint",
        "capture_evidence": "capture_evidence",
        "status_report": "status_report",
        "emergency_stop": "estop",
        "custom": "operator_review",
    }
    return mapping.get(mission_type, "operator_review")


def _risk_reason(risk_tier: str, mission_type: str) -> str:
    if risk_tier == "critical":
        return "critical safety action or safety override request"
    if risk_tier == "high":
        return f"{mission_type} may move the robot and must be approved first"
    if risk_tier == "medium":
        return f"{mission_type} affects runtime state or evidence collection"
    return "read-only or reporting operation"


def _confirmation_prompt(goal: str, risk_tier: str) -> str:
    return (
        f"Confirm {risk_tier} mission before dispatching through the runtime arbiter: {goal}"
    )


def _priority_for_risk(risk_tier: str) -> str:
    mapping = {
        "low": "normal",
        "medium": "normal",
        "high": "high",
        "critical": "critical",
    }
    return mapping.get(_clean_risk(risk_tier), "normal")


def _report_summary(mission: MissionPlan) -> str:
    if mission.evidence:
        return f"{len(mission.evidence)} evidence item(s) recorded for {mission.goal}."
    return f"No evidence recorded yet. Mission status is {mission.status}."


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("metadata", {})
    return dict(meta) if isinstance(meta, dict) else {}


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metadata.items()
        if not str(key).startswith("_")
    }


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_risk(value: Any) -> str:
    risk = str(value).strip().lower()
    if risk not in _VALID_RISKS:
        return "medium"
    return risk


def _risk_at_least(value: str, threshold: str) -> bool:
    return _RISK_ORDER[_clean_risk(value)] >= _RISK_ORDER[_clean_risk(threshold)]


def _new_mission_id() -> str:
    return f"mission-{uuid4().hex[:12]}"


def _utc_now() -> str:
    now = datetime.now(_UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
