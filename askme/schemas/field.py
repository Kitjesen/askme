"""Product-facing schemas for field-operation HTTP payloads.

The field-operation service returns plain dictionaries today. These dataclasses
capture the stable customer-facing contract while preserving unknown keys so the
backend can evolve without silently dropping evidence or audit details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _str(value: Any) -> str:
    return str(value or "")


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class FieldEventAction:
    """One auditable operator action against a field event."""

    action: str
    outcome: str
    operator_id: str = ""
    at: float | None = None
    reason: str = ""
    note: str = ""
    supervisor_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldEventAction:
        known = {
            "action",
            "outcome",
            "operator_id",
            "at",
            "reason",
            "note",
            "supervisor_id",
        }
        return cls(
            action=_str(data.get("action")),
            outcome=_str(data.get("outcome")),
            operator_id=_str(data.get("operator_id")),
            at=_float_or_none(data.get("at")),
            reason=_str(data.get("reason")),
            note=_str(data.get("note")),
            supervisor_id=_str(data.get("supervisor_id")),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "action": self.action,
                "outcome": self.outcome,
                "operator_id": self.operator_id,
            }
        )
        if self.at is not None:
            data["at"] = self.at
        if self.reason:
            data["reason"] = self.reason
        if self.note:
            data["note"] = self.note
        if self.supervisor_id:
            data["supervisor_id"] = self.supervisor_id
        return data


@dataclass(frozen=True)
class FieldEventSla:
    """SLA state attached to a field event view."""

    state: str = ""
    due_at: float | None = None
    remaining_s: float | None = None
    target_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldEventSla:
        known = {"state", "due_at", "remaining_s", "target_s"}
        return cls(
            state=_str(data.get("state")),
            due_at=_float_or_none(data.get("due_at")),
            remaining_s=_float_or_none(data.get("remaining_s")),
            target_s=_float_or_none(data.get("target_s")),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data["state"] = self.state
        if self.due_at is not None:
            data["due_at"] = self.due_at
        if self.remaining_s is not None:
            data["remaining_s"] = self.remaining_s
        if self.target_s is not None:
            data["target_s"] = self.target_s
        return data


@dataclass(frozen=True)
class FieldEventDetail:
    """Customer-visible field event detail payload."""

    event_id: str
    scenario_id: str = ""
    status: str = ""
    location: str = ""
    incident_state: str = ""
    incident_stage: str = ""
    incident_workflow: dict[str, Any] = field(default_factory=dict)
    evidence_media: list[dict[str, Any]] = field(default_factory=list)
    delivery_report: list[dict[str, Any]] = field(default_factory=list)
    voice_delivery: dict[str, Any] = field(default_factory=dict)
    runtime_delivery: dict[str, Any] = field(default_factory=dict)
    runtime_delivery_receipts: list[dict[str, Any]] = field(default_factory=list)
    memory_delivery: dict[str, Any] = field(default_factory=dict)
    action_audit: list[FieldEventAction] = field(default_factory=list)
    sla: FieldEventSla = field(default_factory=FieldEventSla)
    close_approval_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldEventDetail:
        known = {
            "event_id",
            "scenario_id",
            "status",
            "location",
            "incident_state",
            "incident_stage",
            "incident_workflow",
            "evidence_media",
            "delivery_report",
            "voice_delivery",
            "runtime_delivery",
            "runtime_delivery_receipts",
            "memory_delivery",
            "action_audit",
            "sla",
            "close_approval_required",
        }
        return cls(
            event_id=_str(data.get("event_id")),
            scenario_id=_str(data.get("scenario_id")),
            status=_str(data.get("status")),
            location=_str(data.get("location")),
            incident_state=_str(data.get("incident_state")),
            incident_stage=_str(data.get("incident_stage")),
            incident_workflow=_dict(data.get("incident_workflow")),
            evidence_media=_dict_list(data.get("evidence_media")),
            delivery_report=_dict_list(data.get("delivery_report")),
            voice_delivery=_dict(data.get("voice_delivery")),
            runtime_delivery=_dict(data.get("runtime_delivery")),
            runtime_delivery_receipts=_dict_list(data.get("runtime_delivery_receipts")),
            memory_delivery=_dict(data.get("memory_delivery")),
            action_audit=[
                FieldEventAction.from_dict(item)
                for item in _dict_list(data.get("action_audit"))
            ],
            sla=FieldEventSla.from_dict(_dict(data.get("sla"))),
            close_approval_required=bool(data.get("close_approval_required", False)),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "event_id": self.event_id,
                "scenario_id": self.scenario_id,
                "status": self.status,
                "location": self.location,
                "incident_state": self.incident_state,
                "incident_stage": self.incident_stage,
                "incident_workflow": dict(self.incident_workflow),
                "evidence_media": list(self.evidence_media),
                "delivery_report": list(self.delivery_report),
                "voice_delivery": dict(self.voice_delivery),
                "runtime_delivery": dict(self.runtime_delivery),
                "runtime_delivery_receipts": list(self.runtime_delivery_receipts),
                "memory_delivery": dict(self.memory_delivery),
                "action_audit": [item.to_dict() for item in self.action_audit],
                "sla": self.sla.to_dict(),
                "close_approval_required": self.close_approval_required,
            }
        )
        return data


@dataclass(frozen=True)
class FieldEventListResponse:
    """List response returned by GET /api/field/events."""

    events: list[FieldEventDetail]
    total: int = 0
    filtered_total: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    filter: dict[str, Any] = field(default_factory=dict)
    archive_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldEventListResponse:
        known = {"events", "total", "filtered_total", "summary", "filter", "archive_path"}
        return cls(
            events=[
                FieldEventDetail.from_dict(item)
                for item in _dict_list(data.get("events") or data.get("items"))
            ],
            total=int(data.get("total") or 0),
            filtered_total=int(data.get("filtered_total") or 0),
            summary=_dict(data.get("summary")),
            filter=_dict(data.get("filter")),
            archive_path=_str(data.get("archive_path")),
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "events": [event.to_dict() for event in self.events],
                "total": self.total,
                "filtered_total": self.filtered_total,
                "summary": dict(self.summary),
                "filter": dict(self.filter),
                "archive_path": self.archive_path,
            }
        )
        return data


@dataclass(frozen=True)
class FieldEventCreateRequest:
    """Manual field-event creation request."""

    scenario_id: str
    location: str = ""
    description: str = ""
    operator_id: str = "dashboard.operator"
    source: str = "dashboard"
    trigger_source: str = "operator_manual"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "scenario_id": self.scenario_id,
                "location": self.location,
                "description": self.description,
                "operator_id": self.operator_id,
                "source": self.source,
                "trigger_source": self.trigger_source,
            }
        )
        return data


@dataclass(frozen=True)
class FieldEventActionRequest:
    """Operator action request for acknowledge/resend/request-close/close."""

    operator_id: str
    note: str = ""
    supervisor_approved: bool = False
    supervisor_id: str = ""
    approval_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update({"operator_id": self.operator_id, "note": self.note})
        if self.supervisor_approved:
            data["supervisor_approved"] = True
        if self.supervisor_id:
            data["supervisor_id"] = self.supervisor_id
        if self.approval_note:
            data["approval_note"] = self.approval_note
        return data
