"""Adapters from field-operation events into product I/O contracts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from askme.contracts.io import (
    ActionDecision,
    EvidenceRef,
    RiskLevel,
    RobotActionType,
    UserFacingOutput,
)
from askme.schemas.field import FieldEventDetail

FieldProductAction = Literal["notify", "record", "escalate"]


def field_event_to_evidence_refs(event: Any) -> tuple[EvidenceRef, ...]:
    """Convert field-event media/evidence dictionaries into EvidenceRef values."""

    data = _field_event_dict(event)
    event_id = _clean_text(data.get("event_id"))
    scenario_id = _clean_text(data.get("scenario_id"))
    event_confidence = _float_or_none(data.get("confidence"))
    event_observed_at = _float_or_none(data.get("created_at"))
    refs: list[EvidenceRef] = []
    for index, item in enumerate(_list_of_dicts(data.get("evidence_media"))):
        uri = _clean_text(
            item.get("uri")
            or item.get("preview_url")
            or item.get("url")
            or item.get("path")
        )
        source_key = _clean_text(item.get("source_key") or item.get("source"))
        evidence_id = _clean_text(
            item.get("evidence_id")
            or item.get("id")
            or source_key
            or item.get("path")
            or f"{event_id}:evidence:{index}"
        )
        refs.append(
            EvidenceRef(
                evidence_id=evidence_id,
                evidence_type=_clean_text(
                    item.get("evidence_type") or item.get("type") or "field_event"
                ),
                source=source_key or "field_event",
                uri=uri,
                summary=_clean_text(
                    item.get("summary")
                    or item.get("label")
                    or item.get("text")
                    or item.get("path")
                ),
                confidence=_float_or_none(item.get("confidence")) or event_confidence,
                observed_at=_float_or_none(item.get("observed_at") or item.get("timestamp"))
                or event_observed_at,
                metadata={
                    "event_id": event_id,
                    "scenario_id": scenario_id,
                    "field_media": dict(item),
                },
            )
        )
    return tuple(refs)


def field_event_to_action_decision(
    event: Any,
    *,
    product_action: FieldProductAction | None = None,
) -> ActionDecision:
    """Convert a field event into the product action contract."""

    data = _field_event_dict(event)
    action = product_action or _infer_product_action(data)
    evidence_refs = field_event_to_evidence_refs(data)
    action_type = _action_type(action)
    reason = _action_reason(data, action)
    return ActionDecision(
        action_type=action_type,
        reason=reason,
        risk_level=_risk_level(data),
        requires_confirmation=action_type == RobotActionType.ESCALATE,
        parameters={
            "event_id": _clean_text(data.get("event_id")),
            "scenario_id": _clean_text(data.get("scenario_id")),
            "status": _clean_text(data.get("status")),
            "location": _clean_text(data.get("location")),
            "product_action": action,
            "notification_group": _clean_text(data.get("notification_group")),
            "incident_topic": _clean_text(data.get("incident_topic")),
            "delivery_report": _list_of_dicts(data.get("delivery_report")),
            "sla": _dict(data.get("sla")),
            "close_approval_required": bool(data.get("close_approval_required", False)),
        },
        evidence_refs=evidence_refs,
        confidence=_confidence(data),
        metadata={
            "source": "field_operations",
            "incident_state": _clean_text(data.get("incident_state")),
            "incident_stage": _clean_text(data.get("incident_stage")),
        },
    )


def field_event_to_user_output(
    event: Any,
    *,
    product_action: FieldProductAction | None = None,
) -> UserFacingOutput:
    """Convert a field event into operator/customer-facing output."""

    data = _field_event_dict(event)
    decision = field_event_to_action_decision(data, product_action=product_action)
    spoken_text = _clean_text(data.get("voice") or data.get("operator_action"))
    display_text = _display_text(data, decision)
    return UserFacingOutput(
        spoken_text=spoken_text,
        display_text=display_text,
        status=decision.action_type.value,
        next_action=decision.reason,
        evidence=decision.evidence_refs,
        confidence=decision.confidence,
        fallback=spoken_text,
        audit_id=_clean_text(data.get("event_id")),
        metadata={
            "source": "field_operations",
            "product_action": decision.parameters["product_action"],
            "risk_level": decision.risk_level.value,
        },
    )


def field_event_to_product_contracts(
    event: Any,
    *,
    product_action: FieldProductAction | None = None,
) -> tuple[ActionDecision, UserFacingOutput, tuple[EvidenceRef, ...]]:
    """Return the full product I/O slice for a field event."""

    decision = field_event_to_action_decision(event, product_action=product_action)
    output = field_event_to_user_output(event, product_action=product_action)
    return decision, output, decision.evidence_refs


def _field_event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, FieldEventDetail):
        return event.to_dict()
    if isinstance(event, dict):
        data = dict(event)
    elif hasattr(event, "to_dict"):
        data = event.to_dict()
    elif is_dataclass(event) and not isinstance(event, type):
        data = asdict(event)
    else:
        data = {}
    return FieldEventDetail.from_dict(data).to_dict()


def _infer_product_action(data: dict[str, Any]) -> FieldProductAction:
    if _should_escalate(data):
        return "escalate"
    if _clean_text(data.get("incident_topic")) or _clean_text(data.get("notification_group")) not in {
        "",
        "none",
    }:
        return "notify"
    return "record"


def _should_escalate(data: dict[str, Any]) -> bool:
    sla = _dict(data.get("sla"))
    return (
        _clean_text(data.get("priority")) == "P0"
        or _clean_text(data.get("severity")) in {"critical", "error"}
        or _clean_text(sla.get("state")) == "overdue"
        or bool(data.get("close_approval_required", False))
        or _clean_text(data.get("status")) in {"needs_evidence", "pending_close_approval"}
    )


def _action_type(action: FieldProductAction) -> RobotActionType:
    if action == "escalate":
        return RobotActionType.ESCALATE
    if action == "notify":
        return RobotActionType.NOTIFY_HUMAN
    return RobotActionType.RECORD_EVENT


def _action_reason(data: dict[str, Any], action: FieldProductAction) -> str:
    scenario_id = _clean_text(data.get("scenario_id")) or "field_event"
    status = _clean_text(data.get("status")) or "unknown"
    location = _clean_text(data.get("location")) or "-"
    if action == "escalate":
        return f"escalate {scenario_id} at {location}; status={status}"
    if action == "notify":
        group = _clean_text(data.get("notification_group")) or "responder"
        return f"notify {group} for {scenario_id} at {location}; status={status}"
    return f"record {scenario_id} at {location}; status={status}"


def _risk_level(data: dict[str, Any]) -> RiskLevel:
    sla = _dict(data.get("sla"))
    priority = _clean_text(data.get("priority"))
    severity = _clean_text(data.get("severity"))
    status = _clean_text(data.get("status"))
    if priority == "P0" or severity in {"critical", "error"} or _clean_text(sla.get("state")) == "overdue":
        return RiskLevel.CRITICAL
    if priority == "P1" or status in {"needs_evidence", "pending_close_approval"}:
        return RiskLevel.HIGH
    if priority == "P2" or status not in {"", "closed", "duplicate"}:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _display_text(data: dict[str, Any], decision: ActionDecision) -> str:
    scenario_id = _clean_text(data.get("scenario_id")) or "field_event"
    event_id = _clean_text(data.get("event_id")) or "-"
    location = _clean_text(data.get("location")) or "-"
    operator_action = _clean_text(data.get("operator_action"))
    if operator_action:
        return f"{scenario_id} ({event_id}) at {location}: {operator_action}"
    return f"{scenario_id} ({event_id}) at {location}: {decision.reason}"


def _confidence(data: dict[str, Any]) -> float:
    confidence = _float_or_none(data.get("confidence"))
    if confidence is None:
        return 0.8
    return min(1.0, max(0.0, confidence))


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _list(value) if isinstance(item, dict)]


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
