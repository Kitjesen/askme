from __future__ import annotations

from askme.contracts.field_adapters import (
    field_event_to_action_decision,
    field_event_to_evidence_refs,
    field_event_to_product_contracts,
    field_event_to_user_output,
)
from askme.contracts.io import RiskLevel, RobotActionType
from askme.schemas.field import FieldEventDetail


def test_field_event_dict_maps_to_escalation_contracts() -> None:
    event = {
        "event_id": "evt-1",
        "scenario_id": "fire_or_smoke",
        "status": "needs_evidence",
        "priority": "P0",
        "severity": "error",
        "location": "garage-b1",
        "notification_group": "security",
        "incident_topic": "safety.fire_or_smoke",
        "operator_action": "Dispatch guard and request smoke confirmation.",
        "confidence": 0.93,
        "created_at": 123.0,
        "sla": {"state": "overdue", "remaining_s": -30},
        "evidence_media": [
            {
                "type": "image",
                "source_key": "image_path",
                "path": "artifacts/evidence/smoke.jpg",
                "preview_url": "/api/field/evidence?path=smoke",
                "label": "smoke frame",
            }
        ],
    }

    decision = field_event_to_action_decision(event)
    output = field_event_to_user_output(event)
    evidence = field_event_to_evidence_refs(event)

    assert decision.action_type == RobotActionType.ESCALATE
    assert decision.risk_level == RiskLevel.CRITICAL
    assert decision.requires_confirmation is True
    assert decision.confidence == 0.93
    assert decision.parameters["event_id"] == "evt-1"
    assert decision.parameters["notification_group"] == "security"
    assert decision.evidence_refs == evidence
    assert evidence[0].evidence_type == "image"
    assert evidence[0].uri == "/api/field/evidence?path=smoke"
    assert evidence[0].metadata["scenario_id"] == "fire_or_smoke"
    assert output.status == "escalate"
    assert output.audit_id == "evt-1"
    assert "Dispatch guard" in output.display_text
    assert output.validate() == []


def test_field_event_dataclass_can_be_forced_to_notify_action() -> None:
    detail = FieldEventDetail.from_dict(
        {
            "event_id": "evt-2",
            "scenario_id": "illegal_parking",
            "status": "active",
            "priority": "P2",
            "severity": "warning",
            "location": "main-road",
            "notification_group": "operations",
            "operator_action": "Notify road operations.",
            "delivery_report": [{"channel": "log", "status": "sent"}],
            "evidence_media": [{"type": "image", "path": "artifacts/evidence/car.jpg"}],
        }
    )

    decision, output, evidence = field_event_to_product_contracts(
        detail,
        product_action="notify",
    )

    assert decision.action_type == RobotActionType.NOTIFY_HUMAN
    assert decision.risk_level == RiskLevel.MEDIUM
    assert decision.requires_confirmation is False
    assert decision.parameters["delivery_report"] == [{"channel": "log", "status": "sent"}]
    assert output.status == "notify_human"
    assert output.evidence == evidence
    assert evidence[0].uri == "artifacts/evidence/car.jpg"
    assert decision.validate() == []


def test_field_event_without_notification_defaults_to_record_event() -> None:
    event = {
        "event_id": "evt-3",
        "scenario_id": "wayfinding_help_point",
        "status": "closed",
        "location": "lobby",
        "notification_group": "none",
    }

    decision = field_event_to_action_decision(event)

    assert decision.action_type == RobotActionType.RECORD_EVENT
    assert decision.risk_level == RiskLevel.LOW
    assert decision.parameters["product_action"] == "record"
