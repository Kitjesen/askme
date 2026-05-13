"""Machine-readable catalog for askme product I/O contracts."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any

from askme.contracts.io import (
    ActionDecision,
    ActorType,
    AudioInput,
    EvidenceRef,
    Freshness,
    IntentInput,
    IntentType,
    LocationRef,
    PerceptionInput,
    RiskLevel,
    RobotActionType,
    SensorInput,
    UserFacingOutput,
    VisionInput,
)
from askme.contracts.package import (
    CapabilityDependency,
    CapabilityPackageManifest,
    ScenarioPackageManifest,
)

CONTRACT_VERSION = "2026-05-13.v1"


def contract_catalog() -> dict[str, Any]:
    """Return the public contract catalog used by MCP, Dashboard, and tests."""
    return {
        "version": CONTRACT_VERSION,
        "boundary": {
            "purpose": "standard robot product I/O between perception, intent, safety, capability packages, runtime, and UI",
            "runtime_rule": "LLM may propose ActionDecision, but robot movement must pass TaskHandoff/SafetyPreflight/runtime arbiter",
            "customer_rule": "Dashboard and audit logs should show action, reason, risk, evidence, confidence, and next action",
        },
        "flow": [
            "PerceptionInput",
            "IntentInput",
            "ActionDecision",
            "UserFacingOutput",
            "TaskHandoff when movement or long-running work is required",
        ],
        "enums": {
            "ActorType": _enum_values(ActorType),
            "IntentType": _enum_values(IntentType),
            "RobotActionType": _enum_values(RobotActionType),
            "RiskLevel": _enum_values(RiskLevel),
        },
        "contracts": {
            "LocationRef": _contract_fields(LocationRef),
            "EvidenceRef": _contract_fields(EvidenceRef),
            "Freshness": _contract_fields(Freshness),
            "VisionInput": _contract_fields(VisionInput),
            "AudioInput": _contract_fields(AudioInput),
            "SensorInput": _contract_fields(SensorInput),
            "PerceptionInput": _contract_fields(PerceptionInput),
            "IntentInput": _contract_fields(IntentInput),
            "ActionDecision": _contract_fields(ActionDecision),
            "UserFacingOutput": _contract_fields(UserFacingOutput),
            "CapabilityDependency": _contract_fields(CapabilityDependency),
            "CapabilityPackageManifest": _contract_fields(CapabilityPackageManifest),
            "ScenarioPackageManifest": _contract_fields(ScenarioPackageManifest),
        },
        "adapters": {
            "interaction_gate": {
                "input": "InteractionPerceptionSnapshot",
                "outputs": ["PerceptionInput", "ActionDecision"],
            },
            "field_event": {
                "input": "FieldEventDetail or field event dict",
                "outputs": ["EvidenceRef", "ActionDecision", "UserFacingOutput"],
            },
        },
        "rejection_policy": {
            "reject_when": [
                "intent is outside approved site knowledge or customer scenario package",
                "knowledge is expired or conflicting and no operator has approved it",
                "speech is not addressed to the robot",
                "visitor attempts to trigger robot task or hardware control",
                "perception or world-state evidence is stale",
                "route or action is unsafe, unreachable, or requires missing hardware",
            ],
            "escalate_when": [
                "fall_unrecoverable",
                "stuck_unrecoverable",
                "malicious_blocking",
                "motor_fault",
                "fire_smoke",
                "multi_person_command_conflict",
            ],
        },
    }


def contract_examples() -> dict[str, Any]:
    """Return small examples that external MCP clients can copy into tests."""
    evidence = EvidenceRef(
        evidence_id="frame-west-gate-001",
        evidence_type="camera_frame",
        source="vision_bridge",
        summary="visitor standing at west-gate service point",
        confidence=0.91,
    )
    perception = PerceptionInput.from_payload(
        {
            "robot_id": "thunder-01",
            "location": {
                "site_id": "fanmu",
                "map_id": "fanmu-floor-1",
                "point_id": "west_gate_service",
                "name": "西门问询点",
            },
            "vision": {
                "persons": [
                    {
                        "distance_m": 1.4,
                        "facing_robot": True,
                        "visual_attention": True,
                    }
                ],
                "service_points": [{"point_id": "west_gate_service"}],
            },
            "audio": {
                "transcript": "咖啡店怎么走",
                "confidence": 0.88,
                "addressed": True,
            },
            "freshness": {"vision_ms": 300, "audio_ms": 120, "world_state_ms": 600},
            "evidence_refs": [evidence.to_dict()],
        }
    )
    intent = IntentInput(
        intent_type=IntentType.ASK_DIRECTION,
        actor_type=ActorType.VISITOR,
        text="咖啡店怎么走",
        confidence=0.88,
        target="梵木咖啡",
        location=perception.location,
        evidence_refs=(evidence,),
    )
    action = ActionDecision(
        action_type=RobotActionType.GUIDE_BY_VOICE,
        reason="park_wayfinding_answer",
        risk_level=RiskLevel.LOW,
        skill_name="answer_wayfinding",
        parameters={"target_point_id": "fanmu_cafe"},
        evidence_refs=(evidence,),
        confidence=0.91,
    )
    output = action.to_user_output(
        spoken_text="梵木咖啡在二号楼一层。从这里向前约八十米，右转后沿左侧商铺前进。",
        audit_id="audit-demo-001",
    )
    return {
        "version": CONTRACT_VERSION,
        "perception_input": perception.to_dict(),
        "intent_input": intent.to_dict(),
        "action_decision": action.to_dict(),
        "user_facing_output": output.to_dict(),
    }


def _contract_fields(cls: type[Any]) -> list[dict[str, Any]]:
    if not is_dataclass(cls):
        return []
    result: list[dict[str, Any]] = []
    for field in fields(cls):
        has_default = field.default is not MISSING or field.default_factory is not MISSING
        result.append(
            {
                "name": field.name,
                "type": _type_name(field.type),
                "required": not has_default,
            }
        )
    return result


def _enum_values(enum_cls: type[Any]) -> list[str]:
    return [item.value for item in enum_cls]


def _type_name(value: Any) -> str:
    return str(value).replace("typing.", "").replace(" | None", "?")
