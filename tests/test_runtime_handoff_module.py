from __future__ import annotations

from unittest.mock import MagicMock

from askme.runtime.module import ModuleRegistry

from askme.cognition import WorldStateService
from askme.runtime.modules.runtime_handoff_module import RuntimeHandoffModule


def test_runtime_handoff_module_wires_optional_dog_safety_audit_and_store(tmp_path) -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    class FakeSafetyModule:
        name = "safety"

        def __init__(self) -> None:
            self.safety_client = MagicMock()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    safety = FakeSafetyModule()
    registry.register(safety)
    mod = RuntimeHandoffModule()

    mod.build(
        {
            "runtime_handoff": {
                "audit": {"enabled": True, "path": str(tmp_path / "runtime.jsonl")},
                "store": {"enabled": True, "path": str(tmp_path / "runs.json")},
                "require_dog_safety": True,
            }
        },
        registry,
    )

    service = mod.runtime_handoff_service
    assert service.safety_preflight.dog_safety_client is safety.safety_client
    assert service.safety_preflight.require_dog_safety is True
    assert service.run_service._audit_log.enabled is True
    assert service.run_service._store.enabled is True


def test_runtime_handoff_module_exposes_voice_turn_payload() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    mod = RuntimeHandoffModule()
    mod.build({"runtime_handoff": {"profile": "fake"}}, registry)

    payload = mod.voice_turn_payload(
        "确认",
        transcript_id="turn-1",
        confidence=0.9,
        channel="voice",
    )

    assert payload["handled"] is False
    assert payload["reason"] == "no_runtime_control_intent"
    assert payload["voice_turn"]["transcript_id"] == "turn-1"
    assert payload["voice_turn"]["recognized_text"] == "确认"
    assert payload["voice_turn"]["safety_bypass_allowed"] is False


def test_runtime_handoff_module_preserves_voice_turn_session_ids() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    mod = RuntimeHandoffModule()
    mod.build({"runtime_handoff": {"profile": "fake"}}, registry)

    payload = mod.voice_turn_payload(
        "confirm",
        transcript_id="turn-session",
        conversation_session_id="conv-1",
        planning_session_id="plan-1",
    )

    assert payload["voice_turn"]["conversation_session_id"] == "conv-1"
    assert payload["voice_turn"]["planning_session_id"] == "plan-1"
    assert (
        payload["voice_turn"]["conversation_session_id"]
        != payload["voice_turn"]["planning_session_id"]
    )


def test_disabled_runtime_handoff_module_rejects_voice_turn_payload() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    mod = RuntimeHandoffModule()
    mod.build({"runtime_handoff": {"enabled": False}}, registry)

    payload = mod.voice_turn_payload(
        "暂停",
        transcript_id="turn-disabled",
        conversation_session_id="conv-disabled",
        planning_session_id="plan-disabled",
    )

    assert payload["handled"] is False
    assert payload["reason"] == "runtime_handoff_disabled"
    assert payload["voice_turn"]["transcript_id"] == "turn-disabled"
    assert payload["voice_turn"]["conversation_session_id"] == "conv-disabled"
    assert payload["voice_turn"]["planning_session_id"] == "plan-disabled"
    assert payload["voice_turn"]["safety_bypass_allowed"] is False


def test_runtime_handoff_module_wires_external_runtime_client_contract() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    mod = RuntimeHandoffModule()

    mod.build(
        {
            "runtime_handoff": {
                "profile": "lab",
                "endpoint": "http://runtime.local/submit",
                "enable_external_runtime": True,
            }
        },
        registry,
    )

    payload = mod.context_payload()

    assert payload["profile"] == "lab"
    assert payload["runtime_client"]["endpoint_configured"] is True
    assert payload["runtime_client"]["enable_external_runtime"] is True
    assert payload["runtime_client"]["hardware_dispatch"] is False


def test_runtime_handoff_module_forwards_voice_operator_provenance() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    module = RuntimeHandoffModule()
    module.build({"runtime_handoff": {"profile": "fake"}}, registry)

    payload = module.voice_turn_payload(
        "confirm",
        conversation_session_id="voice-module-thread",
        operator_id="operator-7",
        operator_roles=["operator"],
        operator_authenticated=True,
        operator_source="oidc",
        runtime_permission="runtime:submit",
        reason="test provenance",
        risk_acknowledgement=True,
    )

    assert payload["handled"] is False
    assert payload["voice_turn"]["operator"] == {
        "operator_id": "operator-7",
        "roles": ["operator"],
        "authenticated": True,
        "source": "oidc",
        "permission": "runtime:submit",
        "conversation_session_id": "voice-module-thread",
    }
    assert payload["voice_turn"]["runtime_permission"] == "runtime:submit"


def test_runtime_handoff_module_forwards_action_operator_context() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    module = RuntimeHandoffModule()
    module.build({"runtime_handoff": {"profile": "sim"}}, registry)
    submitted = module.submit_plan_payload(
        {
            "plan_id": "operator-context-plan",
            "planning_session_id": "plan-context",
            "intent": "visitor_escort",
            "handoff_ready": True,
            "confirmation_status": "confirmed",
            "mission": {"mission": {"mission_type": "visitor_escort"}},
        }
    )
    run_id = submitted["run"]["run_id"]

    payload = module.pause_payload(
        run_id,
        operator_id="security-1",
        reason="visitor entered path",
        risk_acknowledgement=True,
        operator_context={
            "operator_id": "security-1",
            "roles": ["operator"],
            "authenticated": True,
            "source": "oidc",
            "permission": "runtime:pause",
            "conversation_session_id": "conv-action-1",
        },
    )

    action = payload["run"]["operator_actions"][-1]
    assert action["operator_id"] == "security-1"
    assert action["operator_context"] == {
        "operator_id": "security-1",
        "roles": ["operator"],
        "authenticated": True,
        "source": "oidc",
        "permission": "runtime:pause",
        "conversation_session_id": "conv-action-1",
    }
