from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from askme.runtime.module import ModuleRegistry

from askme.cognition import WorldStateService
from askme.ports.runtime_executor import (
    RuntimeExecutorCancelResult,
    RuntimeExecutorStatusUpdate,
    RuntimeExecutorSubmitResult,
)
from askme.providers.runtime_executor import (
    HttpRuntimeExecutorTransport,
    build_runtime_executor_transport,
)
from askme.runtime.modules.runtime_handoff_module import RuntimeHandoffModule


class _ModuleTransport:
    def __init__(self) -> None:
        self.cancel_calls = []
        self.closed = False

    def submit(self, request):
        return RuntimeExecutorSubmitResult(
            remote_task_id=f"remote:{request.correlation_id}",
            status="queued",
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            cursor="1",
            observed_at=time.time(),
        )

    def get_status(self, request):
        return RuntimeExecutorStatusUpdate(
            remote_task_id=request.remote_task_id,
            status="executing",
            correlation_id=request.correlation_id,
            cursor="2",
            observed_at=time.time(),
        )

    def cancel(self, request):
        self.cancel_calls.append(request)
        return RuntimeExecutorCancelResult(
            remote_task_id=request.remote_task_id,
            status="cancelling",
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            cursor="3",
            observed_at=time.time(),
        )

    def close(self) -> None:
        self.closed = True


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
    mod = RuntimeHandoffModule(executor_transport_factory=build_runtime_executor_transport)

    mod.build(
        {
            "runtime_handoff": {
                "profile": "lab",
                "endpoint": "https://runtime.example",
                "enable_external_runtime": True,
                "credential_env_var": "ASKME_RUNTIME_TOKEN",
            }
        },
        registry,
    )

    payload = mod.context_payload()

    assert payload["profile"] == "lab"
    assert payload["runtime_client"]["endpoint_configured"] is True
    assert payload["runtime_client"]["enable_external_runtime"] is True
    assert payload["runtime_client"]["hardware_dispatch"] is False
    assert isinstance(mod.runtime_handoff_service.executor_transport, HttpRuntimeExecutorTransport)


def test_runtime_handoff_module_keeps_external_transport_disabled_by_default() -> None:
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
                "profile": "external",
                "endpoint": "https://runtime.example",
            }
        },
        registry,
    )

    assert mod.runtime_handoff_service.executor_transport is None


def test_runtime_handoff_module_fails_closed_without_composition_transport() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())

    with pytest.raises(ValueError, match="must be injected by the composition root"):
        RuntimeHandoffModule().build(
            {
                "runtime_handoff": {
                    "profile": "external",
                    "endpoint": "https://runtime.example",
                    "enable_external_runtime": True,
                    "credential_env_var": "ASKME_RUNTIME_TOKEN",
                }
            },
            registry,
        )


@pytest.mark.parametrize(
    ("profile", "endpoint"),
    [
        ("external", "https://runtime.example"),
        ("lab", "https://runtime.example"),
    ],
)
def test_runtime_handoff_module_requires_auth_except_lab_loopback(
    profile: str,
    endpoint: str,
) -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())

    with pytest.raises(ValueError, match="credential_env_var is required"):
        RuntimeHandoffModule(
            executor_transport_factory=build_runtime_executor_transport
        ).build(
            {
                "runtime_handoff": {
                    "profile": profile,
                    "endpoint": endpoint,
                    "enable_external_runtime": True,
                }
            },
            registry,
        )


def test_runtime_handoff_module_allows_authless_lab_loopback() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    module = RuntimeHandoffModule(executor_transport_factory=build_runtime_executor_transport)

    module.build(
        {
            "runtime_handoff": {
                "profile": "lab",
                "endpoint": "http://127.0.0.1:8765",
                "enable_external_runtime": True,
            }
        },
        registry,
    )

    assert isinstance(
        module.runtime_handoff_service.executor_transport, HttpRuntimeExecutorTransport
    )


def test_runtime_handoff_module_rejects_http_loopback_for_external_profile() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())

    with pytest.raises(ValueError, match="external runtime endpoint must use HTTPS"):
        RuntimeHandoffModule(
            executor_transport_factory=build_runtime_executor_transport
        ).build(
            {
                "runtime_handoff": {
                    "profile": "external",
                    "endpoint": "http://127.0.0.1:8765",
                    "enable_external_runtime": True,
                    "credential_env_var": "ASKME_RUNTIME_TOKEN",
                }
            },
            registry,
        )


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
        "thread_id": "voice-module-thread",
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
        "thread_id": "conv-action-1",
    }


@pytest.mark.asyncio
async def test_runtime_handoff_module_keeps_local_cancel_semantics_async() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    module = RuntimeHandoffModule()
    module.build({"runtime_handoff": {"profile": "fake", "fake_auto_complete": False}}, registry)
    submitted = module.submit_plan_payload(
        {
            "plan_id": "local-cancel-plan",
            "planning_session_id": "local-cancel-session",
            "intent": "status_report",
            "handoff_ready": True,
            "confirmation_status": "confirmed",
            "mission": {"mission": {"mission_type": "status_report"}},
        }
    )

    payload = await module.cancel_payload(
        submitted["run"]["run_id"],
        operator_context={
            "operator_id": "askme.operator",
            "roles": ["operator"],
            "authenticated": True,
            "source": "test",
            "permission": "runtime:cancel",
        },
    )

    assert payload["handled"] is True
    assert payload["run"]["current_state"] == "cancelled"


@pytest.mark.asyncio
async def test_runtime_handoff_module_owns_external_supervisor_cancel_and_shutdown() -> None:
    class FakeCognitionModule:
        name = "cognition"

        def __init__(self) -> None:
            self.world_state = WorldStateService()

    transport = _ModuleTransport()
    registry = ModuleRegistry()
    registry.register(FakeCognitionModule())
    module = RuntimeHandoffModule(
        executor_transport_factory=lambda _profile, _config: transport
    )
    module.build(
        {
            "runtime_handoff": {
                "profile": "external",
                "endpoint": "https://runtime.example",
                "enable_external_runtime": True,
                "external_runtime": {
                    "poll_initial_seconds": 0.01,
                    "poll_jitter_ratio": 0.0,
                },
            }
        },
        registry,
    )
    assert module.external_task_supervisor is not None
    assert module.external_task_supervisor.transport is transport
    await module.start()
    submitted = module.submit_plan_payload(
        {
            "plan_id": "external-cancel-plan",
            "planning_session_id": "external-cancel-session",
            "intent": "status_report",
            "handoff_ready": True,
            "confirmation_status": "confirmed",
            "mission": {"mission": {"mission_type": "status_report"}},
        }
    )
    run_id = submitted["run"]["run_id"]

    payload = await module.cancel_payload(
        run_id,
        operator_id="operator-9",
        reason="dashboard cancel",
        operator_context={
            "operator_id": "operator-9",
            "roles": ["operator"],
            "authenticated": True,
            "source": "dashboard",
            "permission": "runtime:cancel",
        },
    )
    await module.stop()

    assert payload["remote_acknowledged"] is True
    assert payload["state"] == "cancel_requested"
    assert payload["run"]["operator_actions"][-1]["operator_id"] == "operator-9"
    assert len(transport.cancel_calls) == 1
    assert transport.closed is True
    assert module.external_task_supervisor.tracked_run_ids == ()
