"""Task handoff, arbiter, mission and runtime audit services."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ActivePerceptionRequest": ("askme.runtime.task.handoff", "ActivePerceptionRequest"),
    "ExternalRuntimeArbiter": ("askme.runtime.task.handoff", "ExternalRuntimeArbiter"),
    "FakeRuntimeArbiter": ("askme.runtime.task.handoff", "FakeRuntimeArbiter"),
    "InspectionReport": ("askme.runtime.task.mission", "InspectionReport"),
    "MissionPlan": ("askme.runtime.task.mission", "MissionPlan"),
    "MissionService": ("askme.runtime.task.mission", "MissionService"),
    "MissionStep": ("askme.runtime.task.mission", "MissionStep"),
    "OperatorPolicyService": ("askme.runtime.task.handoff", "OperatorPolicyService"),
    "ReplanProposal": ("askme.runtime.task.handoff", "ReplanProposal"),
    "RuntimeArbiter": ("askme.runtime.task.handoff", "RuntimeArbiter"),
    "RuntimeArbiterClient": (
        "askme.runtime.task.arbiter_client",
        "RuntimeArbiterClient",
    ),
    "RuntimeArbiterClientError": (
        "askme.runtime.task.arbiter_client",
        "RuntimeArbiterClientError",
    ),
    "RuntimeAuditConfig": ("askme.runtime.task.audit", "RuntimeAuditConfig"),
    "RuntimeAuditLog": ("askme.runtime.task.audit", "RuntimeAuditLog"),
    "RuntimeEvent": ("askme.runtime.task.handoff", "RuntimeEvent"),
    "ExternalCancelOutcome": (
        "askme.runtime.task.executor_supervisor",
        "ExternalCancelOutcome",
    ),
    "ExternalTaskSupervisor": (
        "askme.runtime.task.executor_supervisor",
        "ExternalTaskSupervisor",
    ),
    "RuntimeExecutorTransport": (
        "askme.ports.runtime_executor",
        "RuntimeExecutorTransport",
    ),
    "RuntimeHandoffService": ("askme.runtime.task.handoff", "RuntimeHandoffService"),
    "SafetyAssessment": ("askme.runtime.task.handoff", "SafetyAssessment"),
    "SafetyPreflightService": ("askme.runtime.task.handoff", "SafetyPreflightService"),
    "ShadowRuntimeArbiter": ("askme.runtime.task.handoff", "ShadowRuntimeArbiter"),
    "SimRuntimeArbiter": ("askme.runtime.task.handoff", "SimRuntimeArbiter"),
    "RuntimeSkillDefinition": ("askme.runtime.task.handoff", "SkillDefinition"),
    "SkillDefinition": ("askme.runtime.task.handoff", "SkillDefinition"),
    "SkillRegistry": ("askme.runtime.task.handoff", "SkillRegistry"),
    "SkillResult": ("askme.runtime.task.handoff", "SkillResult"),
    "TaskHandoff": ("askme.runtime.task.handoff", "TaskHandoff"),
    "TaskReportService": ("askme.runtime.task.handoff", "TaskReportService"),
    "TaskRun": ("askme.runtime.task.handoff", "TaskRun"),
    "TaskRunService": ("askme.runtime.task.handoff", "TaskRunService"),
    "TaskRunStore": ("askme.runtime.task.handoff", "TaskRunStore"),
    "TaskRunStoreConfig": ("askme.runtime.task.handoff", "TaskRunStoreConfig"),
    "TaskStep": ("askme.runtime.task.handoff", "TaskStep"),
    "VoiceTaskLifecycleService": (
        "askme.runtime.task.voice_lifecycle",
        "VoiceTaskLifecycleService",
    ),
    "VoiceTaskOperatorContext": (
        "askme.runtime.task.voice_lifecycle",
        "VoiceTaskOperatorContext",
    ),
    "build_field_runtime_callback_payload": (
        "askme.runtime.task.field_callbacks",
        "build_field_runtime_callback_payload",
    ),
    "build_field_runtime_callback_sequence": (
        "askme.runtime.task.field_callbacks",
        "build_field_runtime_callback_sequence",
    ),
    "default_skill_definitions": (
        "askme.runtime.task.handoff",
        "default_skill_definitions",
    ),
    "derive_field_runtime_callback_id": (
        "askme.runtime.task.field_callbacks",
        "derive_field_runtime_callback_id",
    ),
    "field_event_id_from_runtime_result": (
        "askme.runtime.task.field_callbacks",
        "field_event_id_from_runtime_result",
    ),
    "post_field_runtime_callback": (
        "askme.runtime.task.field_callbacks",
        "post_field_runtime_callback",
    ),
    "post_field_runtime_callback_sequence": (
        "askme.runtime.task.field_callbacks",
        "post_field_runtime_callback_sequence",
    ),
    "sign_field_runtime_callback_payload": (
        "askme.runtime.task.field_callbacks",
        "sign_field_runtime_callback_payload",
    ),
    "unsigned_field_runtime_callback_payload": (
        "askme.runtime.task.field_callbacks",
        "unsigned_field_runtime_callback_payload",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve runtime task contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
