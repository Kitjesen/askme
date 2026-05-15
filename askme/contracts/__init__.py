"""Product-level I/O contracts for robot capabilities.

These contracts are the stable boundary between perception, intent routing,
capability packages, safety, runtime handoff, and customer-facing UI.
"""

from askme.contracts.adapters import (
    interaction_decision_to_action_decision,
    perception_snapshot_to_input,
)
from askme.contracts.catalog import CONTRACT_VERSION, contract_catalog, contract_examples
from askme.contracts.field_adapters import (
    field_event_to_action_decision,
    field_event_to_evidence_refs,
    field_event_to_product_contracts,
    field_event_to_user_output,
)
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
    DependencyKind,
    PackageRuntimeInventory,
    PackageStatus,
    ScenarioPackageManifest,
    evaluate_capability_package_readiness,
    evaluate_scenario_package_readiness,
)

__all__ = [
    "ActionDecision",
    "ActorType",
    "AudioInput",
    "CONTRACT_VERSION",
    "CapabilityDependency",
    "CapabilityPackageManifest",
    "DependencyKind",
    "EvidenceRef",
    "Freshness",
    "IntentInput",
    "IntentType",
    "LocationRef",
    "PerceptionInput",
    "PackageStatus",
    "PackageRuntimeInventory",
    "RiskLevel",
    "RobotActionType",
    "ScenarioPackageManifest",
    "SensorInput",
    "UserFacingOutput",
    "VisionInput",
    "contract_catalog",
    "contract_examples",
    "evaluate_capability_package_readiness",
    "evaluate_scenario_package_readiness",
    "field_event_to_action_decision",
    "field_event_to_evidence_refs",
    "field_event_to_product_contracts",
    "field_event_to_user_output",
    "interaction_decision_to_action_decision",
    "perception_snapshot_to_input",
]
