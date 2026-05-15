"""Product manifests for capability and scenario packages.

Capability packages describe reusable robot abilities. Scenario packages bind
those abilities into customer-visible deployments such as patrol, visitor
service, or incident response flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from askme.contracts.io import RiskLevel


class PackageStatus(StrEnum):
    DRAFT = "draft"
    PILOT = "pilot"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class DependencyKind(StrEnum):
    SKILL = "skill"
    TOOL = "tool"
    SERVICE = "service"
    MODEL = "model"
    SENSOR = "sensor"
    DATASET = "dataset"
    HUMAN_APPROVAL = "human_approval"
    CAPABILITY_PACKAGE = "capability_package"


@dataclass(frozen=True)
class PackageRuntimeInventory:
    """Available runtime assets used to decide whether a package can run."""

    skills: frozenset[str] = field(default_factory=frozenset)
    tools: frozenset[str] = field(default_factory=frozenset)
    services: frozenset[str] = field(default_factory=frozenset)
    models: frozenset[str] = field(default_factory=frozenset)
    sensors: frozenset[str] = field(default_factory=frozenset)
    datasets: frozenset[str] = field(default_factory=frozenset)
    capability_packages: frozenset[str] = field(default_factory=frozenset)
    approvals: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_payload(cls, payload: Any) -> PackageRuntimeInventory:
        data = _dict(payload)
        return cls(
            skills=frozenset(_tuple_of_text(data.get("skills"))),
            tools=frozenset(_tuple_of_text(data.get("tools"))),
            services=frozenset(_tuple_of_text(data.get("services"))),
            models=frozenset(_tuple_of_text(data.get("models"))),
            sensors=frozenset(_tuple_of_text(data.get("sensors"))),
            datasets=frozenset(_tuple_of_text(data.get("datasets"))),
            capability_packages=frozenset(_tuple_of_text(data.get("capability_packages"))),
            approvals=frozenset(_tuple_of_text(data.get("approvals"))),
        )

    def has(self, dependency: CapabilityDependency) -> bool:
        values = {
            DependencyKind.SKILL: self.skills,
            DependencyKind.TOOL: self.tools,
            DependencyKind.SERVICE: self.services,
            DependencyKind.MODEL: self.models,
            DependencyKind.SENSOR: self.sensors,
            DependencyKind.DATASET: self.datasets,
            DependencyKind.HUMAN_APPROVAL: self.approvals,
            DependencyKind.CAPABILITY_PACKAGE: self.capability_packages,
        }[dependency.kind]
        return dependency.name in values

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": sorted(self.skills),
            "tools": sorted(self.tools),
            "services": sorted(self.services),
            "models": sorted(self.models),
            "sensors": sorted(self.sensors),
            "datasets": sorted(self.datasets),
            "capability_packages": sorted(self.capability_packages),
            "approvals": sorted(self.approvals),
        }


@dataclass(frozen=True)
class CapabilityDependency:
    name: str
    kind: DependencyKind = DependencyKind.SKILL
    required: bool = True
    version: str = ""
    reason: str = ""
    fallback: str = ""
    customer_visible: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> CapabilityDependency:
        if isinstance(payload, CapabilityDependency):
            return payload
        if not isinstance(payload, dict):
            return cls(name=_clean_text(payload))
        return cls(
            name=_clean_text(payload.get("name") or payload.get("id")),
            kind=_enum_value(DependencyKind, payload.get("kind"), DependencyKind.SKILL),
            required=bool(payload.get("required", True)),
            version=_clean_text(payload.get("version")),
            reason=_clean_text(payload.get("reason")),
            fallback=_clean_text(payload.get("fallback")),
            customer_visible=bool(payload.get("customer_visible", False)),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("dependency name is required")
        if self.required and not self.reason:
            errors.append(f"dependency reason is required for {self.name or '<missing>'}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "required": self.required,
            "version": self.version,
            "reason": self.reason,
            "fallback": self.fallback,
            "customer_visible": self.customer_visible,
        }


def evaluate_capability_package_readiness(
    manifest: CapabilityPackageManifest | dict[str, Any],
    inventory: PackageRuntimeInventory | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether one capability package can be enabled in this runtime."""
    capability = CapabilityPackageManifest.from_payload(manifest)
    inv = _inventory(inventory)
    validation_errors = capability.validate()
    dependency_checks = _dependency_checks(capability.dependencies, inv)
    missing_required = [
        item for item in dependency_checks if item["required"] and item["status"] == "missing"
    ]
    manual_checks = [
        item for item in dependency_checks if item["status"] == "manual_check"
    ]
    if validation_errors or missing_required:
        status = "blocked"
        enableable = False
    elif manual_checks:
        status = "manual_check"
        enableable = False
    else:
        status = "ready"
        enableable = True
    missing_names = [item["name"] for item in missing_required]
    manual_names = [item["name"] for item in manual_checks]
    package_name = capability.customer_visible_name or capability.display_name
    return {
        "kind": "capability_package",
        "package_id": capability.package_id,
        "display_name": capability.display_name,
        "status": status,
        "status_label": _readiness_status_label(status),
        "enableable": enableable,
        "validation_errors": validation_errors,
        "dependency_checks": dependency_checks,
        "missing_required_dependencies": missing_names,
        "customer_missing_dependencies": missing_names,
        "engineering_missing_dependencies": missing_names,
        "manual_check_dependencies": manual_names,
        "risk_level": capability.risk_level.value,
        "required_risk_controls": list(capability.risk_controls),
        "customer_visible_name": capability.customer_visible_name,
        "customer_next_step": _readiness_next_step(
            status=status,
            missing=missing_names,
            manual=manual_names,
            validation_errors=validation_errors,
        ),
        "customer_message": _readiness_customer_message(
            package_name=package_name,
            status=status,
            missing=missing_names,
        ),
        "enablement_decision": _enablement_decision(
            package_kind="capability_package",
            status=status,
            status_label=_readiness_status_label(status),
            next_step=_readiness_next_step(
                status=status,
                missing=missing_names,
                manual=manual_names,
                validation_errors=validation_errors,
            ),
            missing=missing_names,
            manual=manual_names,
        ),
    }


@dataclass(frozen=True)
class CapabilityPackageManifest:
    package_id: str
    display_name: str
    version: str = "1.0.0"
    status: PackageStatus = PackageStatus.DRAFT
    capability: str = ""
    summary: str = ""
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[CapabilityDependency, ...] = field(default_factory=tuple)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_controls: tuple[str, ...] = field(default_factory=tuple)
    customer_visible_name: str = ""
    customer_visible_description: str = ""
    customer_visible_outputs: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> CapabilityPackageManifest:
        if isinstance(payload, CapabilityPackageManifest):
            return payload
        data = _dict(payload)
        return cls(
            package_id=_clean_text(data.get("package_id") or data.get("id")),
            display_name=_clean_text(data.get("display_name") or data.get("name")),
            version=_clean_text(data.get("version")) or "1.0.0",
            status=_enum_value(PackageStatus, data.get("status"), PackageStatus.DRAFT),
            capability=_clean_text(data.get("capability") or data.get("capability_name")),
            summary=_clean_text(data.get("summary") or data.get("description")),
            inputs=_tuple_of_text(data.get("inputs")),
            outputs=_tuple_of_text(data.get("outputs")),
            dependencies=tuple(
                CapabilityDependency.from_payload(item)
                for item in _list(data.get("dependencies"))
            ),
            risk_level=_enum_value(RiskLevel, data.get("risk_level"), RiskLevel.LOW),
            risk_controls=_tuple_of_text(data.get("risk_controls") or data.get("risks")),
            customer_visible_name=_clean_text(data.get("customer_visible_name")),
            customer_visible_description=_clean_text(
                data.get("customer_visible_description")
                or data.get("customer_description")
            ),
            customer_visible_outputs=_tuple_of_text(data.get("customer_visible_outputs")),
            tags=_tuple_of_text(data.get("tags")),
            metadata=_dict(data.get("metadata")),
        )

    @property
    def required_dependencies(self) -> tuple[CapabilityDependency, ...]:
        return tuple(dependency for dependency in self.dependencies if dependency.required)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.package_id:
            errors.append("package_id is required")
        if not self.display_name:
            errors.append("display_name is required")
        if not self.capability:
            errors.append("capability is required")
        if not self.inputs:
            errors.append("inputs are required")
        if not self.outputs:
            errors.append("outputs are required")
        if not (self.customer_visible_description or self.customer_visible_outputs):
            errors.append("customer_visible_description or customer_visible_outputs is required")
        if self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} and not self.risk_controls:
            errors.append("risk_controls are required for high or critical risk packages")
        for dependency in self.dependencies:
            errors.extend(dependency.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "version": self.version,
            "status": self.status.value,
            "capability": self.capability,
            "summary": self.summary,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "risk_level": self.risk_level.value,
            "risk_controls": list(self.risk_controls),
            "customer_visible_name": self.customer_visible_name,
            "customer_visible_description": self.customer_visible_description,
            "customer_visible_outputs": list(self.customer_visible_outputs),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScenarioPackageManifest:
    package_id: str
    display_name: str
    version: str = "1.0.0"
    status: PackageStatus = PackageStatus.DRAFT
    scenario: str = ""
    site_id: str = ""
    customer_name: str = ""
    capability_packages: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[CapabilityDependency, ...] = field(default_factory=tuple)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_controls: tuple[str, ...] = field(default_factory=tuple)
    customer_visible_name: str = ""
    customer_visible_description: str = ""
    customer_visible_steps: tuple[str, ...] = field(default_factory=tuple)
    customer_visible_outputs: tuple[str, ...] = field(default_factory=tuple)
    rollout_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> ScenarioPackageManifest:
        if isinstance(payload, ScenarioPackageManifest):
            return payload
        data = _dict(payload)
        return cls(
            package_id=_clean_text(data.get("package_id") or data.get("id")),
            display_name=_clean_text(data.get("display_name") or data.get("name")),
            version=_clean_text(data.get("version")) or "1.0.0",
            status=_enum_value(PackageStatus, data.get("status"), PackageStatus.DRAFT),
            scenario=_clean_text(data.get("scenario") or data.get("scenario_name")),
            site_id=_clean_text(data.get("site_id")),
            customer_name=_clean_text(data.get("customer_name")),
            capability_packages=_tuple_of_text(
                data.get("capability_packages") or data.get("capabilities")
            ),
            inputs=_tuple_of_text(data.get("inputs")),
            outputs=_tuple_of_text(data.get("outputs")),
            dependencies=tuple(
                CapabilityDependency.from_payload(item)
                for item in _list(data.get("dependencies"))
            ),
            risk_level=_enum_value(RiskLevel, data.get("risk_level"), RiskLevel.LOW),
            risk_controls=_tuple_of_text(data.get("risk_controls") or data.get("risks")),
            customer_visible_name=_clean_text(data.get("customer_visible_name")),
            customer_visible_description=_clean_text(
                data.get("customer_visible_description")
                or data.get("customer_description")
            ),
            customer_visible_steps=_tuple_of_text(data.get("customer_visible_steps")),
            customer_visible_outputs=_tuple_of_text(data.get("customer_visible_outputs")),
            rollout_notes=_clean_text(data.get("rollout_notes")),
            metadata=_dict(data.get("metadata")),
        )

    @property
    def required_dependencies(self) -> tuple[CapabilityDependency, ...]:
        return tuple(dependency for dependency in self.dependencies if dependency.required)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.package_id:
            errors.append("package_id is required")
        if not self.display_name:
            errors.append("display_name is required")
        if not self.scenario:
            errors.append("scenario is required")
        if not self.capability_packages:
            errors.append("capability_packages are required")
        if not (self.customer_visible_description or self.customer_visible_steps):
            errors.append("customer_visible_description or customer_visible_steps is required")
        if self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} and not self.risk_controls:
            errors.append("risk_controls are required for high or critical risk packages")
        for dependency in self.dependencies:
            errors.extend(dependency.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "version": self.version,
            "status": self.status.value,
            "scenario": self.scenario,
            "site_id": self.site_id,
            "customer_name": self.customer_name,
            "capability_packages": list(self.capability_packages),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "risk_level": self.risk_level.value,
            "risk_controls": list(self.risk_controls),
            "customer_visible_name": self.customer_visible_name,
            "customer_visible_description": self.customer_visible_description,
            "customer_visible_steps": list(self.customer_visible_steps),
            "customer_visible_outputs": list(self.customer_visible_outputs),
            "rollout_notes": self.rollout_notes,
            "metadata": dict(self.metadata),
        }


def evaluate_scenario_package_readiness(
    manifest: ScenarioPackageManifest | dict[str, Any],
    inventory: PackageRuntimeInventory | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a customer-visible scenario package can be released."""
    scenario = ScenarioPackageManifest.from_payload(manifest)
    inv = _inventory(inventory)
    validation_errors = scenario.validate()
    dependency_checks = _dependency_checks(scenario.dependencies, inv)
    declared_capability_checks = [
        {
            "name": package_id,
            "kind": DependencyKind.CAPABILITY_PACKAGE.value,
            "required": True,
            "status": "available" if package_id in inv.capability_packages else "missing",
            "reason": "Scenario requires this capability package.",
            "fallback": "",
            "customer_visible": True,
        }
        for package_id in scenario.capability_packages
    ]
    all_checks = [*declared_capability_checks, *dependency_checks]
    missing_required = [
        item for item in all_checks if item["required"] and item["status"] == "missing"
    ]
    manual_checks = [item for item in all_checks if item["status"] == "manual_check"]
    if validation_errors or missing_required:
        status = "blocked"
        releasable = False
    elif manual_checks:
        status = "manual_check"
        releasable = False
    else:
        status = "ready"
        releasable = True
    missing_names = [item["name"] for item in missing_required]
    customer_missing_names = _customer_missing_dependency_names(missing_required)
    manual_names = [item["name"] for item in manual_checks]
    package_name = scenario.customer_visible_name or scenario.display_name
    return {
        "kind": "scenario_package",
        "package_id": scenario.package_id,
        "display_name": scenario.display_name,
        "status": status,
        "status_label": _readiness_status_label(status),
        "releasable": releasable,
        "validation_errors": validation_errors,
        "dependency_checks": all_checks,
        "missing_required_dependencies": missing_names,
        "customer_missing_dependencies": customer_missing_names,
        "engineering_missing_dependencies": missing_names,
        "manual_check_dependencies": manual_names,
        "risk_level": scenario.risk_level.value,
        "required_risk_controls": list(scenario.risk_controls),
        "customer_visible_name": scenario.customer_visible_name,
        "customer_next_step": _readiness_next_step(
            status=status,
            missing=customer_missing_names,
            manual=manual_names,
            validation_errors=validation_errors,
        ),
        "customer_message": _readiness_customer_message(
            package_name=package_name,
            status=status,
            missing=customer_missing_names,
        ),
        "enablement_decision": _enablement_decision(
            package_kind="scenario_package",
            status=status,
            status_label=_readiness_status_label(status),
            next_step=_readiness_next_step(
                status=status,
                missing=customer_missing_names,
                manual=manual_names,
                validation_errors=validation_errors,
            ),
            missing=customer_missing_names,
            manual=manual_names,
        ),
    }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _tuple_of_text(value: Any) -> tuple[str, ...]:
    return tuple(item for item in (_clean_text(raw) for raw in _list(value)) if item)


def _enum_value(enum_cls: type[StrEnum], value: Any, default: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is not None:
        try:
            return enum_cls(str(value))
        except ValueError:
            pass
    return default


def _inventory(value: PackageRuntimeInventory | dict[str, Any] | None) -> PackageRuntimeInventory:
    if isinstance(value, PackageRuntimeInventory):
        return value
    return PackageRuntimeInventory.from_payload(value or {})


def _dependency_checks(
    dependencies: tuple[CapabilityDependency, ...],
    inventory: PackageRuntimeInventory,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for dependency in dependencies:
        available = inventory.has(dependency)
        if available:
            status = "available"
        elif dependency.required:
            status = "missing"
        else:
            status = "manual_check" if not dependency.fallback else "fallback_available"
        checks.append(
            {
                "name": dependency.name,
                "kind": dependency.kind.value,
                "required": dependency.required,
                "status": status,
                "reason": dependency.reason,
                "fallback": dependency.fallback,
                "customer_visible": dependency.customer_visible,
            }
        )
    return checks


def _customer_missing_dependency_names(missing_required: list[dict[str, Any]]) -> list[str]:
    names = [str(item.get("name") or "").strip() for item in missing_required if item.get("name")]
    package_names = {
        name
        for name, item in zip(names, missing_required, strict=False)
        if item.get("kind") == DependencyKind.CAPABILITY_PACKAGE.value
    }
    customer_names: list[str] = []
    for name, item in zip(names, missing_required, strict=False):
        if item.get("kind") == DependencyKind.SKILL.value and f"capability.{name}" in package_names:
            continue
        if name not in customer_names:
            customer_names.append(name)
    return customer_names


def _readiness_customer_message(*, package_name: str, status: str, missing: list[str]) -> str:
    if status == "ready":
        return f"{package_name} 已满足启用条件，可进入现场验证或发布流程。"
    if status == "manual_check":
        return f"{package_name} 仍有人工确认项，确认完成前不要对客户承诺自动运行。"
    missing_label = ", ".join(missing) if missing else "required fields"
    return f"{package_name} 缺少必要依赖：{missing_label}，暂不能启用。"


def _readiness_status_label(status: str) -> str:
    return {
        "ready": "可进入现场验证",
        "manual_check": "需要人工确认",
        "blocked": "阻断启用",
    }.get(status, "未知状态")


def _readiness_next_step(
    *,
    status: str,
    missing: list[str],
    manual: list[str],
    validation_errors: list[str],
) -> str:
    if validation_errors:
        return "先修正能力包合同字段，再重新提交启用检查。"
    if status == "ready":
        return "安排现场联调，验证真实传感器、通知通道、机器人执行器和人工接管流程。"
    if manual:
        return "完成主管或现场负责人确认后，再进入客户试点启用。"
    if missing:
        return "补齐缺失依赖或从本客户项目的启用范围中移除该场景。"
    return "重新运行启用检查并保留审计记录。"


def _enablement_decision(
    *,
    package_kind: str,
    status: str,
    status_label: str,
    next_step: str,
    missing: list[str],
    manual: list[str],
) -> dict[str, Any]:
    if status == "ready":
        return {
            "package_kind": package_kind,
            "decision": "site_validation_allowed",
            "status": status,
            "status_label": status_label,
            "can_run_controlled_demo": True,
            "can_enter_customer_pilot": True,
            "can_claim_unattended_production": False,
            "release_claim": "可进入现场验证或客户试点，不能声明无人值守生产上线。",
            "next_action": next_step,
            "blocking_dependencies": [],
            "manual_acceptance_dependencies": [],
        }
    if status == "manual_check":
        return {
            "package_kind": package_kind,
            "decision": "human_acceptance_required",
            "status": status,
            "status_label": status_label,
            "can_run_controlled_demo": True,
            "can_enter_customer_pilot": False,
            "can_claim_unattended_production": False,
            "release_claim": "只能做受控演示或内部验证，完成主管或现场负责人确认前不能进入客户试点。",
            "next_action": next_step,
            "blocking_dependencies": [],
            "manual_acceptance_dependencies": list(manual),
        }
    return {
        "package_kind": package_kind,
        "decision": "blocked",
        "status": status,
        "status_label": status_label,
        "can_run_controlled_demo": False,
        "can_enter_customer_pilot": False,
        "can_claim_unattended_production": False,
        "release_claim": "缺失必要依赖，不能对客户启用、演示或声明可交付。",
        "next_action": next_step,
        "blocking_dependencies": list(missing),
        "manual_acceptance_dependencies": list(manual),
    }


__all__ = [
    "CapabilityDependency",
    "CapabilityPackageManifest",
    "DependencyKind",
    "PackageStatus",
    "PackageRuntimeInventory",
    "ScenarioPackageManifest",
    "evaluate_capability_package_readiness",
    "evaluate_scenario_package_readiness",
]
