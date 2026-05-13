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


__all__ = [
    "CapabilityDependency",
    "CapabilityPackageManifest",
    "DependencyKind",
    "PackageStatus",
    "ScenarioPackageManifest",
]
