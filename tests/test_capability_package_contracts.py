from __future__ import annotations

from askme.contracts.package import (
    CapabilityDependency,
    CapabilityPackageManifest,
    DependencyKind,
    PackageStatus,
    ScenarioPackageManifest,
)


def test_capability_package_manifest_round_trips_product_boundary() -> None:
    manifest = CapabilityPackageManifest.from_payload(
        {
            "package_id": "cap-fire-smoke",
            "display_name": "Fire and smoke detection",
            "status": "pilot",
            "capability": "detect_fire_smoke",
            "summary": "Detect visible smoke and fire risk from robot perception.",
            "inputs": ["PerceptionInput.vision", "SensorInput.temperature_c"],
            "outputs": ["ActionDecision.NOTIFY_HUMAN", "UserFacingOutput.display_text"],
            "dependencies": [
                {
                    "name": "vision_bridge",
                    "kind": "service",
                    "required": True,
                    "reason": "Provides fresh camera detections.",
                },
                {
                    "name": "thermal_sensor",
                    "kind": "sensor",
                    "required": False,
                    "fallback": "Use vision-only confidence.",
                },
            ],
            "risk_level": "high",
            "risk_controls": ["Human confirmation before evacuation notice."],
            "customer_visible_name": "Fire and smoke alert",
            "customer_visible_description": "Alerts staff when robot evidence suggests smoke or fire.",
            "customer_visible_outputs": ["Alert card", "Incident record"],
            "tags": ["security", "incident"],
        }
    )

    payload = manifest.to_dict()

    assert manifest.status == PackageStatus.PILOT
    assert manifest.dependencies[0].kind == DependencyKind.SERVICE
    assert [dependency.name for dependency in manifest.required_dependencies] == ["vision_bridge"]
    assert manifest.validate() == []
    assert payload["risk_level"] == "high"
    assert payload["dependencies"][1]["fallback"] == "Use vision-only confidence."
    assert payload["customer_visible_outputs"] == ["Alert card", "Incident record"]


def test_capability_dependency_accepts_scalar_payload_for_small_manifests() -> None:
    dependency = CapabilityDependency.from_payload("navigate")

    assert dependency.name == "navigate"
    assert dependency.kind == DependencyKind.SKILL
    assert dependency.required is True
    assert dependency.validate() == ["dependency reason is required for navigate"]


def test_scenario_package_manifest_expresses_customer_visible_bundle() -> None:
    manifest = ScenarioPackageManifest.from_payload(
        {
            "package_id": "scenario-night-patrol",
            "display_name": "Night patrol scenario",
            "status": "active",
            "scenario": "night_security_patrol",
            "site_id": "park-a",
            "customer_name": "Demo Park",
            "capability_packages": [
                "cap-route-patrol",
                "cap-night-intruder",
                "cap-alert-dispatch",
            ],
            "inputs": ["Patrol route", "PerceptionInput", "Site policy"],
            "outputs": ["Patrol report", "Security alert"],
            "dependencies": [
                {
                    "name": "cap-night-intruder",
                    "kind": "capability_package",
                    "reason": "Detects people in restricted areas.",
                }
            ],
            "risk_level": "medium",
            "customer_visible_name": "Night security patrol",
            "customer_visible_description": "Robot patrols fixed routes and reports restricted-area activity.",
            "customer_visible_steps": ["Start patrol", "Inspect checkpoints", "Notify security"],
            "customer_visible_outputs": ["Route summary", "Evidence snapshots"],
            "rollout_notes": "Pilot on west gate route first.",
        }
    )

    payload = manifest.to_dict()

    assert manifest.status == PackageStatus.ACTIVE
    assert manifest.validate() == []
    assert manifest.required_dependencies[0].kind == DependencyKind.CAPABILITY_PACKAGE
    assert payload["capability_packages"] == [
        "cap-route-patrol",
        "cap-night-intruder",
        "cap-alert-dispatch",
    ]
    assert payload["customer_visible_steps"] == [
        "Start patrol",
        "Inspect checkpoints",
        "Notify security",
    ]


def test_package_manifests_validate_minimum_product_fields() -> None:
    capability = CapabilityPackageManifest.from_payload(
        {
            "package_id": "cap-dangerous-motion",
            "display_name": "Dangerous motion",
            "capability": "dog_control",
            "inputs": ["IntentInput"],
            "outputs": ["ActionDecision"],
            "risk_level": "high",
        }
    )
    scenario = ScenarioPackageManifest(
        package_id="scenario-empty",
        display_name="Empty scenario",
        scenario="",
    )

    assert capability.validate() == [
        "customer_visible_description or customer_visible_outputs is required",
        "risk_controls are required for high or critical risk packages",
    ]
    assert scenario.validate() == [
        "scenario is required",
        "capability_packages are required",
        "customer_visible_description or customer_visible_steps is required",
    ]
