from __future__ import annotations

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


def test_capability_package_readiness_blocks_missing_required_dependencies() -> None:
    manifest = CapabilityPackageManifest.from_payload(
        {
            "package_id": "cap-fire-smoke",
            "display_name": "Fire and smoke detection",
            "capability": "detect_fire_smoke",
            "inputs": ["vision", "temperature"],
            "outputs": ["incident_event"],
            "dependencies": [
                {
                    "name": "vision_bridge",
                    "kind": "service",
                    "reason": "Provides fresh camera detections.",
                },
                {
                    "name": "smoke_sensor",
                    "kind": "sensor",
                    "required": False,
                    "fallback": "Use vision-only confidence.",
                },
            ],
            "risk_level": "high",
            "risk_controls": ["Notify security before evacuation broadcast."],
            "customer_visible_name": "Fire and smoke alert",
            "customer_visible_description": "Detects fire and smoke risk.",
        }
    )

    blocked = evaluate_capability_package_readiness(manifest, inventory={})
    ready = evaluate_capability_package_readiness(
        manifest,
        inventory=PackageRuntimeInventory(services=frozenset({"vision_bridge"})),
    )

    assert blocked["status"] == "blocked"
    assert blocked["status_label"] == "阻断启用"
    assert blocked["enableable"] is False
    assert blocked["missing_required_dependencies"] == ["vision_bridge"]
    assert blocked["customer_missing_dependencies"] == ["vision_bridge"]
    assert blocked["engineering_missing_dependencies"] == ["vision_bridge"]
    assert blocked["dependency_checks"][1]["status"] == "fallback_available"
    assert "vision_bridge" in blocked["customer_message"]
    assert blocked["customer_next_step"] == "补齐缺失依赖或从本客户项目的启用范围中移除该场景。"
    assert blocked["enablement_decision"] == {
        "package_kind": "capability_package",
        "decision": "blocked",
        "status": "blocked",
        "status_label": "阻断启用",
        "can_run_controlled_demo": False,
        "can_enter_customer_pilot": False,
        "can_claim_unattended_production": False,
        "release_claim": "缺失必要依赖，不能对客户启用、演示或声明可交付。",
        "next_action": "补齐缺失依赖或从本客户项目的启用范围中移除该场景。",
        "blocking_dependencies": ["vision_bridge"],
        "manual_acceptance_dependencies": [],
    }
    assert ready["status"] == "ready"
    assert ready["status_label"] == "可进入现场验证"
    assert ready["enableable"] is True
    assert ready["customer_next_step"] == (
        "安排现场联调，验证真实传感器、通知通道、机器人执行器和人工接管流程。"
    )
    assert ready["enablement_decision"]["decision"] == "site_validation_allowed"
    assert ready["enablement_decision"]["can_enter_customer_pilot"] is True
    assert ready["enablement_decision"]["can_claim_unattended_production"] is False


def test_capability_package_readiness_flags_optional_manual_checks() -> None:
    manifest = CapabilityPackageManifest.from_payload(
        {
            "package_id": "cap-wayfinding",
            "display_name": "Wayfinding answer",
            "capability": "answer_wayfinding",
            "inputs": ["visitor_query"],
            "outputs": ["voice_answer"],
            "dependencies": [
                {
                    "name": "site_map_review",
                    "kind": "human_approval",
                    "required": False,
                    "reason": "Confirms the route wording is customer approved.",
                }
            ],
            "customer_visible_description": "Answers visitor destination questions.",
        }
    )

    payload = evaluate_capability_package_readiness(manifest, inventory={})

    assert payload["status"] == "manual_check"
    assert payload["status_label"] == "需要人工确认"
    assert payload["enableable"] is False
    assert payload["manual_check_dependencies"] == ["site_map_review"]
    assert payload["customer_next_step"] == "完成主管或现场负责人确认后，再进入客户试点启用。"
    assert payload["enablement_decision"]["decision"] == "human_acceptance_required"
    assert payload["enablement_decision"]["can_run_controlled_demo"] is True
    assert payload["enablement_decision"]["can_enter_customer_pilot"] is False
    assert payload["enablement_decision"]["manual_acceptance_dependencies"] == [
        "site_map_review"
    ]


def test_scenario_package_readiness_requires_declared_capability_packages() -> None:
    manifest = ScenarioPackageManifest.from_payload(
        {
            "package_id": "scenario-visitor-guide",
            "display_name": "Visitor guide",
            "scenario": "visitor_wayfinding_and_escort",
            "capability_packages": ["cap-wayfinding", "cap-escort"],
            "customer_visible_name": "Visitor guide service",
            "customer_visible_steps": ["Ask destination", "Confirm", "Guide or escort"],
            "outputs": ["interaction_record", "escort_handoff"],
        }
    )

    blocked = evaluate_scenario_package_readiness(
        manifest,
        inventory={"capability_packages": ["cap-wayfinding"]},
    )
    ready = evaluate_scenario_package_readiness(
        manifest,
        inventory={"capability_packages": ["cap-wayfinding", "cap-escort"]},
    )

    assert blocked["status"] == "blocked"
    assert blocked["status_label"] == "阻断启用"
    assert blocked["releasable"] is False
    assert blocked["missing_required_dependencies"] == ["cap-escort"]
    assert blocked["customer_missing_dependencies"] == ["cap-escort"]
    assert blocked["engineering_missing_dependencies"] == ["cap-escort"]
    assert blocked["enablement_decision"]["package_kind"] == "scenario_package"
    assert blocked["enablement_decision"]["blocking_dependencies"] == ["cap-escort"]
    assert ready["status"] == "ready"
    assert ready["status_label"] == "可进入现场验证"
    assert ready["releasable"] is True
    assert ready["customer_next_step"] == (
        "安排现场联调，验证真实传感器、通知通道、机器人执行器和人工接管流程。"
    )
def test_scenario_package_readiness_separates_customer_and_engineering_missing_dependencies() -> None:
    manifest = ScenarioPackageManifest.from_payload(
        {
            "package_id": "scenario-escort",
            "display_name": "Visitor escort",
            "scenario": "visitor_escort",
            "capability_packages": ["capability.navigate", "capability.escort_visitor"],
            "dependencies": [
                {
                    "name": "navigate",
                    "kind": "skill",
                    "reason": "Navigation skill is required by the navigate capability package.",
                },
                {
                    "name": "escort_visitor",
                    "kind": "skill",
                    "reason": "Escort skill is required by the escort capability package.",
                },
            ],
            "customer_visible_name": "Visitor escort service",
            "customer_visible_steps": ["Confirm destination", "Lead visitor"],
        }
    )

    payload = evaluate_scenario_package_readiness(manifest, inventory={})

    assert payload["status"] == "blocked"
    assert payload["missing_required_dependencies"] == [
        "capability.navigate",
        "capability.escort_visitor",
        "navigate",
        "escort_visitor",
    ]
    assert payload["engineering_missing_dependencies"] == [
        "capability.navigate",
        "capability.escort_visitor",
        "navigate",
        "escort_visitor",
    ]
    assert payload["customer_missing_dependencies"] == [
        "capability.navigate",
        "capability.escort_visitor",
    ]
    assert "navigate, escort_visitor" not in payload["customer_message"]
