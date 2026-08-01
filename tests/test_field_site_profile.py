from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from askme.pipeline.field_operations import FieldOperationsService
from askme.pipeline.field_site_profile import (
    archive_customer_project_profile,
    build_customer_project_acceptance_registry,
    build_customer_project_catalog,
    build_customer_project_execution_bindings,
    build_customer_project_resource_catalog,
    build_site_profile_catalog,
    build_site_profile_report,
    build_solution_delivery_readiness,
    create_customer_project_from_template,
    create_customer_project_template_release_request,
    create_delivery_resource_governance_request,
    customer_project_acceptance_closure,
    customer_project_acceptance_report,
    customer_project_template_release_notes,
    delete_managed_object,
    disable_delivery_resource,
    escalate_overdue_delivery_resource_governance_requests,
    export_customer_project_acceptance_dossier,
    export_customer_project_package,
    export_customer_project_proposal_bundle,
    export_customer_project_template_release_notes_bundle,
    field_operations_config_from_site_profile,
    get_customer_project_profile,
    import_customer_project_package,
    list_customer_project_customer_signoffs,
    list_customer_project_onsite_evidence,
    list_customer_project_revisions,
    list_customer_project_template_release_requests,
    list_customer_project_template_revisions,
    list_customer_project_templates,
    list_delivery_resource_governance_requests,
    list_delivery_resource_registry,
    list_delivery_resource_revisions,
    load_field_site_profile,
    managed_object_catalog_from_site_profile,
    register_customer_project_acceptance_review,
    register_customer_project_customer_signoff,
    register_customer_project_onsite_evidence,
    render_site_profile_env_template,
    review_customer_project_template_release_request,
    review_delivery_resource_governance_request,
    rollback_customer_project_profile,
    rollback_delivery_resource_registry,
    site_profile_env_references,
    update_customer_project_template_release,
    upsert_customer_project_profile,
    upsert_delivery_resource,
    upsert_managed_object,
    validate_field_site_profile,
    verify_customer_project_acceptance_dossier,
    verify_customer_project_package,
    verify_customer_project_proposal_bundle,
)

from askme.pipeline.field import customer_project_acceptance as acceptance_module


def _set_demo_device_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET", "camera-main-road-secret")
    monkeypatch.setenv("ASKME_FIELD_CAMERA_GUIDE_SECRET", "camera-guide-secret")
    monkeypatch.setenv("ASKME_FIELD_SMOKE_WAREHOUSE_SECRET", "smoke-warehouse-secret")
    monkeypatch.setenv("ASKME_FIELD_ROBOT_THUNDER_SECRET", "robot-thunder-secret")


def _artifact_test_root(tmp_path: Path, name: str) -> Path:
    root = Path("artifacts/test-field-site-profile") / f"{name}-{tmp_path.name}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_real_link_acceptance_reports(root: Path) -> dict[str, str]:
    archive = root / "field-events.jsonl"
    scenario = root / "scenario.json"
    ingest = root / "ingest.json"
    voice = root / "voice.json"
    notification = root / "notification.json"
    runtime = root / "runtime.json"
    archive.write_text(
        json.dumps(
            {
                "event_id": "field-real-1",
                "scenario_id": "illegal_parking",
                "status": "archived",
                "created_at": time.time(),
                "payload": {
                    "source": "camera",
                    "device_trust": {
                        "trusted": True,
                        "device_id": "camera-main-road-1",
                        "source": "camera",
                        "status": "trusted",
                        "signature_verified": True,
                        "reason": "",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scenario.write_text(
        json.dumps(
            {
                "status": "passed",
                "scenario_count": 1,
                "passed": 1,
                "failed": 0,
                "external_services": True,
                "hardware_dispatch": True,
            }
        ),
        encoding="utf-8",
    )
    ingest.write_text(
        json.dumps({"status": "passed", "event_count": 1, "local_server": False}),
        encoding="utf-8",
    )
    voice.write_text(
        json.dumps(
            {
                "status": "passed",
                "local_server": False,
                "live_tts": True,
                "voice_delivery": {"status": "queued"},
                "voice_directive": {"resolved_profile": "emergency_short"},
            }
        ),
        encoding="utf-8",
    )
    notification.write_text(
        json.dumps(
            {
                "status": "passed",
                "local_server": False,
                "external_services": True,
                "collector_request_count": 1,
            }
        ),
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "shadow",
                "receipt_count": 3,
                "callback_status_codes": [200, 200, 200],
                "runtime_statuses": ["created", "submitted", "shadowed"],
                "final_runtime_delivery": {
                    "status": "shadowed",
                    "runtime_callback_trust": {"trusted": True, "status": "trusted"},
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "archive_path": str(archive),
        "scenario_report_path": str(scenario),
        "smoke_report_path": str(ingest),
        "voice_smoke_report_path": str(voice),
        "notification_smoke_report_path": str(notification),
        "runtime_roundtrip_report_path": str(runtime),
    }


def test_demo_field_site_profile_passes_and_exports_runtime_config() -> None:
    profile_path = Path("deploy/site-profiles/park-demo.yaml")

    report = build_site_profile_report(profile_path)

    assert report["status"] == "passed"
    assert report["summary"]["site_id"] == "inovx-demo-park"
    assert report["summary"]["parking_restricted_count"] >= 1
    assert report["summary"]["help_point_count"] >= 1
    assert report["summary"]["device_sources"]["camera"] >= 1
    assert report["summary"]["device_sources"]["sensor"] >= 1
    assert report["summary"]["device_sources"]["robot"] >= 1
    assert report["readiness"]["wayfinding_configured"] is True
    assert (
        report["field_operations_config"]["site_map"]["zones"]["main-road-1"]["parking_allowed"]
        is False
    )
    assert report["field_operations_config"]["dingtalk_webhooks"]["security"] == (
        "${ASKME_DINGTALK_SECURITY_WEBHOOK}"
    )


def test_julong_site_profile_scopes_guide_and_patrol_for_commissioning() -> None:
    profile_path = Path("deploy/site-profiles/julong-tech-e-valley.yaml")

    report = build_site_profile_report(profile_path)

    assert report["status"] == "passed"
    assert report["summary"]["site_id"] == "julong-tech-e-valley"
    assert report["summary"]["site_name"] == "聚龙科创e谷"
    assert report["summary"]["project_id"] == "julong-guide-patrol"
    assert report["summary"]["help_point_count"] == 1
    assert report["summary"]["device_sources"] == {
        "camera": 1,
        "robot": 1,
        "sensor": 1,
    }
    objects = report["field_operations_config"]["managed_objects"]
    assert set(objects) == {"patrol_checkpoints", "visitors"}
    assert "capability.patrol_scan" in objects["patrol_checkpoints"]["bindings"]["skill_packages"]
    assert "capability.answer_wayfinding" in objects["visitors"]["bindings"]["skill_packages"]
    zones = report["field_operations_config"]["site_map"]["zones"]
    assert all("待现场标定" in zone["name"] for zone in zones.values())


def test_field_site_profile_env_check_warns_for_unset_references(monkeypatch) -> None:
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)

    report = build_site_profile_report(
        Path("deploy/site-profiles/park-demo.yaml"),
        check_env=True,
    )

    assert report["status"] == "passed"
    assert (
        "responder_groups.security.webhook_env references unset environment variable "
        "ASKME_DINGTALK_SECURITY_WEBHOOK"
    ) in report["warnings"]


def test_field_site_profile_rejects_missing_product_critical_sections() -> None:
    report = validate_field_site_profile(
        {
            "site": {"site_id": "bad"},
            "zones": {
                "parking": {"type": "parking_area", "parking_allowed": True},
            },
            "responder_groups": {},
            "devices": {},
            "thresholds": {},
        }
    )

    assert report["status"] == "failed"
    assert "site.name is required" in report["errors"]
    assert "zones must include at least one main_channel" in report["errors"]
    assert "zones must include at least one help_point" in report["errors"]
    assert "devices must contain at least one registered device" in report["errors"]
    assert "responder_groups.security is required" in report["errors"]


def test_field_site_profile_exports_device_registry_with_env_placeholders() -> None:
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    config = field_operations_config_from_site_profile(profile)

    assert config["device_registry"]["camera-main-road-1"]["secret"] == (
        "${ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET}"
    )
    assert config["device_registry"]["robot-thunder-1"]["robot_id"] == "thunder-1"
    assert "vehicles" in config["managed_objects"]
    assert config["managed_objects"]["vehicles"]["responder_group"] == "security"


def test_site_profile_managed_objects_are_customer_specific_delivery_scope() -> None:
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    catalog = managed_object_catalog_from_site_profile(profile)

    assert catalog["object_type_count"] >= 4
    assert {"traffic", "safety", "cleaning", "visitor_service"} <= set(catalog["categories"])
    assert "illegal_parking" in catalog["scenario_ids"]
    assert "visitor_escort" in catalog["scenario_ids"]
    assert catalog["objects_by_id"]["trash_bins"]["responder_group"] == "cleaning"
    assert catalog["acceptance_summary"]["overall_status"] == "ready"
    assert catalog["acceptance_summary"]["ready_object_count"] >= 4
    vehicle_status = catalog["objects_by_id"]["vehicles"]["acceptance_status"]
    assert vehicle_status["status"] == "ready"
    assert not vehicle_status["missing"]
    assert vehicle_status["acceptance_checks"][0]["status"] == "linked"
    assert vehicle_status["acceptance_checks"][0]["resolved_by"] == "scenario_alias"
    assert vehicle_status["acceptance_checks"][0]["matched"] == "illegal_parking_camera_ingest"
    resource_status = catalog["objects_by_id"]["vehicles"]["resource_binding_status"]
    assert resource_status["overall_status"] == "ready"
    assert resource_status["linked_count"] >= 4
    assert catalog["binding_readiness_summary"]["overall_status"] == "ready"
    assert catalog["resource_catalog_summary"]["vision_model_count"] >= 1


def test_managed_object_resource_binding_reports_unregistered_resources() -> None:
    catalog = managed_object_catalog_from_site_profile(
        {
            "managed_objects": {
                "custom_camera_gate": {
                    "display_name": "Custom camera gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": ["tenant-only-gate-model"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": [
                            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
                        ],
                    },
                }
            }
        }
    )

    resource_status = catalog["objects_by_id"]["custom_camera_gate"]["resource_binding_status"]
    assert resource_status["overall_status"] == "manual_check"
    assert resource_status["manual_check_count"] == 1
    assert catalog["binding_readiness_summary"]["overall_status"] == "manual_check"
    assert catalog["binding_readiness_summary"]["unregistered_resource_count"] == 1
    assert catalog["binding_readiness_summary"]["unregistered_resources"] == [
        {
            "object_id": "custom_camera_gate",
            "resource_type": "vision_models",
            "resource_id": "tenant-only-gate-model",
        }
    ]


def test_profile_delivery_resources_register_project_specific_bindings() -> None:
    catalog = managed_object_catalog_from_site_profile(
        {
            "delivery_resources": {
                "vision_models": {
                    "tenant-only-gate-model": {
                        "display_name": "Tenant gate detector",
                        "version": "v1.0.0",
                        "owner": "delivery.team",
                        "source": "project",
                        "description": "Customer-specific gate model.",
                    }
                }
            },
            "managed_objects": {
                "custom_camera_gate": {
                    "display_name": "Custom camera gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": ["tenant-only-gate-model"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": [
                            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
                        ],
                    },
                }
            },
        }
    )

    resource_status = catalog["objects_by_id"]["custom_camera_gate"]["resource_binding_status"]
    assert resource_status["overall_status"] == "ready"
    assert catalog["binding_readiness_summary"]["overall_status"] == "ready"
    assert catalog["binding_readiness_summary"]["unregistered_resource_count"] == 0
    checks = resource_status["checks"]
    custom_model = next(
        item
        for item in checks
        if item["resource_type"] == "vision_models"
        and item["resource_id"] == "tenant-only-gate-model"
    )
    assert custom_model["status"] == "linked"
    assert custom_model["display_name"] == "Tenant gate detector"
    assert custom_model["version"] == "v1.0.0"
    assert custom_model["source"] == "project"


def test_shared_delivery_resource_registry_resolves_object_bindings(tmp_path: Path) -> None:
    resource_root = tmp_path / "delivery-resources"
    result = upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v2.1.0",
            "owner": "solution.resource.team",
            "description": "Reusable gate detector for solution projects.",
        },
        operator_id="delivery.operator",
        reason="register shared model for customer bindings",
    )

    assert result["accepted"] is True
    assert result["created"] is True
    registry = list_delivery_resource_registry(resource_root)
    assert registry["summary"]["resource_count"] == 1
    assert registry["resources"][0]["resource_id"] == "shared-gate-model"
    assert registry["resources"][0]["source"] == "shared_registry"

    catalog = managed_object_catalog_from_site_profile(
        {
            "managed_objects": {
                "custom_camera_gate": {
                    "display_name": "Custom camera gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": ["shared-gate-model"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": [
                            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
                        ],
                    },
                }
            },
        },
        delivery_resource_root=resource_root,
    )

    resource_status = catalog["objects_by_id"]["custom_camera_gate"]["resource_binding_status"]
    assert resource_status["overall_status"] == "ready"
    assert catalog["binding_readiness_summary"]["overall_status"] == "ready"
    shared_model = next(
        item
        for item in resource_status["checks"]
        if item["resource_type"] == "vision_models" and item["resource_id"] == "shared-gate-model"
    )
    assert shared_model["status"] == "linked"
    assert shared_model["display_name"] == "Shared gate detector"
    assert shared_model["version"] == "v2.1.0"
    assert shared_model["source"] == "shared_registry"


def test_default_delivery_resource_registry_seed_covers_customer_projects() -> None:
    resource_root = Path("deploy/delivery-resources")
    registry = list_delivery_resource_registry(resource_root)

    assert registry["found"] is True
    assert registry["summary"]["resource_count"] >= 35
    assert registry["summary"]["unregistered_resource_count"] == 0
    assert (
        registry["delivery_resources"]["vision_models"]["vehicle-detection"]["publish_status"]
        == "published"
    )
    assert (
        registry["delivery_resources"]["sensor_protocols"]["voice-turn-json"]["owner"]
        == "voice.platform"
    )
    assert (
        registry["delivery_resources"]["skill_packages"]["capability.answer_wayfinding"][
            "publish_status"
        ]
        == "published"
    )
    assert (
        registry["delivery_resources"]["acceptance_tests"][
            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
        ]["publish_status"]
        == "published"
    )

    catalog = build_customer_project_resource_catalog(
        Path("deploy/site-profiles"),
        template_root=Path("deploy/customer-project-templates"),
        delivery_resource_root=resource_root,
    )

    assert catalog["summary"]["unregistered_resource_count"] == 0
    assert catalog["summary"]["used_resource_count"] >= 20
    vehicle = next(
        item
        for item in catalog["resources"]
        if item["resource_type"] == "vision_models" and item["resource_id"] == "vehicle-detection"
    )
    assert vehicle["source"] == "shared_registry"
    assert vehicle["publish_status"] == "published"
    wayfinding = next(
        item
        for item in catalog["resources"]
        if item["resource_type"] == "skill_packages"
        and item["resource_id"] == "capability.answer_wayfinding"
    )
    assert wayfinding["consumer_count"] >= 2
    assert wayfinding["source"] == "shared_registry"


def test_delivery_resource_registry_history_disable_and_rollback(tmp_path: Path) -> None:
    resource_root = tmp_path / "delivery-resources"
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v1.0.0",
            "owner": "solution.resource.team",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="initial registration",
    )
    updated = upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v2.0.0",
            "owner": "solution.resource.team",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="model version update",
    )
    assert updated["accepted"] is True
    assert updated["revision"]["revision_id"]

    history = list_delivery_resource_revisions(resource_root)
    assert history["revision_count"] == 1
    revision_id = history["revisions"][0]["revision_id"]

    disabled = disable_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        operator_id="delivery.operator",
        reason="bad field accuracy",
    )
    assert disabled["accepted"] is True
    assert disabled["resource"]["publish_status"] == "disabled"

    catalog = managed_object_catalog_from_site_profile(
        {
            "managed_objects": {
                "custom_camera_gate": {
                    "display_name": "Custom camera gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": ["shared-gate-model"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": [
                            "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
                        ],
                    },
                }
            },
        },
        delivery_resource_root=resource_root,
    )
    resource_status = catalog["objects_by_id"]["custom_camera_gate"]["resource_binding_status"]
    assert resource_status["overall_status"] == "blocked"
    shared_model = next(
        item
        for item in resource_status["checks"]
        if item["resource_type"] == "vision_models" and item["resource_id"] == "shared-gate-model"
    )
    assert shared_model["status"] == "blocked"
    assert shared_model["publish_status"] == "disabled"

    preview = rollback_delivery_resource_registry(
        resource_root,
        revision_id,
        operator_id="delivery.operator",
        reason="preview rollback",
        dry_run=True,
    )
    assert preview["accepted"] is True
    assert preview["dry_run"] is True
    assert preview["would_write"] is True
    assert preview["target_summary"]["resource_count"] == 1

    rollback = rollback_delivery_resource_registry(
        resource_root,
        revision_id,
        operator_id="delivery.operator",
        reason="restore previous stable model",
    )
    assert rollback["accepted"] is True
    restored = list_delivery_resource_registry(resource_root)
    restored_model = restored["delivery_resources"]["vision_models"]["shared-gate-model"]
    assert restored_model["version"] == "v1.0.0"
    assert restored_model["publish_status"] == "published"


def test_delivery_resource_governance_request_requires_second_approver_before_disable(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "delivery-resources"
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "vehicle-detection",
        {
            "display_name": "Vehicle detector",
            "version": "v1.0.0",
            "owner": "solution.resource.team",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="initial registration",
    )

    request = create_delivery_resource_governance_request(
        resource_root,
        "disable_resource",
        {
            "resource_type": "vision_models",
            "resource_id": "vehicle-detection",
        },
        operator_id="delivery.operator",
        reason="bad field accuracy",
        sla_target_s=60,
    )

    assert request["accepted"] is True
    assert request["request"]["status"] == "pending"
    assert request["request"]["action"] == "disable_resource"
    assert request["request"]["operation"]["resource_type"] == "vision_models"
    assert request["request"]["operation"]["resource_id"] == "vehicle-detection"
    assert request["request"]["sla_target_s"] == 60
    assert request["request"]["due_at"] == pytest.approx(request["request"]["requested_at"] + 60)
    assert request["request"]["review_sla"]["state"] in {"active", "due_soon"}
    assert request["request"]["review_sla"]["escalation_required"] is False
    assert request["preview"]["accepted"] is True
    assert request["preview"]["dry_run"] is True
    assert request["preview"]["target_publish_status"] == "disabled"
    assert request["preview"]["impact"]["analysis_status"] == "complete"
    assert request["preview"]["impact"]["affected_consumer_count"] >= 1
    assert request["preview"]["impact"]["affected_customer_project_count"] >= 1
    assert request["preview"]["impact"]["affected_object_count"] >= 1
    assert request["preview"]["impact"]["affected_template_count"] >= 1
    assert request["preview"]["impact"]["truncated"] is False
    assert any(
        item["project_id"] == "demo-field-ops"
        for item in request["preview"]["impact"]["affected_projects"]
    )
    assert any(
        item["template_id"] == "park-visitor-service"
        for item in request["preview"]["impact"]["affected_templates"]
    )
    assert any(
        item["project_id"] == "demo-field-ops"
        and item["object_id"] == "vehicles"
        and item["resource_id"] == "vehicle-detection"
        for item in request["preview"]["impact"]["affected_consumers"]
    )
    still_published = list_delivery_resource_registry(resource_root)
    assert (
        still_published["delivery_resources"]["vision_models"]["vehicle-detection"][
            "publish_status"
        ]
        == "published"
    )

    pending = list_delivery_resource_governance_requests(resource_root, status="pending")
    assert pending["summary"]["pending_count"] == 1
    assert pending["summary"]["overdue_count"] == 0
    overdue = list_delivery_resource_governance_requests(
        resource_root,
        status="pending",
        overdue_only=True,
        now=request["request"]["requested_at"] + 61,
    )
    assert overdue["overdue_only"] is True
    assert overdue["request_count"] == 1
    assert overdue["summary"]["overdue_count"] == 1
    assert overdue["requests"][0]["review_sla"]["state"] == "overdue"
    assert overdue["requests"][0]["review_sla"]["escalation_required"] is True
    escalation = escalate_overdue_delivery_resource_governance_requests(
        resource_root,
        operator_id="delivery.reviewer",
        reason="approval SLA missed",
        now=request["request"]["requested_at"] + 61,
    )
    assert escalation["accepted"] is True
    assert escalation["checked_count"] == 1
    assert escalation["escalated_count"] == 1
    assert escalation["escalations"][0]["request_id"] == request["request"]["request_id"]
    assert escalation["escalations"][0]["notification"]["channel"] == "delivery_owner_queue"
    assert escalation["requests"][0]["escalation_count"] == 1
    assert escalation["requests"][0]["last_escalation"]["status"] == "queued"
    callback_root = tmp_path / "callback-delivery"
    upsert_delivery_resource(
        callback_root,
        "vision_models",
        "vehicle-detection",
        {
            "display_name": "Vehicle detector",
            "version": "v1.0.0",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="initial registration",
    )
    callback_request = create_delivery_resource_governance_request(
        callback_root,
        action="disable_resource",
        operation={
            "resource_type": "vision_models",
            "resource_id": "vehicle-detection",
        },
        operator_id="delivery.operator",
        reason="bad onsite accuracy",
        sla_target_s=60,
    )
    delivered_request_path = Path(callback_request["request"]["request_path"])
    delivered_payload = json.loads(delivered_request_path.read_text(encoding="utf-8"))
    delivered_payload["requested_at"] = callback_request["request"]["requested_at"]
    delivered_payload["due_at"] = callback_request["request"]["requested_at"] - 1
    delivered_request_path.write_text(
        json.dumps(delivered_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    captured_escalations: list[dict[str, Any]] = []

    def fake_delivery(escalation_record: dict[str, Any]) -> dict[str, Any]:
        captured_escalations.append(escalation_record)
        return {
            "status": "sent",
            "delivery_mode": "configured_channels",
            "sent_channels": ["log"],
            "delivery_report": [
                {
                    "channel": "log",
                    "status": "sent",
                    "reason": "",
                }
            ],
        }

    delivered = escalate_overdue_delivery_resource_governance_requests(
        callback_root,
        operator_id="delivery.reviewer",
        reason="approval SLA missed",
        now=callback_request["request"]["requested_at"] + 61,
        notification_delivery=fake_delivery,
    )
    assert captured_escalations
    assert delivered["escalations"][0]["status"] == "sent"
    assert delivered["escalations"][0]["notification"]["delivery_mode"] == ("configured_channels")
    assert delivered["escalations"][0]["notification"]["sent_channels"] == ["log"]
    assert delivered["escalations"][0]["delivery_report"][0]["channel"] == "log"
    duplicate_escalation = escalate_overdue_delivery_resource_governance_requests(
        resource_root,
        operator_id="delivery.reviewer",
        reason="second scan",
        now=request["request"]["requested_at"] + 62,
    )
    assert duplicate_escalation["escalated_count"] == 0
    assert duplicate_escalation["skipped_count"] == 1
    assert duplicate_escalation["skipped"][0]["reason"] == (
        "resource_governance_request_already_escalated"
    )
    request_id = request["request"]["request_id"]

    self_review = review_delivery_resource_governance_request(
        resource_root,
        request_id,
        decision="approve",
        operator_id="delivery.operator",
        reason="self approve",
    )
    assert self_review["accepted"] is False
    assert self_review["reason"] == "resource_governance_request_requires_second_approver"
    assert self_review["request"]["escalation_count"] == 1

    approved = review_delivery_resource_governance_request(
        resource_root,
        request_id,
        decision="approve",
        operator_id="delivery.reviewer",
        reason="confirm bad field accuracy",
    )
    assert approved["accepted"] is True
    assert approved["request"]["status"] == "approved"
    assert approved["request"]["reviewed_by"] == "delivery.reviewer"
    assert approved["request"]["review_sla"]["state"] == "closed"
    assert approved["apply_result"]["accepted"] is True
    disabled = list_delivery_resource_registry(resource_root)
    assert (
        disabled["delivery_resources"]["vision_models"]["vehicle-detection"]["publish_status"]
        == "disabled"
    )

    second_review = review_delivery_resource_governance_request(
        resource_root,
        request_id,
        decision="reject",
        operator_id="delivery.reviewer",
        reason="duplicate",
    )
    assert second_review["accepted"] is False
    assert second_review["reason"] == "resource_governance_request_not_pending"
    closed_overdue = list_delivery_resource_governance_requests(
        resource_root,
        overdue_only=True,
        now=request["request"]["requested_at"] + 120,
    )
    assert closed_overdue["request_count"] == 0


def test_delivery_resource_governance_request_approves_registry_rollback(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "delivery-resources"
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v1.0.0",
            "owner": "solution.resource.team",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="initial registration",
    )
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v2.0.0",
            "owner": "solution.resource.team",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="model version update",
    )
    revision_id = list_delivery_resource_revisions(resource_root)["revisions"][0]["revision_id"]

    request = create_delivery_resource_governance_request(
        resource_root,
        "rollback_registry",
        {"revision_id": revision_id},
        operator_id="delivery.operator",
        reason="restore stable model",
    )
    assert request["accepted"] is True
    assert request["request"]["status"] == "pending"
    assert request["preview"]["accepted"] is True
    assert request["preview"]["dry_run"] is True
    assert request["preview"]["would_write"] is True
    current = list_delivery_resource_registry(resource_root)
    assert (
        current["delivery_resources"]["vision_models"]["shared-gate-model"]["version"] == "v2.0.0"
    )

    approved = review_delivery_resource_governance_request(
        resource_root,
        request["request"]["request_id"],
        decision="approve",
        operator_id="delivery.reviewer",
        reason="approve rollback",
    )

    assert approved["accepted"] is True
    assert approved["request"]["status"] == "approved"
    assert approved["apply_result"]["accepted"] is True
    restored = list_delivery_resource_registry(resource_root)
    assert (
        restored["delivery_resources"]["vision_models"]["shared-gate-model"]["version"] == "v1.0.0"
    )
    approved_requests = list_delivery_resource_governance_requests(resource_root, status="approved")
    assert approved_requests["summary"]["approved_count"] == 1


def test_delivery_resource_governance_review_rejects_registry_drift(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "delivery-resources"
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "vehicle-detection",
        {
            "display_name": "Vehicle detector",
            "version": "v1.0.0",
            "publish_status": "published",
        },
        operator_id="delivery.operator",
        reason="initial registration",
    )
    request = create_delivery_resource_governance_request(
        resource_root,
        "disable_resource",
        {
            "resource_type": "vision_models",
            "resource_id": "vehicle-detection",
        },
        operator_id="delivery.operator",
        reason="bad field accuracy",
    )
    assert request["accepted"] is True

    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "vehicle-detection",
        {
            "display_name": "Vehicle detector",
            "version": "v2.0.0",
            "publish_status": "published",
        },
        operator_id="resource.owner",
        reason="new model published before review",
    )
    reviewed = review_delivery_resource_governance_request(
        resource_root,
        request["request"]["request_id"],
        decision="approve",
        operator_id="delivery.reviewer",
        reason="approve stale request",
    )

    assert reviewed["accepted"] is False
    assert reviewed["reason"] == "resource_governance_registry_changed_since_request"
    assert reviewed["request"]["status"] == "pending"
    assert reviewed["request_registry_sha256"] == request["request"]["current_registry_sha256"]
    assert reviewed["current_registry_sha256"] != request["request"]["current_registry_sha256"]
    registry = list_delivery_resource_registry(resource_root)
    current = registry["delivery_resources"]["vision_models"]["vehicle-detection"]
    assert current["version"] == "v2.0.0"
    assert current["publish_status"] == "published"


def test_disabled_acceptance_resource_blocks_binding_readiness(tmp_path: Path) -> None:
    resource_root = tmp_path / "delivery-resources"
    acceptance_ref = "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
    registered = upsert_delivery_resource(
        resource_root,
        "acceptance_tests",
        acceptance_ref,
        {
            "display_name": "Illegal parking acceptance",
            "version": "v1.0.0",
            "publish_status": "disabled",
        },
        operator_id="delivery.operator",
        reason="acceptance test temporarily disabled",
    )
    assert registered["accepted"] is True

    catalog = managed_object_catalog_from_site_profile(
        {
            "managed_objects": {
                "parking_lane": {
                    "display_name": "Parking lane",
                    "category": "traffic",
                    "bindings": {
                        "vision_models": ["vehicle-detection"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.detect_illegal_parking"],
                        "acceptance_tests": [acceptance_ref],
                    },
                }
            },
        },
        delivery_resource_root=resource_root,
    )
    resource_status = catalog["objects_by_id"]["parking_lane"]["resource_binding_status"]
    assert resource_status["overall_status"] == "blocked"
    acceptance_check = next(
        item for item in resource_status["checks"] if item["resource_type"] == "acceptance_tests"
    )
    assert acceptance_check["status"] == "blocked"
    assert acceptance_check["publish_status"] == "disabled"


def test_managed_object_acceptance_gate_blocks_missing_test_file() -> None:
    catalog = managed_object_catalog_from_site_profile(
        {
            "managed_objects": {
                "bad_gate": {
                    "display_name": "Bad gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": ["gate-detection"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": ["tests/nope.py::missing_gate"],
                    },
                }
            }
        }
    )

    status = catalog["objects_by_id"]["bad_gate"]["acceptance_status"]
    assert status["status"] == "blocked"
    assert status["acceptance_checks"][0]["status"] == "file_missing"
    assert catalog["acceptance_summary"]["overall_status"] == "blocked"


def test_customer_project_acceptance_registry_resolves_project_and_template_references() -> None:
    registry = build_customer_project_acceptance_registry(
        Path("deploy/site-profiles"),
        template_root=Path("deploy/customer-project-templates"),
    )

    assert registry["summary"]["reference_count"] >= 4
    assert registry["summary"]["consumer_count"] >= registry["summary"]["reference_count"]
    assert registry["summary"]["linked_count"] >= 1
    assert registry["references"]
    assert registry["consumers"]
    vehicle_consumer = next(
        item
        for item in registry["consumers"]
        if item.get("project_id") == "demo-field-ops" and item.get("object_id") == "vehicles"
    )
    assert vehicle_consumer["status"] == "linked"
    assert vehicle_consumer["matched"] == "illegal_parking_camera_ingest"
    assert any(item["scope_type"] == "template" for item in registry["consumers"])
    assert registry["next_step"]


def test_customer_project_resource_catalog_resolves_object_bindings() -> None:
    registry = build_customer_project_resource_catalog(
        Path("deploy/site-profiles"),
        template_root=Path("deploy/customer-project-templates"),
    )

    assert registry["summary"]["resource_count"] >= 10
    assert registry["summary"]["used_resource_count"] >= 4
    assert registry["summary"]["consumer_count"] >= registry["summary"]["used_resource_count"]
    assert registry["summary"]["unregistered_resource_count"] == 0
    assert registry["summary"]["overall_status"] == "ready"
    assert any(
        item["resource_type"] == "vision_models"
        and item["resource_id"] == "vehicle-detection"
        and item["consumer_count"] >= 1
        for item in registry["resources"]
    )
    assert any(
        item["resource_type"] == "skill_packages"
        and item["resource_id"] == "capability.answer_wayfinding"
        for item in registry["resources"]
    )
    vehicle_consumer = next(
        item
        for item in registry["consumers"]
        if item.get("project_id") == "demo-field-ops"
        and item.get("object_id") == "vehicles"
        and item.get("resource_id") == "vehicle-detection"
    )
    assert vehicle_consumer["status"] == "linked"


def test_customer_project_resource_catalog_uses_custom_delivery_resource_root(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "delivery-resources"
    upsert_delivery_resource(
        resource_root,
        "vision_models",
        "shared-gate-model",
        {
            "display_name": "Shared gate detector",
            "version": "v2.1.0",
            "owner": "solution.resource.team",
            "description": "Reusable gate detector for customer project bindings.",
        },
        operator_id="delivery.operator",
        reason="register shared model for customer project catalog",
    )
    profile_root = tmp_path / "site-profiles"
    profile_root.mkdir(parents=True)
    profile = profile_root / "custom-site.yaml"
    profile.write_text(
        json.dumps(
            {
                "site": {
                    "site_id": "custom-site",
                    "name": "Custom Site",
                },
                "customer": {
                    "customer_id": "custom-customer",
                    "customer_name": "Custom Customer",
                    "project_id": "custom-project",
                    "project_name": "Custom Project",
                },
                "managed_objects": {
                    "custom_camera_gate": {
                        "display_name": "Custom camera gate",
                        "category": "access",
                        "bindings": {
                            "vision_models": ["shared-gate-model"],
                            "sensor_protocols": ["camera-detection-json"],
                            "skill_packages": ["capability.detect_illegal_parking"],
                            "acceptance_tests": [
                                "tests/scenario_tests/test_field_operations_evaluation.py::illegal_parking"
                            ],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = build_customer_project_resource_catalog(
        profile_root,
        template_root=None,
        delivery_resource_root=resource_root,
    )

    assert catalog["summary"]["unregistered_resource_count"] == 0
    consumer = next(
        item
        for item in catalog["consumers"]
        if item["object_id"] == "custom_camera_gate" and item["resource_type"] == "vision_models"
    )
    assert consumer["status"] == "linked"
    assert consumer["source"] == "shared_registry"
    resource = next(
        item
        for item in catalog["resources"]
        if item["resource_type"] == "vision_models" and item["resource_id"] == "shared-gate-model"
    )
    assert resource["source"] == "shared_registry"
    assert resource["consumer_count"] == 1


def test_customer_project_execution_bindings_expose_ingest_and_runtime_plan() -> None:
    plan = build_customer_project_execution_bindings(
        Path("deploy/site-profiles"),
        "demo-field-ops",
    )

    assert plan["found"] is True
    assert plan["summary"]["object_count"] >= 4
    assert plan["summary"]["overall_status"] in {"ready", "manual_check"}
    vehicles = plan["plans_by_object_id"]["vehicles"]
    assert vehicles["overall_status"] == "ready"
    assert vehicles["scope_constraints"] == {
        "tenant_ids": [],
        "delivery_namespaces": [],
        "customer_ids": [],
        "project_ids": [],
        "site_ids": [],
    }
    assert vehicles["ingest_contract"]["endpoint"] == "/api/field/ingest"
    assert vehicles["ingest_contract"]["sample_payload"]["managed_object_id"] == "vehicles"
    assert vehicles["runtime_contract"]["callback_endpoint"] == (
        "/api/field/events/{event_id}/runtime-delivery"
    )
    assert vehicles["input_adapters"][0]["adapter"] == "camera_detection_json"
    adapter_contract = vehicles["input_adapters"][0]["adapter_contract"]
    assert adapter_contract["normalizer"] == (
        "askme.pipeline.field.field_ingest_adapters.normalize_field_ingest_payload"
    )
    assert adapter_contract["bridge"] == "field-ingest-bridge"
    assert adapter_contract["ingest_endpoint"] == "/api/field/ingest"
    assert adapter_contract["device_signature_required"] is True
    assert "ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET" in adapter_contract["device_secret_envs"]
    assert "--dry-run" in adapter_contract["dry_run_command"]
    assert "--watch" in adapter_contract["live_command"]
    assert adapter_contract["sample_fixture"].endswith("site-a-device-events.jsonl")
    assert "summary.accepted" in adapter_contract["verification_outputs"]
    assert vehicles["bridge_contract"]["live_post_required_for_customer_signoff"] is True
    assert vehicles["skill_routes"][0]["resource_id"] == "capability.detect_illegal_parking"
    assert vehicles["skill_routes"][0]["capability"] == "detect_illegal_parking"
    assert vehicles["skill_routes"][0]["installed_contract"] is True
    assert vehicles["skill_routes"][0]["safety_level"] == "dangerous"
    assert vehicles["skill_routes"][0]["confirm_before_execute"] is True
    assert vehicles["skill_routes"][0]["approval_policy"] == "supervisor_required"
    assert vehicles["skill_routes"][0]["tool"] == "field_event_trigger"
    assert vehicles["skill_routes"][0]["output_contract"] == "field_event"
    assert "image_path" in vehicles["skill_routes"][0]["required_inputs"]
    assert "runtime arbiter" in vehicles["skill_routes"][0]["hardware_boundary"]
    assert vehicles["acceptance_tests"][0]["status"] == "ready"
    assert vehicles["source_plans"][0]["device_count"] >= 1
    visitors = plan["plans_by_object_id"]["visitors"]
    assert visitors["overall_status"] in {"ready", "manual_check"}
    assert any(item["protocol_id"] == "voice-turn-json" for item in visitors["input_adapters"])
    assert plan["customer_claim"]
    assert plan["next_step"]


def test_customer_project_acceptance_report_summarizes_delivery_gates(tmp_path: Path) -> None:
    archive = tmp_path / "field-events.jsonl"
    scenario = tmp_path / "scenario.json"
    ingest = tmp_path / "ingest.json"
    voice = tmp_path / "voice.json"
    notification = tmp_path / "notification.json"
    runtime = tmp_path / "runtime.json"
    archive.write_text(
        json.dumps(
            {
                "event_id": "field-1",
                "scenario_id": "illegal_parking",
                "payload": {"source": "camera"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scenario.write_text(
        json.dumps({"status": "passed", "scenario_count": 1, "passed": 1, "failed": 0}),
        encoding="utf-8",
    )
    ingest.write_text(
        json.dumps({"status": "passed", "event_count": 1, "local_server": True}), encoding="utf-8"
    )
    voice.write_text(
        json.dumps(
            {
                "status": "passed",
                "local_server": True,
                "live_tts": False,
                "voice_delivery": {"status": "queued"},
                "voice_directive": {"resolved_profile": "emergency_short"},
            }
        ),
        encoding="utf-8",
    )
    notification.write_text(
        json.dumps(
            {
                "status": "passed",
                "local_server": True,
                "external_services": False,
                "collector_request_count": 1,
            }
        ),
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            {
                "ok": True,
                "mode": "local_server",
                "receipt_count": 1,
                "callback_status_codes": [200],
                "runtime_statuses": ["created", "shadowed"],
                "final_runtime_delivery": {
                    "status": "shadowed",
                    "runtime_callback_trust": {"trusted": True, "status": "trusted"},
                },
            }
        ),
        encoding="utf-8",
    )
    report = customer_project_acceptance_report(
        Path("deploy/site-profiles"),
        "demo-field-ops",
        check_env=False,
        field_evidence_config={
            "archive_path": str(archive),
            "scenario_report_path": str(scenario),
            "smoke_report_path": str(ingest),
            "voice_smoke_report_path": str(voice),
            "notification_smoke_report_path": str(notification),
            "runtime_roundtrip_report_path": str(runtime),
        },
    )

    assert report["found"] is True
    assert report["overall_status"] == "manual_check"
    assert report["acceptance_summary"]["overall_status"] == "ready"
    assert len(report["env_missing"]) > 0
    gates_by_id = {gate["gate_id"]: gate for gate in report["gates"]}
    assert gates_by_id["managed_object_execution_bindings"]["status"] == "ready"
    assert "具备可执行接入计划" in gates_by_id["managed_object_execution_bindings"]["evidence"]
    assert gates_by_id["deployment_credentials"]["status"] == "manual_check"
    assert gates_by_id["onsite_acceptance_boundary"]["status"] == "manual_check"
    assert gates_by_id["field_readiness"]["status"] == "manual_check"
    assert gates_by_id["field_smoke_evidence"]["status"] == "manual_check"
    assert gates_by_id["voice_notification_evidence"]["status"] == "manual_check"
    assert gates_by_id["runtime_audit_trust"]["status"] == "manual_check"
    assert gates_by_id["field_device_onboarding"]["status"] == "manual_check"
    assert {gate["gate_id"] for gate in report["gates"]} == {
        "site_profile",
        "managed_object_acceptance",
        "managed_object_execution_bindings",
        "deployment_credentials",
        "onsite_acceptance_boundary",
        "field_readiness",
        "field_smoke_evidence",
        "voice_notification_evidence",
        "runtime_audit_trust",
        "field_device_onboarding",
    }
    assert report["execution_bindings"]["summary"]["overall_status"] == "ready"
    assert report["execution_bindings"]["customer_claim"]
    vehicle_contract = next(
        item
        for item in report["execution_bindings"]["object_contracts"]
        if item["object_id"] == "vehicles"
    )
    assert vehicle_contract["input_adapters"][0]["bridge"] == "field-ingest-bridge"
    assert vehicle_contract["input_adapters"][0]["device_signature_required"] is True
    checklist = report["site_acceptance_checklist"]
    assert checklist["overall_status"] in {"blocked", "manual_check"}
    checklist_by_id = {item["item_id"]: item for item in checklist["items"]}
    assert checklist_by_id["device_ingest"]["status"] in {"blocked", "manual_check"}
    assert checklist_by_id["voice_playback"]["status"] in {"blocked", "manual_check"}
    assert checklist_by_id["notification_delivery"]["status"] in {"blocked", "manual_check"}
    assert checklist_by_id["runtime_roundtrip"]["status"] in {"blocked", "manual_check"}
    assert report["field_readiness"]["status"] == "ready_for_lab"
    assert report["field_readiness"]["reports"]["voice_smoke"]["voice_profile"] == "emergency_short"
    assert report["field_readiness"]["device_onboarding"]["ready"] == 0
    launch = report["launch_readiness"]
    assert launch["readiness_type"] == "askme.customer_project_launch_readiness.v1"
    assert launch["overall_status"] == "blocked"
    assert launch["launch_stage"] == "demo_or_integration_only"
    assert launch["production_ready"] is False
    assert "不能声明客户可上线" in launch["release_claim"]
    assert {gate["gate_id"] for gate in launch["gates"]} == {
        "project_acceptance_report",
        "managed_object_execution_bindings",
        "deployment_credentials",
        "onsite_required_evidence",
        "field_real_link",
        "field_device_onboarding",
        "site_acceptance_checklist",
    }
    onsite_summary = report["onsite_acceptance_evidence"]["summary"]
    assert onsite_summary["overall_status"] == "manual_check"
    assert onsite_summary["missing_required_types"] == [
        "device_ingest",
        "voice_playback",
        "notification_delivery",
        "runtime_roundtrip",
    ]
    assert not [
        receipt
        for receipt in report["onsite_acceptance_evidence"]["receipts"]
        if receipt.get("source") == "field_readiness_auto_backfill"
    ]
    workflow = report["delivery_workflow"]
    assert workflow["steps"]
    assert workflow["overall_status"] in {"ready", "manual_check", "blocked"}
    assert {step["step_id"] for step in workflow["steps"]} == {
        "customer_scope",
        "managed_objects",
        "runtime_bindings",
        "site_map_devices",
        "responder_credentials",
        "acceptance_evidence",
        "handoff_package",
    }
    assert "生产上线仍需要设备、通知、语音和机器人运行的独立现场证据" in report["release_claim"]


def test_acceptance_report_auto_backfills_required_onsite_evidence_from_real_link_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_demo_device_secrets(monkeypatch)
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_root = _artifact_test_root(tmp_path, "auto-real")
    try:
        evidence_config = _write_real_link_acceptance_reports(evidence_root)

        report = customer_project_acceptance_report(
            profile_root,
            "demo-field-ops",
            check_env=False,
            field_evidence_config=evidence_config,
        )

        assert report["found"] is True
        onsite = report["onsite_acceptance_evidence"]
        onsite_summary = onsite["summary"]
        assert onsite_summary["overall_status"] == "ready"
        assert onsite_summary["passed_required_count"] == 4
        assert onsite_summary["missing_required_types"] == []
        receipts_by_type = {
            receipt["evidence_type"]: receipt
            for receipt in onsite["receipts"]
            if receipt.get("source") == "field_readiness_auto_backfill"
        }
        assert set(receipts_by_type) == {
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        }
        assert receipts_by_type["device_ingest"]["path"].endswith("field-events.jsonl")
        assert receipts_by_type["voice_playback"]["sha256"]
        assert receipts_by_type["notification_delivery"]["exists"] is True
        assert (
            receipts_by_type["runtime_roundtrip"]["external_reference"] == "final_status=shadowed"
        )
        assert report["field_readiness"]["device_onboarding"]["ready"] == 1
        assert report["field_readiness"]["device_onboarding"]["manual_check"] == 3
        assert report["field_readiness"]["device_onboarding"]["blocked"] == 0
        gates_by_id = {gate["gate_id"]: gate for gate in report["gates"]}
        assert gates_by_id["onsite_acceptance_boundary"]["status"] == "ready"
        assert gates_by_id["field_device_onboarding"]["status"] == "manual_check"
        assert (
            "4/4 个必需现场证据回执已通过" in gates_by_id["onsite_acceptance_boundary"]["evidence"]
        )
        checklist = report["site_acceptance_checklist"]
        checklist_by_id = {item["item_id"]: item for item in checklist["items"]}
        assert checklist_by_id["device_ingest"]["status"] == "ready"
        assert checklist_by_id["voice_playback"]["status"] == "ready"
        assert checklist_by_id["notification_delivery"]["status"] == "ready"
        assert checklist_by_id["runtime_roundtrip"]["status"] == "ready"
        assert checklist_by_id["voice_playback"]["source"] == "field_readiness_auto_backfill"

        dossier_result = export_customer_project_acceptance_dossier(
            profile_root,
            "demo-field-ops",
            output_root=tmp_path / "dossiers",
            check_env=False,
            field_evidence_config=evidence_config,
        )
        assert dossier_result["accepted"] is True
        assert dossier_result["dossier"]["manifest"]["onsite_evidence_status"] == "ready"
        assert dossier_result["dossier"]["manifest"]["onsite_required_evidence_ready"] is True
        assert dossier_result["dossier"]["launch_readiness"]["overall_status"] == "manual_check"
        assert dossier_result["dossier"]["manifest"]["launch_readiness_status"] == "manual_check"
        assert dossier_result["dossier"]["manifest"]["launch_stage"] == "pilot_or_site_trial"
        launch_gates = {
            gate["gate_id"]: gate for gate in dossier_result["dossier"]["launch_readiness"]["gates"]
        }
        assert launch_gates["field_device_onboarding"]["status"] == "manual_check"
        assert "ready=1" in launch_gates["field_device_onboarding"]["evidence"]
        assert dossier_result["dossier"]["site_acceptance_checklist"]["items"]
        listing = list_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            check_env=False,
            field_evidence_config=evidence_config,
        )
        assert listing["readiness_auto_included"] is True
        assert listing["onsite_acceptance_evidence"]["summary"]["overall_status"] == "ready"
        assert {
            receipt["evidence_type"]
            for receipt in listing["onsite_acceptance_evidence"]["receipts"]
            if receipt.get("source") == "field_readiness_auto_backfill"
        } == {
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        }
    finally:
        shutil.rmtree(evidence_root, ignore_errors=True)


def test_acceptance_report_auto_backfill_is_read_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_demo_device_secrets(monkeypatch)
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    profile_file = profile_root / "park-demo.yaml"
    profile_before = profile_file.read_text(encoding="utf-8")
    evidence_root = _artifact_test_root(tmp_path, "auto-idempotent")
    try:
        evidence_config = _write_real_link_acceptance_reports(evidence_root)

        first = customer_project_acceptance_report(
            profile_root,
            "demo-field-ops",
            check_env=False,
            field_evidence_config=evidence_config,
        )
        second = customer_project_acceptance_report(
            profile_root,
            "demo-field-ops",
            check_env=False,
            field_evidence_config=evidence_config,
        )

        first_receipts = first["onsite_acceptance_evidence"]["receipts"]
        second_receipts = second["onsite_acceptance_evidence"]["receipts"]
        assert [item["receipt_id"] for item in first_receipts] == [
            item["receipt_id"] for item in second_receipts
        ]
        assert (
            first["onsite_acceptance_evidence"]["summary"]
            == second["onsite_acceptance_evidence"]["summary"]
        )
        assert profile_file.read_text(encoding="utf-8") == profile_before
        listing = list_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            check_env=False,
            field_evidence_config=evidence_config,
        )
        assert listing["onsite_acceptance_evidence"]["summary"]["overall_status"] == "ready"
        manual_listing = list_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            include_readiness_auto=False,
        )
        assert manual_listing["readiness_auto_included"] is False
        assert not [
            receipt
            for receipt in manual_listing["onsite_acceptance_evidence"]["receipts"]
            if receipt.get("source") == "field_readiness_auto_backfill"
        ]
    finally:
        shutil.rmtree(evidence_root, ignore_errors=True)


def _forged_onsite_evidence_payload(evidence_type: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "evidence_type": evidence_type,
        "status": "passed",
        "local_server": False,
    }
    if evidence_type == "device_ingest":
        payload["trusted_device_event_count"] = 1
    elif evidence_type == "voice_playback":
        payload["live_tts"] = True
    elif evidence_type == "notification_delivery":
        payload["external_services"] = True
        payload["collector_request_count"] = 1
    elif evidence_type == "runtime_roundtrip":
        payload["trusted_callbacks"] = True
        payload["final_status_verified"] = True
    return payload


def _ready_customer_signoff_closure(*, onsite_gate_eligible: bool = True) -> dict[str, Any]:
    dossier_sha = "d" * 64
    audit_sha = "b" * 64
    receipt_ids = {
        "device_ingest": "auto-device-ingest",
        "voice_playback": "auto-voice-playback",
        "notification_delivery": "auto-notification-delivery",
        "runtime_roundtrip": "auto-runtime-roundtrip",
    }
    receipts = [
        {
            "receipt_id": receipt_id,
            "evidence_type": evidence_type,
            "status": "passed",
            "source": "field_readiness_auto_backfill",
            "sha256": "c" * 64,
            "trust_status": "verified" if onsite_gate_eligible else "manual_check",
            "acceptance_gate_eligible": onsite_gate_eligible,
        }
        for evidence_type, receipt_id in receipt_ids.items()
    ]
    return {
        "found": True,
        "overall_status": "ready_for_customer_signoff",
        "customer_claim": "证据和内部复核已具备，可提交客户签收。",
        "next_step": "归档客户签收结果。",
        "gates": [
            {"gate_id": "customer_signoff", "status": "manual_check"},
        ],
        "acceptance_report": {
            "overall_status": "ready_for_onsite_acceptance",
            "customer_status": "可进入试点验收",
            "release_claim": "仅用于试点验收",
        },
        "onsite_acceptance_evidence": {
            "receipts": receipts,
            "summary": {
                "overall_status": "ready" if onsite_gate_eligible else "manual_check",
                "passed_required_count": len(receipts) if onsite_gate_eligible else 0,
                "latest_receipt_id": receipts[0]["receipt_id"],
            },
        },
        "site_acceptance_checklist": {
            "overall_status": "ready",
            "items": [],
        },
        "manual_review": {
            "latest": {"decision": "accepted"},
            "reviews": [],
            "review_count": 1,
        },
        "customer_signoff": {
            "latest": {},
            "signoffs": [],
            "signoff_count": 0,
            "base_ready_for_signoff": True,
        },
        "artifact_verification": {
            "acceptance_dossier": {
                "valid": True,
                "reason": "ok",
                "manifest": {"payload_sha256": dossier_sha},
            },
            "proposal_bundle": {
                "status": "ready",
                "evidence": "proposal valid",
                "proposal_path": "artifacts/customer-project-proposals/demo-proposal-bundle.json",
                "verification": {"payload_sha256": "e" * 64},
            },
            "audit_export": {
                "status": "ready",
                "evidence": "audit export valid",
                "manifest_path": "artifacts/audit_exports/demo.manifest.json",
                "sha256": audit_sha,
                "export_id": "audit-demo",
            },
        },
        "evidence_timeline": [],
    }


def _accepted_customer_signoff_payload(evidence_refs: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decision": "accepted",
        "signatory_name": "Fanmu Operator",
        "signatory_role": "Customer operations owner",
        "organization": "Fanmu Creative Park",
        "reason": "Customer accepts the pilot handoff.",
        "risk_acknowledgement": True,
        "credential_ref": "customer-signoff.pdf",
        "credential_sha256": "a" * 64,
    }
    if evidence_refs is not None:
        payload["evidence_refs"] = evidence_refs
    return payload


def _complete_customer_signoff_evidence_refs() -> list[str]:
    return [
        "onsite:auto-device-ingest",
        "onsite:auto-voice-playback",
        "onsite:auto-notification-delivery",
        "onsite:auto-runtime-roundtrip",
        f"acceptance_dossier:{'d' * 64}",
        "proposal_bundle",
        "audit_export:audit-demo",
    ]


def test_customer_project_onsite_evidence_updates_acceptance_dossier(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_root = Path("artifacts/test-onsite-evidence") / str(int(time.time() * 1000))
    evidence_root.mkdir(parents=True, exist_ok=True)
    try:
        for evidence_type in (
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        ):
            evidence_path = evidence_root / f"{evidence_type}.json"
            evidence_path.write_text(
                json.dumps(_forged_onsite_evidence_payload(evidence_type)),
                encoding="utf-8",
            )
            registered = register_customer_project_onsite_evidence(
                profile_root,
                "demo-field-ops",
                {
                    "evidence_type": evidence_type,
                    "status": "passed",
                    "path": str(evidence_path),
                    "summary": f"{evidence_type} passed onsite smoke.",
                    "source": "pytest",
                },
                operator_id="delivery.lead",
                reason="Onsite smoke passed.",
            )
            assert registered["accepted"] is True
            assert registered["receipt"]["sha256"]
            assert registered["receipt"]["evidence_tier"] == "acceptance_candidate"
            assert registered["receipt"]["production_eligible"] is False
            assert registered["receipt"]["trust_status"] == "manual_check"
            assert registered["receipt"]["acceptance_gate_eligible"] is False

        listing = list_customer_project_onsite_evidence(profile_root, "demo-field-ops")
        assert all(
            receipt["production_eligible"] is False
            for receipt in listing["onsite_acceptance_evidence"]["receipts"]
        )
        onsite_summary = listing["onsite_acceptance_evidence"]["summary"]
        assert onsite_summary["overall_status"] == "manual_check"
        assert onsite_summary["passed_required_count"] == 0
        assert not onsite_summary["missing_required_types"]
        assert sorted(onsite_summary["manual_check_required_types"]) == sorted(
            [
                "device_ingest",
                "voice_playback",
                "notification_delivery",
                "runtime_roundtrip",
            ]
        )

        report = customer_project_acceptance_report(
            profile_root,
            "demo-field-ops",
            check_env=False,
        )
        gates_by_id = {gate["gate_id"]: gate for gate in report["gates"]}
        assert gates_by_id["onsite_acceptance_boundary"]["status"] == "manual_check"
        assert (
            "0/4 个必需现场证据回执已通过" in gates_by_id["onsite_acceptance_boundary"]["evidence"]
        )

        review_result = register_customer_project_acceptance_review(
            profile_root,
            "demo-field-ops",
            {
                "decision": "accepted",
                "reason": "Onsite evidence is complete enough for customer signoff.",
                "risk_acknowledgement": True,
                "evidence_refs": [
                    listing["onsite_acceptance_evidence"]["summary"]["latest_receipt_id"]
                ],
            },
            operator_id="delivery.lead",
            reason="Delivery owner review.",
        )
        assert review_result["accepted"] is True
        closure = customer_project_acceptance_closure(
            profile_root,
            "demo-field-ops",
            check_env=False,
        )
        closure_gates = {gate["gate_id"]: gate for gate in closure["gates"]}
        assert "site_acceptance_checklist" in closure_gates
        assert closure_gates["site_acceptance_checklist"]["status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert closure_gates["manual_acceptance_review"]["status"] == "ready"
        assert closure_gates["dossier_verification"]["status"] == "ready"
        assert closure_gates["proposal_verification"]["status"] in {"ready", "manual_check"}
        assert closure_gates["audit_export"]["status"] in {"ready", "manual_check"}
        assert closure["artifact_verification"]["acceptance_dossier"]["valid"] is True
        assert closure["site_acceptance_checklist"]["items"]
        assert closure["manual_review"]["latest"]["decision"] == "accepted"
        assert closure["evidence_timeline"]
        assert closure_gates["customer_signoff"]["status"] == "manual_check"

        early_accept = register_customer_project_customer_signoff(
            profile_root,
            "demo-field-ops",
            {
                "decision": "accepted",
                "signatory_name": "Fanmu Operator",
                "signatory_role": "Customer operations owner",
                "organization": "Fanmu Creative Park",
                "reason": "Customer accepts the pilot handoff.",
                "risk_acknowledgement": True,
                "evidence_refs": [
                    listing["onsite_acceptance_evidence"]["summary"]["latest_receipt_id"]
                ],
            },
            operator_id="delivery.lead",
            reason="Customer signoff attempt.",
        )
        assert early_accept["accepted"] is False
        assert early_accept["reason"] == "project_not_ready_for_customer_signoff"

        customer_fix = register_customer_project_customer_signoff(
            profile_root,
            "demo-field-ops",
            {
                "decision": "needs_fix",
                "signatory_name": "Fanmu Operator",
                "signatory_role": "Customer operations owner",
                "organization": "Fanmu Creative Park",
                "reason": "Customer asks delivery to attach final audit export.",
                "evidence_refs": [
                    listing["onsite_acceptance_evidence"]["summary"]["latest_receipt_id"]
                ],
            },
            operator_id="delivery.lead",
            reason="Customer asks for final audit export.",
        )
        assert customer_fix["accepted"] is True
        signoffs = list_customer_project_customer_signoffs(profile_root, "demo-field-ops")
        assert signoffs["signoff_count"] == 1
        assert signoffs["latest"]["decision"] == "needs_fix"
        assert signoffs["latest"]["integrity_valid"] is True
        assert signoffs["latest"]["signoff_payload_sha256"]
        assert signoffs["latest"]["gate_snapshot"]["overall_status"] == closure["overall_status"]
        assert signoffs["latest"]["handoff_materials"]["acceptance_dossier"]["valid"] is True
        closure_after_signoff = customer_project_acceptance_closure(
            profile_root,
            "demo-field-ops",
            check_env=False,
        )
        signoff_gate = {gate["gate_id"]: gate for gate in closure_after_signoff["gates"]}[
            "customer_signoff"
        ]
        assert signoff_gate["status"] == "manual_check"
        assert (
            closure_after_signoff["customer_signoff"]["latest"]["signatory_name"]
            == "Fanmu Operator"
        )
        assert any(
            item["type"] == "customer_signoff"
            for item in closure_after_signoff["evidence_timeline"]
        )

        dossier_result = export_customer_project_acceptance_dossier(
            profile_root,
            "demo-field-ops",
            output_root=tmp_path / "dossiers",
            check_env=False,
        )
        dossier = dossier_result["dossier"]
        assert dossier["manifest"]["onsite_evidence_status"] == "manual_check"
        assert dossier["manifest"]["onsite_evidence_count"] == 4
        assert dossier["manifest"]["onsite_required_evidence_ready"] is False
        assert dossier["manifest"]["site_acceptance_checklist_status"] in {
            "ready",
            "manual_check",
            "blocked",
        }
        assert dossier["manifest"]["site_acceptance_checklist_ready_count"] >= 1
        assert dossier["manifest"]["manual_review_decision"] == "accepted"
        assert dossier["manifest"]["manual_review_count"] == 1
        assert dossier["manifest"]["customer_signoff_decision"] == "needs_fix"
        assert dossier["manifest"]["customer_signoff_count"] == 1
        assert dossier["manifest"]["customer_signoff_payload_sha256"]
        assert dossier["manifest"]["customer_signoff_integrity_valid"] is True
        onsite_inventory = [
            item
            for item in dossier["evidence_inventory"]
            if item.get("evidence_type") == "onsite_acceptance"
        ]
        assert len(onsite_inventory) == 4
        assert all(item["sha256"] for item in onsite_inventory)
        assert all(item["exists"] is True for item in onsite_inventory)
        assert all(item["size_bytes"] > 0 for item in onsite_inventory)
        assert all(
            item["evidence_url"].startswith("/api/field/evidence?path=")
            for item in onsite_inventory
        )
        assert {item["onsite_evidence_type"] for item in onsite_inventory} == {
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        }
        assert all(item["receipt_id"] for item in onsite_inventory)
        assert verify_customer_project_acceptance_dossier(dossier)["valid"] is True
    finally:
        shutil.rmtree(evidence_root, ignore_errors=True)
        try:
            evidence_root.parent.rmdir()
        except OSError:
            pass


def test_customer_project_customer_signoff_rejects_missing_unknown_and_incomplete_evidence_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_closure",
        lambda *args, **kwargs: _ready_customer_signoff_closure(),
    )

    missing = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(),
        operator_id="delivery.lead",
        reason="Missing evidence refs should block customer acceptance.",
    )
    unknown = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(["arbitrary-string"]),
        operator_id="delivery.lead",
        reason="Unknown evidence refs should block customer acceptance.",
    )
    incomplete = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(
            [
                "onsite:auto-device-ingest",
                f"acceptance_dossier:{'d' * 64}",
            ]
        ),
        operator_id="delivery.lead",
        reason="Incomplete evidence refs should block customer acceptance.",
    )

    assert missing["accepted"] is False
    assert missing["reason"] == "customer_signoff_evidence_refs_required"
    assert missing["evidence_ref_assessment"]["missing_material_types"] == [
        "acceptance_dossier",
        "proposal_bundle",
        "audit_export",
    ]
    assert unknown["accepted"] is False
    assert unknown["reason"] == "customer_signoff_evidence_refs_unresolved"
    assert unknown["evidence_ref_assessment"]["unresolved_refs"] == ["arbitrary-string"]
    assert incomplete["accepted"] is False
    assert incomplete["reason"] == "customer_signoff_evidence_refs_incomplete"
    assert sorted(incomplete["evidence_ref_assessment"]["missing_onsite_evidence_types"]) == [
        "notification_delivery",
        "runtime_roundtrip",
        "voice_playback",
    ]
    assert incomplete["evidence_ref_assessment"]["missing_material_types"] == [
        "proposal_bundle",
        "audit_export",
    ]
    signoffs = acceptance_module.list_customer_project_customer_signoffs(
        profile_root,
        "demo-field-ops",
    )
    assert signoffs["signoff_count"] == 0


def test_customer_project_customer_signoff_accepts_complete_verified_evidence_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_closure",
        lambda *args, **kwargs: _ready_customer_signoff_closure(),
    )

    accepted = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(_complete_customer_signoff_evidence_refs()),
        operator_id="delivery.lead",
        reason="Customer accepts the verified handoff package.",
    )

    assert accepted["accepted"] is True
    assessment = accepted["signoff"]["evidence_ref_assessment"]
    assert assessment["valid"] is True
    assert assessment["reason"] == "ok"
    assert assessment["resolved_count"] == len(_complete_customer_signoff_evidence_refs())
    assert assessment["missing_onsite_evidence_types"] == []
    assert assessment["missing_material_types"] == []
    assert set(assessment["available_ref_types"]) == {
        "acceptance_dossier",
        "audit_export",
        "onsite_receipt",
        "proposal_bundle",
    }
    signoffs = acceptance_module.list_customer_project_customer_signoffs(
        profile_root,
        "demo-field-ops",
    )
    assert signoffs["signoff_count"] == 1
    assert signoffs["latest"]["decision"] == "accepted"
    assert signoffs["latest"]["integrity_valid"] is True
    assert signoffs["latest"]["evidence_ref_assessment"]["valid"] is True


def test_customer_project_accepted_signoff_still_blocks_unattended_production_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    detail = get_customer_project_profile(profile_root, "demo-field-ops")
    profile_path = Path(detail["profile_path"])
    profile = load_field_site_profile(profile_path)
    profile["acceptance_reviews"] = [
        {
            "review_id": "ready-review",
            "reviewed_at": time.time(),
            "operator_id": "delivery.lead",
            "decision": "accepted",
            "reason": "Internal delivery review accepted the pilot handoff.",
            "risk_acknowledgement": True,
        }
    ]
    profile_path.write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    ready_closure = _ready_customer_signoff_closure()
    original_closure = acceptance_module.customer_project_acceptance_closure
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_closure",
        lambda *args, **kwargs: ready_closure,
    )

    accepted = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(_complete_customer_signoff_evidence_refs()),
        operator_id="delivery.lead",
        reason="Customer accepts the verified pilot handoff package.",
    )

    assert accepted["accepted"] is True

    monkeypatch.setattr(acceptance_module, "customer_project_acceptance_closure", original_closure)
    ready_report = {
        "found": True,
        "overall_status": "ready_for_onsite_acceptance",
        "customer_status": "ready",
        "release_claim": "pilot acceptance only",
        "customer": {"customer_id": "demo"},
        "site": {"site_id": "demo-field-ops"},
        "onsite_acceptance_evidence": ready_closure["onsite_acceptance_evidence"],
        "site_acceptance_checklist": {
            "overall_status": "ready",
            "ready_count": 4,
            "manual_check_count": 0,
            "blocked_count": 0,
            "customer_message": "Checklist is ready.",
        },
    }
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_report",
        lambda *args, **kwargs: ready_report,
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_acceptance_dossier_verification",
        lambda report: ready_closure["artifact_verification"]["acceptance_dossier"],
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_latest_proposal_verification",
        lambda profile: ready_closure["artifact_verification"]["proposal_bundle"],
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_latest_audit_export",
        lambda profile: ready_closure["artifact_verification"]["audit_export"],
    )

    closure = acceptance_module.customer_project_acceptance_closure(
        profile_root,
        "demo-field-ops",
        check_env=False,
    )

    assert closure["overall_status"] == "accepted_by_customer"
    assert "试点验收结论" in closure["customer_claim"]
    assert "无人值守生产上线" in closure["blocked_uses"]


def test_customer_project_customer_signoff_rejects_manual_check_onsite_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_closure",
        lambda *args, **kwargs: _ready_customer_signoff_closure(onsite_gate_eligible=False),
    )

    rejected = acceptance_module.register_customer_project_customer_signoff(
        profile_root,
        "demo-field-ops",
        _accepted_customer_signoff_payload(_complete_customer_signoff_evidence_refs()),
        operator_id="delivery.lead",
        reason="Manual-check onsite receipts should not satisfy customer signoff.",
    )

    assert rejected["accepted"] is False
    assert rejected["reason"] == "customer_signoff_evidence_refs_unresolved"
    assert sorted(rejected["evidence_ref_assessment"]["unresolved_refs"]) == sorted(
        [
            "onsite:auto-device-ingest",
            "onsite:auto-voice-playback",
            "onsite:auto-notification-delivery",
            "onsite:auto-runtime-roundtrip",
        ]
    )
    signoffs = acceptance_module.list_customer_project_customer_signoffs(
        profile_root,
        "demo-field-ops",
    )
    assert signoffs["signoff_count"] == 0


def test_customer_project_customer_signoff_downgrades_legacy_accepted_without_evidence_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    detail = get_customer_project_profile(profile_root, "demo-field-ops")
    profile_path = Path(detail["profile_path"])
    profile = load_field_site_profile(profile_path)
    profile["acceptance_reviews"] = [
        {
            "review_id": "legacy-ready-review",
            "reviewed_at": time.time(),
            "operator_id": "delivery.lead",
            "decision": "accepted",
            "reason": "Internal delivery review accepted the pilot handoff.",
            "risk_acknowledgement": True,
        }
    ]
    profile["customer_signoffs"] = [
        {
            "signoff_type": "askme.customer_project_customer_signoff",
            "signoff_version": 1,
            "signoff_id": "legacy-accepted-without-evidence-assessment",
            "signed_at": time.time(),
            "operator_id": "legacy.delivery",
            "decision": "accepted",
            "signatory_name": "Legacy Customer",
            "signatory_role": "Customer owner",
            "organization": "Legacy Org",
            "reason": "Legacy signoff before evidence refs were enforced.",
            "risk_acknowledgement": True,
            "credential_ref": "legacy-signoff.pdf",
            "credential_sha256": "a" * 64,
            "evidence_refs": ["legacy-free-text-ref"],
        }
    ]
    profile_path.write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    ready_closure = _ready_customer_signoff_closure()
    ready_report = {
        "found": True,
        "overall_status": "ready_for_onsite_acceptance",
        "customer_status": "ready",
        "release_claim": "pilot acceptance only",
        "customer": {"customer_id": "demo"},
        "site": {"site_id": "demo-field-ops"},
        "onsite_acceptance_evidence": ready_closure["onsite_acceptance_evidence"],
        "site_acceptance_checklist": {
            "overall_status": "ready",
            "ready_count": 4,
            "manual_check_count": 0,
            "blocked_count": 0,
            "customer_message": "Checklist is ready.",
        },
    }
    monkeypatch.setattr(
        acceptance_module,
        "customer_project_acceptance_report",
        lambda *args, **kwargs: ready_report,
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_acceptance_dossier_verification",
        lambda report: ready_closure["artifact_verification"]["acceptance_dossier"],
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_latest_proposal_verification",
        lambda profile: ready_closure["artifact_verification"]["proposal_bundle"],
    )
    monkeypatch.setattr(
        acceptance_module,
        "_customer_project_latest_audit_export",
        lambda profile: ready_closure["artifact_verification"]["audit_export"],
    )

    signoffs = acceptance_module.list_customer_project_customer_signoffs(
        profile_root,
        "demo-field-ops",
    )
    latest = signoffs["latest"]
    assert latest["decision"] == "accepted"
    assert latest["integrity_valid"] is True
    assert latest["evidence_ref_assessment"]["valid"] is False
    assert latest["evidence_ref_assessment"]["reason"] == (
        "customer_signoff_evidence_refs_legacy_unverified"
    )
    assert latest["evidence_ref_assessment"]["legacy_unverified"] is True

    gate = acceptance_module._customer_project_customer_signoff_gate(
        latest,
        base_ready_for_signoff=True,
    )
    assert gate["status"] == "manual_check"
    assert "evidence refs are not verified" in gate["evidence"]

    closure = acceptance_module.customer_project_acceptance_closure(
        profile_root,
        "demo-field-ops",
        check_env=False,
    )
    closure_gates = {gate["gate_id"]: gate for gate in closure["gates"]}
    assert closure["customer_signoff"]["base_ready_for_signoff"] is True
    assert closure_gates["customer_signoff"]["status"] == "manual_check"
    assert closure["overall_status"] == "ready_for_customer_signoff"
    assert closure["overall_status"] != "accepted_by_customer"


def test_customer_project_onsite_evidence_rejects_invalid_receipts(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    detail = get_customer_project_profile(profile_root, "demo-field-ops")
    profile_path = Path(detail["profile_path"])
    before = profile_path.read_text(encoding="utf-8")
    evidence_path = tmp_path / "verified-device-ingest.json"
    evidence_path.write_text(
        json.dumps(_forged_onsite_evidence_payload("device_ingest")), encoding="utf-8"
    )

    bad_type = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {"evidence_type": "random_photo", "status": "passed"},
        operator_id="delivery.lead",
        reason="Invalid evidence type should not write.",
    )
    bad_status = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {"evidence_type": "device_ingest", "status": "not_a_status"},
        operator_id="delivery.lead",
        reason="Invalid evidence status should not write.",
    )
    fake_pass = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "summary": "A required onsite gate cannot pass without a verifiable artifact.",
        },
        operator_id="delivery.lead",
        reason="Unverified passed evidence should not write.",
    )
    missing_file = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "runtime_roundtrip",
            "status": "passed",
            "path": str(tmp_path / "missing-runtime.json"),
        },
        operator_id="delivery.lead",
        reason="Missing passed evidence should not write.",
    )
    bad_tier = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "path": str(evidence_path),
            "evidence_tier": "production_launch",
        },
        operator_id="delivery.lead",
        reason="Production trust tier should not write.",
    )
    bad_tier_missing_file = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "path": str(tmp_path / "missing-production-launch.json"),
            "evidence_tier": "production_launch",
        },
        operator_id="delivery.lead",
        reason="Unsupported trust tier should be rejected before artifact checks.",
    )

    assert bad_type["accepted"] is False
    assert bad_type["reason"] == "unsupported_onsite_evidence_type"
    assert bad_status["accepted"] is False
    assert bad_status["reason"] == "unsupported_onsite_evidence_status"
    assert fake_pass["accepted"] is False
    assert fake_pass["reason"] == "passed_required_onsite_evidence_requires_path"
    assert fake_pass["trust"]["status"] == "unverified"
    assert missing_file["accepted"] is False
    assert missing_file["reason"] == "passed_required_onsite_evidence_path_not_found"
    assert bad_tier["accepted"] is False
    assert bad_tier["reason"] == "unsupported_onsite_evidence_trust_tier"
    assert bad_tier_missing_file["accepted"] is False
    assert bad_tier_missing_file["reason"] == "unsupported_onsite_evidence_trust_tier"
    assert profile_path.read_text(encoding="utf-8") == before


def test_customer_project_onsite_evidence_sanitizes_client_production_claims(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_path = tmp_path / "verified-device-ingest.json"
    evidence_path.write_text(
        json.dumps(_forged_onsite_evidence_payload("device_ingest")), encoding="utf-8"
    )

    registered = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "path": str(evidence_path),
            "production_eligible": True,
        },
        operator_id="delivery.lead",
        reason="Client-side production claim must be ignored.",
    )

    assert registered["accepted"] is True
    receipt = registered["receipt"]
    assert receipt["evidence_tier"] == "acceptance_candidate"
    assert receipt["production_eligible"] is False
    assert receipt["verified_evidence"] is True
    assert receipt["trust_status"] == "manual_check"
    assert receipt["acceptance_gate_eligible"] is False


@pytest.mark.parametrize(
    "evidence_type",
    [
        "device_ingest",
        "voice_playback",
        "notification_delivery",
        "runtime_roundtrip",
    ],
)
def test_customer_project_onsite_evidence_forged_json_does_not_open_acceptance_gate(
    tmp_path: Path,
    evidence_type: str,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_path = tmp_path / f"forged-{evidence_type}.json"
    evidence_path.write_text(
        json.dumps(_forged_onsite_evidence_payload(evidence_type)),
        encoding="utf-8",
    )

    registered = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": evidence_type,
            "status": "passed",
            "path": str(evidence_path),
            "summary": "This file self-claims real-link success but is not system verified.",
        },
        operator_id="delivery.lead",
        reason="Forged JSON must not open onsite acceptance.",
    )

    assert registered["accepted"] is True
    receipt = registered["receipt"]
    assert receipt["status"] == "passed"
    assert receipt["verified_evidence"] is True
    assert receipt["trust_status"] == "manual_check"
    assert receipt["acceptance_gate_eligible"] is False
    summary = registered["onsite_acceptance_evidence"]["summary"]
    assert summary["overall_status"] == "manual_check"
    assert summary["passed_required_count"] == 0
    assert summary["manual_check_required_types"] == [evidence_type]


def test_customer_project_onsite_evidence_ignores_claimed_sha256(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_path = tmp_path / "manual-device-ingest.json"
    evidence_path.write_text(
        json.dumps(_forged_onsite_evidence_payload("device_ingest")),
        encoding="utf-8",
    )
    claimed_sha = "f" * 64

    registered = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "path": str(evidence_path),
            "sha256": claimed_sha,
        },
        operator_id="delivery.lead",
        reason="Client-supplied hash must not override file inventory.",
    )

    actual_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert registered["accepted"] is True
    assert registered["receipt"]["sha256"] == actual_sha
    assert registered["receipt"]["sha256"] != claimed_sha


def test_customer_project_onsite_evidence_accepts_profile_bundle_sibling_evidence() -> None:
    with tempfile.TemporaryDirectory(prefix="askme-field-evidence-") as temp_dir:
        temp_path = Path(temp_dir)
        profile_root = temp_path / "profiles"
        shutil.copytree(Path("deploy/site-profiles"), profile_root)
        evidence_path = temp_path / "manual-device-ingest.json"
        evidence_path.write_text(
            json.dumps(_forged_onsite_evidence_payload("device_ingest")),
            encoding="utf-8",
        )
        claimed_sha = "f" * 64

        registered = register_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            {
                "evidence_type": "device_ingest",
                "status": "passed",
                "path": str(evidence_path),
                "sha256": claimed_sha,
                "production_eligible": True,
            },
            operator_id="delivery.lead",
            reason="Profile-bundle-local manual evidence should remain auditable.",
        )

        actual_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        assert registered["accepted"] is True
        receipt = registered["receipt"]
        assert receipt["sha256"] == actual_sha
        assert receipt["sha256"] != claimed_sha
        assert receipt["production_eligible"] is False
        assert receipt["trust_status"] == "manual_check"
        assert receipt["acceptance_gate_eligible"] is False


def test_customer_project_onsite_evidence_plain_file_does_not_open_acceptance_gate(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_path = tmp_path / "manual-device-ingest.json"
    evidence_path.write_text(
        json.dumps({"evidence_type": "device_ingest", "status": "passed"}),
        encoding="utf-8",
    )

    registered = register_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "path": str(evidence_path),
            "summary": "A plain JSON file is not enough to prove live device ingest.",
        },
        operator_id="delivery.lead",
        reason="Manual file should require delivery review.",
    )

    assert registered["accepted"] is True
    receipt = registered["receipt"]
    assert receipt["status"] == "passed"
    assert receipt["verified_evidence"] is True
    assert receipt["trust_status"] == "manual_check"
    assert receipt["acceptance_gate_eligible"] is False
    summary = registered["onsite_acceptance_evidence"]["summary"]
    assert summary["overall_status"] == "manual_check"
    assert summary["passed_required_count"] == 0
    assert summary["manual_check_required_types"] == ["device_ingest"]
    report = customer_project_acceptance_report(
        profile_root,
        "demo-field-ops",
        check_env=False,
    )
    checklist_by_id = {
        item["item_id"]: item for item in report["site_acceptance_checklist"]["items"]
    }
    assert checklist_by_id["device_ingest"]["status"] == "manual_check"
    assert report["site_acceptance_checklist"]["overall_status"] != "ready"


def test_customer_project_onsite_evidence_downgrades_legacy_unverified_passed_receipts(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    detail = get_customer_project_profile(profile_root, "demo-field-ops")
    profile_path = Path(detail["profile_path"])
    profile = load_field_site_profile(profile_path)
    profile["onsite_acceptance_evidence"] = [
        {
            "evidence_type": "device_ingest",
            "status": "passed",
            "summary": "Legacy manual receipt without a file.",
            "evidence_tier": "site_acceptance",
            "production_eligible": True,
        },
        {
            "evidence_type": "voice_playback",
            "status": "passed",
            "summary": "Legacy manual receipt without a file.",
            "evidence_tier": "site_acceptance",
            "production_eligible": True,
        },
        {
            "evidence_type": "notification_delivery",
            "status": "passed",
            "summary": "Legacy manual receipt without a file.",
            "evidence_tier": "site_acceptance",
            "production_eligible": True,
        },
        {
            "evidence_type": "runtime_roundtrip",
            "status": "passed",
            "summary": "Legacy manual receipt without a file.",
            "evidence_tier": "site_acceptance",
            "production_eligible": True,
        },
    ]
    profile_path.write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    listing = list_customer_project_onsite_evidence(
        profile_root,
        "demo-field-ops",
        include_readiness_auto=False,
    )

    receipts = listing["onsite_acceptance_evidence"]["receipts"]
    assert {receipt["status"] for receipt in receipts} == {"manual_check"}
    assert {receipt["original_status"] for receipt in receipts} == {"passed"}
    assert all(receipt["verified_evidence"] is False for receipt in receipts)
    assert all(receipt["trust_status"] == "unverified" for receipt in receipts)
    assert all(receipt["production_eligible"] is False for receipt in receipts)
    assert {receipt["evidence_tier"] for receipt in receipts} == {"acceptance_candidate"}
    summary = listing["onsite_acceptance_evidence"]["summary"]
    assert summary["overall_status"] == "manual_check"
    assert summary["passed_required_count"] == 0
    assert sorted(summary["manual_check_required_types"]) == sorted(
        [
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        ]
    )


def test_customer_project_onsite_failed_receipt_blocks_acceptance(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    shutil.copytree(Path("deploy/site-profiles"), profile_root)
    evidence_root = Path("artifacts/test-onsite-evidence") / str(int(time.time() * 1000))
    evidence_root.mkdir(parents=True, exist_ok=True)
    try:
        for evidence_type in (
            "device_ingest",
            "voice_playback",
            "notification_delivery",
            "runtime_roundtrip",
        ):
            evidence_path = evidence_root / f"{evidence_type}.json"
            evidence_path.write_text(
                json.dumps(_forged_onsite_evidence_payload(evidence_type)),
                encoding="utf-8",
            )
            assert (
                register_customer_project_onsite_evidence(
                    profile_root,
                    "demo-field-ops",
                    {
                        "evidence_type": evidence_type,
                        "status": "passed",
                        "path": str(evidence_path),
                    },
                    operator_id="delivery.lead",
                    reason="Required smoke passed.",
                )["accepted"]
                is True
            )

        failed_path = evidence_root / "device_ingest_failed.json"
        failed_path.write_text(
            json.dumps({"evidence_type": "device_ingest", "status": "failed"}),
            encoding="utf-8",
        )
        failed = register_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            {
                "evidence_type": "device_ingest",
                "status": "failed",
                "path": str(failed_path),
            },
            operator_id="delivery.lead",
            reason="Latest device ingest smoke failed.",
        )
        assert failed["accepted"] is True
        blocked_summary = failed["onsite_acceptance_evidence"]["summary"]
        assert blocked_summary["overall_status"] == "blocked"
        assert blocked_summary["failed_required_types"] == ["device_ingest"]

        report = customer_project_acceptance_report(
            profile_root,
            "demo-field-ops",
            check_env=False,
        )
        gates_by_id = {gate["gate_id"]: gate for gate in report["gates"]}
        assert gates_by_id["onsite_acceptance_boundary"]["status"] == "blocked"

        recovered_path = evidence_root / "device_ingest_recovered.json"
        recovered_path.write_text(
            json.dumps(_forged_onsite_evidence_payload("device_ingest")),
            encoding="utf-8",
        )
        recovered = register_customer_project_onsite_evidence(
            profile_root,
            "demo-field-ops",
            {
                "evidence_type": "device_ingest",
                "status": "passed",
                "path": str(recovered_path),
            },
            operator_id="delivery.lead",
            reason="Device ingest smoke recovered.",
        )
        recovered_summary = recovered["onsite_acceptance_evidence"]["summary"]
        assert recovered_summary["overall_status"] == "manual_check"
        assert recovered_summary["failed_required_types"] == []
        assert recovered_summary["passed_required_count"] == 0
    finally:
        shutil.rmtree(evidence_root, ignore_errors=True)
        try:
            evidence_root.parent.rmdir()
        except OSError:
            pass


def test_customer_project_acceptance_dossier_exports_hash_manifest(tmp_path: Path) -> None:
    dossier_result = export_customer_project_acceptance_dossier(
        Path("deploy/site-profiles"),
        "demo-field-ops",
        output_root=tmp_path / "dossiers",
        check_env=False,
    )

    assert dossier_result["accepted"] is True
    assert Path(dossier_result["dossier_path"]).exists()
    html_path = Path(dossier_result["html_path"])
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "AskMe 客户验收资料包" in html
    assert "清单 SHA-256" in html
    assert "field-ingest-smoke" in html
    dossier = dossier_result["dossier"]
    assert dossier["dossier_type"] == "askme.customer_project_acceptance"
    assert dossier["overall_status"] in {"blocked", "manual_check", "ready_for_onsite_acceptance"}
    assert dossier["manifest"]["payload_sha256"]
    assert dossier["manifest"]["project_id"] == "demo-field-ops"
    assert dossier["manifest"]["evidence_count"] >= 1
    assert "site_acceptance_checklist_status" in dossier["manifest"]
    assert (
        dossier["launch_readiness"]["readiness_type"]
        == "askme.customer_project_launch_readiness.v1"
    )
    assert (
        dossier["manifest"]["launch_readiness_status"]
        == dossier["launch_readiness"]["overall_status"]
    )
    assert dossier["manifest"]["launch_stage"] == dossier["launch_readiness"]["launch_stage"]
    assert (
        dossier["manifest"]["production_ready"] is dossier["launch_readiness"]["production_ready"]
    )
    assert dossier["manifest"]["site_acceptance_checklist_ready_count"] >= 0
    assert dossier["delivery_workflow"]["steps"]
    assert dossier["site_acceptance_checklist"]["items"]
    assert dossier["evidence_inventory"]
    hashed_inventory = [item for item in dossier["evidence_inventory"] if item["sha256"]]
    assert hashed_inventory
    assert all(item["exists"] is True for item in hashed_inventory)
    assert all(item["size_bytes"] > 0 for item in hashed_inventory)
    assert all(len(item["sha256"]) == 64 for item in hashed_inventory)
    assert all(item["path"] for item in hashed_inventory)
    assert all(
        item["evidence_url"].startswith("/api/field/evidence?path=") for item in hashed_inventory
    )
    assert "交付流程" in html
    assert "上线准入" in html
    assert dossier["launch_readiness"]["launch_stage"] in html
    assert "\ufffd" not in html
    verification = verify_customer_project_acceptance_dossier(dossier)
    assert verification["valid"] is True


def test_customer_project_acceptance_dossier_verify_rejects_tamper_and_bad_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_HMAC_SECRET", "test-signing-secret")
    dossier_result = export_customer_project_acceptance_dossier(
        Path("deploy/site-profiles"),
        "demo-field-ops",
        output_root=tmp_path / "dossiers",
        check_env=False,
    )
    dossier = dossier_result["dossier"]

    assert verify_customer_project_acceptance_dossier(dossier)["valid"] is True

    tampered = json.loads(json.dumps(dossier))
    tampered["customer"]["project_name"] = "tampered project name"
    tampered_result = verify_customer_project_acceptance_dossier(tampered)
    assert tampered_result["valid"] is False
    assert "manifest.payload_sha256 mismatch" in tampered_result["errors"]

    tampered_launch = json.loads(json.dumps(dossier))
    tampered_launch["launch_readiness"]["overall_status"] = "ready"
    tampered_launch_result = verify_customer_project_acceptance_dossier(tampered_launch)
    assert tampered_launch_result["valid"] is False
    assert "manifest.payload_sha256 mismatch" in tampered_launch_result["errors"]

    bad_signature = json.loads(json.dumps(dossier))
    bad_signature["manifest"]["payload_signature"] = "bad-signature"
    bad_signature_result = verify_customer_project_acceptance_dossier(bad_signature)
    assert bad_signature_result["valid"] is False
    assert "manifest.payload_signature mismatch" in bad_signature_result["errors"]

    monkeypatch.delenv("ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_HMAC_SECRET", raising=False)
    missing_secret_result = verify_customer_project_acceptance_dossier(dossier)
    assert missing_secret_result["valid"] is False
    assert (
        "signature present but verification secret is not configured"
        in missing_secret_result["errors"]
    )


def test_customer_project_proposal_bundle_binds_package_dossier_and_release_notes(
    tmp_path: Path,
) -> None:
    template_root = tmp_path / "templates"
    shutil.copytree(Path("deploy/customer-project-templates"), template_root)
    request = create_customer_project_template_release_request(
        template_root,
        "factory-inspection",
        {"version": "0.1.2", "publish_status": "published", "release_channel": "stable"},
        operator_id="product.owner",
        reason="Approve proposal material.",
    )
    approved = review_customer_project_template_release_request(
        template_root,
        request["request"]["request_id"],
        decision="approve",
        operator_id="product.reviewer",
        reason="Second approver.",
    )
    assert approved["accepted"] is True

    result = export_customer_project_proposal_bundle(
        Path("deploy/site-profiles"),
        template_root,
        "demo-field-ops",
        output_root=tmp_path / "proposals",
        check_env=False,
    )

    assert result["accepted"] is True
    assert Path(result["proposal_path"]).exists()
    assert Path(result["html_path"]).exists()
    proposal = result["proposal"]
    assert proposal["proposal_type"] == "askme.customer_project_proposal"
    assert proposal["customer_project_package"]["manifest"]["project_id"] == "demo-field-ops"
    assert proposal["acceptance_dossier"]["manifest"]["project_id"] == "demo-field-ops"
    readable = proposal["customer_readable_delivery"]
    assert readable["applicability_scope"]["scenarios"]
    assert readable["customer_prerequisites"]
    assert readable["scenario_acceptance_criteria"]
    assert readable["dependency_matrix"]
    assert proposal["manifest"]["proposal_scenario_acceptance_criteria_count"] == len(
        readable["scenario_acceptance_criteria"]
    )
    assert proposal["manifest"]["proposal_customer_prerequisite_count"] == len(
        readable["customer_prerequisites"]
    )
    assert (
        proposal["launch_readiness"]["readiness_type"]
        == "askme.customer_project_launch_readiness.v1"
    )
    assert (
        proposal["acceptance_dossier"]["launch_readiness"]["overall_status"]
        == proposal["launch_readiness"]["overall_status"]
    )
    assert (
        proposal["manifest"]["launch_readiness_status"]
        == proposal["launch_readiness"]["overall_status"]
    )
    assert proposal["manifest"]["launch_stage"] == proposal["launch_readiness"]["launch_stage"]
    assert proposal["approved_template_release_bundle"]["summary"]["approved_release_count"] == 1
    assert (
        proposal["approved_template_release_bundle"]["release_notes"][0]["template_id"]
        == "factory-inspection"
    )
    assert proposal["proposal_insert"]["safe_claims"]
    assert proposal["manifest"]["payload_sha256"]
    assert proposal["manifest"]["tenant_id"] == proposal["customer"]["tenant_id"]
    assert proposal["manifest"]["delivery_namespace"] == proposal["customer"]["delivery_namespace"]
    verification = verify_customer_project_proposal_bundle(proposal)
    assert verification["valid"] is True
    assert verification["proposal_scope"]["project_id"] == "demo-field-ops"
    tampered = json.loads(json.dumps(proposal))
    tampered["manifest"]["tenant_id"] = "other-tenant"
    tampered["delivery_boundary"] = "changed after export"
    rejected = verify_customer_project_proposal_bundle(tampered)
    assert rejected["valid"] is False
    assert "manifest.tenant_id mismatch" in rejected["errors"]
    assert "manifest.payload_sha256 mismatch" in rejected["errors"]
    tampered_launch = json.loads(json.dumps(proposal))
    tampered_launch["launch_readiness"]["production_ready"] = True
    tampered_launch["launch_readiness"]["launch_stage"] = "production_acceptance_ready"
    launch_rejected = verify_customer_project_proposal_bundle(tampered_launch)
    assert launch_rejected["valid"] is False
    assert "manifest.payload_sha256 mismatch" in launch_rejected["errors"]
    html = Path(result["html_path"]).read_text(encoding="utf-8")
    assert "AskMe 客户项目提案包" in html
    assert "factory-inspection" in html
    assert "上线准入" in html
    assert proposal["launch_readiness"]["launch_stage"] in html
    assert "验收门禁" in html


def test_field_site_profile_env_references_cover_responder_and_device_envs() -> None:
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    refs = site_profile_env_references(profile)
    by_name = {str(item["env_name"]): item for item in refs}

    assert "ASKME_DINGTALK_SECURITY_WEBHOOK" in by_name
    assert "ASKME_DINGTALK_SECURITY_SECRET" in by_name
    assert "ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET" in by_name
    assert "ASKME_FIELD_ROBOT_THUNDER_SECRET" in by_name
    assert by_name["ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET"]["category"] == "field_device_secret"
    assert by_name["ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET"]["reference"] == (
        "devices.camera-main-road-1.secret_env"
    )


def test_field_site_profile_env_references_report_configured_state(monkeypatch) -> None:
    monkeypatch.setenv("ASKME_FIELD_ROBOT_THUNDER_SECRET", "robot-secret")
    monkeypatch.delenv("ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET", raising=False)
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    refs = site_profile_env_references(profile)
    by_name = {str(item["env_name"]): item for item in refs}

    assert by_name["ASKME_FIELD_ROBOT_THUNDER_SECRET"]["configured"] is True
    assert by_name["ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET"]["configured"] is False


def test_field_site_profile_env_template_is_customer_deployment_handoff() -> None:
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    template = render_site_profile_env_template(profile)

    assert "# AskMe field site environment template" in template
    assert "# DingTalk responder webhooks" in template
    assert "# Field device HMAC ingest secrets" in template
    assert "ASKME_DINGTALK_SECURITY_WEBHOOK=" in template
    assert "ASKME_FIELD_ROBOT_THUNDER_SECRET=" in template
    assert template.count("ASKME_DINGTALK_SECURITY_WEBHOOK=") == 1
    assert template.count("ASKME_FIELD_ROBOT_THUNDER_SECRET=") == 1
    assert template.endswith("\n")


def test_site_profile_catalog_supports_multi_site_rollout(tmp_path: Path, monkeypatch) -> None:
    source = Path("deploy/site-profiles/park-demo.yaml")
    first = tmp_path / "site-a.yaml"
    second = tmp_path / "site-b.yaml"
    broken = tmp_path / "broken.yaml"
    first.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    second.write_text(
        source.read_text(encoding="utf-8").replace("inovx-demo-park", "factory-zone-b"),
        encoding="utf-8",
    )
    broken.write_text("site:\n  site_id: broken\nzones: {}\n", encoding="utf-8")
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)

    catalog = build_site_profile_catalog(tmp_path, check_env=True)

    assert catalog["summary"]["site_count"] == 3
    assert catalog["summary"]["configured_count"] == 2
    assert catalog["summary"]["blocked_count"] == 1
    assert catalog["summary"]["env_missing_count"] > 0
    assert catalog["summary"]["multi_site_ready"] is False
    assert catalog["sites"][0]["site_id"] == "broken"
    assert catalog["sites"][0]["deployment_stage"] == "blocked"
    assert catalog["sites"][0]["delivery_workflow"]["overall_status"] == "blocked"
    valid_sites = [item for item in catalog["sites"] if item["status"] == "passed"]
    assert {item["deployment_stage"] for item in valid_sites} == {"site_config_ready"}
    assert all(site["delivery_workflow"]["steps"] for site in valid_sites)
    assert "Fix blocked site profiles" in catalog["next_step"]


def test_customer_project_catalog_groups_sites_by_customer_and_objects(tmp_path: Path) -> None:
    source = Path("deploy/site-profiles/park-demo.yaml").read_text(encoding="utf-8")
    first = tmp_path / "site-a.yaml"
    second = tmp_path / "site-b.yaml"
    first.write_text(source, encoding="utf-8")
    second.write_text(
        source.replace("demo-customer", "factory-customer")
        .replace("creative_park", "manufacturing")
        .replace("demo-field-ops", "factory-line-patrol")
        .replace("inovx-demo-park", "factory-site-b"),
        encoding="utf-8",
    )

    catalog = build_customer_project_catalog(tmp_path)

    assert catalog["summary"]["customer_count"] == 2
    assert catalog["summary"]["project_count"] == 2
    assert catalog["summary"]["managed_object_type_count"] >= 8
    assert {"creative_park", "manufacturing"} <= {
        industry for customer in catalog["customers"] for industry in customer["industries"]
    }
    assert len(catalog["projects"]) == 2
    assert catalog["delivery_acceptance_gate"]["gate_type"] == (
        "askme.solution_delivery_catalog_acceptance_gate"
    )
    assert catalog["delivery_acceptance_gate"]["project_count"] == 2
    assert catalog["summary"]["delivery_acceptance_gate_status"] in {
        "ready",
        "manual_check",
        "blocked",
    }
    assert all(project["managed_objects"] for project in catalog["projects"])
    assert all(project["delivery_workflow"]["steps"] for project in catalog["projects"])
    assert all(
        project["delivery_workflow"]["overall_status"] in {"ready", "manual_check", "blocked"}
        for project in catalog["projects"]
    )
    demo = next(
        project for project in catalog["projects"] if project["project_id"] == "demo-field-ops"
    )
    assert demo["product_acceptance_gate"]["gate_type"] == (
        "askme.solution_delivery_product_acceptance_gate"
    )
    assert demo["product_acceptance_gate"]["gates"]
    assert {gate["gate_id"] for gate in demo["product_acceptance_gate"]["gates"]} >= {
        "customer_scope",
        "site_profile",
        "managed_object_catalog",
        "resource_bindings",
        "acceptance_references",
        "object_change_audit",
        "handoff_artifacts",
    }
    workflow_by_id = {item["step_id"]: item for item in demo["delivery_workflow"]["steps"]}
    assert workflow_by_id["customer_scope"]["status"] == "ready"
    assert workflow_by_id["managed_objects"]["status"] == "ready"
    assert workflow_by_id["runtime_bindings"]["status"] == "ready"
    assert workflow_by_id["handoff_package"]["status"] == "manual_check"
    filtered = build_customer_project_catalog(
        tmp_path,
        industry="manufacturing",
        customer_id="factory",
    )
    assert filtered["filters"] == {
        "customer_id": "factory",
        "industry": "manufacturing",
    }
    assert filtered["summary"]["filtered"] is True
    assert filtered["summary"]["customer_count"] == 1
    assert filtered["delivery_acceptance_gate"]["project_count"] == 1
    assert filtered["projects"][0]["project_id"] == "factory-line-patrol"


def test_customer_project_templates_are_valid_solution_starters() -> None:
    payload = list_customer_project_templates(Path("deploy/customer-project-templates"))

    assert payload["summary"]["template_count"] >= 4
    assert payload["summary"]["valid_count"] >= 4
    assert payload["summary"]["manual_check_count"] >= 4
    industries = {item["industry"] for item in payload["templates"]}
    assert {"manufacturing", "creative_park", "warehouse", "scenic_area"} <= industries
    assert all(
        item["managed_objects_summary"]["object_type_count"] >= 2 for item in payload["templates"]
    )
    assert all(
        item["template_package"]["package_schema"] == "askme.customer_project_template.v1"
        for item in payload["templates"]
    )
    assert all(item["template_package"]["version"] == "0.1.0" for item in payload["templates"])
    assert all(
        item["template_package"]["publish_status"] == "pilot" for item in payload["templates"]
    )
    assert all(
        item["template_package"]["product_status"] == "manual_check"
        for item in payload["templates"]
    )
    assert all(
        item["template_package"]["dependencies"]["skill_package_count"] >= 1
        for item in payload["templates"]
    )
    assert all(
        item["delivery_summary"]["default_object_count"] >= 2 for item in payload["templates"]
    )
    assert all(
        item["delivery_summary"]["template_version"] == "0.1.0" for item in payload["templates"]
    )
    assert all(item["delivery_summary"]["scenario_ids"] for item in payload["templates"])
    assert all(item["delivery_summary"]["skill_packages"] for item in payload["templates"])
    assert all(item["delivery_summary"]["acceptance_tests"] for item in payload["templates"])
    assert payload["summary"]["runtime_blueprint_bound_count"] >= 4
    assert payload["summary"]["runtime_blueprint_manual_check_count"] >= 4
    assert all(
        item["runtime_blueprint_binding"]["binding_type"]
        == "askme.customer_project_template.runtime_blueprint_binding.v1"
        for item in payload["templates"]
    )
    assert all(
        item["runtime_blueprint_binding"]["selected_blueprint"]["name"]
        for item in payload["templates"]
    )
    assert all(
        item["runtime_blueprint_binding"]["policy"][
            "template_must_bind_runtime_blueprint_before_delivery"
        ]
        is True
        for item in payload["templates"]
    )
    assert all(
        item["applicability_scope"]["scope_type"]
        == "askme.customer_delivery_applicability_scope.v1"
        for item in payload["templates"]
    )
    assert all(item["applicability_scope"]["industries"] for item in payload["templates"])
    assert all(item["applicability_scope"]["scenarios"] for item in payload["templates"])
    assert all(item["applicability_scope"]["managed_object_types"] for item in payload["templates"])
    assert all(item["out_of_scope"] for item in payload["templates"])
    assert all(item["customer_prerequisites"] for item in payload["templates"])
    assert all(item["scenario_acceptance_criteria"] for item in payload["templates"])
    assert all(item["dependency_matrix"] for item in payload["templates"])
    assert all(item["delivery_checklist"] for item in payload["templates"])
    for item in payload["templates"]:
        deps = item["template_package"]["dependencies"]
        delivery = item["delivery_summary"]
        assert deps["vision_model_count"] == len(delivery["vision_models"])
        assert deps["sensor_protocol_count"] == len(delivery["sensor_protocols"])
        assert deps["skill_package_count"] == len(delivery["skill_packages"])
        assert deps["acceptance_test_count"] == len(delivery["acceptance_tests"])
        for obj in delivery["default_objects"]:
            assert obj["object_id"]
            assert obj["display_name"]
            assert obj["category"] != "uncategorized"
    factory = next(
        item for item in payload["templates"] if item["template_id"] == "factory-inspection"
    )
    assert factory["tenant_id"] == "default"
    assert factory["delivery_namespace"] == "default"
    assert factory["product_status"] == "manual_check"
    assert factory["runtime_blueprint_binding"]["status"] == "manual_check"
    assert factory["runtime_blueprint_binding"]["selected_blueprint"]["name"] == "edge_robot"
    assert factory["runtime_blueprint_binding"]["match_reason"] == "industry_default:manufacturing"
    checklist_by_id = {item["step_id"]: item for item in factory["delivery_checklist"]}
    assert checklist_by_id["validate_template"]["status"] == "ready"
    assert checklist_by_id["review_template_release"]["status"] == "manual_check"
    assert checklist_by_id["replace_customer_scope"]["status"] == "manual_check"
    assert checklist_by_id["run_acceptance"]["status"] in {"ready", "manual_check", "blocked"}

    filtered = list_customer_project_templates(
        Path("deploy/customer-project-templates"),
        industry="manufacturing",
        publish_status="pilot",
        product_status="manual_check",
    )
    assert filtered["filters"] == {
        "industry": "manufacturing",
        "product_status": "manual_check",
        "publish_status": "pilot",
    }
    assert filtered["summary"]["filtered"] is True
    assert filtered["summary"]["template_count"] == 1
    assert filtered["summary"]["industry_count"] == 1
    assert filtered["templates"][0]["template_id"] == "factory-inspection"


def test_solution_delivery_readiness_rolls_up_product_gates() -> None:
    project_catalog = build_customer_project_catalog(Path("deploy/site-profiles"), check_env=True)
    template_catalog = list_customer_project_templates(Path("deploy/customer-project-templates"))
    resource_catalog = build_customer_project_resource_catalog(
        Path("deploy/site-profiles"),
        template_root=Path("deploy/customer-project-templates"),
    )
    governance_requests = list_delivery_resource_governance_requests(
        Path("deploy/delivery-resources"),
    )

    readiness = build_solution_delivery_readiness(
        project_catalog=project_catalog,
        template_catalog=template_catalog,
        resource_catalog=resource_catalog,
        governance_requests=governance_requests,
    )

    assert readiness["readiness_type"] == "askme.solution_delivery_readiness"
    assert readiness["overall_status"] in {"ready", "manual_check", "blocked"}
    assert readiness["production_ready"] is (readiness["overall_status"] == "ready")
    assert readiness["summary"]["project_count"] >= 1
    assert readiness["summary"]["template_count"] >= 4
    assert readiness["summary"]["resource_count"] >= 10
    gate_ids = {gate["gate_id"] for gate in readiness["gates"]}
    assert {
        "customer_project_acceptance",
        "template_market",
        "delivery_resource_bindings",
        "delivery_resource_governance",
    } <= gate_ids
    assert readiness["customer_status"]
    assert readiness["release_claim"]


def test_customer_project_template_release_governance_records_revisions(tmp_path: Path) -> None:
    template_root = tmp_path / "templates"
    shutil.copytree(Path("deploy/customer-project-templates"), template_root)

    direct_publish = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {
            "version": "0.1.1",
            "publish_status": "published",
            "release_channel": "stable",
            "release_note": "Validated for reusable factory pilot handoff.",
        },
        operator_id="product.owner",
        reason="Promote after template package review.",
    )
    assert direct_publish["accepted"] is False
    assert direct_publish["reason"] == "published_release_requires_approval_request"

    updated = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {
            "version": "0.1.1",
            "publish_status": "published",
            "release_channel": "stable",
            "release_note": "Validated for reusable factory pilot handoff.",
        },
        operator_id="product.reviewer",
        reason="Promote after approved template package review.",
        allow_published=True,
        approval_request_id="approved-release-request",
    )

    assert updated["accepted"] is True
    assert updated["template"]["version"] == "0.1.1"
    assert updated["template"]["publish_status"] == "published"
    assert updated["template"]["release_approval_request_id"] == "approved-release-request"
    assert updated["template_package"]["product_status"] == "manual_check"
    assert updated["template_package"]["manual_checks"] == [
        "Template acceptance references require manual review before signoff."
    ]
    assert updated["revision"]["revision_type"] == "askme.customer_project_template_revision"
    assert Path(updated["revision"]["revision_path"]).exists()

    payload = list_customer_project_templates(template_root)
    factory = next(
        item for item in payload["templates"] if item["template_id"] == "factory-inspection"
    )
    assert factory["template_package"]["version"] == "0.1.1"
    assert factory["template_package"]["publish_status"] == "published"
    assert factory["template_package"]["product_status"] == "manual_check"
    checklist_by_id = {item["step_id"]: item for item in factory["delivery_checklist"]}
    assert checklist_by_id["review_template_release"]["status"] == "ready"

    history = list_customer_project_template_revisions(template_root, "factory-inspection")
    assert history["found"] is True
    assert history["revision_count"] == 1
    assert history["revisions"][0]["operator_id"] == "product.reviewer"
    assert history["revisions"][0]["template_release"]["publish_status"] == "pilot"
    assert history["revisions"][0]["template_release"]["version"] == "0.1.0"

    invalid_status = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {"publish_status": "ship-it"},
        operator_id="product.owner",
    )
    assert invalid_status["accepted"] is False
    assert invalid_status["reason"] == "invalid_publish_status"

    invalid_version = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {"version": "next", "publish_status": "published"},
        operator_id="product.owner",
    )
    assert invalid_version["accepted"] is False
    assert invalid_version["reason"] == "invalid_template_version"


def test_customer_project_template_release_request_requires_second_approver(tmp_path: Path) -> None:
    template_root = tmp_path / "templates"
    shutil.copytree(Path("deploy/customer-project-templates"), template_root)

    request = create_customer_project_template_release_request(
        template_root,
        "factory-inspection",
        {
            "version": "0.1.1",
            "publish_status": "published",
            "release_channel": "stable",
        },
        operator_id="product.owner",
        reason="Request customer-visible factory template release.",
    )

    assert request["accepted"] is True
    request_id = request["request"]["request_id"]
    assert request["request"]["status"] == "pending"
    assert request["request"]["release"]["publish_status"] == "published"
    still_pilot = list_customer_project_templates(template_root)
    factory_before_review = next(
        item for item in still_pilot["templates"] if item["template_id"] == "factory-inspection"
    )
    assert factory_before_review["template_package"]["publish_status"] == "pilot"

    pending = list_customer_project_template_release_requests(template_root, status="pending")
    assert pending["summary"]["pending_count"] == 1
    assert pending["requests"][0]["request_id"] == request_id

    self_approval = review_customer_project_template_release_request(
        template_root,
        request_id,
        decision="approve",
        operator_id="product.owner",
        reason="Self approval must be rejected.",
    )
    assert self_approval["accepted"] is False
    assert self_approval["reason"] == "release_request_requires_second_approver"

    approved = review_customer_project_template_release_request(
        template_root,
        request_id,
        decision="approve",
        operator_id="product.reviewer",
        reason="Second product owner approval.",
    )
    assert approved["accepted"] is True
    assert approved["request"]["status"] == "approved"
    assert approved["request"]["reviewed_by"] == "product.reviewer"
    assert approved["release_result"]["template"]["publish_status"] == "published"

    payload = list_customer_project_templates(template_root)
    factory_after_review = next(
        item for item in payload["templates"] if item["template_id"] == "factory-inspection"
    )
    assert factory_after_review["template_package"]["version"] == "0.1.1"
    assert factory_after_review["template_package"]["publish_status"] == "published"
    assert approved["release_result"]["template"]["release_approval_request_id"] == request_id

    omitted_status_direct_write = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {"release_note": "Try to edit an already-published template without approval."},
        operator_id="product.owner",
        reason="Direct edit should not bypass the approval gate.",
    )
    assert omitted_status_direct_write["accepted"] is False
    assert omitted_status_direct_write["reason"] == "published_release_requires_approval_request"

    approved_request_storage = json.loads(
        Path(approved["request"]["request_path"]).read_text(encoding="utf-8")
    )
    assert approved_request_storage["status"] == "approved"
    assert "request_path" not in approved_request_storage
    assert (
        list_customer_project_template_release_requests(template_root)["summary"]["applying_count"]
        == 0
    )

    notes = customer_project_template_release_notes(template_root)
    assert notes["summary"]["approved_release_count"] == 1
    assert notes["notes"][0]["template_id"] == "factory-inspection"
    assert notes["notes"][0]["version"] == "0.1.1"
    assert notes["notes"][0]["approved_by"] == "product.reviewer"
    assert "现场验收证据" in notes["notes"][0]["customer_claim"]
    assert notes["notes"][0]["applicability_scope"]["industries"] == ["manufacturing"]
    assert notes["notes"][0]["scenario_acceptance_criteria"]
    assert notes["notes"][0]["dependency_matrix"]
    assert notes["notes"][0]["delivery_summary"]["default_objects"]

    bundle = export_customer_project_template_release_notes_bundle(
        template_root,
        customer_context={
            "customer_name": "ACME Factory",
            "project_name": "Robot Patrol Pilot",
            "site_name": "Line One",
        },
    )
    assert bundle["accepted"] is True
    assert bundle["bundle"]["bundle_schema"] == "askme.template_release_notes_bundle.v1"
    assert bundle["bundle"]["customer_context"]["customer_name"] == "ACME Factory"
    assert bundle["bundle"]["release_note_count"] == 1
    assert bundle["bundle"]["proposal_insert"]["section_title"] == (
        "Robot Patrol Pilot approved reusable capabilities"
    )
    assert (
        bundle["bundle"]["proposal_insert"]["scenario_coverage"][0]["acceptance_criteria_count"]
        >= 1
    )
    assert bundle["bundle"]["proposal_insert"]["dependency_summary"]["dependency_count"] >= 1
    assert (
        "approved reusable robot-service template"
        in (bundle["bundle"]["proposal_insert"]["safe_claims"][0])
    )
    assert bundle["bundle"]["manifest"]["release_note_count"] == 1
    assert (
        bundle["bundle"]["files"]["json_filename"]
        == "robot-patrol-pilot-template-release-notes.json"
    )
    assert "ACME Factory" in bundle["bundle"]["html"]
    assert "approved reusable capabilities" in bundle["bundle"]["html"]
    assert "factory-inspection" in bundle["bundle"]["html"]
    assert bundle["bundle"]["manifest"]["bundle_sha256"]

    second_review = review_customer_project_template_release_request(
        template_root,
        request_id,
        decision="approve",
        operator_id="product.reviewer",
    )
    assert second_review["accepted"] is False
    assert second_review["reason"] == "release_request_not_pending"


def test_customer_project_template_release_request_records_apply_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from askme.pipeline.field import customer_project_template_release as release_module

    template_root = tmp_path / "templates"
    shutil.copytree(Path("deploy/customer-project-templates"), template_root)

    rejected_request = create_customer_project_template_release_request(
        template_root,
        "factory-inspection",
        {
            "version": "0.1.1",
            "publish_status": "published",
            "release_channel": "stable",
        },
        operator_id="product.owner",
        reason="Exercise failed release apply path.",
    )
    time.sleep(0.01)
    exception_request = create_customer_project_template_release_request(
        template_root,
        "factory-inspection",
        {
            "version": "0.1.2",
            "publish_status": "published",
            "release_channel": "stable",
        },
        operator_id="product.owner",
        reason="Exercise exception release apply path.",
    )
    assert rejected_request["accepted"] is True
    assert exception_request["accepted"] is True

    def reject_update(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"accepted": False, "reason": "simulated_apply_denied"}

    monkeypatch.setattr(release_module, "update_customer_project_template_release", reject_update)
    rejected = review_customer_project_template_release_request(
        template_root,
        rejected_request["request"]["request_id"],
        decision="approve",
        operator_id="product.reviewer",
        reason="Second product owner approval.",
    )
    assert rejected["accepted"] is False
    assert rejected["reason"] == "simulated_apply_denied"
    assert rejected["request"]["status"] == "apply_failed"
    assert rejected["request"]["apply_failure_reason"] == "simulated_apply_denied"

    rejected_storage = json.loads(
        Path(rejected["request"]["request_path"]).read_text(encoding="utf-8")
    )
    assert rejected_storage["status"] == "apply_failed"
    assert rejected_storage["apply_failure_reason"] == "simulated_apply_denied"

    retry = review_customer_project_template_release_request(
        template_root,
        rejected_request["request"]["request_id"],
        decision="approve",
        operator_id="product.reviewer",
    )
    assert retry["accepted"] is False
    assert retry["reason"] == "release_request_not_pending"
    assert retry["request"]["status"] == "apply_failed"

    def raise_update(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated apply crash")

    monkeypatch.setattr(release_module, "update_customer_project_template_release", raise_update)
    crashed = review_customer_project_template_release_request(
        template_root,
        exception_request["request"]["request_id"],
        decision="approve",
        operator_id="product.reviewer",
        reason="Second product owner approval.",
    )
    assert crashed["accepted"] is False
    assert crashed["reason"] == "release_apply_exception"
    assert crashed["request"]["status"] == "apply_failed"
    assert crashed["request"]["apply_failure_reason"] == "simulated apply crash"
    assert crashed["release_result"]["error"] == "simulated apply crash"

    crashed_storage = json.loads(
        Path(crashed["request"]["request_path"]).read_text(encoding="utf-8")
    )
    assert crashed_storage["status"] == "apply_failed"
    assert crashed_storage["apply_failure_reason"] == "simulated apply crash"
    assert (
        list_customer_project_template_release_requests(template_root)["summary"][
            "apply_failed_count"
        ]
        == 2
    )


def test_customer_project_template_create_update_export_import_and_archive(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    result = create_customer_project_from_template(
        template_root=Path("deploy/customer-project-templates"),
        profile_root=profile_root,
        template_id="factory-inspection",
        customer={
            "tenant_id": "tenant-acme",
            "delivery_namespace": "pilot",
            "customer_id": "acme-factory",
            "customer_name": "ACME Factory",
            "industry": "manufacturing",
            "project_id": "line-one",
            "project_name": "Line One Inspection",
        },
        site={"site_id": "acme-site", "name": "ACME Site"},
    )
    assert result["accepted"] is True
    assert Path(result["profile_path"]).exists()
    assert "tenant-acme" in result["profile_path"]
    assert "pilot" in result["profile_path"]
    assert result["implementation_handoff"]["handoff_schema"] == (
        "askme.customer_project_implementation_handoff.v1"
    )
    assert result["implementation_handoff"]["project_id"] == "line-one"
    assert result["implementation_handoff"]["status"] in {
        "needs_object_binding",
        "ready_for_acceptance_evidence",
    }
    assert result["implementation_handoff"]["summary"]["object_count"] >= 2
    assert result["implementation_handoff"]["next_steps"][1]["label"] == "补齐对象能力绑定"

    detail = get_customer_project_profile(profile_root, "line-one")
    assert detail["found"] is True
    assert detail["customer"]["tenant_id"] == "tenant-acme"
    assert detail["customer"]["delivery_namespace"] == "pilot"
    assert detail["customer"]["customer_id"] == "acme-factory"
    assert detail["managed_objects"]["object_type_count"] >= 2
    assert detail["implementation_handoff"]["handoff_schema"] == (
        "askme.customer_project_implementation_handoff.v1"
    )
    assert detail["implementation_handoff"]["project_id"] == "line-one"
    assert detail["next_step"] == detail["implementation_handoff"]["customer_status"]

    updated = upsert_managed_object(
        profile_root,
        "line-one",
        "custom_gate",
        {
            "display_name": "Custom gate",
            "category": "access",
            "object_labels": ["gate"],
            "scenario_ids": ["gate_inspection"],
            "zone_types": ["main_channel"],
            "device_sources": ["camera"],
            "responder_group": "operations",
            "evidence_required": ["photo", "location"],
            "bindings": {
                "vision_models": ["gate-detection"],
                "sensor_protocols": ["camera-detection-json"],
                "skill_packages": ["capability.inspect_gate"],
                "acceptance_tests": [
                    "tests/scenario_tests/test_field_operations_evaluation.py::gate_inspection"
                ],
            },
        },
        operator_id="delivery.manager",
        reason="Add custom gate for customer pilot scope.",
    )
    assert updated["accepted"] is True
    assert updated["managed_object"]["bindings"]["skill_packages"] == ["capability.inspect_gate"]
    assert updated["managed_object"]["acceptance_status"]["status"] == "manual_check"
    assert updated["managed_object"]["acceptance_status"]["acceptance_checks"][0]["status"] == (
        "node_unresolved"
    )
    assert updated["object_change"]["action"] == "created"
    assert updated["object_change"]["operator_id"] == "delivery.manager"
    assert updated["implementation_handoff"]["handoff_schema"] == (
        "askme.customer_project_implementation_handoff.v1"
    )
    assert updated["implementation_handoff"]["project_id"] == "line-one"
    assert updated["implementation_handoff"]["summary"]["object_count"] >= 3
    assert updated["next_step"] == updated["implementation_handoff"]["customer_status"]
    detail_after_update = get_customer_project_profile(profile_root, "line-one")
    assert detail_after_update["object_change_log"][-1]["object_id"] == "custom_gate"
    assert detail_after_update["object_change_log"][-1]["action"] == "created"

    exported = export_customer_project_package(
        profile_root, "line-one", output_root=tmp_path / "packages"
    )
    assert exported["accepted"] is True
    assert exported["package"]["package_type"] == "askme.customer_project"
    assert exported["package"]["manifest"]["payload_sha256"]
    assert exported["package"]["manifest"]["profile_sha256"]
    assert exported["package"]["customer"]["tenant_id"] == "tenant-acme"
    assert exported["package"]["customer"]["delivery_namespace"] == "pilot"
    assert exported["package"]["manifest"]["tenant_id"] == "tenant-acme"
    assert exported["package"]["manifest"]["delivery_namespace"] == "pilot"
    assert exported["package"]["manifest"]["customer_id"] == "acme-factory"
    assert exported["package"]["acceptance_summary"]["overall_status"] == "manual_check"
    assert exported["package"]["manifest"]["acceptance_overall_status"] == "manual_check"
    assert exported["package"]["manifest"]["acceptance_manual_check_object_count"] >= 1
    assert exported["package"]["binding_readiness_summary"]["overall_status"] == "manual_check"
    assert exported["package"]["manifest"]["resource_binding_overall_status"] == "manual_check"
    assert (
        exported["package"]["manifest"]["resource_binding_ready_object_count"]
        == (exported["package"]["binding_readiness_summary"]["ready_object_count"])
    )
    assert (
        exported["package"]["manifest"]["resource_binding_manual_check_object_count"]
        == (exported["package"]["binding_readiness_summary"]["manual_check_object_count"])
    )
    assert (
        exported["package"]["manifest"]["resource_binding_blocked_object_count"]
        == (exported["package"]["binding_readiness_summary"]["blocked_object_count"])
    )
    assert exported["package"]["manifest"]["resource_binding_manual_check_object_count"] >= 1
    assert exported["package"]["manifest"]["resource_binding_unregistered_resource_count"] == 0
    assert exported["package"]["manifest"]["delivery_resource_count"] >= 10
    assert exported["package"]["reuse_assessment"]["status"] == "manual_check"
    assert exported["package"]["reuse_assessment"]["manual_check_count"] >= 1
    assert (
        exported["package"]["deployment_dependencies"]["binding_readiness"]["overall_status"]
        == "manual_check"
    )
    assert exported["package"]["deployment_dependencies"]["device_count"] >= 3
    assert exported["package"]["deployment_dependencies"]["missing_env_count"] >= 1
    assert exported["package"]["manifest"]["reuse_status"] == "manual_check"
    assert exported["package"]["manifest"]["dependency_missing_env_count"] >= 1
    assert exported["package"]["applicability_scope"]["industries"] == ["manufacturing"]
    assert exported["package"]["out_of_scope"]
    assert exported["package"]["customer_prerequisites"]
    assert exported["package"]["scenario_acceptance_criteria"]
    assert exported["package"]["dependency_matrix"]
    assert exported["package"]["manifest"]["customer_prerequisite_count"] == len(
        exported["package"]["customer_prerequisites"]
    )
    assert exported["package"]["manifest"]["scenario_acceptance_criteria_count"] == len(
        exported["package"]["scenario_acceptance_criteria"]
    )
    assert exported["package"]["manifest"]["dependency_matrix_count"] == len(
        exported["package"]["dependency_matrix"]
    )
    assert (
        exported["package"]["managed_object_action_plan"]["overall_status"]
        == "manual_check_required"
    )
    assert (
        exported["package"]["package_delivery_gate"]["delivery_gate_status"]
        == "manual_check_required"
    )
    assert exported["package"]["package_delivery_gate"]["export_allowed"] is True
    assert exported["package"]["package_delivery_gate"]["import_allowed"] is True
    assert exported["package"]["package_delivery_gate"]["customer_handoff_ready"] is False
    assert (
        exported["package"]["manifest"]["package_delivery_gate_status"] == "manual_check_required"
    )
    assert exported["package"]["manifest"]["package_delivery_import_allowed"] is True
    assert exported["package"]["manifest"]["package_delivery_customer_handoff_ready"] is False
    verification = verify_customer_project_package(exported["package"])
    assert verification["valid"] is True
    assert verification["delivery_gate_status"] == "manual_check_required"
    assert verification["import_allowed"] is True
    assert Path(exported["package_path"]).exists()

    imported_root = tmp_path / "imported"
    dry_run = import_customer_project_package(imported_root, exported["package"], dry_run=True)
    assert dry_run["accepted"] is True
    assert dry_run["dry_run"] is True
    assert dry_run["diff"]["change_type"] == "create"
    assert dry_run["diff"]["incoming_acceptance_summary"]["overall_status"] == "manual_check"
    assert dry_run["diff"]["incoming_binding_readiness_summary"]["overall_status"] == "manual_check"
    assert (
        dry_run["diff"]["incoming_binding_readiness_summary"]["ready_object_count"]
        == (exported["package"]["binding_readiness_summary"]["ready_object_count"])
    )
    assert dry_run["diff"]["incoming_binding_readiness_summary"]["manual_check_object_count"] >= 1
    assert dry_run["diff"]["incoming_binding_readiness_summary"]["blocked_object_count"] == 0
    assert dry_run["diff"]["incoming_binding_readiness_summary"]["unregistered_resource_count"] == 0
    assert dry_run["diff"]["incoming_reuse_assessment"]["status"] == "manual_check"
    assert (
        dry_run["diff"]["incoming_delivery_gate"]["delivery_gate_status"] == "manual_check_required"
    )
    assert dry_run["diff"]["incoming_delivery_gate"]["import_allowed"] is True
    assert dry_run["import_gate_result"] == "accepted_with_manual_check"
    assert dry_run["would_write"] is True
    assert dry_run["implementation_handoff"]["handoff_schema"] == (
        "askme.customer_project_implementation_handoff.v1"
    )
    assert dry_run["implementation_handoff"]["project_id"] == "line-one"
    assert dry_run["implementation_handoff"]["summary"]["object_count"] >= 2
    imported = import_customer_project_package(imported_root, exported["package"])
    assert imported["accepted"] is True
    assert imported["import_gate_result"] == "accepted_with_manual_check"
    assert imported["implementation_handoff"]["project_id"] == "line-one"
    assert imported["implementation_handoff"]["next_steps"][2]["label"] == "登记现场验收证据"
    assert "tenant-acme" in imported["profile_path"]
    assert "pilot" in imported["profile_path"]
    assert get_customer_project_profile(imported_root, "line-one")["found"] is True

    rejected_delete = delete_managed_object(profile_root, "line-one", "custom_gate")
    assert rejected_delete["accepted"] is False
    assert rejected_delete["reason"] == "delete_reason_required"

    deleted = delete_managed_object(
        profile_root,
        "line-one",
        "custom_gate",
        operator_id="delivery.manager",
        reason="Customer removed this gate from the onsite scope.",
    )
    assert deleted["accepted"] is True
    assert deleted["offline_reason"] == "Customer removed this gate from the onsite scope."
    assert deleted["deleted_object"]["display_name"] == "Custom gate"
    assert deleted["object_change"]["action"] == "offline"
    assert deleted["object_change"]["operator_id"] == "delivery.manager"
    assert deleted["implementation_handoff"]["handoff_schema"] == (
        "askme.customer_project_implementation_handoff.v1"
    )
    assert deleted["implementation_handoff"]["project_id"] == "line-one"
    assert deleted["implementation_handoff"]["summary"]["object_count"] >= 2
    assert deleted["next_step"] == deleted["implementation_handoff"]["customer_status"]
    assert (
        "custom_gate"
        not in get_customer_project_profile(profile_root, "line-one")["managed_objects"][
            "objects_by_id"
        ]
    )
    after_delete = get_customer_project_profile(profile_root, "line-one")
    assert [item["action"] for item in after_delete["object_change_log"][-2:]] == [
        "created",
        "offline",
    ]
    catalog_after_delete = build_customer_project_catalog(profile_root)
    assert catalog_after_delete["projects"][0]["object_change_log"][-1]["action"] == "offline"

    history = list_customer_project_revisions(profile_root, "line-one")
    assert history["found"] is True
    assert history["count"] >= 2
    actions = [item["action"] for item in history["revisions"]]
    assert "managed_object_upsert" in actions
    assert "managed_object_delete" in actions
    restore_revision = next(
        item for item in history["revisions"] if item["action"] == "managed_object_delete"
    )

    dry_rollback = rollback_customer_project_profile(
        profile_root,
        "line-one",
        restore_revision["revision_id"],
        operator_id="delivery.manager",
        reason="Preview restore deleted object.",
        dry_run=True,
    )
    assert dry_rollback["accepted"] is True
    assert dry_rollback["dry_run"] is True
    assert dry_rollback["would_write"] is True
    assert dry_rollback["field_changes"]

    rollback = rollback_customer_project_profile(
        profile_root,
        "line-one",
        restore_revision["revision_id"],
        operator_id="delivery.manager",
        reason="Restore object removed by mistake.",
    )
    assert rollback["accepted"] is True
    assert rollback["rollback_snapshot"]["action"] == "rollback_current"
    restored_detail = get_customer_project_profile(profile_root, "line-one")
    assert "custom_gate" in restored_detail["managed_objects"]["objects_by_id"]
    assert list_customer_project_revisions(profile_root, "line-one")["revisions"][0]["action"] == (
        "rollback_current"
    )

    archived = archive_customer_project_profile(
        profile_root, "line-one", archive_root=tmp_path / "archive"
    )
    assert archived["accepted"] is True
    assert Path(archived["archived_path"]).exists()
    assert get_customer_project_profile(profile_root, "line-one")["found"] is False


def test_customer_project_package_blocks_import_when_delivery_gate_is_blocked(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profiles"
    created = create_customer_project_from_template(
        template_root=Path("deploy/customer-project-templates"),
        profile_root=profile_root,
        template_id="factory-inspection",
        customer={
            "tenant_id": "tenant-acme",
            "delivery_namespace": "pilot",
            "customer_id": "acme-factory",
            "customer_name": "ACME Factory",
            "industry": "manufacturing",
            "project_id": "blocked-line",
            "project_name": "Blocked Line Inspection",
        },
        site={"site_id": "blocked-site", "name": "Blocked Site"},
    )
    assert created["accepted"] is True
    updated = upsert_managed_object(
        profile_root,
        "blocked-line",
        "bad_gate",
        {
            "display_name": "Bad gate",
            "category": "access",
            "object_labels": ["gate"],
            "scenario_ids": ["gate_inspection"],
            "zone_types": ["main_channel"],
            "device_sources": ["camera"],
            "responder_group": "operations",
            "evidence_required": ["photo", "location"],
            "bindings": {
                "vision_models": ["missing-gate-model"],
                "sensor_protocols": ["camera-detection-json"],
                "skill_packages": ["capability.inspect_gate"],
                "acceptance_tests": [
                    "tests/scenario_tests/missing_customer_acceptance.py::bad_gate"
                ],
            },
        },
        operator_id="delivery.manager",
        reason="Add intentionally blocked object for package gate regression.",
    )
    assert updated["accepted"] is True
    assert updated["managed_object"]["acceptance_status"]["status"] == "blocked"

    exported = export_customer_project_package(
        profile_root,
        "blocked-line",
        output_root=tmp_path / "packages",
    )
    assert exported["accepted"] is True
    package = exported["package"]
    assert package["managed_object_action_plan"]["overall_status"] == "blocked"
    assert package["managed_object_action_plan"]["blocked_action_count"] >= 1
    assert package["package_delivery_gate"]["delivery_gate_status"] == "blocked"
    assert package["package_delivery_gate"]["export_allowed"] is False
    assert package["package_delivery_gate"]["import_allowed"] is False
    assert package["package_delivery_gate"]["customer_handoff_ready"] is False
    assert package["manifest"]["package_delivery_gate_status"] == "blocked"
    assert package["manifest"]["package_delivery_import_allowed"] is False
    assert package["manifest"]["package_delivery_blocked_action_count"] >= 1

    verification = verify_customer_project_package(package)
    assert verification["valid"] is True
    assert verification["delivery_gate_status"] == "blocked"
    assert verification["import_allowed"] is False
    assert any(
        reason["reason_code"] == "acceptance_test_blocked"
        for reason in verification["delivery_gate"]["delivery_gate_reasons"]
    )

    imported_root = tmp_path / "imported"
    dry_run = import_customer_project_package(imported_root, package, dry_run=True)
    assert dry_run["accepted"] is True
    assert dry_run["dry_run"] is True
    assert dry_run["import_gate_result"] == "rejected"
    assert dry_run["would_write"] is False
    assert dry_run["diff"]["incoming_delivery_gate"]["delivery_gate_status"] == "blocked"

    imported = import_customer_project_package(imported_root, package)
    assert imported["accepted"] is False
    assert imported["reason"] == "package_delivery_gate_blocked"
    assert imported["import_gate_result"] == "rejected"
    assert imported["delivery_gate"]["delivery_gate_status"] == "blocked"
    assert get_customer_project_profile(imported_root, "blocked-line")["found"] is False


def test_customer_project_profile_upsert_updates_metadata_without_losing_objects(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profiles"
    created = create_customer_project_from_template(
        template_root=Path("deploy/customer-project-templates"),
        profile_root=profile_root,
        template_id="factory-inspection",
        customer={
            "tenant_id": "tenant-acme",
            "delivery_namespace": "pilot",
            "customer_id": "acme-factory",
            "customer_name": "ACME Factory",
            "industry": "manufacturing",
            "project_id": "line-one",
            "project_name": "Line One Inspection",
        },
        site={"site_id": "acme-site", "name": "ACME Site"},
    )
    assert created["accepted"] is True
    detail = get_customer_project_profile(profile_root, "line-one")
    profile = detail["profile"]
    object_count = detail["managed_objects"]["object_type_count"]

    profile["customer"]["customer_name"] = "ACME Factory Customer"
    profile["customer"]["project_name"] = "Line One Pilot"
    profile["customer"]["object_scope_note"] = (
        "Customer-visible metadata can change without losing objects."
    )
    profile["site"]["name"] = "ACME Factory Site"

    updated = upsert_customer_project_profile(profile_root, profile)

    assert updated["accepted"] is True
    assert "tenant-acme" in updated["profile_path"]
    assert "pilot" in updated["profile_path"]
    assert updated["implementation_handoff"]["project_id"] == "line-one"
    assert updated["implementation_handoff"]["summary"]["object_count"] == object_count
    assert updated["implementation_handoff"]["next_steps"][0]["label"] == "核对项目基础信息"
    after = get_customer_project_profile(profile_root, "line-one")
    assert after["customer"]["customer_name"] == "ACME Factory Customer"
    assert after["customer"]["project_name"] == "Line One Pilot"
    assert after["site"]["name"] == "ACME Factory Site"
    assert after["managed_objects"]["object_type_count"] == object_count
    assert after["managed_objects"]["objects_by_id"]


def test_customer_project_package_import_isolates_same_project_by_delivery_namespace(
    tmp_path: Path,
) -> None:
    packages = {}
    for namespace in ("pilot", "production"):
        source_root = tmp_path / f"source-{namespace}"
        created = create_customer_project_from_template(
            template_root=Path("deploy/customer-project-templates"),
            profile_root=source_root,
            template_id="factory-inspection",
            customer={
                "tenant_id": "tenant-a",
                "delivery_namespace": namespace,
                "customer_id": "acme-factory",
                "customer_name": "ACME Factory",
                "industry": "manufacturing",
                "project_id": "line-one",
                "project_name": "Line One Inspection",
            },
            site={"site_id": "acme-site", "name": "ACME Site"},
        )
        assert created["accepted"] is True
        exported = export_customer_project_package(
            source_root,
            "line-one",
            output_root=tmp_path / "packages",
        )
        assert exported["accepted"] is True
        packages[namespace] = exported["package"]

    imported_root = tmp_path / "imported"
    pilot_import = import_customer_project_package(imported_root, packages["pilot"])
    production_dry_run = import_customer_project_package(
        imported_root, packages["production"], dry_run=True
    )
    production_import = import_customer_project_package(imported_root, packages["production"])

    assert pilot_import["accepted"] is True
    assert production_dry_run["accepted"] is True
    assert production_import["accepted"] is True
    assert pilot_import["diff"]["change_type"] == "create"
    assert production_dry_run["diff"]["change_type"] == "create"
    assert production_dry_run["diff"]["collision_candidates"]
    assert pilot_import["profile_path"] != production_import["profile_path"]
    assert Path(pilot_import["profile_path"]).exists()
    assert Path(production_import["profile_path"]).exists()

    pilot_profile = load_field_site_profile(Path(pilot_import["profile_path"]))
    production_profile = load_field_site_profile(Path(production_import["profile_path"]))
    assert pilot_profile["customer"]["delivery_namespace"] == "pilot"
    assert production_profile["customer"]["delivery_namespace"] == "production"
    assert (
        pilot_profile["customer"]["project_id"]
        == production_profile["customer"]["project_id"]
        == "line-one"
    )


def test_customer_project_package_rejects_manifest_scope_tamper(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    created = create_customer_project_from_template(
        template_root=Path("deploy/customer-project-templates"),
        profile_root=source_root,
        template_id="factory-inspection",
        customer={
            "tenant_id": "tenant-a",
            "delivery_namespace": "pilot",
            "customer_id": "acme-factory",
            "customer_name": "ACME Factory",
            "industry": "manufacturing",
            "project_id": "line-one",
            "project_name": "Line One Inspection",
        },
        site={"site_id": "acme-site", "name": "ACME Site"},
    )
    assert created["accepted"] is True
    exported = export_customer_project_package(
        source_root, "line-one", output_root=tmp_path / "packages"
    )
    assert exported["accepted"] is True

    tampered = json.loads(json.dumps(exported["package"]))
    tampered["manifest"]["tenant_id"] = "tenant-b"
    tampered["manifest"]["delivery_namespace"] = "production"
    tampered["manifest"]["reuse_status"] = "ready"
    tampered["manifest"]["resource_binding_overall_status"] = "blocked"
    tampered["manifest"]["delivery_resource_count"] = 999
    tampered["manifest"]["scenario_acceptance_criteria_count"] = 999

    verification = verify_customer_project_package(tampered)

    assert verification["valid"] is False
    assert verification["reason"] == "integrity_errors"
    assert "manifest.tenant_id mismatch" in verification["errors"]
    assert "manifest.delivery_namespace mismatch" in verification["errors"]
    assert "manifest.reuse_status mismatch" in verification["errors"]
    assert "manifest.resource_binding_overall_status mismatch" in verification["errors"]
    assert "manifest.delivery_resource_count mismatch" in verification["errors"]
    assert "manifest.scenario_acceptance_criteria_count mismatch" in verification["errors"]


@pytest.mark.asyncio
async def test_field_operations_service_loads_site_profile_for_real_ingest(tmp_path: Path) -> None:
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "site_profile_path": "deploy/site-profiles/park-demo.yaml",
        }
    )

    result = await service.ingest_payload(
        {
            "source": "camera",
            "observed_at": time.time(),
            "zone_id": "main-road-1",
            "detections": [{"label": "vehicle", "confidence": 0.92}],
            "duration_s": 180,
            "image_path": "artifacts/evidence/car.jpg",
        }
    )

    assert result["accepted"] is True
    assert result["normalized"]["location"] == "B区主通道"
    assert result["event"]["incident_topic"] == "traffic.illegal_parking"
    assert result["event"]["customer_id"] == "demo-customer"
    assert result["event"]["project_id"] == "demo-field-ops"
    assert result["event"]["site_id"] == "inovx-demo-park"
    assert result["event"]["managed_object_id"] == "vehicles"
    assert result["event"]["managed_object_bindings"]["skill_packages"] == [
        "capability.detect_illegal_parking"
    ]
    execution = result["event"]["resource_execution_context"]
    assert execution["overall_status"] == "ready"
    assert execution["managed_object_id"] == "vehicles"
    assert execution["source"] == "camera"
    assert execution["selected_skill_package"] == "capability.detect_illegal_parking"
    assert execution["selected_capability"] == "detect_illegal_parking"
    assert execution["approval_required"] is True
    assert execution["output_contract"] == "field_event"
    assert execution["capability_routes"][0]["tool"] == "field_event_trigger"
    assert execution["capability_routes"][0]["installed_contract"] is True
    assert execution["capability_routes"][0]["safety_level"] == "dangerous"
    assert execution["capability_routes"][0]["confirm_before_execute"] is True
    assert "image_path" in execution["capability_routes"][0]["required_inputs"]
    assert execution["ingest_endpoint"] == "/api/field/ingest"
    assert execution["runtime_callback_endpoint"] == "/api/field/events/{event_id}/runtime-delivery"
    listed = service.list_payload(project_id="demo-field-ops", managed_object_id="vehicles")
    assert listed["filtered_total"] == 1
    assert listed["summary"]["by_project"]["demo-field-ops"] == 1
    assert listed["summary"]["by_managed_object"]["vehicles"] == 1
    assert result["event"]["location"] == "B区主通道"


def test_field_operations_service_rejects_invalid_site_profile(tmp_path: Path) -> None:
    profile = tmp_path / "bad-site.yaml"
    profile.write_text(
        "site:\n  site_id: bad\nzones: {}\nresponder_groups: {}\ndevices: {}\nthresholds: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field site profile validation failed"):
        FieldOperationsService(config={"site_profile_path": str(profile)})


def test_field_operations_service_can_surface_site_profile_env_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "site_profile_path": "deploy/site-profiles/park-demo.yaml",
            "site_profile_check_env": True,
        }
    )

    payload = service.readiness_payload()

    assert any(
        item.startswith("field site profile: responder_groups.security.webhook_env")
        for item in payload["warnings"]
    )
    assert (
        "Set site profile environment variables for DingTalk responders and field devices"
        in payload["next_actions"]
    )
