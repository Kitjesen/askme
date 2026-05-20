from __future__ import annotations

from askme.pipeline.field.customer_project_package_assessment import (
    _customer_project_package_acceptance_summary,
    _customer_project_package_reuse_assessment,
    _customer_project_reuse_dependencies,
    _managed_object_acceptance_summary,
    _managed_object_binding_readiness_summary,
)


def test_managed_object_readiness_summaries_are_blocked_when_any_object_blocks() -> None:
    objects = [
        {
            "object_id": "trash-bin-01",
            "resource_binding_status": {
                "overall_status": "blocked",
                "checks": [
                    {
                        "status": "unregistered",
                        "resource_type": "vision_model",
                        "resource_id": "trash-full-detector",
                    }
                ],
            },
            "acceptance_status": {"status": "manual_check"},
        },
        {
            "object_id": "gate-01",
            "resource_binding_status": {"overall_status": "ready", "checks": []},
            "acceptance_status": {"status": "ready"},
        },
    ]

    binding = _managed_object_binding_readiness_summary(objects)
    acceptance = _managed_object_acceptance_summary(objects)

    assert binding["overall_status"] == "blocked"
    assert binding["ready_object_count"] == 1
    assert binding["blocked_object_count"] == 1
    assert binding["unregistered_resource_count"] == 1
    assert acceptance["overall_status"] == "manual_check"
    assert acceptance["manual_check_object_count"] == 1


def test_customer_project_acceptance_summary_lists_blocked_and_manual_objects() -> None:
    summary = _customer_project_package_acceptance_summary(
        {
            "acceptance_summary": {
                "overall_status": "manual_check",
                "ready_object_count": 1,
                "manual_check_object_count": 1,
                "blocked_object_count": 1,
                "object_count": 3,
            },
            "objects": [
                {
                    "object_id": "camera-01",
                    "display_name": "主入口相机",
                    "acceptance_status": {"status": "blocked", "next_step": "绑定验收用例"},
                },
                {
                    "object_id": "trash-bin-01",
                    "acceptance_status": {"status": "manual_check", "next_step": "补现场证据"},
                },
            ],
        }
    )

    assert summary["overall_status"] == "manual_check"
    assert summary["blocked_objects"][0]["object_id"] == "camera-01"
    assert summary["manual_check_objects"][0]["object_id"] == "trash-bin-01"
    assert "不能声明生产上线" in summary["release_claim"]


def test_reuse_assessment_tracks_missing_env_and_dependencies() -> None:
    profile = {
        "devices": {"camera": {"enabled": True}},
        "responder_groups": {"security": {"webhook_env": "ASKME_DINGTALK"}},
    }
    objects = [
        {
            "object_id": "camera-01",
            "device_sources": ["camera"],
            "responder_group": "security",
            "bindings": {
                "vision_models": ["night-stranger"],
                "sensor_protocols": ["rtsp"],
                "skill_packages": ["security-patrol"],
                "acceptance_tests": ["tests/test_security.py::test_night_stranger"],
            },
            "resource_binding_status": {"overall_status": "ready", "checks": []},
        }
    ]
    env_references = [
        {"env_name": "ASKME_DINGTALK", "required": True, "configured": False},
    ]

    dependencies = _customer_project_reuse_dependencies(profile, objects, env_references)
    assessment = _customer_project_package_reuse_assessment(
        profile=profile,
        report={"errors": [], "warnings": []},
        managed_object_catalog={
            "objects": objects,
            "binding_readiness_summary": {"overall_status": "ready"},
        },
        acceptance_summary={"overall_status": "ready"},
        env_references=env_references,
    )

    assert dependencies["device_sources"] == ["camera"]
    assert dependencies["missing_env_count"] == 1
    assert dependencies["vision_models"] == ["night-stranger"]
    assert assessment["status"] == "manual_check"
    assert assessment["manual_check_count"] == 1
