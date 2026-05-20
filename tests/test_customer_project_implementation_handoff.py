from __future__ import annotations

from pathlib import Path

from askme.pipeline.field.customer_project_implementation_handoff import (
    _customer_project_implementation_handoff,
)


def _profile(managed_objects: dict[str, object]) -> dict[str, object]:
    return {
        "customer": {
            "project_id": "line-one",
            "project_name": "Line One Inspection",
            "customer_id": "acme",
            "customer_name": "ACME",
        },
        "site": {
            "site_id": "site-a",
            "name": "Site A",
        },
        "managed_objects": managed_objects,
    }


def test_customer_project_implementation_handoff_reports_missing_bindings() -> None:
    handoff = _customer_project_implementation_handoff(
        _profile(
            {
                "boiler": {
                    "display_name": "Boiler",
                    "category": "equipment",
                    "bindings": {
                        "vision_models": ["thermal-detector"],
                    },
                }
            }
        ),
        template_id="factory-inspection",
        profile_path=Path("profiles/line-one.yaml"),
    )

    assert handoff["handoff_schema"] == "askme.customer_project_implementation_handoff.v1"
    assert handoff["status"] == "needs_object_binding"
    assert handoff["customer_status"] == "项目已创建，待补齐现场对象能力绑定。"
    assert handoff["summary"] == {
        "object_count": 1,
        "object_ready_count": 0,
        "object_needs_binding_count": 1,
    }
    assert handoff["object_binding_todo"][0]["missing_binding_types"] == [
        "sensor_protocols",
        "skill_packages",
        "acceptance_tests",
    ]
    assert handoff["object_binding_todo"][0]["missing_binding_labels"] == [
        "传感器协议",
        "能力包",
        "验收项",
    ]


def test_customer_project_implementation_handoff_accepts_scalar_bindings_and_skips_noise() -> None:
    handoff = _customer_project_implementation_handoff(
        _profile(
            {
                "gate": {
                    "display_name": "Gate",
                    "category": "access",
                    "bindings": {
                        "vision_models": "gate-detector",
                        "sensor_protocols": ("camera-json",),
                        "skill_packages": ["capability.inspect_gate"],
                        "acceptance_tests": "tests/test_gate.py::test_gate",
                    },
                },
                "noise": "not-a-managed-object",
            }
        ),
        template_id="factory-inspection",
        profile_path="profiles/line-one.yaml",
    )

    assert handoff["status"] == "ready_for_acceptance_evidence"
    assert handoff["customer_status"] == "项目已创建，对象能力绑定已齐，待登记现场验收证据。"
    assert handoff["summary"] == {
        "object_count": 1,
        "object_ready_count": 1,
        "object_needs_binding_count": 0,
    }
    assert handoff["object_binding_todo"][0]["ready_for_site_acceptance"] is True
    assert handoff["object_binding_todo"][0]["missing_binding_types"] == []
    assert handoff["next_steps"][1]["status"] == "ready"


def test_customer_project_implementation_handoff_sorts_and_limits_object_todo() -> None:
    managed_objects = {
        f"object_{index:03d}": {
            "display_name": f"Object {index:03d}",
            "category": "asset",
            "bindings": {},
        }
        for index in range(55)
    }

    handoff = _customer_project_implementation_handoff(
        _profile(managed_objects),
        template_id="factory-inspection",
        profile_path="profiles/line-one.yaml",
    )

    assert handoff["summary"]["object_count"] == 55
    assert len(handoff["object_binding_todo"]) == 50
    assert handoff["object_binding_todo"][0]["object_id"] == "object_000"
    assert handoff["object_binding_todo"][-1]["object_id"] == "object_049"
