from __future__ import annotations

import pytest

from askme.blueprints import (
    BLUEPRINTS,
    blueprint_delivery_package,
    blueprint_readiness,
    catalog_payload,
    get_blueprint_spec,
    inspect_blueprint,
    list_blueprints,
    load_blueprint_runtime,
)
from askme.runtime.module import Runtime


@pytest.mark.parametrize("spec", BLUEPRINTS)
def test_blueprint_catalog_matches_runtime_modules(spec) -> None:
    inspection = inspect_blueprint(spec.name)

    assert inspection["valid"] is True
    assert inspection["modules"] == list(spec.modules)
    assert inspection["duplicate_modules"] == []
    assert inspection["missing_declared_modules"] == []
    assert inspection["undeclared_runtime_modules"] == []


def test_blueprint_catalog_payload_is_customer_readable() -> None:
    payload = catalog_payload(config={"voice": {}, "perception": {}, "interaction_gate": {}, "cognition": {}, "runtime_handoff": {}})
    edge = next(item for item in payload["items"] if item["name"] == "edge_robot")

    assert payload["summary"]["blueprint_count"] >= 6
    assert payload["summary"]["customer_visible_count"] >= 3
    assert payload["summary"]["valid_count"] == payload["summary"]["blueprint_count"]
    assert "configuration_incomplete_count" in payload["summary"]
    assert edge["title"] == "Park Patrol Robot Runtime"
    assert "visitor wayfinding and escort handoff" in edge["scenarios"]
    assert "runtime_handoff" in edge["modules"]
    assert "LLM and voice layers do not control hardware directly." in edge["safety_boundaries"]
    assert edge["inspection"]["startup_command"] == "python -m askme.blueprints.edge_robot"
    assert edge["readiness"]["status"] == "configuration_incomplete"
    assert "field_operations" in edge["readiness"]["missing_config"]
    assert edge["delivery_package"]["package_id"] == "blueprint.edge_robot"
    assert edge["delivery_package"]["deliverables"]["scenario_acceptance"]


def test_blueprint_readiness_reports_config_and_validation_gates() -> None:
    missing = blueprint_readiness("voice", config={})
    ready = blueprint_readiness(
        "voice",
        config={
            "voice": {"asr": {}, "tts": {}},
            "llm": {},
            "memory": {},
            "dashboard": {},
        },
    )

    assert missing["status"] == "configuration_incomplete"
    assert missing["production_ready"] is False
    assert missing["missing_config"] == ["voice.asr", "voice.tts", "llm", "memory", "dashboard"]
    assert missing["config_evidence"][2]["paths_checked"] == ["llm", "brain"]
    assert ready["status"] == "ready_for_validation"
    assert ready["missing_config"] == []
    assert ready["gates"][0]["gate_id"] == "runtime_composition"
    assert ready["gates"][1]["gate_id"] == "required_config"
    assert ready["gates"][2]["gate_id"] == "external_services"
    assert ready["gates"][3]["gate_id"] == "validation_commands"
    assert "tests/test_voice_loop.py" in " ".join(ready["validation_commands"])


def test_blueprint_readiness_uses_product_config_aliases() -> None:
    voice_ready = blueprint_readiness(
        "voice",
        config={
            "voice": {"asr": {}, "tts": {}},
            "brain": {},
            "memory": {},
            "health_server": {},
        },
    )
    edge_ready = blueprint_readiness(
        "edge_robot",
        config={
            "voice": {},
            "perception": {},
            "field_operations": {
                "dingtalk_webhooks": {"security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}"},
            },
            "runtime_handoff": {},
            "runtime": {"dog_control": {"base_url": "${DOG_CONTROL_SERVICE_URL}"}},
        },
    )

    assert voice_ready["status"] == "ready_for_validation"
    assert voice_ready["missing_config"] == []
    assert voice_ready["config_evidence"][2]["matched_path"] == "brain"
    assert voice_ready["config_evidence"][4]["matched_path"] == "health_server"
    assert edge_ready["status"] == "ready_for_validation"
    assert edge_ready["missing_config"] == []
    assert edge_ready["config_evidence"][4]["matched_path"] == "field_operations.dingtalk_webhooks"
    assert edge_ready["config_evidence"][5]["matched_path"] == "runtime.dog_control"


def test_blueprint_delivery_package_is_actionable_for_customer_pilot() -> None:
    package = blueprint_delivery_package(
        "park",
        config={
            "voice": {},
            "perception": {},
            "field_operations": {
                "dingtalk_webhooks": {"security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}"},
            },
            "runtime_handoff": {},
            "runtime": {"dog_control": {"base_url": "${DOG_CONTROL_SERVICE_URL}"}},
        },
    )

    assert package["status"] == "ready_for_site_validation"
    assert package["release_boundary"].startswith("Can be used for lab or customer pilot")
    assert package["deliverables"]["runtime_composition"]["status"] == "ready"
    assert package["deliverables"]["external_service_checklist"]
    assert any(
        step["step"] == "generate_site_env_template"
        for step in package["handoff_steps"]
    )
    assert any(
        item["customer_scenario"] == "visitor wayfinding and escort handoff"
        for item in package["deliverables"]["scenario_acceptance"]
    )
    assert "Stop if the runtime composition gate fails." in package["stop_conditions"]


def test_blueprint_delivery_package_blocks_missing_config() -> None:
    package = blueprint_delivery_package("voice", config={})

    assert package["status"] == "missing_configuration"
    assert package["deliverables"]["configuration_checklist"][0]["status"] == "missing"
    assert any("voice.asr" in item for item in package["stop_conditions"])


def test_blueprint_aliases_and_runtime_loading() -> None:
    assert get_blueprint_spec("park").name == "edge_robot"
    assert get_blueprint_spec("lingtu").name == "lingtu_voice"

    runtime = load_blueprint_runtime("park")

    assert isinstance(runtime, Runtime)
    assert [module.name for module in runtime._module_classes][-1] == "proactive"


def test_customer_visible_blueprint_filter() -> None:
    customer_names = {item.name for item in list_blueprints(customer_visible=True)}
    internal_names = {item.name for item in list_blueprints(customer_visible=False)}

    assert {"voice", "voice_perception", "edge_robot", "lingtu_voice"} <= customer_names
    assert {"text", "mcp"} <= internal_names
