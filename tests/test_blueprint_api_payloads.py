from __future__ import annotations

import logging

from fastapi import FastAPI

from askme.api.routes.capabilities import register_capability_routes
from askme.api.schemas.blueprints import (
    BlueprintCatalogResponse,
    BlueprintDeliveryPackageResponse,
    BlueprintDetailResponse,
)
from askme.api.services.blueprint_payloads import (
    available_blueprint_names,
    blueprint_item_from_payload,
    blueprint_runtime_summary,
    load_blueprints_payload,
)


def test_load_blueprints_payload_uses_provider_when_present() -> None:
    payload = {"items": [{"name": "custom"}]}

    assert load_blueprints_payload(lambda: payload) == payload


def test_load_blueprints_payload_falls_back_to_catalog() -> None:
    payload = load_blueprints_payload(None)

    assert isinstance(payload.get("items"), list)
    assert "edge_robot" in available_blueprint_names(payload)


def test_blueprint_item_from_payload_matches_public_alias() -> None:
    payload = {
        "items": [
            {"name": "voice"},
            {"name": "edge_robot", "delivery_package": {"package_id": "blueprint.edge_robot"}},
        ]
    }

    item = blueprint_item_from_payload(payload, "park")

    assert item == {"name": "edge_robot", "delivery_package": {"package_id": "blueprint.edge_robot"}}


def test_blueprint_item_from_payload_handles_bad_payloads() -> None:
    assert blueprint_item_from_payload({"items": "bad"}, "park") is None
    assert blueprint_item_from_payload({"items": []}, "") is None


def test_available_blueprint_names_returns_sorted_names() -> None:
    payload = {"items": [{"name": "voice"}, {"name": "edge_robot"}, {"missing": True}]}

    assert available_blueprint_names(payload) == ["edge_robot", "voice"]


def test_blueprint_runtime_summary_is_dashboard_ready() -> None:
    payload = {
        "items": [
            {
                "name": "edge_robot",
                "title": "园区巡检机器人运行时",
                "product_stage": "pilot",
                "customer_visible": True,
                "primary_loop": "voice",
                "deployment_targets": ["robot_edge_pc"],
                "capabilities": ["语音交互"],
                "scenarios": ["访客问路和带路服务"],
                "external_services": ["ASR"],
                "safety_boundaries": ["LLM must not directly control hardware"],
                "validation_commands": ["python -m pytest tests/test_voice_loop.py -q"],
                "readiness": {"status": "ready_for_validation", "missing_config": []},
                "delivery_package": {
                    "package_id": "blueprint.edge_robot",
                    "status": "ready_for_site_validation",
                    "release_boundary": "pilot only",
                    "customer_claim": "ready for site validation",
                },
            },
            {
                "name": "voice",
                "customer_visible": True,
                "readiness": {"status": "configuration_incomplete", "missing_config": ["voice.tts"]},
            },
        ]
    }

    summary = blueprint_runtime_summary(payload)

    assert summary["summary"] == {
        "blueprint_count": 2,
        "customer_visible_count": 2,
        "ready_for_validation_count": 1,
        "missing_configuration_count": 1,
    }
    assert summary["items"][0]["package_id"] == "blueprint.edge_robot"
    assert summary["items"][0]["external_services"] == ["ASR"]
    assert summary["items"][0]["safety_boundaries"] == ["LLM must not directly control hardware"]
    assert summary["items"][0]["validation_commands"] == [
        "python -m pytest tests/test_voice_loop.py -q"
    ]
    assert summary["items"][0]["release_boundary"] == "pilot only"
    assert summary["items"][0]["acceptance_boundary"] == "pilot only"
    assert summary["items"][0]["customer_claim"] == "ready for site validation"
    assert summary["items"][0]["customer_status"] == "可进入现场验证"
    assert summary["items"][0]["customer_next_step"] == "先做现场验证，再对客户声明可交付。"
    assert summary["items"][0]["delivery_actions"] == [
        "运行现场验证用例。",
        "归档语音、通知、机器人运行和客户复核证据。",
        "签收前复核安全边界和人工接管方案。",
    ]
    assert summary["items"][1]["missing_config"] == ["voice.tts"]
    assert summary["items"][1]["customer_status"] == "运行配置未补齐"
    assert summary["items"][1]["acceptance_boundary"] == (
        "运行配置、外部服务和现场验证证据补齐前，不能作为客户验收依据。"
    )
    assert "voice.tts" in summary["items"][1]["customer_next_step"]
    assert summary["items"][1]["delivery_actions"][0] == "补齐运行配置：voice.tts"
    assert summary["policy"]["runtime_blueprints_are_delivery_profiles"] is True


def test_blueprint_catalog_payload_matches_api_schema() -> None:
    payload = load_blueprints_payload(None)

    schema_payload = BlueprintCatalogResponse.model_validate(payload)
    edge = next(item for item in schema_payload.items if item.name == "edge_robot")

    assert schema_payload.summary["blueprint_count"] >= 6
    assert edge.delivery_package is not None
    assert edge.delivery_package.package_id == "blueprint.edge_robot"
    assert edge.delivery_package.customer_status
    assert edge.delivery_package.customer_next_step
    assert edge.delivery_package.acceptance_boundary
    assert edge.delivery_package.delivery_actions


def test_blueprint_detail_and_delivery_package_schemas_require_customer_handoff_fields() -> None:
    payload = load_blueprints_payload(None)
    item = blueprint_item_from_payload(payload, "park")
    assert item is not None
    package = item["delivery_package"]

    detail = BlueprintDetailResponse.model_validate(
        {
            "ok": True,
            "blueprint": item,
            "policy": {"site_validation_required_before_customer_claim": True},
        }
    )
    handoff = BlueprintDeliveryPackageResponse.model_validate(
        {
            "ok": True,
            "blueprint": item["name"],
            "delivery_package": package,
            "policy": {"delivery_package_is_customer_handoff": True},
        }
    )

    assert detail.blueprint.name == "edge_robot"
    assert handoff.delivery_package.customer_status == package["customer_status"]
    assert handoff.delivery_package.delivery_actions


def test_blueprint_routes_expose_product_response_schemas_in_openapi() -> None:
    app = FastAPI()
    register_capability_routes(
        app,
        capabilities_provider=lambda: {},
        blueprints_provider=None,
        logger=logging.getLogger("test"),
    )

    paths = app.openapi()["paths"]

    assert (
        paths["/api/blueprints"]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        .endswith("/BlueprintCatalogResponse")
    )
    assert (
        paths["/api/blueprints/{blueprint_name}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        .endswith("/BlueprintDetailResponse")
    )
    assert (
        paths["/api/blueprints/{blueprint_name}/delivery-package"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]["$ref"]
        .endswith("/BlueprintDeliveryPackageResponse")
    )
