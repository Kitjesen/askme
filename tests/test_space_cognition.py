from __future__ import annotations

import json
from pathlib import Path

from askme.runtime.handoff import RuntimeHandoffService
from fastapi.testclient import TestClient

from askme.api.schemas.space import (
    SpaceGuideResponse,
    SpaceHealthResponse,
    SpaceHistoryResponse,
    SpaceInteractionsResponse,
    SpaceManageResponse,
    SpacePointsResponse,
    SpaceProposalCreateResponse,
    SpaceProposalReviewResponse,
    SpaceProposalsResponse,
    SpaceResolveDestinationResponse,
    SpaceRollbackResponse,
    SpaceRoutesResponse,
    SpaceServicePointTriggerResponse,
    SpaceServicePointsResponse,
)
from askme.cognition import WorldStateService
from askme.health_server import create_health_app
from askme.space import ParkSpaceService


def _text(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape")


def _space_service() -> ParkSpaceService:
    return ParkSpaceService.from_config(
        {
            "space_cognition": {
                "park_id": "fanmu",
                "points": [
                    {
                        "point_id": "sp-west-gate",
                        "point_name": _text("\\u897f\\u95e8\\u95ee\\u8be2\\u70b9"),
                        "point_type": "service",
                        "aliases": [
                            _text("\\u897f\\u95e8"),
                            _text("\\u5927\\u95e8\\u53e3"),
                        ],
                        "x": 0,
                        "y": 0,
                    },
                    {
                        "point_id": "poi-fanmu-coffee",
                        "point_name": _text("\\u68b5\\u6728\\u5496\\u5561"),
                        "point_type": "restaurant",
                        "aliases": [
                            _text("\\u5496\\u5561\\u5e97"),
                            _text("\\u5496\\u5561\\u9986"),
                            _text("\\u559d\\u5496\\u5561\\u7684\\u5730\\u65b9"),
                        ],
                        "building": _text("2\\u53f7\\u697c"),
                        "floor": _text("\\u4e00\\u5c42"),
                        "x": 80,
                        "y": 20,
                        "guide_mode": "escort",
                    },
                    {
                        "point_id": "poi-restroom-1",
                        "point_name": _text("3\\u53f7\\u697c\\u4e00\\u5c42\\u536b\\u751f\\u95f4"),
                        "point_type": "restroom",
                        "aliases": [
                            _text("\\u5395\\u6240"),
                            _text("\\u6d17\\u624b\\u95f4"),
                        ],
                        "x": 40,
                        "y": 0,
                    },
                ],
                "service_points": [
                    {
                        "service_point_id": "guide-west-gate",
                        "point_id": "sp-west-gate",
                        "service_point_name": _text("\\u897f\\u95e8\\u95ee\\u8be2\\u670d\\u52a1\\u70b9"),
                        "dwell_seconds": 3,
                        "greeting_prompt": _text("\\u4f60\\u597d\\uff0c\\u8bf7\\u95ee\\u9700\\u8981\\u6307\\u8def\\u5417\\uff1f"),
                        "supported_intents": ["wayfinding", "escort"],
                    }
                ],
                "routes": [
                    {
                        "route_id": "route-west-coffee",
                        "from_point_id": "sp-west-gate",
                        "to_point_id": "poi-fanmu-coffee",
                        "instructions": _text(
                            "\\u68b5\\u6728\\u5496\\u5561\\u57282\\u53f7\\u697c\\u4e00\\u5c42\\u3002"
                            "\\u4ece\\u897f\\u95e8\\u6cbf\\u4e3b\\u901a\\u9053\\u5411\\u524d\\u7ea680\\u7c73\\uff0c"
                            "\\u53f3\\u8f6c\\u540e\\u6cbf\\u5de6\\u4fa7\\u5546\\u94fa\\u524d\\u8fdb\\u5373\\u53ef\\u5230\\u8fbe\\u3002"
                        ),
                        "guide_mode": "escort",
                        "robot_passable": True,
                        "distance_m": 95,
                    }
                ],
            }
        }
    )


def test_resolves_destination_by_alias_and_requires_confirmation() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload(
        {"query": _text("\\u5496\\u5561\\u5e97\\u5728\\u54ea"), "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is True
    assert payload["point"]["point_id"] == "poi-fanmu-coffee"
    assert payload["match_reason"] == "single_restaurant_candidate"
    assert payload["candidate_count"] == 1
    assert payload["selection_policy"] == "single_category_candidate"
    assert payload["requires_confirmation"] is True
    assert payload["confirmation_prompt"] == _text("\\u4f60\\u662f\\u8981\\u53bb\\u68b5\\u6728\\u5496\\u5561\\u5417\\uff1f")


def test_category_query_lists_multiple_matching_destinations() -> None:
    service = _space_service()
    service.manage_payload(
        {
            "entity": "point",
            "action": "upsert",
            "item": {
                "point_id": "poi-roastery",
                "point_name": _text("\\u897f\\u95e8\\u70d8\\u7119\\u5496\\u5561"),
                "point_type": "restaurant",
                "aliases": [
                    _text("\\u5496\\u5561\\u5e97"),
                    _text("\\u624b\\u51b2\\u5496\\u5561"),
                ],
                "building": _text("1\\u53f7\\u697c"),
                "floor": _text("\\u4e00\\u5c42"),
                "x": 8,
                "y": 1,
                "guide_mode": "voice",
            },
        }
    )

    payload = service.resolve_destination_payload(
        {"query": _text("\\u5496\\u5561\\u5e97\\u5728\\u54ea"), "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is False
    assert payload["reason"] == "multiple_destinations"
    assert payload["requires_clarification"] is True
    assert payload["candidate_count"] == 2
    assert [candidate["point_id"] for candidate in payload["candidates"]] == [
        "poi-roastery",
        "poi-fanmu-coffee",
    ]
    assert _text("\\u8bf7\\u544a\\u8bc9\\u6211\\u4f60\\u60f3\\u53bb\\u54ea\\u4e00\\u4e2a") in payload["reply"]


def test_category_listing_query_returns_candidates_before_routing() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload(
        {"query": _text("\\u6709\\u54ea\\u4e9b\\u5496\\u5561\\u5e97"), "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is False
    assert payload["reason"] == "category_candidates_found"
    assert payload["requires_clarification"] is False
    assert payload["listing_only"] is True
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["point_id"] == "poi-fanmu-coffee"
    assert _text("\\u8bf7\\u544a\\u8bc9\\u6211") not in payload["reply"]


def test_resolves_nearest_type_query() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload(
        {"query": _text("\\u6700\\u8fd1\\u7684\\u5395\\u6240\\u5728\\u54ea\\u91cc"), "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is True
    assert payload["point"]["point_id"] == "poi-restroom-1"
    assert payload["match_reason"] == "nearest_restroom"


def test_unknown_destination_refuses_instead_of_hallucinating() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload({"query": _text("\\u53bb\\u4e0d\\u5b58\\u5728\\u7684\\u5c55\\u9986")})

    assert payload["resolved"] is False
    assert payload["reason"] == "destination_not_found"
    assert payload["reply"] == _text(
        "\\u6211\\u8fd8\\u6ca1\\u6709\\u5728\\u56ed\\u533a\\u70b9\\u4f4d\\u5e93\\u91cc\\u627e\\u5230\\u8fd9\\u4e2a\\u5730\\u70b9\\uff0c"
        "\\u8bf7\\u6362\\u4e00\\u79cd\\u8bf4\\u6cd5\\u6216\\u8054\\u7cfb\\u5de5\\u4f5c\\u4eba\\u5458\\u786e\\u8ba4\\u3002"
    )
    assert payload["requires_operator_update"] is True


def test_service_point_trigger_requires_person_and_dwell_time() -> None:
    service = _space_service()

    no_person = service.service_point_trigger_payload(
        {"service_point_id": "guide-west-gate", "person_present": False, "dwell_seconds": 5}
    )
    too_short = service.service_point_trigger_payload(
        {"service_point_id": "guide-west-gate", "person_present": True, "dwell_seconds": 1.5}
    )
    prompt = service.service_point_trigger_payload(
        {"service_point_id": "guide-west-gate", "person_present": True, "dwell_seconds": 3.5}
    )

    assert no_person["should_prompt"] is False
    assert no_person["reason"] == "no_person"
    assert too_short["should_prompt"] is False
    assert too_short["reason"] == "dwell_time_too_short"
    assert too_short["required_dwell_seconds"] == 3.0
    assert prompt["should_prompt"] is True
    assert prompt["reason"] == "visitor_dwelling_at_service_point"
    assert prompt["speech_text"] == _text("\\u4f60\\u597d\\uff0c\\u8bf7\\u95ee\\u9700\\u8981\\u6307\\u8def\\u5417\\uff1f")
    assert prompt["next_expected_input"] == "visitor_destination"


def test_guide_returns_voice_text_or_escort_payload() -> None:
    service = _space_service()

    payload = service.guide_payload(
        {
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u9986"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
            "guide_mode": "escort",
        }
    )

    assert payload["guide_ready"] is True
    assert payload["mode"] == "escort"
    assert payload["speech_text"].startswith(_text("\\u68b5\\u6728\\u5496\\u5561\\u57282\\u53f7\\u697c"))
    assert payload["field_event_payload"]["scenario_id"] == "visitor_escort"
    assert payload["field_event_payload"]["destination_point_id"] == "poi-fanmu-coffee"
    assert payload["runtime_handoff_ready"] is False
    assert payload["runtime_handoff_reason"] == "visitor_destination_confirmation_required"
    assert payload["runtime_handoff_preview"]["task_type"] == "visitor_escort"
    assert payload["runtime_handoff_preview"]["handoff_ready"] is False
    assert payload["runtime_handoff_plan"]["handoff_ready"] is False
    assert payload["runtime_handoff_plan"]["intent"] == "visitor_escort"
    assert payload["runtime_handoff_preview"]["steps"][0]["skill_name"] == "low_speed_escort"
    assert "operator_confirmation_required" in payload["runtime_handoff_validation"]
    assert payload["interaction_id"].startswith("space-interaction-")


def test_space_interaction_records_capture_prompt_guide_and_refusal(tmp_path: Path) -> None:
    store_path = tmp_path / "space-catalog.json"
    service = ParkSpaceService.from_config(
        {
            "space_cognition": {
                **_space_service()._snapshot_payload(),
                "park_id": "fanmu",
                "store_path": str(store_path),
            }
        }
    )

    prompt = service.service_point_trigger_payload(
        {"service_point_id": "guide-west-gate", "person_present": True, "dwell_seconds": 4}
    )
    guide = service.guide_payload(
        {
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u5e97"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
            "guide_mode": "escort",
        }
    )
    refused = service.guide_payload({"query": _text("\\u53bb\\u4e0d\\u5b58\\u5728\\u7684\\u5c55\\u9986")})
    interactions = service.interactions_payload({"limit": 10})
    persisted = json.loads(store_path.read_text(encoding="utf-8"))

    assert prompt["interaction_id"].startswith("space-interaction-")
    assert guide["interaction_id"].startswith("space-interaction-")
    assert refused["interaction_id"].startswith("space-interaction-")
    assert interactions["count"] == 3
    assert interactions["interactions"][0]["status"] == "refused"
    assert interactions["interactions"][1]["destination_point_id"] == "poi-fanmu-coffee"
    assert interactions["interactions"][2]["status"] == "prompted"
    assert persisted["interactions"][0]["service_point_id"] == "guide-west-gate"


def test_space_interactions_payload_filters_records() -> None:
    service = _space_service()
    service.service_point_trigger_payload(
        {"service_point_id": "guide-west-gate", "person_present": True, "dwell_seconds": 4}
    )
    service.guide_payload(
        {
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u9986"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
        }
    )

    filtered = service.interactions_payload({"destination_point_id": "poi-fanmu-coffee"})

    assert filtered["count"] == 1
    assert filtered["interactions"][0]["destination_point_id"] == "poi-fanmu-coffee"


def test_confirmed_escort_returns_runtime_handoff_contract() -> None:
    service = _space_service()

    payload = service.guide_payload(
        {
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u9986"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
            "guide_mode": "escort",
            "visitor_confirmed": True,
            "operator_id": "guide.operator",
            "operator_roles": ["operator"],
            "world_state_snapshot": {"updated_at": 100, "fact_count": 8},
        }
    )

    handoff = payload["runtime_handoff"]
    assert payload["runtime_handoff_ready"] is True
    assert payload["runtime_handoff_validation"] == []
    assert payload["runtime_handoff_plan"]["handoff_ready"] is True
    assert payload["runtime_handoff_plan"]["mission"]["mission"]["mission_type"] == "visitor_escort"
    assert handoff["handoff_ready"] is True
    assert handoff["operator_id"] == "guide.operator"
    assert handoff["task_type"] == "visitor_escort"
    assert handoff["risk_level"] == "high"
    assert handoff["target_area"] == "route-west-coffee"
    assert handoff["world_state_snapshot_id"] == "world-100000-8"
    assert [step["skill_name"] for step in handoff["steps"]] == [
        "low_speed_escort",
        "generate_report",
    ]
    assert handoff["steps"][0]["parameters"] == {
        "area_id": "route-west-coffee",
        "destination": _text("\\u68b5\\u6728\\u5496\\u5561"),
        "destination_point_id": "poi-fanmu-coffee",
        "route_id": "route-west-coffee",
        "map_id": "default",
        "service_point_id": "guide-west-gate",
        "speed_limit": "low",
        "interaction_policy": "visitor_must_remain_tracked",
    }


def test_confirmed_escort_plan_submits_to_fake_runtime() -> None:
    service = _space_service()
    payload = service.guide_payload(
        {
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u9986"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
            "guide_mode": "escort",
            "visitor_confirmed": True,
            "operator_id": "guide.operator",
            "operator_roles": ["operator"],
        }
    )
    world = WorldStateService()
    world.update_robot_state(
        {
            "online": True,
            "battery_percent": 86,
            "estop_active": False,
            "localized": True,
        },
        stale_after_s=60.0,
    )
    runtime = RuntimeHandoffService(world_state=world)

    result = runtime.submit_plan_payload(payload["runtime_handoff_plan"])

    assert result["accepted"] is True
    assert result["handoff"]["task_type"] == "visitor_escort"
    assert result["run"]["current_state"] == "completed"
    assert result["run"]["report"]["completed_steps"] == ["low_speed_escort", "generate_report"]


def test_space_manage_persists_catalog_updates(tmp_path: Path) -> None:
    store_path = tmp_path / "space-catalog.json"
    config = {
        "space_cognition": {
            "park_id": "fanmu",
            "store_path": str(store_path),
            "points": [
                {
                    "point_id": "sp-west-gate",
                    "point_name": _text("\\u897f\\u95e8\\u95ee\\u8be2\\u70b9"),
                    "point_type": "service",
                    "aliases": [_text("\\u897f\\u95e8")],
                    "x": 0,
                    "y": 0,
                }
            ],
        }
    }
    service = ParkSpaceService.from_config(config)

    added = service.manage_payload(
        {
            "entity": "point",
            "action": "upsert",
            "item": {
                "point_id": "poi-bookstore",
                "point_name": _text("\\u68b5\\u6728\\u4e66\\u5e97"),
                "point_type": "shop",
                "aliases": [_text("\\u4e66\\u5e97"), _text("\\u4e70\\u4e66\\u7684\\u5730\\u65b9")],
                "x": 35,
                "y": 12,
            },
        }
    )

    assert added["ok"] is True
    assert added["revision"] == 1
    assert added["change"]["operator_id"] == "unknown"
    assert added["change"]["item_id"] == "poi-bookstore"
    assert added["persisted"]["written"] is True
    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert persisted["revision"] == 1
    assert persisted["change_log"][0]["entity"] == "point"
    assert any(point["point_id"] == "poi-bookstore" for point in persisted["points"])

    reloaded = ParkSpaceService.from_config(config)
    assert reloaded.history_payload()["revision"] == 1
    assert reloaded.history_payload()["changes"][0]["item_id"] == "poi-bookstore"
    resolved = reloaded.resolve_destination_payload({"query": _text("\\u4e66\\u5e97\\u5728\\u54ea")})
    assert resolved["resolved"] is True
    assert resolved["point"]["point_id"] == "poi-bookstore"

    disabled = reloaded.manage_payload({"entity": "point", "action": "disable", "item": {"point_id": "poi-bookstore"}})
    assert disabled["ok"] is True
    unavailable = ParkSpaceService.from_config(config).resolve_destination_payload({"query": _text("\\u4e66\\u5e97")})
    assert unavailable["resolved"] is False
    restored = ParkSpaceService.from_config(config).rollback_payload(
        {"revision": 1, "operator_id": "supervisor-1", "reason": "restore bookstore"}
    )
    assert restored["ok"] is True
    assert restored["revision"] == 3
    assert restored["restored_revision"] == 1
    assert restored["change"]["action"] == "rollback"
    assert restored["change"]["operator_id"] == "supervisor-1"
    available = ParkSpaceService.from_config(config).resolve_destination_payload({"query": _text("\\u4e66\\u5e97")})
    assert available["resolved"] is True


def test_space_manage_validates_references_before_persisting(tmp_path: Path) -> None:
    store_path = tmp_path / "space-catalog.json"
    config = {
        "_project_root": str(tmp_path),
        "space_cognition": {
            "park_id": "fanmu",
            "store_path": "space-catalog.json",
            "points": [
                {
                    "point_id": "sp-west-gate",
                    "point_name": _text("\\u897f\\u95e8\\u95ee\\u8be2\\u70b9"),
                    "point_type": "service",
                },
                {
                    "point_id": "poi-coffee",
                    "point_name": _text("\\u5496\\u5561\\u5e97"),
                    "point_type": "restaurant",
                },
            ],
        },
    }
    service = ParkSpaceService.from_config(config)

    missing_service_point = service.manage_payload(
        {
            "entity": "service_point",
            "action": "upsert",
            "item": {
                "service_point_id": "guide-missing",
                "point_id": "missing-point",
            },
        }
    )
    missing_route = service.manage_payload(
        {
            "entity": "route",
            "action": "upsert",
            "item": {
                "route_id": "route-missing",
                "from_point_id": "sp-west-gate",
                "to_point_id": "missing-point",
            },
        }
    )
    route = service.manage_payload(
        {
            "entity": "route",
            "action": "upsert",
            "item": {
                "route_id": "route-coffee",
                "from_point_id": "sp-west-gate",
                "to_point_id": "poi-coffee",
                "instructions": _text("\\u5f80\\u524d\\u8d70\\u5373\\u53ef\\u5230\\u8fbe\\u3002"),
            },
        }
    )
    blocked_delete = service.manage_payload(
        {
            "entity": "point",
            "action": "delete",
            "item": {"point_id": "poi-coffee"},
        }
    )

    assert missing_service_point["ok"] is False
    assert missing_service_point["reason"] == "service_point_point_not_found"
    assert missing_route["ok"] is False
    assert missing_route["reason"] == "route_point_not_found"
    assert missing_route["missing_point_ids"] == ["missing-point"]
    assert route["ok"] is True
    assert route["persisted"]["path"] == str(store_path)
    assert blocked_delete["ok"] is False
    assert blocked_delete["reason"] == "point_in_use"
    assert blocked_delete["references"] == [{"entity": "route", "id": "route-coffee"}]


def test_space_proposal_requires_review_before_catalog_changes(tmp_path: Path) -> None:
    store_path = tmp_path / "space-catalog.json"
    config = {
        "space_cognition": {
            "park_id": "fanmu",
            "store_path": str(store_path),
            "points": [
                {
                    "point_id": "sp-west-gate",
                    "point_name": _text("\\u897f\\u95e8\\u95ee\\u8be2\\u70b9"),
                    "point_type": "service",
                }
            ],
        }
    }
    service = ParkSpaceService.from_config(config)

    proposal = service.propose_payload(
        {
            "operator_id": "field-editor",
            "entity": "point",
            "action": "upsert",
            "item": {
                "point_id": "poi-gallery",
                "point_name": _text("\\u827a\\u672f\\u5c55\\u5385"),
                "aliases": [_text("\\u5c55\\u5385")],
            },
            "reason": "site update",
        }
    )
    unresolved = service.resolve_destination_payload({"query": _text("\\u5c55\\u5385")})
    proposals = service.proposals_payload({"status": "pending"})
    approved = service.review_proposal_payload(
        {
            "operator_id": "supervisor-1",
            "proposal_id": proposal["proposal"]["proposal_id"],
            "decision": "approve",
        }
    )
    resolved = service.resolve_destination_payload({"query": _text("\\u5c55\\u5385")})
    persisted = json.loads(store_path.read_text(encoding="utf-8"))

    assert proposal["ok"] is True
    assert proposal["proposal_created"] is True
    assert proposal["proposal"]["status"] == "pending"
    assert unresolved["resolved"] is False
    assert proposals["pending_count"] == 1
    assert approved["ok"] is True
    assert approved["reviewed"] is True
    assert approved["proposal"]["status"] == "approved"
    assert approved["change"]["proposal_id"] == proposal["proposal"]["proposal_id"]
    assert resolved["resolved"] is True
    assert persisted["pending_changes"][0]["status"] == "approved"


def test_space_routes_expose_product_contract() -> None:
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=_space_service(),
    )
    client = TestClient(app)

    health = client.get("/api/space/health")
    history = client.get("/api/space/history")
    assert health.status_code == 200
    SpaceHealthResponse.model_validate(health.json())
    assert "service_point_trigger" in health.json()["capabilities"]
    assert history.status_code == 200
    SpaceHistoryResponse.model_validate(history.json())
    assert history.json()["revision"] == 0

    trigger = client.post(
        "/api/space/service-point-trigger",
        json={"service_point_id": "guide-west-gate", "person_present": True, "dwell_seconds": 4},
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )
    response = client.post(
        "/api/space/guide",
        json={
            "query": _text("\\u6211\\u8981\\u53bb\\u5496\\u5561\\u5e97"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
        },
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )

    assert trigger.status_code == 200
    SpaceServicePointTriggerResponse.model_validate(trigger.json())
    assert trigger.json()["should_prompt"] is True
    assert response.status_code == 200
    payload = response.json()
    SpaceGuideResponse.model_validate(payload)
    assert payload["guide_ready"] is True
    assert payload["mode"] == "escort"
    interactions = client.get("/api/space/interactions")
    assert interactions.status_code == 200
    SpaceInteractionsResponse.model_validate(interactions.json())
    assert interactions.json()["count"] == 2
    assert interactions.json()["interactions"][0]["event_type"] == "guide_request"


def test_space_routes_expose_response_schemas_in_openapi() -> None:
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=_space_service(),
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        ("/api/space/health", "get"): "SpaceHealthResponse",
        ("/api/space/points", "get"): "SpacePointsResponse",
        ("/api/space/service-points", "get"): "SpaceServicePointsResponse",
        ("/api/space/routes", "get"): "SpaceRoutesResponse",
        ("/api/space/history", "get"): "SpaceHistoryResponse",
        ("/api/space/proposals", "get"): "SpaceProposalsResponse",
        ("/api/space/interactions", "get"): "SpaceInteractionsResponse",
        ("/api/space/resolve-destination", "post"): "SpaceResolveDestinationResponse",
        ("/api/space/guide", "post"): "SpaceGuideResponse",
        ("/api/space/service-point-trigger", "post"): "SpaceServicePointTriggerResponse",
        ("/api/space/manage", "post"): "SpaceManageResponse",
        ("/api/space/proposals", "post"): "SpaceProposalCreateResponse",
        ("/api/space/proposals/review", "post"): "SpaceProposalReviewResponse",
        ("/api/space/rollback", "post"): "SpaceRollbackResponse",
    }
    for (path, method), schema_name in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith(f"/{schema_name}")

    client = TestClient(app)
    SpacePointsResponse.model_validate(client.get("/api/space/points").json())
    SpaceServicePointsResponse.model_validate(
        client.get("/api/space/service-points").json()
    )
    SpaceRoutesResponse.model_validate(client.get("/api/space/routes").json())
    SpaceProposalsResponse.model_validate(client.get("/api/space/proposals").json())
    SpaceResolveDestinationResponse.model_validate(
        client.post(
            "/api/space/resolve-destination",
            json={"query": _text("\\u5496\\u5561\\u5e97\\u5728\\u54ea")},
            headers={"X-Askme-Operator-Id": "dashboard.operator"},
        ).json()
    )


def test_scenario_preview_includes_space_candidates_for_wayfinding() -> None:
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=_space_service(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/scenario-intents/preview",
        json={
            "text": _text("\\u6709\\u54ea\\u4e9b\\u5496\\u5561\\u5e97"),
            "current_point_id": "sp-west-gate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy"]["does_not_start_guide"] is True
    assert payload["space_resolution"]["available"] is True
    resolution = payload["space_resolution"]["resolution"]
    assert resolution["listing_only"] is True
    assert resolution["reason"] == "category_candidates_found"
    assert resolution["candidate_count"] == 1
    assert resolution["candidates"][0]["point_id"] == "poi-fanmu-coffee"


def test_chat_endpoint_answers_wayfinding_with_space_evidence_without_starting_guide() -> None:
    async def chat_handler(text: str, *, speak: bool = False):
        return {"reply": _text("\\u6211\\u9700\\u8981\\u67e5\\u4e00\\u4e0b\\u70b9\\u4f4d\\u5e93")}

    service = _space_service()
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        chat_handler=chat_handler,
        space_handler=service,
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "text": _text("\\u6709\\u54ea\\u4e9b\\u5496\\u5561\\u5e97"),
            "current_point_id": "sp-west-gate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == _text("\\u6211\\u627e\\u52301\\u4e2a\\u5496\\u5561\\u5e97\\uff1a\\u68b5\\u6728\\u5496\\u5561\\u3002")
    assert payload["reply_source"] == "space_cognition"
    assert payload["space_answered"] is True
    assert payload["scenario_preview"]["policy"]["does_not_start_guide"] is True
    assert payload["space_resolution"]["does_not_start_guide"] is True
    assert payload["space_resolution"]["resolution"]["listing_only"] is True
    assert payload["evidence"][0]["source"] == _text("\\u56ed\\u533a\\u7a7a\\u95f4\\u8ba4\\u77e5\\u5e93")
    assert payload["evidence"][0]["source_system"] == "space_cognition"
    assert payload["evidence"][0]["record_id"] == "poi-fanmu-coffee"

    interactions = client.get("/api/space/interactions")
    assert interactions.status_code == 200
    assert interactions.json()["count"] == 0


def test_dashboard_only_chat_answers_space_question_without_chat_handler() -> None:
    service = _space_service()
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=service,
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "text": _text("\\u5496\\u5561\\u5e97\\u5728\\u54ea"),
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == _text("\\u76ee\\u524d\\u70b9\\u4f4d\\u5e93\\u91cc\\u627e\\u5230\\u4e00\\u4e2a\\u5496\\u5561\\u5e97\\uff1a\\u68b5\\u6728\\u5496\\u5561\\u3002")
    assert payload["reply_source"] == "space_cognition"
    assert payload["space_answered"] is True
    assert payload["chat_backend"]["configured"] is False
    assert payload["space_resolution"]["does_not_start_guide"] is True
    assert payload["evidence"][0]["source_system"] == "space_cognition"

    interactions = client.get("/api/space/interactions")
    assert interactions.status_code == 200
    assert interactions.json()["count"] == 0


def test_space_post_routes_reject_non_object_json_body() -> None:
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=_space_service(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/space/resolve-destination",
        json=["coffee"],
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_space_manage_route_requires_supervisor_permission(tmp_path: Path) -> None:
    service = ParkSpaceService.from_config(
        {
            "space_cognition": {
                "park_id": "fanmu",
                "store_path": str(tmp_path / "space-catalog.json"),
                "points": [],
            }
        }
    )
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=service,
    )
    client = TestClient(app)

    body = {
        "entity": "point",
        "action": "upsert",
        "item": {
            "point_id": "poi-bookstore",
            "point_name": _text("\\u68b5\\u6728\\u4e66\\u5e97"),
            "aliases": [_text("\\u4e66\\u5e97")],
        },
    }
    denied = client.post(
        "/api/space/manage",
        json=body,
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )
    allowed = client.post(
        "/api/space/manage",
        json=body,
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    service_point = client.post(
        "/api/space/manage",
        json={
            "entity": "service_point",
            "action": "upsert",
            "item": {
                "service_point_id": "guide-bookstore",
                "point_id": "poi-bookstore",
                "service_point_name": _text("\\u4e66\\u5e97\\u95ee\\u8be2\\u70b9"),
            },
        },
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    route = client.post(
        "/api/space/manage",
        json={
            "entity": "route",
            "action": "upsert",
            "item": {
                "route_id": "route-bookstore",
                "from_point_id": "poi-bookstore",
                "to_point_id": "poi-bookstore",
                "instructions": _text("\\u5df2\\u5230\\u8fbe\\u4e66\\u5e97\\u95ee\\u8be2\\u70b9\\u3002"),
            },
        },
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    rollback = client.post(
        "/api/space/rollback",
        json={"revision": 1, "reason": "restore point-only catalog"},
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )
    proposed = client.post(
        "/api/space/proposals",
        json={
            "entity": "point",
            "action": "upsert",
            "item": {
                "point_id": "poi-gallery",
                "point_name": _text("\\u827a\\u672f\\u5c55\\u5385"),
                "aliases": [_text("\\u5c55\\u5385")],
            },
        },
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )
    reviewed = client.post(
        "/api/space/proposals/review",
        json={
            "proposal_id": proposed.json()["proposal"]["proposal_id"],
            "decision": "approve",
        },
        headers={"X-Askme-Operator-Id": "supervisor-1"},
    )

    assert denied.status_code == 403
    assert denied.json()["operator_auth"]["permission"] == "knowledge:approve"
    assert allowed.status_code == 200
    SpaceManageResponse.model_validate(allowed.json())
    assert allowed.json()["ok"] is True
    assert allowed.json()["change"]["operator_id"] == "supervisor-1"
    assert service_point.status_code == 200
    SpaceManageResponse.model_validate(service_point.json())
    assert service_point.json()["service_point"]["service_point_id"] == "guide-bookstore"
    assert route.status_code == 200
    SpaceManageResponse.model_validate(route.json())
    assert route.json()["route"]["route_id"] == "route-bookstore"
    assert rollback.status_code == 200
    SpaceRollbackResponse.model_validate(rollback.json())
    assert rollback.json()["revision"] == 4
    assert rollback.json()["restored_revision"] == 1
    assert proposed.status_code == 200
    SpaceProposalCreateResponse.model_validate(proposed.json())
    assert proposed.json()["proposal"]["operator_id"] == "dashboard.operator"
    assert reviewed.status_code == 200
    SpaceProposalReviewResponse.model_validate(reviewed.json())
    assert reviewed.json()["proposal"]["status"] == "approved"

    history = client.get("/api/space/history")
    proposals = client.get("/api/space/proposals")
    assert history.status_code == 200
    assert proposals.status_code == 200
    assert proposals.json()["pending_count"] == 0
    assert history.json()["revision"] == 5
    assert [change["entity"] for change in history.json()["changes"]] == [
        "point",
        "catalog",
        "route",
        "service_point",
        "point",
    ]
    assert client.get("/api/space/service-points").json()["service_points"] == []
    assert client.get("/api/space/routes").json()["routes"] == []
