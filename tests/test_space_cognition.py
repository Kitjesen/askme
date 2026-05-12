from __future__ import annotations

from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from askme.space import ParkSpaceService


def _space_service() -> ParkSpaceService:
    return ParkSpaceService.from_config(
        {
            "space_cognition": {
                "park_id": "fanmu",
                "points": [
                    {
                        "point_id": "sp-west-gate",
                        "point_name": "西门问询点",
                        "point_type": "service",
                        "aliases": ["西门", "大门口"],
                        "x": 0,
                        "y": 0,
                    },
                    {
                        "point_id": "poi-fanmu-coffee",
                        "point_name": "梵木咖啡",
                        "point_type": "restaurant",
                        "aliases": ["咖啡店", "咖啡馆", "喝咖啡的地方"],
                        "building": "2号楼",
                        "floor": "一层",
                        "x": 80,
                        "y": 20,
                        "guide_mode": "escort",
                    },
                    {
                        "point_id": "poi-restroom-1",
                        "point_name": "3号楼一层卫生间",
                        "point_type": "restroom",
                        "aliases": ["厕所", "洗手间"],
                        "x": 40,
                        "y": 0,
                    },
                ],
                "service_points": [
                    {
                        "service_point_id": "guide-west-gate",
                        "point_id": "sp-west-gate",
                        "service_point_name": "西门问询服务点",
                        "dwell_seconds": 3,
                        "greeting_prompt": "你好，请问需要指路吗？",
                        "supported_intents": ["wayfinding", "escort"],
                    }
                ],
                "routes": [
                    {
                        "route_id": "route-west-coffee",
                        "from_point_id": "sp-west-gate",
                        "to_point_id": "poi-fanmu-coffee",
                        "instructions": "梵木咖啡在2号楼一层。从西门沿主通道向前约80米，右转后沿左侧商铺前进即可到达。",
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
        {"query": "咖啡店在哪", "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is True
    assert payload["point"]["point_id"] == "poi-fanmu-coffee"
    assert payload["match_reason"] == "partial_name_or_alias"
    assert payload["requires_confirmation"] is True
    assert payload["confirmation_prompt"] == "你是要去梵木咖啡吗？"


def test_resolves_nearest_type_query() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload(
        {"query": "最近的厕所在哪里", "current_point_id": "sp-west-gate"}
    )

    assert payload["resolved"] is True
    assert payload["point"]["point_id"] == "poi-restroom-1"
    assert payload["match_reason"] == "nearest_restroom"


def test_unknown_destination_refuses_instead_of_hallucinating() -> None:
    service = _space_service()

    payload = service.resolve_destination_payload({"query": "去不存在的展馆"})

    assert payload["resolved"] is False
    assert payload["reason"] == "destination_not_found"
    assert payload["requires_operator_update"] is True


def test_guide_returns_voice_text_or_escort_payload() -> None:
    service = _space_service()

    payload = service.guide_payload(
        {
            "query": "我要去咖啡馆",
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
            "guide_mode": "escort",
        }
    )

    assert payload["guide_ready"] is True
    assert payload["mode"] == "escort"
    assert payload["speech_text"].startswith("梵木咖啡在2号楼")
    assert payload["field_event_payload"]["scenario_id"] == "visitor_escort"
    assert payload["field_event_payload"]["destination_point_id"] == "poi-fanmu-coffee"


def test_space_routes_expose_product_contract() -> None:
    app = create_health_app(
        health_provider=lambda: {"ok": True},
        space_handler=_space_service(),
    )
    client = TestClient(app)

    health = client.get("/api/space/health")
    assert health.status_code == 200
    assert health.json()["capabilities"]

    response = client.post(
        "/api/space/guide",
        json={
            "query": "我要去咖啡店",
            "current_point_id": "sp-west-gate",
            "service_point_id": "guide-west-gate",
        },
        headers={"X-Askme-Operator-Id": "dashboard.operator"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["guide_ready"] is True
    assert payload["mode"] == "escort"
