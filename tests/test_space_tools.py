from __future__ import annotations

import json

from askme.space import ParkSpaceService
from askme.tools.builtin_tools import register_builtin_tools
from askme.tools.space_tool import SpaceLookupPlaceTool, SpaceRecommendRouteTool
from askme.tools.tool_registry import ToolRegistry


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
                        "aliases": ["咖啡店", "咖啡馆"],
                        "building": "2号楼",
                        "floor": "一层",
                        "x": 80,
                        "y": 20,
                        "guide_mode": "escort",
                    },
                ],
                "routes": [
                    {
                        "route_id": "route-west-coffee",
                        "from_point_id": "sp-west-gate",
                        "to_point_id": "poi-fanmu-coffee",
                        "instructions": "梵木咖啡在2号楼一层。从西门沿主通道向前约80米即可到达。",
                        "guide_mode": "escort",
                        "robot_passable": True,
                        "distance_m": 95,
                    }
                ],
            }
        }
    )


def test_space_lookup_place_tool_returns_confirmable_point() -> None:
    service = _space_service()
    tool = SpaceLookupPlaceTool(service_factory=lambda _config: service)

    payload = json.loads(
        tool.execute(query="咖啡店在哪", current_point_id="sp-west-gate")
    )

    assert payload["resolved"] is True
    assert payload["point_id"] == "poi-fanmu-coffee"
    assert payload["confirmation_prompt"] == "你是要去梵木咖啡吗？"


def test_space_recommend_route_tool_returns_speech_and_escort_handoff() -> None:
    service = _space_service()
    tool = SpaceRecommendRouteTool(service_factory=lambda _config: service)

    payload = json.loads(
        tool.execute(
            query="我要去咖啡馆",
            current_point_id="sp-west-gate",
            service_point_id="guide-west-gate",
            guide_mode="escort",
        )
    )

    assert payload["guide_ready"] is True
    assert payload["mode"] == "escort"
    assert payload["route_id"] == "route-west-coffee"
    assert payload["escort_handoff"]["scenario_id"] == "visitor_escort"


def test_space_tools_are_registered_as_agent_allowed_builtin_tools() -> None:
    registry = ToolRegistry(config={"default_timeout": 1.0})

    register_builtin_tools(registry, production_mode=True)

    allowed = registry.get_agent_allowed_names()
    assert "space_lookup_place" in allowed
    assert "space_recommend_route" in allowed
