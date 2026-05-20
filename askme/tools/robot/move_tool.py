"""Robot movement tool routed through runtime safety services.

Movement commands must flow through runtime services that own planning,
collision checks, emergency-stop state, and execution policy. This tool does
not publish directly to ROS2 topics or call vendor instruction endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from askme.tools.core.tool_registry import BaseTool
from askme.tools.robot.runtime_api import call_runtime_api

logger = logging.getLogger(__name__)


def _call_runtime_api(
    service: str,
    method: str,
    path: str,
    body: dict | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for historical imports."""
    return call_runtime_api(service, method, path, body)


class MoveRobotTool(BaseTool):
    """Control robot movement through runtime safety-checked APIs."""

    name = "move_robot"
    description = (
        "控制机器人运动（通过 runtime 安全层）。支持以下动作：\n"
        "- action='go_to', target='厨房' -> 语义导航（通过 nav-gateway，有路径规划和避障）\n"
        "- action='rotate', angle=90 -> 原地旋转（正=左转，负=右转，单位度）\n"
        "- action='forward', distance=1.0 -> 前进（单位米，负=后退）\n"
        "- action='stop' -> 立即停止\n"
        "注意：rotate/forward 需要 dog-control-service 支持，服务未配置时会返回错误。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["rotate", "forward", "go_to", "stop"],
                "description": "动作类型",
            },
            "angle": {
                "type": "number",
                "description": "旋转角度（度），正=左转，负=右转。仅 action=rotate 时有效",
            },
            "distance": {
                "type": "number",
                "description": "前进距离（米），负=后退。仅 action=forward 时有效",
            },
            "target": {
                "type": "string",
                "description": "目标位置名称（如'厨房'、'仓库'）。仅 action=go_to 时有效",
            },
        },
        "required": ["action"],
    }
    safety_level = "dangerous"
    agent_allowed = True
    voice_label = "移动机器人"

    def __init__(
        self,
        robot_control_client: Any | None = None,
        navigation_client: Any | None = None,
    ) -> None:
        self._robot_control_client = robot_control_client
        self._navigation_client = navigation_client

    def set_robot_control_client(self, robot_control_client: Any) -> None:
        self._robot_control_client = robot_control_client

    def set_navigation_client(self, navigation_client: Any) -> None:
        self._navigation_client = navigation_client

    def execute(
        self,
        *,
        action: str = "",
        angle: float = 0,
        distance: float = 0,
        target: str = "",
        **kwargs: Any,
    ) -> str:
        if action == "go_to":
            return self._go_to(target)
        if action == "rotate":
            return self._dispatch_control("rotate", {"angle_deg": angle})
        if action == "forward":
            return self._dispatch_control("walk_forward", {"distance_m": distance})
        if action == "stop":
            return self._dispatch_control("stop")
        return f"[错误] 未知动作: {action}"

    def _go_to(self, target: str) -> str:
        """Semantic navigation via Thunder nav-gateway dispatch path."""
        if not target:
            return "[错误] 请指定目标位置"
        return self._go_to_thunder(target)

    def _go_to_thunder(self, target: str) -> str:
        """Dispatch semantic navigation through nav-gateway."""
        from uuid import uuid4

        mission_id = uuid4().hex[:16]
        if self._navigation_client is not None:
            result = self._navigation_client.dispatch_navigation(
                "nav.semantic.execute",
                {"semantic_target": target},
                mission_type="voice_command",
                mission_id=mission_id,
            )
            return self._format_navigation_result(target, mission_id, result)

        result = _call_runtime_api(
            "nav",
            "POST",
            "/api/v1/navigation/dispatch",
            {
                "mission_id": mission_id,
                "mission_type": "voice_command",
                "requested_capability": "nav.semantic.execute",
                "parameters": {"semantic_target": target},
            },
        )
        return self._format_navigation_result(target, mission_id, result)

    def _format_navigation_result(
        self,
        target: str,
        mission_id: str,
        result: dict[str, Any],
    ) -> str:
        if "error" in result:
            err = result["error"]
            if "服务不可达" in err or "NAV_GATEWAY_URL" in err:
                return f"[导航不可用] nav-gateway 未运行。无法导航到 {target}。"
            return f"[导航错误] {err}"

        session = result.get("session", {})
        task_id = session.get(
            "mission_id",
            result.get("task_id", result.get("id", result.get("mission_id", mission_id))),
        )
        return f"导航任务已下发: 前往{target} (task_id={task_id})"

    def _dispatch_control(self, capability: str, params: dict | None = None) -> str:
        """Dispatch a movement capability to dog-control-service."""
        from uuid import uuid4

        if self._robot_control_client is not None:
            result = self._robot_control_client.dispatch_capability(capability, params or {})
            return self._format_control_result(capability, result)

        body = {
            "mission_id": uuid4().hex[:12],
            "mission_type": "motion_command",
            "requested_capability": capability,
            "parameters": params or {},
        }
        result = _call_runtime_api("control", "POST", "/api/v1/control/executions", body)
        return self._format_control_result(capability, result)

    def _format_control_result(self, capability: str, result: dict[str, Any]) -> str:
        if "error" in result:
            err = result["error"]
            if "服务不可达" in err:
                return (
                    "[控制不可用] dog-control-service 未运行。"
                    f"无法执行 {capability}。"
                    "请确认 runtime 服务已启动。"
                )
            return f"[控制错误] {err}"

        return f"已执行: {capability}"


def register_move_tools(
    registry: Any,
    robot_control_client: Any | None = None,
    navigation_client: Any | None = None,
) -> None:
    """Register movement tools."""
    registry.register(
        MoveRobotTool(
            robot_control_client=robot_control_client,
            navigation_client=navigation_client,
        )
    )
