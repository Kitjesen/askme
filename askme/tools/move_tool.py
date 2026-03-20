"""Robot movement tool — dispatches motion commands through runtime services.

Routes movement through the proper safety-checked path:
- go_to: nav-gateway API (semantic navigation with task tracking)
- rotate/forward/stop: dog-control-service API (capability dispatch)

DOES NOT directly publish to ROS2 topics — all motion goes through
runtime services that handle collision avoidance, safety checks, and
state management.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .tool_registry import BaseTool

logger = logging.getLogger(__name__)


def _call_runtime_api(
    service: str, method: str, path: str, body: dict | None = None
) -> dict[str, Any]:
    """Call a runtime service via HTTP. Returns parsed response or error dict."""
    import os
    import urllib.request
    import urllib.error

    port_map = {
        "control": 5080,
        "nav": 5090,
        "safety": 5070,
    }
    port = port_map.get(service)
    if not port:
        return {"error": f"unknown service: {service}"}

    # Check if service URL is configured via env
    env_key = f"DOG_{'CONTROL' if service == 'control' else service.upper()}_SERVICE_URL"
    base_url = os.environ.get(env_key, f"http://localhost:{port}")

    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        return {"error": f"服务不可达 ({service}:{port}): {exc.reason}"}
    except Exception as exc:
        return {"error": f"请求失败: {exc}"}


class MoveRobotTool(BaseTool):
    """Control robot movement through runtime safety-checked APIs."""

    name = "move_robot"
    description = (
        "控制机器人运动（通过 runtime 安全层）。支持以下动作：\n"
        "- action='go_to', target='厨房' → 语义导航（通过 nav-gateway，有路径规划和避障）\n"
        "- action='rotate', angle=90 → 原地旋转（正=左转，负=右转，单位度）\n"
        "- action='forward', distance=1.0 → 前进（单位米，负=后退）\n"
        "- action='stop' → 立即停止\n"
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
    safety_level = "normal"

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
        elif action == "rotate":
            return self._dispatch_control("rotate", {"angle_deg": angle})
        elif action == "forward":
            return self._dispatch_control("walk_forward", {"distance_m": distance})
        elif action == "stop":
            return self._dispatch_control("stop")
        else:
            return f"[错误] 未知动作: {action}"

    def _go_to(self, target: str) -> str:
        """Semantic navigation via nav-gateway API."""
        if not target:
            return "[错误] 请指定目标位置"

        from uuid import uuid4
        result = _call_runtime_api("nav", "POST", "/api/v1/nav/tasks", {
            "task_type": "SEMANTIC_NAV",
            "target_name": target,
            "mission_id": uuid4().hex[:12],
        })

        if "error" in result:
            err = result["error"]
            if "服务不可达" in err:
                return f"[导航不可用] nav-gateway 未运行。无法导航到 {target}。"
            return f"[导航错误] {err}"

        task_id = result.get("task_id", result.get("id", ""))
        return f"导航任务已下发: 前往{target} (task_id={task_id})"

    def _dispatch_control(self, capability: str, params: dict | None = None) -> str:
        """Dispatch a capability to dog-control-service."""
        from uuid import uuid4
        body = {
            "mission_id": uuid4().hex[:12],
            "mission_type": "motion_command",
            "requested_capability": capability,
            "parameters": params or {},
        }
        result = _call_runtime_api("control", "POST", "/api/v1/control/executions", body)

        if "error" in result:
            err = result["error"]
            if "服务不可达" in err:
                return (
                    f"[控制不可用] dog-control-service 未运行。"
                    f"无法执行 {capability}。"
                    f"请确认 runtime 服务已启动。"
                )
            return f"[控制错误] {err}"

        return f"已执行: {capability}"


def register_move_tools(registry: Any) -> None:
    """Register movement tools."""
    registry.register(MoveRobotTool())
