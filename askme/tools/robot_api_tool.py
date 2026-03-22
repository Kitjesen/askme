"""Unified Robot API Tool — wraps all 7 Thunder runtime REST services.

Agents use this single tool instead of remembering per-service ports.
All requests go through http://localhost:{port}/path with optional
Bearer token from config runtime.api_key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from askme.config import get_section
from .tool_registry import BaseTool

# Service name → default localhost port
_SERVICE_PORTS: dict[str, int] = {
    "arbiter":   5050,
    "telemetry": 5060,
    "safety":    5070,
    "control":   5080,
    "nav":       5090,
    "arm":       5100,
    "ops":       5110,
}


class RobotApiTool(BaseTool):
    """Unified Thunder runtime API tool for agents.

    Abstracts all runtime service endpoints behind a single tool so
    agents don't need to know ports or construct URLs manually.

    Services:
      - arbiter   (5050): mission lifecycle, multi-skill coordination
      - telemetry (5060): sensor data, health metrics, battery, IMU
      - safety    (5070): estop state, safety policy
      - control   (5080): posture, motion capabilities (stand/sit/move)
      - nav       (5090): navigation tasks, map management
      - arm       (5100): robot arm control (if equipped)
      - ops       (5110): OTA updates, config management
    """

    name = "robot_api"
    description = (
        "调用 Thunder 机器人 runtime 服务 API。\n"
        "服务说明：\n"
        "  arbiter(5050) — mission生命周期、多技能协调\n"
        "  telemetry(5060) — 传感器数据、电量、IMU健康\n"
        "  safety(5070) — 急停状态、安全策略\n"
        "  control(5080) — 姿态/运动（站立/坐下/移动）\n"
        "  nav(5090) — 导航任务、地图管理\n"
        "  arm(5100) — 机械臂控制\n"
        "  ops(5110) — OTA更新、配置管理"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["arbiter", "telemetry", "safety", "control", "nav", "arm", "ops"],
                "description": "目标服务名称",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP 方法",
            },
            "path": {
                "type": "string",
                "description": "API 路径，如 /api/v1/missions 或 /api/v1/safety/modes/estop",
            },
            "body": {
                "type": "object",
                "description": "请求体（JSON），仅 POST/PUT/PATCH 使用（可选）",
            },
        },
        "required": ["service", "method", "path"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "查询机器人"  # runtime services have their own safety layer

    _TIMEOUT = 10.0
    _MAX_RESPONSE = 4096

    def execute(
        self,
        *,
        service: str = "",
        method: str = "GET",
        path: str = "",
        body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if service not in _SERVICE_PORTS:
            return (
                f"[Error] 未知服务 '{service}'。"
                f"可用服务: {', '.join(_SERVICE_PORTS)}"
            )
        if not path:
            return "[Error] path 不能为空，如 /api/v1/missions"

        port = _SERVICE_PORTS[service]
        url = f"http://localhost:{port}{path}"
        method = method.upper()

        # Build request
        data: bytes | None = None
        headers: dict[str, str] = {"Accept": "application/json"}

        # Optional Bearer auth from runtime config
        try:
            api_key = get_section("runtime").get("api_key", "")
            if not api_key:
                api_key = os.environ.get("RUNTIME_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        except Exception:
            pass

        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                raw = resp.read(self._MAX_RESPONSE).decode("utf-8", errors="replace")
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    try:
                        parsed = json.loads(raw)
                        return json.dumps(
                            {"status": status, "body": parsed},
                            ensure_ascii=False,
                            indent=2,
                        )
                    except json.JSONDecodeError:
                        pass
                return json.dumps(
                    {"status": status, "body": raw[:2000]},
                    ensure_ascii=False,
                )
        except urllib.error.HTTPError as exc:
            body_text = exc.read(512).decode("utf-8", errors="replace")
            return json.dumps(
                {"status": exc.code, "error": exc.reason, "body": body_text},
                ensure_ascii=False,
            )
        except urllib.error.URLError as exc:
            return (
                f"[Error] {service} 服务不可达 (localhost:{port}): {exc.reason}。"
                "请确认服务是否已启动。"
            )
        except (TimeoutError, OSError):
            return f"[Error] {service} 服务请求超时 ({self._TIMEOUT}s)。"
        except Exception as exc:
            return f"[Error] {exc}"
