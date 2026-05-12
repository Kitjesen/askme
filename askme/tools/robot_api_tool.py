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
    "nav":       8088,
    "arm":       5100,
    "ops":       5110,
}

_SERVICE_ENV_URLS: dict[str, str] = {
    "control": "DOG_CONTROL_SERVICE_URL",
    "safety": "DOG_SAFETY_SERVICE_URL",
    "nav": "NAV_GATEWAY_URL",
}

_SERVICE_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "control": ("dog_control",),
    "safety": ("dog_safety",),
    "nav": ("nav_gateway", "dog_nav"),
}


def _service_base_url(service: str) -> str:
    """Resolve service URL with canonical env vars before localhost fallback."""
    env_key = _SERVICE_ENV_URLS.get(service)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key].rstrip("/")

    try:
        runtime_cfg = get_section("runtime")
        for cfg_key in _SERVICE_CONFIG_KEYS.get(service, ()):
            base_url = runtime_cfg.get(cfg_key, {}).get("base_url", "")
            if base_url:
                return str(base_url).rstrip("/")
    except Exception:
        pass

    return f"http://localhost:{_SERVICE_PORTS[service]}"


def _runtime_bearer_token() -> str:
    """Resolve runtime auth token with current env names before legacy fallback."""
    for key in ("RUNTIME_BEARER_TOKEN", "NOVA_DOG_RUNTIME_API_KEY"):
        value = os.environ.get(key, "")
        if value:
            return value

    try:
        runtime_cfg = get_section("runtime")
        api_key = runtime_cfg.get("api_key", "")
        if api_key:
            return str(api_key)
        voice_bridge_key = runtime_cfg.get("voice_bridge", {}).get("api_key", "")
        if voice_bridge_key:
            return str(voice_bridge_key)
    except Exception:
        pass

    return os.environ.get("RUNTIME_API_KEY", "")


class RobotApiTool(BaseTool):
    """Unified Thunder runtime API tool for agents.

    Abstracts all runtime service endpoints behind a single tool so
    agents don't need to know ports or construct URLs manually.

    Services:
      - arbiter   (5050): mission lifecycle, multi-skill coordination
      - telemetry (5060): sensor data, health metrics, battery, IMU
      - safety    (5070): estop state, safety policy
      - control   (5080): posture, motion capabilities (stand/sit/move)
      - nav       (8088): navigation tasks, map management
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
        "  nav(8088) — 导航任务、地图管理\n"
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
    safety_level = "dangerous"
    agent_allowed = True
    voice_label = "操作机器人 API"

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

        base_url = _service_base_url(service)
        if not path.startswith("/"):
            path = "/" + path
        url = f"{base_url}{path}"
        method = method.upper()

        # Build request
        data: bytes | None = None
        headers: dict[str, str] = {"Accept": "application/json"}

        token = _runtime_bearer_token()
        if token:
            headers["Authorization"] = (
                token if token.lower().startswith("bearer ") else f"Bearer {token}"
            )

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
                f"[Error] {service} 服务不可达 ({base_url}): {exc.reason}。"
                "请确认服务是否已启动。"
            )
        except (TimeoutError, OSError):
            return f"[Error] {service} 服务请求超时 ({self._TIMEOUT}s)。"
        except Exception as exc:
            return f"[Error] {exc}"
