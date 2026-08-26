"""
Built-in tools for askme.

Ports the four original tools (get_time, run_command, read_file, list_directory)
from the prototype as proper BaseTool subclasses.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from askme.config import get_section, project_root
from askme.tools.core.tool_registry import BaseTool, ToolRegistry
from askme.tools.field.field_event_tool import FieldEventTriggerTool
from askme.tools.spatial.space_tool import SpaceLookupPlaceTool, SpaceRecommendRouteTool

# Directories the LLM is allowed to read. Data produced by askme itself
# (sessions, memory, logs) is fair game; source code and credentials are not.
_ALLOWED_READ_ROOTS: tuple[Path, ...] = (
    project_root() / "data",
    project_root() / "logs",
    project_root() / "askme" / "skills",  # skill SKILL.md files
)


def _is_path_allowed(raw_path: str) -> bool:
    """Return True if *raw_path* resolves to a directory allowed for LLM reads.

    Prevents prompt-injection attacks that try to exfiltrate credentials
    (e.g. ``read_file(".env")`` or ``read_file("config.yaml")``).
    """
    try:
        resolved = Path(raw_path).resolve()
    except Exception:
        return False
    return any(
        resolved == allowed or allowed in resolved.parents
        for allowed in _ALLOWED_READ_ROOTS
    )


class GetTimeTool(BaseTool):
    """Return the current system date and time."""

    name = "get_current_time"
    description = "获取当前系统时间"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"
    agent_allowed = True
    voice_label = "获取时间"

    def execute(self, **kwargs: Any) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunCommandTool(BaseTool):
    """Execute a shell command and return its output."""

    name = "run_command"
    description = "在本地执行一条 shell 命令并返回输出（最多 2000 字符）"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令",
            },
        },
        "required": ["command"],
    }
    safety_level = "dangerous"
    dev_only = True  # excluded in production_mode

    def execute(self, *, command: str = "", **kwargs: Any) -> str:
        if not command:
            return "[Error] No command provided."
        try:
            shlex.split(command)
        except ValueError as exc:
            return f"[Error] Invalid command syntax: {exc}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout or result.stderr or "(no output)"
            return output[:2000]
        except subprocess.TimeoutExpired:
            return "[Error] Command timed out (10s)."
        except Exception as exc:
            return f"[Error] {exc}"


class ReadFileTool(BaseTool):
    """Read the contents of a local file."""

    name = "read_file"
    description = "读取本地文件内容（最多 3000 字符）"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
        },
        "required": ["path"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "读取文件"

    def execute(self, *, path: str = "", **kwargs: Any) -> str:
        if not path:
            return "[Error] No path provided."
        # Resolve relative paths against project root (e.g. "data/qp_memory/...")
        if not Path(path).is_absolute():
            path = str(project_root() / path)
        if not _is_path_allowed(path):
            return (
                "[Error] 路径不在允许的读取范围内。"
                " LLM 只能读取 data/、logs/、askme/skills/ 目录下的文件。"
            )
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read(3000)
        except Exception as exc:
            return f"[Error] Reading file failed: {exc}"


class ListDirectoryTool(BaseTool):
    """List files and directories in a given path."""

    name = "list_directory"
    description = "列出目录中的文件和文件夹"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径，默认当前目录",
            },
        },
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "查看目录"

    def execute(self, *, path: str = ".", **kwargs: Any) -> str:
        target = path or "."
        if not _is_path_allowed(target):
            return (
                "[Error] 路径不在允许的列目录范围内。"
                " LLM 只能列出 data/、logs/、askme/skills/ 目录。"
            )
        try:
            entries = os.listdir(target)
            return "\n".join(entries[:50])
        except Exception as exc:
            return f"[Error] {exc}"


def _http_allowlist() -> list[str]:
    """Load allowed URL prefixes from config; auto-append runtime service URLs."""
    try:
        cfg = list(get_section("tools").get("http_allowlist", []))
    except Exception:
        cfg = []
    # Auto-append configured runtime service base URLs so GET diagnostics work
    # while generic writes are still blocked below by runtime identity.
    try:
        runtime_cfg = get_section("runtime")
        for svc in (
            "arbiter",
            "telemetry",
            "dog_safety",
            "dog_control",
            "nav_gateway",
            "dog_nav",
            "arm",
            "ops",
            "voice_bridge",
        ):
            url = runtime_cfg.get(svc, {}).get("base_url", "")
            if url:
                cfg.append(url.rstrip("/"))
    except Exception:
        pass
    cfg.extend(_robot_runtime_base_urls())
    # Auto-include standard runtime service ports on localhost
    for port in (5050, 5060, 5070, 5080, 8088, 5100, 5110):
        cfg.append(f"http://localhost:{port}")
    return list(dict.fromkeys(cfg))


def _is_url_allowed(url: str, allowlist: list[str]) -> bool:
    """Return True if url matches at least one prefix in allowlist.

    Empty allowlist = allow localhost/127.0.0.1/[::1] only.
    """
    if not allowlist:
        parsed = urllib.parse.urlparse(url)
        return parsed.hostname in ("localhost", "127.0.0.1", "::1")
    return any(url.startswith(prefix) for prefix in allowlist)


_ROBOT_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_ROBOT_RUNTIME_PORTS = {5050, 5060, 5070, 5080, 8088, 5100, 5110}
_ROBOT_RUNTIME_ENV_URLS = (
    "RUNTIME_ARBITER_URL",
    "RUNTIME_TELEMETRY_URL",
    "DOG_SAFETY_SERVICE_URL",
    "DOG_CONTROL_SERVICE_URL",
    "NAV_GATEWAY_URL",
    "DOG_NAV_SERVICE_URL",
    "DOG_ARM_SERVICE_URL",
    "DOG_OPS_SERVICE_URL",
)
_ROBOT_RUNTIME_CONFIG_KEYS = (
    "arbiter",
    "telemetry",
    "dog_safety",
    "dog_control",
    "nav_gateway",
    "dog_nav",
    "arm",
    "ops",
)


def _robot_runtime_base_urls() -> set[str]:
    """Return configured runtime service base URLs used to identify robot writes."""
    urls = {
        value.rstrip("/")
        for key in _ROBOT_RUNTIME_ENV_URLS
        if (value := os.environ.get(key, "").strip())
    }
    try:
        runtime_cfg = get_section("runtime")
        for key in _ROBOT_RUNTIME_CONFIG_KEYS:
            section = runtime_cfg.get(key, {})
            if isinstance(section, dict):
                base_url = str(section.get("base_url", "")).strip()
                if base_url:
                    urls.add(base_url.rstrip("/"))
    except Exception:
        pass
    return urls


def _same_service_base(url: str, base_url: str) -> bool:
    """Return True when url is inside a configured service base URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        base = urllib.parse.urlparse(base_url)
    except Exception:
        return False
    if not parsed.scheme or not parsed.hostname or not base.scheme or not base.hostname:
        return False
    if parsed.scheme.lower() != base.scheme.lower():
        return False
    if (parsed.hostname or "").lower() != (base.hostname or "").lower():
        return False
    parsed_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_port = base.port or (443 if base.scheme == "https" else 80)
    if parsed_port != base_port:
        return False
    base_path = (base.path or "/").rstrip("/")
    if not base_path:
        return True
    return parsed.path == base_path or parsed.path.startswith(base_path + "/")


def _is_robot_runtime_url(url: str) -> bool:
    """Return True when url points at a robot runtime service."""
    parsed = urllib.parse.urlparse(url)
    try:
        if parsed.port in _ROBOT_RUNTIME_PORTS:
            return True
    except ValueError:
        return False
    return any(_same_service_base(url, base_url) for base_url in _robot_runtime_base_urls())


def _is_robot_write_url(method: str, url: str) -> bool:
    """Return True for generic writes aimed at robot runtime services."""
    if method.upper() not in _ROBOT_WRITE_METHODS:
        return False
    return _is_robot_runtime_url(url)


def _extract_http_urls(text: str) -> list[str]:
    """Extract HTTP(S) URLs from shell snippets without trying to parse the shell."""
    return [
        match.rstrip(".,;)]}'\"")
        for match in re.findall(r"https?://[^\s\"'<>]+", text, flags=re.IGNORECASE)
    ]


def _command_uses_robot_write_http(command: str) -> bool:
    """Detect curl/wget/python snippets that write directly to robot runtime APIs."""
    urls = [url for url in _extract_http_urls(command) if _is_robot_runtime_url(url)]
    if not urls:
        return False

    cmd_lower = command.lower()
    method_pattern = r"(?:-x|--request|--method(?:=|\s+))\s*['\"]?(post|put|patch|delete)\b"
    has_write_method = re.search(method_pattern, cmd_lower) is not None
    has_write_payload = re.search(
        r"(?:^|\s)(?:-d|--data(?:-[\w-]+)?|--json|--form|--post-data|--post-file)(?:=|\s|$)",
        cmd_lower,
    ) is not None or any(token in cmd_lower for token in (" data=", ".data ="))
    has_python_write_call = re.search(
        r"(?:requests|httpx)\s*\.\s*(post|put|patch|delete)\s*\(",
        cmd_lower,
    ) is not None
    has_python_request_write_call = (
        re.search(r"(?:requests|httpx)\s*\.\s*request\s*\(", cmd_lower) is not None
        and re.search(r"['\"](post|put|patch|delete)['\"]", cmd_lower) is not None
    )
    has_python_method_kwarg = re.search(
        r"method\s*=\s*['\"]?(post|put|patch|delete)\b",
        cmd_lower,
    ) is not None
    has_http_client_write = re.search(
        r"\.request\s*\(\s*['\"](post|put|patch|delete)['\"]",
        cmd_lower,
    ) is not None
    return (
        has_write_method
        or has_write_payload
        or has_python_write_call
        or has_python_request_write_call
        or has_python_method_kwarg
        or has_http_client_write
    )


class HttpRequestTool(BaseTool):
    """Make an HTTP request to a configured service endpoint.

    Skills use this to call real services (dog-control, nav-gateway,
    custom REST APIs, etc.).  Allowed URL prefixes are configured in
    config.yaml ``tools.http_allowlist``.
    """

    name = "http_request"
    description = (
        "向外部服务发送 HTTP 请求（GET/POST/PUT/DELETE）并返回响应。"
        "用于调用机器人控制服务、导航服务等 REST API。"
        "URL 必须在 config.yaml tools.http_allowlist 白名单中。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP 方法",
            },
            "url": {
                "type": "string",
                "description": "请求 URL，如 http://localhost:8080/api/v1/action/stand",
            },
            "body": {
                "type": "object",
                "description": "请求体（JSON），仅 POST/PUT/PATCH 使用",
            },
            "headers": {
                "type": "object",
                "description": "额外的 HTTP 请求头（可选）",
            },
            "timeout": {
                "type": "number",
                "description": "超时秒数，默认 5",
            },
        },
        "required": ["method", "url"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "调用接口"

    def execute(
        self,
        *,
        method: str = "GET",
        url: str = "",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> str:
        if not url:
            return "[Error] URL is required."

        method = method.upper()
        if _is_robot_write_url(method, url):
            return (
                "[Error] Robot runtime write requests are blocked through "
                "http_request. Use safety-gated robot tools instead."
            )

        allowlist = _http_allowlist()
        if not _is_url_allowed(url, allowlist):
            allowed_display = ", ".join(allowlist) if allowlist else "localhost only"
            return (
                f"[Error] URL '{url}' 不在白名单中。"
                f" 允许的前缀：{allowed_display}。"
                " 请在 config.yaml tools.http_allowlist 中添加。"
            )

        data: bytes | None = None
        req_headers: dict[str, str] = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(8192).decode("utf-8", errors="replace")
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                # Try to pretty-print JSON responses
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
            return f"[Error] 请求失败: {exc.reason}"
        except (TimeoutError, OSError):
            return f"[Error] 请求超时 ({timeout}s): {url}"
        except Exception as exc:
            return f"[Error] {exc}"


class NavStatusTool(BaseTool):
    """Query navigation system status from the runtime service."""

    name = "nav_status"
    description = "查询机器人当前导航状态（当前任务、位置、运行状态）"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"
    read_only = True

    def __init__(self, navigation_client: Any | None = None) -> None:
        self._navigation_client = navigation_client

    def set_navigation_client(self, navigation_client: Any) -> None:
        self._navigation_client = navigation_client

    @staticmethod
    def _format_status(result: dict[str, Any]) -> str:
        reason_codes = result.get("reason_codes") or []
        readiness = result.get("readiness") or {}
        blockers = readiness.get("blockers") or [] if isinstance(readiness, dict) else []
        if (
            result.get("has_odometry") is False
            or "odometry_missing" in reason_codes
            or "odometry_missing" in blockers
        ):
            return "定位未就绪，暂时无法获取当前位置。"
        odometry = result.get("odometry") or result.get("position")
        if isinstance(odometry, dict):
            try:
                x = float(odometry["x"])
                y = float(odometry["y"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                frame = str(odometry.get("frame_id") or "地图")
                return (
                    f"当前位置：{frame} 坐标系，横坐标 {x:.2f} 米，"
                    f"纵坐标 {y:.2f} 米。"
                )
        return json.dumps(result, ensure_ascii=False, indent=2)

    def execute(self, **kwargs: Any) -> str:
        import json as _json
        import urllib.request

        if self._navigation_client is not None:
            result = self._navigation_client.status()
            if "error" in result:
                return f"[导航状态] 查询失败: {result['error']}"
            return self._format_status(result)

        url = os.environ.get("NAV_GATEWAY_URL", "").rstrip("/")
        if not url:
            return "[导航状态] 导航服务未配置（NAV_GATEWAY_URL 未设置）"
        try:
            with urllib.request.urlopen(url + "/api/v1/navigation/status", timeout=3) as resp:
                data = _json.loads(resp.read())
                return self._format_status(data)
        except Exception as exc:
            return f"[导航状态] 查询失败: {exc}"


class NavDispatchTool(BaseTool):
    """Dispatch a navigation task to Thunder nav-gateway.

    This is the actual execution tool for the navigate/mapping/follow_person
    skills.  Without this, skills could only confirm — not execute.

    Requires NAV_GATEWAY_URL environment variable pointing to nav-gateway
    (e.g. http://localhost:8088).
    """

    # capability strings expected by the nav-gateway capability router
    _CAPABILITY_MAP: dict[str, str] = {
        "navigate":      "nav.semantic.execute",
        "mapping":       "nav.mapping.start",
        "follow_person": "nav.follow_person.start",
    }

    name = "nav_dispatch"
    description = (
        "向导航服务下发导航任务（语义导航/建图/跟随）。"
        "成功后机器人开始移动，失败时返回错误原因。"
        "需要配置 NAV_GATEWAY_URL 环境变量。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "目标位置描述，如'仓库A'、'出口'、'充电桩'",
            },
            "task_type": {
                "type": "string",
                "enum": ["navigate", "mapping", "follow_person"],
                "description": "任务类型：navigate=语义导航（默认），mapping=SLAM建图，follow_person=跟随人",
            },
            "params": {
                "type": "object",
                "description": "额外参数（可选），如 {\"map_scope\": \"全区\"}",
            },
        },
        "required": ["destination"],
    }
    safety_level = "dangerous"
    cancelable = True
    cancel_on_turn_interrupt = True
    side_effect_class = "physical_motion"

    def __init__(self, navigation_client: Any | None = None) -> None:
        self._navigation_client = navigation_client
        self._dispatch_done = threading.Event()
        self._dispatch_done.set()

    def set_navigation_client(self, navigation_client: Any) -> None:
        self._navigation_client = navigation_client

    def request_cancel(self, reason: str) -> bool:
        """Gracefully cancel navigation without invoking the E-STOP route."""
        dispatch_was_running = not self._dispatch_done.is_set()
        accepted = self._send_graceful_cancel(reason)
        if dispatch_was_running and self._dispatch_done.wait(timeout=5.5):
            # Cancellation can race a dispatch acknowledgement. Repeating the
            # idempotent graceful-cancel route after the acknowledgement keeps
            # a late-created mission from escaping the interrupted turn.
            accepted = self._send_graceful_cancel(reason) or accepted
        return accepted

    def _send_graceful_cancel(self, reason: str) -> bool:
        cancel_navigation = getattr(self._navigation_client, "cancel_navigation", None)
        if callable(cancel_navigation):
            try:
                result = cancel_navigation(reason=reason)
                return not isinstance(result, dict) or "error" not in result
            except Exception:
                return False

        url = os.environ.get("NAV_GATEWAY_URL", "").rstrip("/")
        if not url:
            return False
        data = json.dumps({"reason": str(reason or "turn_interrupted")}).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            url + "/api/v1/navigation/cancel",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                return True
        except Exception:
            return False

    def _build_parameters(
        self, task_type: str, destination: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        p = params or {}
        if task_type == "navigate":
            return {"semantic_target": destination}
        if task_type == "mapping":
            return {"map_name": p.get("map_scope") or destination, "save_on_complete": True}
        # follow_person: no extra parameters needed
        return {}

    def execute(
        self,
        *,
        destination: str = "",
        task_type: str = "navigate",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        self._dispatch_done.clear()
        try:
            return self._execute_dispatch(
                destination=destination,
                task_type=task_type,
                params=params,
                **kwargs,
            )
        finally:
            self._dispatch_done.set()

    def _execute_dispatch(
        self,
        *,
        destination: str = "",
        task_type: str = "navigate",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        import json as _json
        import urllib.request
        from uuid import uuid4

        url = os.environ.get("NAV_GATEWAY_URL", "").rstrip("/")
        if not url and self._navigation_client is None:
            return (
                "[导航] 导航服务未配置。"
                "请设置 NAV_GATEWAY_URL 环境变量，例如: http://localhost:8088"
            )
        if not destination and task_type != "follow_person":
            return "[Error] 目标位置不能为空"

        capability = self._CAPABILITY_MAP.get(task_type, "nav.semantic.execute")
        params_payload = self._build_parameters(task_type, destination, params)
        if self._navigation_client is not None:
            result = self._navigation_client.dispatch_navigation(
                capability,
                params_payload,
                mission_type="voice_command",
            )
            return self._format_dispatch_result(result, fallback_mission_id="")

        body: dict[str, Any] = {
            "mission_id": uuid4().hex[:16],
            "mission_type": "voice_command",
            "requested_capability": capability,
            "parameters": params_payload,
        }
        data = _json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url + "/api/v1/navigation/dispatch",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read(4096).decode("utf-8", errors="replace")
                try:
                    result = _json.loads(raw)
                    return self._format_dispatch_result(
                        result,
                        fallback_mission_id=body["mission_id"],
                    )
                except _json.JSONDecodeError:
                    return f"[导航] 任务已下发 (HTTP {resp.status})"
        except urllib.error.HTTPError as exc:
            body_text = exc.read(256).decode("utf-8", errors="replace")
            return f"[导航] 下发失败 (HTTP {exc.code}): {body_text}"
        except urllib.error.URLError as exc:
            return f"[导航] 导航服务不可达: {exc.reason}。请检查 NAV_GATEWAY_URL={url}"
        except (TimeoutError, OSError):
            return "[导航] 请求超时 (5s)，请检查导航服务是否在线"
        except Exception as exc:
            return f"[导航] 请求异常: {exc}"

    def _format_dispatch_result(
        self,
        result: dict[str, Any],
        *,
        fallback_mission_id: str,
    ) -> str:
        if "error" in result:
            err = str(result["error"])
            if "NAV_GATEWAY_URL" in err:
                return (
                    "[导航] 导航服务未配置。"
                    "请设置 NAV_GATEWAY_URL 环境变量，例如: http://localhost:8088"
                )
            status = result.get("http_status")
            if status:
                return f"[导航] 下发失败 (HTTP {status}): {err}"
            return f"[导航] 导航服务不可达: {err}"

        session = result.get("session", {}) if isinstance(result, dict) else {}
        mission_id = session.get(
            "mission_id",
            result.get("mission_id", result.get("task_id", fallback_mission_id)),
        )
        state = session.get("state", result.get("state", "submitted"))
        return f"[导航] 任务已下发 (state={state}, mission_id={mission_id})"


class DogControlDispatchTool(BaseTool):
    """Dispatch posture/motion capability to Thunder via dog-control-service.

    This is the actual execution tool for dog_control skill.
    Uses DogControlClient which enforces the safety layer contract:
    all motion commands go through dog-control-service, never bypassing
    dog-safety-service.

    Requires an injected RobotControlPort. Runtime and MCP startup wire that
    port from the configured DOG_CONTROL_SERVICE_URL provider.
    """

    name = "dog_control_dispatch"
    description = (
        "向 Thunder 机器人下发姿态/运动指令（站立、坐下、趴下等）。"
        "指令经由 dog-control-service 安全层执行，确保不绕过安全检查。"
        "需要配置 DOG_CONTROL_SERVICE_URL 环境变量。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "capability": {
                "type": "string",
                "description": (
                    "机器人能力名称，支持：stand（站立）、sit（坐下）、"
                    "lie_down（趴下）、start_patrol（开始巡逻）、stop（停止）"
                ),
            },
            "params": {
                "type": "object",
                "description": "额外参数（可选），如 {\"speed\": 0.5}",
            },
        },
        "required": ["capability"],
    }
    safety_level = "dangerous"

    def __init__(self, robot_control_client: Any | None = None) -> None:
        self._robot_control_client = robot_control_client

    def set_robot_control_client(self, robot_control_client: Any) -> None:
        self._robot_control_client = robot_control_client

    def execute(
        self,
        *,
        capability: str = "",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if not capability:
            return "[Error] 未指定能力名称"

        client = self._robot_control_client
        if client is None:
            return (
                "[Thunder] Robot control service is not configured. "
                "Inject a RobotControlPort through runtime/control or MCP context, "
                "or start through runtime/MCP so DOG_CONTROL_SERVICE_URL can be wired "
                "by the provider layer."
            )
        if not client.is_configured():
            return (
                "[Thunder] 机器人控制服务未配置。"
                "请设置 DOG_CONTROL_SERVICE_URL 环境变量，例如: http://localhost:5080"
            )

        result = client.dispatch_capability(capability, params)
        if "error" in result:
            return f"[Thunder] 指令下发失败: {result['error']}"

        status = result.get("status", "dispatched")
        exec_id = result.get("execution_id", result.get("id", ""))
        msg = f"[Thunder] 已下发 '{capability}' 指令 (status={status}"
        if exec_id:
            msg += f", id={exec_id}"
        msg += ")"
        return msg


class DispatchSkillTool(BaseTool):
    """Meta-tool: lets the LLM invoke a named skill during general conversation.

    This enables natural language → skill composition without voice triggers.
    The LLM sees the skill catalog in its system prompt and can decide to
    dispatch skills based on user intent.
    """

    name = "dispatch_skill"
    description = (
        "执行一个机器人技能（如导航、检查、建图等）。"
        "可用技能列表见系统提示。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "要执行的技能名称",
            },
            "reason": {
                "type": "string",
                "description": "调用原因/传给技能的上下文",
            },
        },
        "required": ["skill_name"],
    }
    safety_level = "normal"  # meta-tool only; actual dangerous skills carry their own safety_level

    def __init__(self) -> None:
        # Dispatcher is set after construction via set_dispatcher()
        self._dispatcher: Any = None

    def set_dispatcher(self, dispatcher: Any) -> None:
        """Wire the SkillDispatcher after both are constructed."""
        self._dispatcher = dispatcher

    def execute(self, *, skill_name: str = "", reason: str = "", **kwargs: Any) -> str:
        if not skill_name:
            return "[Error] 未指定技能名称"
        if self._dispatcher is None:
            return "[Error] SkillDispatcher 未初始化"
        return self._dispatcher.execute_skill_sync(skill_name, reason)


class SandboxedBashTool(BaseTool):
    """Execute bash commands sandboxed to data/agent_workspace/."""

    name = "bash"
    description = (
        "在机器人本地执行 bash/shell 命令（限沙箱 workspace 目录）。"
        "适合运行脚本、查看文件、执行 Python 代码等操作。"
        "工作区路径：data/agent_workspace/"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令",
            },
            "cwd": {
                "type": "string",
                "description": "工作目录（相对于 agent_workspace/），可选",
            },
        },
        "required": ["command"],
    }
    safety_level = "dangerous"
    dev_only = False  # production-safe: sandboxed

    agent_allowed = True
    voice_label = "运行命令"

    _WORKSPACE = project_root() / "data" / "agent_workspace"
    _MAX_OUTPUT = 4000
    _TIMEOUT = 30

    # Commands that must NEVER be executed by the agent
    _BLOCKED_COMMANDS = {
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){ :|:& };:",
        "reboot", "shutdown", "poweroff", "halt",
        "passwd", "useradd", "userdel", "visudo",
    }
    _BLOCKED_PREFIXES = (
        "rm -rf /",     # recursive delete from root
        "apt remove", "apt purge", "apt autoremove",
        "pip uninstall",
        "systemctl stop", "systemctl disable",  # stopping services
        "kill -9", "killall",
    )
    # These prefixes are allowed even though they match blocked patterns
    _SAFE_PREFIXES = (
        "systemctl status", "systemctl is-active",
        "journalctl", "df", "free", "top -bn1", "uptime",
        "cat /var/log", "tail", "head", "grep", "wc",
        "ls", "pwd", "whoami", "hostname", "date",
        "ping -c", "curl", "wget -q",
        "python", "pip list", "pip show",
        "ros2 topic list", "ros2 node list", "ros2 topic echo",
    )

    def _is_command_safe(self, command: str) -> str | None:
        """Return error message if command is blocked, None if safe."""
        cmd_lower = command.strip().lower()

        if _command_uses_robot_write_http(command):
            return (
                "[安全拒绝] 机器人 runtime 写请求不能通过 bash/curl/wget/python "
                "绕过安全工具。请使用 robot_api 或专用安全门控工具。"
            )

        # Exact match block
        for blocked in self._BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return f"[安全拒绝] 命令 '{blocked}' 被禁止执行。"

        # Prefix block (but allow safe overrides)
        for prefix in self._BLOCKED_PREFIXES:
            if cmd_lower.startswith(prefix):
                # Check if it matches a safe prefix
                is_safe = any(cmd_lower.startswith(s) for s in self._SAFE_PREFIXES)
                if not is_safe:
                    return f"[安全拒绝] 以 '{prefix}' 开头的命令被禁止。仅允许诊断类命令。"

        return None

    def execute(self, *, command: str = "", cwd: str = "", **kwargs: Any) -> str:
        if not command:
            return "[Error] No command provided."

        # Safety check
        blocked = self._is_command_safe(command)
        if blocked:
            return blocked

        workspace = self._WORKSPACE
        workspace.mkdir(parents=True, exist_ok=True)

        # Resolve working directory — must stay inside workspace
        if cwd:
            candidate = (workspace / cwd).resolve()
        else:
            candidate = workspace.resolve()

        try:
            workspace_resolved = workspace.resolve()
        except Exception as exc:
            return f"[Error] Cannot resolve workspace: {exc}"

        if candidate != workspace_resolved and workspace_resolved not in candidate.parents:
            return (
                f"[Error] 路径逃逸被拒绝。cwd '{cwd}' 解析到 '{candidate}'，"
                f"不在允许的工作区 '{workspace_resolved}' 内。"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(candidate),
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            if not output:
                output = f"(exit code {result.returncode}, no output)"
            if len(output) > self._MAX_OUTPUT:
                output = output[: self._MAX_OUTPUT] + "\n...[输出已截断]"
            return output
        except subprocess.TimeoutExpired:
            return f"[Timeout] 命令执行超过 {self._TIMEOUT}s，已终止。"
        except Exception as exc:
            return f"[Error] {exc}"


class WriteFileTool(BaseTool):
    """Write or create a file inside data/agent_workspace/."""

    name = "write_file"
    description = (
        "在 agent workspace 创建或覆写文件。"
        "路径相对于 data/agent_workspace/，不能逃逸到工作区外。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于 agent_workspace/），如 'output.txt' 或 'scripts/hello.py'",
            },
            "content": {
                "type": "string",
                "description": "文件内容",
            },
        },
        "required": ["path", "content"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "写入文件"

    _ALLOWED_ROOT = project_root() / "data" / "agent_workspace"

    def execute(self, *, path: str = "", content: str = "", **kwargs: Any) -> str:
        if not path:
            return "[Error] No path provided."

        allowed_root = self._ALLOWED_ROOT
        allowed_root.mkdir(parents=True, exist_ok=True)

        try:
            allowed_resolved = allowed_root.resolve()
            target = (allowed_root / path).resolve()
        except Exception as exc:
            return f"[Error] 路径解析失败: {exc}"

        if target != allowed_resolved and allowed_resolved not in target.parents:
            return (
                f"[Error] 路径逃逸被拒绝。'{path}' 解析到 '{target}'，"
                f"不在允许的工作区 '{allowed_resolved}' 内。"
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"已写入 {target}（{len(content)} 字符）"
        except Exception as exc:
            return f"[Error] 写入文件失败: {exc}"


class EditFileTool(BaseTool):
    """Surgical string replacement in a file — old_string must match exactly once."""

    name = "edit_file"
    description = (
        "在文件中做精确字符串替换（old_string → new_string）。"
        "old_string 必须在文件中唯一出现，否则返回错误。"
        "支持绝对路径，或相对于 agent_workspace/ 的相对路径。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（绝对路径，或相对于 agent_workspace/ 的相对路径）",
            },
            "old_string": {
                "type": "string",
                "description": "要替换的原始字符串（必须在文件中唯一出现，含缩进/换行）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新字符串",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "编辑文件"

    _WORKSPACE = project_root() / "data" / "agent_workspace"

    def execute(
        self, *, path: str = "", old_string: str = "", new_string: str = "", **kwargs: Any
    ) -> str:
        if not path:
            return "[Error] No path provided."
        if not old_string:
            return "[Error] old_string cannot be empty."

        target = Path(path)
        if not target.is_absolute():
            target = self._WORKSPACE / path
        try:
            target = target.resolve()
        except Exception as exc:
            return f"[Error] 路径解析失败: {exc}"

        if not target.exists():
            return f"[Error] 文件不存在: {target}"
        if not target.is_file():
            return f"[Error] 不是文件: {target}"

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"[Error] 读取文件失败: {exc}"

        count = content.count(old_string)
        if count == 0:
            return (
                "[Error] 未找到目标字符串，编辑失败。"
                "请检查 old_string 是否与文件内容完全一致（含缩进和换行）。"
            )
        if count > 1:
            return (
                f"[Error] old_string 在文件中出现了 {count} 次，无法唯一定位。"
                "请提供更多上下文使其唯一。"
            )

        new_content = content.replace(old_string, new_string, 1)
        try:
            target.write_text(new_content, encoding="utf-8")
            return f"已编辑 {target}：替换了 1 处（{len(old_string)} → {len(new_string)} 字符）"
        except Exception as exc:
            return f"[Error] 写入失败: {exc}"


class SpeakProgressTool(BaseTool):
    """Non-blocking TTS progress announcements during long agent tasks."""

    name = "speak_progress"
    description = (
        "向用户实时播报执行进度（非阻塞，继续执行下一步）。"
        "在长任务中定期汇报，防止用户以为系统卡住。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要播报的进度文本，如'正在分析数据，请稍候...'",
            },
        },
        "required": ["text"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = ""

    def __init__(self, audio_agent: Any = None) -> None:
        self._audio = audio_agent

    def execute(self, *, text: str = "", **kwargs: Any) -> str:
        if not text:
            return "[Error] No text provided."
        if self._audio is not None:
            try:
                self._audio.speak(text)
            except Exception:
                pass  # best-effort; never crash the agent task
        return "已播报"


class WebFetchTool(BaseTool):
    """Fetch a web page and return cleaned text content."""

    name = "web_fetch"
    description = (
        "抓取网页内容并返回清洁后的文本（去除 HTML 标签）。"
        "用于访问文档、GitHub、新闻页面等。最多返回 6000 字符。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的 URL，如 https://docs.python.org/3/library/asyncio.html",
            },
            "max_chars": {
                "type": "integer",
                "description": "返回最大字符数，默认 6000",
            },
        },
        "required": ["url"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "抓取网页"
    _TIMEOUT = 15
    _DEFAULT_MAX = 6000

    # Headers to mimic a real browser (avoid bot blocks)
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def execute(self, *, url: str = "", max_chars: int = 0, **kwargs: Any) -> str:
        import html as _html
        import re as _re

        if not url:
            return "[Error] URL 不能为空。"
        if not url.startswith(("http://", "https://")):
            return "[Error] URL 必须以 http:// 或 https:// 开头。"

        limit = max_chars if max_chars and max_chars > 0 else self._DEFAULT_MAX

        req = urllib.request.Request(url, headers=self._HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    raw = resp.read(limit * 2).decode("utf-8", errors="replace")
                    try:
                        import json as _json
                        parsed = _json.loads(raw)
                        text = _json.dumps(parsed, ensure_ascii=False, indent=2)
                    except Exception:
                        text = raw
                else:
                    raw = resp.read(limit * 4).decode("utf-8", errors="replace")
                    # Strip script/style blocks first
                    text = _re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", raw, flags=_re.IGNORECASE)
                    # Strip HTML tags
                    text = _re.sub(r"<[^>]+>", " ", text)
                    # Decode HTML entities
                    text = _html.unescape(text)
                    # Collapse whitespace
                    text = _re.sub(r"\s{3,}", "\n\n", text)
                    text = text.strip()

                if len(text) > limit:
                    text = text[:limit] + "\n...[已截断]"
                return text or "(页面为空)"
        except urllib.error.HTTPError as exc:
            return f"[Error] HTTP {exc.code}: {exc.reason} — {url}"
        except urllib.error.URLError as exc:
            return f"[Error] 无法访问 {url}: {exc.reason}"
        except (TimeoutError, OSError):
            return f"[Error] 请求超时 ({self._TIMEOUT}s): {url}"
        except Exception as exc:
            return f"[Error] {exc}"


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo Instant Answer API (no API key needed)."""

    name = "web_search"
    description = (
        "搜索互联网获取最新信息（使用 DuckDuckGo，无需 API key）。"
        "返回摘要和相关链接。适合查询最新资讯、技术文档、定义等。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如 'Python asyncio best practices 2024'",
            },
        },
        "required": ["query"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "搜索网络"
    _TIMEOUT = 10

    def execute(self, *, query: str = "", **kwargs: Any) -> str:
        if not query:
            return "[Error] 搜索关键词不能为空。"

        # China-first search order: Bing → Baidu (DDG is blocked in China)
        result = self._bing_fallback(query)
        if result and not result.startswith("[Error]"):
            return result
        return self._baidu_search(query)

    def _bing_fallback(self, query: str) -> str:
        """Bing HTML search (primary search engine for China)."""
        import html as _html
        import re as _re

        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded}&setlang=zh-hans"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                raw = resp.read(65536).decode("utf-8", errors="replace")

            results: list[str] = []

            # Extract result blocks (Bing uses <li class="b_algo">)
            blocks = _re.findall(r'<li[^>]+class="b_algo"[^>]*>(.*?)</li>', raw, _re.DOTALL)
            for block in blocks[:5]:
                # URL from <cite>
                cite_m = _re.search(r"<cite[^>]*>(.*?)</cite>", block, _re.DOTALL)
                # Snippet from <p class="b_lineclamp..."> or plain <p>
                snip_m = _re.search(
                    r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', block, _re.DOTALL
                ) or _re.search(r"<p>(.*?)</p>", block, _re.DOTALL)

                cite = (
                    _html.unescape(_re.sub(r"<[^>]+>", "", cite_m.group(1))).strip()
                    if cite_m else ""
                )
                snip = (
                    _html.unescape(_re.sub(r"<[^>]+>", "", snip_m.group(1))).strip()
                    if snip_m else ""
                )

                if snip:
                    results.append(f"• {snip[:160]}")
                    if cite:
                        results.append(f"  {cite}")

            if results:
                return f"搜索结果（{query}）：\n" + "\n".join(results)

        except Exception:
            pass

        return f"[Error] Bing 搜索失败: {query}"

    def _baidu_search(self, query: str) -> str:
        """Baidu HTML search — reliable in China when Bing is slow."""
        import html as _html
        import re as _re

        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.baidu.com/s?wd={encoded}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Linux; aarch64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                raw = resp.read(65536).decode("utf-8", errors="replace")

            results: list[str] = []
            # Extract Baidu result blocks: <div class="result c-container ...">
            # Title in <h3><a>...</a></h3>, snippet in <span class="content-right_...">
            titles = _re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', raw, _re.DOTALL)
            snippets = _re.findall(
                r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>',
                raw, _re.DOTALL,
            )
            if not snippets:
                # Fallback pattern for mobile/lite Baidu
                snippets = _re.findall(r'<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>', raw, _re.DOTALL)

            for i, title in enumerate(titles[:5]):
                clean_title = _re.sub(r"<[^>]+>", "", _html.unescape(title)).strip()
                if not clean_title or len(clean_title) < 4:
                    continue
                snippet = ""
                if i < len(snippets):
                    snippet = _re.sub(r"<[^>]+>", "", _html.unescape(snippets[i])).strip()[:120]
                entry = f"• {clean_title}"
                if snippet:
                    entry += f"\n  {snippet}"
                results.append(entry)

            if results:
                return f"搜索结果（{query}）：\n" + "\n".join(results)

        except Exception:
            pass

        return (
            f"[搜索] 未找到 '{query}' 的相关结果。\n"
            "建议：尝试更具体的关键词，或用 web_fetch 直接访问目标页面。"
        )


# ── Convenience registration ────────────────────────────────────

_BUILTIN_TOOLS: list[type[BaseTool]] = [
    GetTimeTool,
    RunCommandTool,
    ReadFileTool,
    ListDirectoryTool,
    HttpRequestTool,
    NavStatusTool,
    NavDispatchTool,
    FieldEventTriggerTool,
    SpaceLookupPlaceTool,
    SpaceRecommendRouteTool,
    DogControlDispatchTool,
    SandboxedBashTool,
    WriteFileTool,
    EditFileTool,
    WebFetchTool,
    WebSearchTool,
]


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    production_mode: bool = False,
    robot_control_client: Any | None = None,
    navigation_client: Any | None = None,
) -> None:
    """Instantiate and register all built-in tools into the given registry.

    Args:
        registry: the ToolRegistry to populate.
        production_mode: when True, tools marked ``dev_only = True`` are
            skipped — this prevents development/debug tools (e.g.
            RunCommandTool) from being available in production deployments.
    """
    for tool_cls in _BUILTIN_TOOLS:
        if production_mode and getattr(tool_cls, "dev_only", False):
            import logging as _logging
            _logging.getLogger(__name__).info(
                "production_mode: skipping dev_only tool '%s'", tool_cls.name
            )
            continue
        if tool_cls in {NavDispatchTool, NavStatusTool}:
            registry.register(tool_cls(navigation_client=navigation_client))
        elif tool_cls is DogControlDispatchTool:
            registry.register(tool_cls(robot_control_client=robot_control_client))
        else:
            registry.register(tool_cls())
    # dispatch_skill is registered separately in app.py after SkillDispatcher
    # is constructed (circular dependency: tool needs dispatcher, dispatcher
    # needs pipeline which needs tools).
