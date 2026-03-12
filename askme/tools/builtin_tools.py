"""
Built-in tools for askme.

Ports the four original tools (get_time, run_command, read_file, list_directory)
from the prototype as proper BaseTool subclasses.
"""

from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from askme.config import get_section, project_root
from .tool_registry import BaseTool, ToolRegistry

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
            args = shlex.split(command)
        except ValueError as exc:
            return f"[Error] Invalid command syntax: {exc}"
        try:
            result = subprocess.run(
                args,
                shell=False,
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

    def execute(self, *, path: str = "", **kwargs: Any) -> str:
        if not path:
            return "[Error] No path provided."
        if not _is_path_allowed(path):
            return (
                "[Error] 路径不在允许的读取范围内。"
                " LLM 只能读取 data/、logs/、askme/skills/ 目录下的文件。"
            )
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
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
    """Load allowed URL prefixes from config tools.http_allowlist."""
    try:
        return get_section("tools").get("http_allowlist", [])
    except Exception:
        return []


def _is_url_allowed(url: str, allowlist: list[str]) -> bool:
    """Return True if url matches at least one prefix in allowlist.

    Empty allowlist = allow localhost/127.0.0.1/[::1] only.
    """
    if not allowlist:
        parsed = urllib.parse.urlparse(url)
        return parsed.hostname in ("localhost", "127.0.0.1", "::1")
    return any(url.startswith(prefix) for prefix in allowlist)


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

        allowlist = _http_allowlist()
        if not _is_url_allowed(url, allowlist):
            allowed_display = ", ".join(allowlist) if allowlist else "localhost only"
            return (
                f"[Error] URL '{url}' 不在白名单中。"
                f" 允许的前缀：{allowed_display}。"
                " 请在 config.yaml tools.http_allowlist 中添加。"
            )

        method = method.upper()
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
        except TimeoutError:
            return f"[Error] 请求超时 ({timeout}s): {url}"
        except Exception as exc:
            return f"[Error] {exc}"


class NavStatusTool(BaseTool):
    """Query navigation system status from the runtime service."""

    name = "nav_status"
    description = "查询机器人当前导航状态（当前任务、位置、运行状态）"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        import json as _json
        import urllib.request

        url = os.environ.get("NAV_GATEWAY_URL", "").rstrip("/")
        if not url:
            return "[导航状态] 导航服务未配置（NAV_GATEWAY_URL 未设置）"
        try:
            with urllib.request.urlopen(url + "/api/v1/nav/status", timeout=3) as resp:
                data = _json.loads(resp.read())
                return _json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"[导航状态] 查询失败: {exc}"


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
    safety_level = "dangerous"  # requires operator confirmation before execution

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


# ── Convenience registration ────────────────────────────────────

_BUILTIN_TOOLS: list[type[BaseTool]] = [
    GetTimeTool,
    RunCommandTool,
    ReadFileTool,
    ListDirectoryTool,
    HttpRequestTool,
    NavStatusTool,
]


def register_builtin_tools(registry: ToolRegistry, *, production_mode: bool = False) -> None:
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
        registry.register(tool_cls())
    # dispatch_skill is registered separately in app.py after SkillDispatcher
    # is constructed (circular dependency: tool needs dispatcher, dispatcher
    # needs pipeline which needs tools).
