"""DEPRECATED — agentic shell (replaced by ZeroClaw MCP Agent).

This module previously contained a full Python-based ReAct loop (~827 lines)
that has been superseded by the ZeroClaw MCP Agent.  ZeroClaw is now the
single agent decision-maker for the Askme runtime; Askme no longer maintains
two separate ReAct loops.

The ThunderAgentShell class below is a minimal compat stub that logs a
deprecation warning on construction and returns error messages from
run_task().  Import sites should migrate to ZeroClaw MCP calls.

Retained for reference / compat:
  - ThunderAgentShell       — class stub with deprecation warning
  - AgentShell              — alias in agent_shell.py (unchanged)
  - _build_agent_system_prompt — prompt template (stateless utility)
  - _SPAWN_AGENT_SCHEMA     — inline schema constant
  - _MAX_DEPTH, _MAX_ITERATIONS — module-level constants
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from askme.agent_shell.agent_hooks import AgentHookRunner
from askme.agent_shell.agent_profile import AgentProfile, AgentProfileRegistry

logger = logging.getLogger(__name__)

# ── Constants (kept for compat) ────────────────────────────────────────────────

_DEFAULT_AGENT_MODEL = "MiniMax-M2.7-highspeed"
_MAX_ITERATIONS = 5
_DEFAULT_TIMEOUT = 120.0
_MAX_DEPTH = 1
_SENSITIVE_KEYS = ("authorization", "api_key", "apikey", "bearer", "password", "secret", "token")

_SPAWN_AGENT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_agent",
        "description": (
            "启动子 Agent 自主完成一个专注的子任务（最多嵌套1层）。"
            "子 Agent 拥有独立执行上下文和工具权限，完成后返回结果字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string", "description": "子 Agent 类型"},
                "task": {"type": "string", "description": "子 Agent 要完成的具体任务"},
                "context": {"type": "string", "description": "给子 Agent 的额外上下文（可选）"},
            },
            "required": ["task"],
        },
    },
}


# ── Prompt template (kept for compat) ──────────────────────────────────────────


def _build_agent_system_prompt(workspace: Path, profile: AgentProfile | None = None) -> str:
    """Build the system prompt for the agentic loop (DEPRECATED compat)."""
    profile_block = ""
    if profile is not None:
        profile_block = (
            f"当前 Agent Profile: {profile.display_name} ({profile.name})\n"
            f"职责: {profile.description}\n"
            f"专属指令: {profile.instructions}\n\n"
            f"Agent model policy: {profile.model}; permission mode: {profile.permission_mode}\n"
            f"Max turns: {profile.max_iterations or _MAX_ITERATIONS}; "
            f"timeout: {profile.timeout_seconds or _DEFAULT_TIMEOUT}s\n"
            f"Preloaded skills: {', '.join(profile.preloaded_skills) or 'none'}\n"
            f"MCP servers: {', '.join(profile.mcp_servers) or 'none'}; "
            f"hooks: {', '.join(profile.hooks.keys()) or 'none'}\n"
            f"Effort: {profile.effort or 'inherit'}; isolation: {profile.isolation or 'none'}\n"
            f"Memory scope: {profile.memory_scope or 'none'}\n\n"
        )
    return (
        profile_block
        + "你是现场机器人上运行的自主执行 Agent，拥有真实的执行能力。\n"
        + "你可以运行 shell 命令、读写文件、搜索网络、调用机器人 API、发送 HTTP 请求。\n\n"
        + f"工作区：{workspace}（所有文件操作默认在此目录内）\n\n"
        + "【工具使用指南】\n"
        + "  bash         — shell 命令执行，支持 python/pip/curl 等；超时 30s\n"
        + "  write_file   — 写文件到工作区；path 用相对路径（如 result.py）\n"
        + "  edit_file    — 精确替换文件内容（old_string → new_string）\n"
        + "  read_file    — 读取文件；path 为绝对路径\n"
        + "  web_search   — 搜索网络获取摘要和链接\n"
        + "  web_fetch    — 抓取指定网页完整内容\n"
        + "  http_request — 调用 REST API\n"
        + "  robot_api    — 机器人运行时快捷接口\n"
        + "  spawn_agent  — 启动子 Agent 执行独立子任务（最多嵌套1层）\n"
        + "  speak_progress — 主动向用户播报进度（不阻塞执行）\n"
        + "  create_skill — 把当前解决方案固化为新语音技能\n\n"
        + "【执行原则】\n"
        + "1. 行动优先：直接用工具做，不要先说'我将会...'再做\n"
        + "2. 搜索后深读：web_search 拿到链接 → web_fetch 读全文 → 再综合回答\n"
        + "3. 验证每步：bash/http_request 执行后检查输出，失败时换策略\n"
        + "4. 并行子任务：多个独立子任务用 spawn_agent 并行处理\n"
        + "5. 进度播报：超过15秒的操作前用 speak_progress 告知用户在做什么\n"
        + "6. 保存结果：有价值的输出写入工作区文件（write_file），避免丢失\n"
        + "7. 固化方案：如果任务会重复执行，用 create_skill 固化为语音技能\n"
        + "8. 口语回复：最终回复用简洁中文口语，说清楚'做了什么+结果是什么'"
    )


# ── Deprecation-stub class ─────────────────────────────────────────────────────


class ThunderAgentShell:
    """DEPRECATED: Agentic execution shell — replaced by ZeroClaw MCP Agent.

    Previously wrapped the LLM client in an autonomous tool-use ReAct loop.
    That functionality now lives in the ZeroClaw MCP Agent, which is the
    single agent decision-maker for the Askme runtime.
    """

    execution_status = "deprecated"
    deprecated_replacement = "ZeroClaw MCP Agent"

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Any,
        audio: Any,
        *,
        model: str | None = None,
        workspace: Path | None = None,
        agent_profile: str = "field_operator",
        _depth: int = 0,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._audio = audio
        self._workspace = workspace or (
            Path(__file__).parent.parent.parent / "data" / "agent_workspace"
        )
        self._depth = _depth
        self._profile_registry = AgentProfileRegistry(project_dir=Path.cwd())
        self._profile = self._profile_registry.get(agent_profile)
        self._hook_runner = AgentHookRunner(self._profile.hooks)
        profile_model = "" if self._profile.model in {"", "inherit"} else self._profile.model
        self._model = (
            os.environ.get("AGENT_MODEL") or model or profile_model or _DEFAULT_AGENT_MODEL
        )
        self._iteration_limit = self._profile.max_iterations or _MAX_ITERATIONS
        self._default_timeout = float(
            os.environ.get("AGENT_TIMEOUT", self._profile.timeout_seconds or _DEFAULT_TIMEOUT)
        )
        self._current_action = ""
        self._active_run_summary: dict[str, Any] | None = None
        self._last_run_summary: dict[str, Any] = {}

        logger.warning(
            "[AgentShell] ThunderAgentShell is DEPRECATED. Use ZeroClaw MCP Agent instead. (%s)",
            agent_profile,
        )

    # -- Public compat interface ------------------------------------------------

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent (no-op in stub)."""
        self._audio = audio

    async def run_task(
        self,
        task: str,
        *,
        context: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run an agentic task loop (DEPRECATED — returns error message)."""
        logger.warning(
            "[AgentShell] run_task() is DEPRECATED. Migrate to ZeroClaw MCP. Task not executed: %s",
            task[:80],
        )
        return (
            "[DEPRECATED] AgentShell.run_task() has been replaced by "
            "ZeroClaw MCP Agent. Task not executed."
        )

    def last_run_summary(self) -> dict[str, Any]:
        """Return the latest product-readable run summary (stub returns empty)."""
        return dict(self._last_run_summary)

    # -- Internal helpers kept to avoid AttributeError on downstream access -----

    async def _execute_tool(self, tc: dict[str, str]) -> str:
        return "[DEPRECATED] AgentShell tools disabled."

    async def _spawn_child_agent(self, args_json: str) -> str:
        return "[DEPRECATED] AgentShell sub-agent spawning disabled."

    async def _run_agent_loop(
        self,
        messages: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        system_prompt: str,
    ) -> str:
        return "[DEPRECATED] AgentShell loop disabled."

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        system_prompt: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        return "", []

    def _start_run_summary(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    def _finish_run_summary(
        self, summary: dict[str, Any], *, status: str, final_response: str
    ) -> None:
        pass

    def _record_tool_summary(self, **kwargs: Any) -> None:
        pass

    def _persist_run_summary(self, summary: dict[str, Any]) -> None:
        pass
