"""
Tool registry system for askme.

Provides an abstract BaseTool class and a ToolRegistry that manages
tool registration, OpenAI-format definition export, and execution dispatch.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Iterable

from askme.config import get_config

logger = logging.getLogger(__name__)

_SAFETY_ORDER = {
    "normal": 0,
    "dangerous": 1,
    "critical": 2,
}
_APPROVAL_REQUIRED_PREFIX = "[Approval Required]"
_APPROVAL_CANCELLED_PREFIX = "[Approval Cancelled]"
_APPROVAL_EXPIRED_PREFIX = "[Approval Expired]"
_APPROVAL_PENDING_PREFIX = "[Approval Pending]"
_DEFAULT_CONFIRMATION_PHRASES = {
    "确认执行",
    "继续执行",
    "批准执行",
    "确认",
    "批准",
    "同意",
    "是",
    "好的",
    "ok",
    "yes",
    "approve",
    "confirm",
}
_DEFAULT_REJECTION_PHRASES = {
    "取消",
    "取消执行",
    "放弃",
    "不",
    "不行",
    "拒绝",
    "no",
    "cancel",
    "deny",
}
_DEFAULT_CONFIRMATION_BYPASS_TOOLS: set[str] = set()
# NOTE: robot_emergency_stop is intentionally NOT in this bypass set.
# LLM-triggered emergency stop requires explicit operator confirmation.
# Voice-triggered E-STOP goes through IntentRouter → pipeline.handle_estop()
# which is a separate, confirmation-free path for genuine emergencies.


def _json_type_matches(value: object, expected: str) -> bool:
    """Return True if *value* matches the JSON Schema *expected* type string."""
    _MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    py_type = _MAP.get(expected)
    if py_type is None:
        return True  # Unknown type — don't block
    # JSON numbers: Python bools are subclasses of int; distinguish them.
    if expected == "integer" and isinstance(value, bool):
        return False
    if expected == "number" and isinstance(value, bool):
        return False
    return isinstance(value, py_type)


def _normalize_safety_level(level: str | None) -> str:
    if level in _SAFETY_ORDER:
        return level
    return "critical"


class ToolExecutionTimeoutError(TimeoutError):
    """Raised when a tool exceeds its configured execution timeout."""


@dataclass
class PendingToolApproval:
    """A dangerous tool invocation waiting for explicit operator approval."""

    tool_name: str
    kwargs: dict[str, Any]
    args_json: str | None
    safety_level: str
    requested_at: float


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
      - name: unique tool identifier
      - description: human-readable description
      - parameters: JSON Schema dict for the tool's parameters
      - execute(**kwargs) -> str: the tool's implementation

    Optional class attributes:
      - dev_only: if True, the tool is skipped when production_mode is enabled
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    safety_level: str = "normal"  # normal | dangerous | critical
    dev_only: bool = False  # if True, excluded when production_mode=True
    agent_allowed: bool = False  # if True, available in ThunderAgentShell
    voice_label: str = ""  # Chinese TTS label (e.g. "观察环境"), empty = no announce

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given keyword arguments.

        Returns:
            A string result to feed back to the LLM.
        """
        ...

    def get_definition(self) -> dict[str, Any]:
        """Return the OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolRegistry:
    """Registry that holds tools and dispatches execution requests."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = get_config().get("tools", {}) if config is None else config
        default_timeout = float(cfg.get("default_timeout", 8.0))

        self._tools: dict[str, BaseTool] = {}
        self._timeout_by_safety: dict[str, float] = {
            "normal": default_timeout,
            "dangerous": float(cfg.get("dangerous_timeout", default_timeout)),
            "critical": float(cfg.get("critical_timeout", default_timeout)),
        }
        self._timeout_cooldown: float = max(0.0, float(cfg.get("timeout_cooldown", 30.0)))
        self._cooldown_until: dict[str, float] = {}
        self._approval_timeout_seconds: float = max(
            0.0,
            float(cfg.get("approval_timeout_seconds", 30.0)),
        )
        self._require_confirmation_levels = {
            _normalize_safety_level(level)
            for level in cfg.get(
                "require_confirmation_levels",
                ["dangerous", "critical"],
            )
        }
        self._confirmation_phrases = {
            self._normalize_phrase(phrase)
            for phrase in cfg.get(
                "confirmation_phrases",
                sorted(_DEFAULT_CONFIRMATION_PHRASES),
            )
            if self._normalize_phrase(phrase)
        }
        self._rejection_phrases = {
            self._normalize_phrase(phrase)
            for phrase in cfg.get(
                "rejection_phrases",
                sorted(_DEFAULT_REJECTION_PHRASES),
            )
            if self._normalize_phrase(phrase)
        }
        self._confirmation_bypass_tools = {
            str(name).strip()
            for name in cfg.get(
                "confirmation_bypass_tools",
                sorted(_DEFAULT_CONFIRMATION_BYPASS_TOOLS),
            )
            if str(name).strip()
        }
        self._pending_approval: PendingToolApproval | None = None

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. Overwrites if name already exists."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty 'name'.")
        logger.debug("Registered tool: %s", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        removed = self._tools.pop(name, None)
        if removed:
            logger.debug("Unregistered tool: %s", name)
        self._cooldown_until.pop(name, None)
        if self._pending_approval and self._pending_approval.tool_name == name:
            self._pending_approval = None
        return removed is not None

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name, or None."""
        return self._tools.get(name)

    def get_agent_allowed_names(self) -> set[str]:
        """Return names of all tools with agent_allowed=True."""
        return {name for name, tool in self._tools.items() if tool.agent_allowed}

    def get_voice_labels(self) -> dict[str, str]:
        """Return {name: voice_label} for tools with non-empty voice_label."""
        return {name: tool.voice_label for name, tool in self._tools.items()
                if tool.voice_label}

    def get_definitions(
        self,
        *,
        allowed_names: Iterable[str] | None = None,
        max_safety_level: str = "critical",
    ) -> list[dict[str, Any]]:
        """Return visible tool definitions in OpenAI function-calling format."""
        allowed = set(allowed_names) if allowed_names is not None else None
        return [
            tool.get_definition()
            for tool in self._tools.values()
            if self._is_tool_exposed(
                tool,
                allowed_names=allowed,
                max_safety_level=max_safety_level,
            )
        ]

    def list_names(
        self,
        *,
        allowed_names: Iterable[str] | None = None,
        max_safety_level: str = "critical",
    ) -> list[str]:
        """Return a sorted list of visible registered tool names."""
        allowed = set(allowed_names) if allowed_names is not None else None
        return sorted(
            tool.name
            for tool in self._tools.values()
            if self._is_tool_exposed(
                tool,
                allowed_names=allowed,
                max_safety_level=max_safety_level,
            )
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def has_pending_approval(self) -> bool:
        """Whether a dangerous tool invocation is waiting for approval."""
        self._expire_pending_approval()
        return self._pending_approval is not None

    def matches_confirmation(self, text: str) -> bool:
        """Return True when *text* confirms the pending dangerous action."""
        if not self.has_pending_approval():
            return False
        return self._normalize_phrase(text) in self._confirmation_phrases

    def matches_rejection(self, text: str) -> bool:
        """Return True when *text* rejects the pending dangerous action."""
        if not self.has_pending_approval():
            return False
        return self._normalize_phrase(text) in self._rejection_phrases

    def handle_pending_input(self, text: str) -> str | None:
        """Resolve or restate the pending high-risk action for arbitrary operator input."""
        expired = self._expire_pending_approval()
        if expired is not None:
            return self._format_approval_expired(expired)

        pending = self._pending_approval
        if pending is None:
            return None

        normalized = self._normalize_phrase(text)
        if normalized in self._confirmation_phrases:
            return self.approve_pending()
        if normalized in self._rejection_phrases:
            return self.reject_pending()
        return self._format_approval_pending(pending)

    def approve_pending(self) -> str:
        """Execute the currently pending dangerous tool invocation."""
        expired = self._expire_pending_approval()
        if expired is not None:
            return self._format_approval_expired(expired)

        pending = self._pending_approval
        if pending is None:
            return "[Approval] No pending high-risk operation."

        tool = self._tools.get(pending.tool_name)
        self._pending_approval = None
        if tool is None:
            logger.warning("Approved tool disappeared before execution: %s", pending.tool_name)
            return f"[Error] Tool not found: {pending.tool_name}"

        logger.warning(
            "Operator approved %s tool: %s(%s)",
            pending.safety_level,
            pending.tool_name,
            pending.kwargs,
        )
        return self._execute_tool(tool, pending.kwargs, timeout=self._resolve_timeout(tool, None))

    def reject_pending(self) -> str:
        """Cancel the currently pending dangerous tool invocation."""
        expired = self._expire_pending_approval()
        if expired is not None:
            return self._format_approval_expired(expired)

        pending = self._pending_approval
        if pending is None:
            return "[Approval] No pending high-risk operation."

        self._pending_approval = None
        return (
            f"{_APPROVAL_CANCELLED_PREFIX} 已取消高风险操作: "
            f"{pending.tool_name}({self._format_kwargs(pending.kwargs)})"
        )

    def execute(
        self,
        name: str,
        args_json: str | None = None,
        *,
        allowed_names: Iterable[str] | None = None,
        max_safety_level: str = "critical",
        timeout: float | None = None,
    ) -> str:
        """Execute a tool by name with JSON-encoded arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return f"[Error] Tool not found: {name}"

        allowed = set(allowed_names) if allowed_names is not None else None
        access_error = self._get_access_error(
            tool,
            allowed_names=allowed,
            max_safety_level=max_safety_level,
        )
        if access_error:
            return access_error

        try:
            kwargs = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return f"[Error] Invalid JSON arguments: {exc}"
        if not isinstance(kwargs, dict):
            return "[Error] Tool arguments must decode to an object."

        # Schema validation: check required fields declared in parameters schema
        schema = getattr(tool, "parameters", None)
        if schema and isinstance(schema, dict):
            validation_error = self._validate_args(tool.name, kwargs, schema)
            if validation_error:
                logger.warning(
                    "Tool '%s' argument validation failed: %s", tool.name, validation_error
                )
                return f"[Error] {validation_error}"

        if self._requires_confirmation(tool):
            self._expire_pending_approval()
            if self._pending_approval is not None:
                return self._format_approval_pending(self._pending_approval)
            self._pending_approval = PendingToolApproval(
                tool_name=name,
                kwargs=kwargs,
                args_json=args_json,
                safety_level=_normalize_safety_level(tool.safety_level),
                requested_at=time.monotonic(),
            )
            logger.warning(
                "Queued %s tool for operator approval: %s(%s)",
                _normalize_safety_level(tool.safety_level),
                name,
                kwargs,
            )
            return self._format_approval_required(tool, kwargs)

        return self._execute_tool(
            tool,
            kwargs,
            timeout=self._resolve_timeout(tool, timeout),
        )

    _RESULT_MAX_CHARS = 5000
    _RESULT_TRUNCATION_SUFFIX = "...[截断]"

    def _execute_tool(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
        *,
        timeout: float,
    ) -> str:
        safety = _normalize_safety_level(tool.safety_level)
        try:
            logger.info(
                "Executing tool: %s(%s) [safety=%s timeout=%.1fs]",
                tool.name,
                kwargs,
                safety,
                timeout,
            )
            result = str(self._run_with_timeout(tool, kwargs, timeout=timeout))

            # Task 4: truncate oversized results to prevent information leakage
            if len(result) > self._RESULT_MAX_CHARS:
                result = result[: self._RESULT_MAX_CHARS] + self._RESULT_TRUNCATION_SUFFIX

            # Task 3: structured audit log for dangerous / critical tools
            if safety in ("dangerous", "critical"):
                _args_preview = self._format_kwargs(kwargs)
                if len(_args_preview) > 200:
                    _args_preview = _args_preview[:200] + "..."
                logger.warning(
                    "[AUDIT] tool_call tool=%s safety=%s args=%s result_len=%d",
                    tool.name,
                    safety,
                    _args_preview,
                    len(result),
                )

            return result
        except ToolExecutionTimeoutError:
            self._mark_timed_out(tool.name)
            logger.error("Tool execution timed out: %s", tool.name)
            if self._timeout_cooldown > 0:
                return (
                    f"[Timeout] Tool '{tool.name}' exceeded {timeout:.1f}s and "
                    f"is unavailable for {self._timeout_cooldown:.0f}s."
                )
            return f"[Timeout] Tool '{tool.name}' exceeded {timeout:.1f}s."
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool.name)
            return f"[Error] Tool '{tool.name}' execution failed: {exc}"

    @staticmethod
    def _validate_args(tool_name: str, kwargs: dict[str, Any], schema: dict[str, Any]) -> str | None:
        """Lightweight schema validation — checks required fields and basic types.

        Returns an error message string on failure, or None if valid.

        Uses the ``required`` and ``properties`` fields of the JSON Schema.
        Does NOT perform deep nested validation — that would require jsonschema.
        """
        required = schema.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            return (
                f"Tool '{tool_name}' missing required argument(s): "
                + ", ".join(f"'{m}'" for m in missing)
            )

        properties = schema.get("properties", {})
        for key, value in kwargs.items():
            if key not in properties:
                # Extra keys are allowed unless additionalProperties: false
                if schema.get("additionalProperties") is False:
                    return f"Tool '{tool_name}' received unexpected argument '{key}'"
                continue
            prop_schema = properties[key]
            expected_type = prop_schema.get("type")
            if expected_type and not _json_type_matches(value, expected_type):
                return (
                    f"Tool '{tool_name}' argument '{key}' expected type "
                    f"'{expected_type}', got '{type(value).__name__}'"
                )

        return None

    def _get_access_error(
        self,
        tool: BaseTool,
        *,
        allowed_names: set[str] | None,
        max_safety_level: str,
    ) -> str | None:
        if allowed_names is not None and tool.name not in allowed_names:
            return f"[Error] Tool '{tool.name}' is not enabled for this request."

        tool_level = _normalize_safety_level(tool.safety_level)
        allowed_level = _normalize_safety_level(max_safety_level)
        if _SAFETY_ORDER[tool_level] > _SAFETY_ORDER[allowed_level]:
            return (
                f"[Error] Tool '{tool.name}' requires safety level '{tool_level}', "
                f"but this request only allows '{allowed_level}'."
            )

        cooldown_remaining = self._cooldown_remaining(tool.name)
        if cooldown_remaining > 0:
            return (
                f"[Error] Tool '{tool.name}' is temporarily unavailable after a timeout. "
                f"Retry in {math.ceil(cooldown_remaining)}s."
            )

        return None

    def _requires_confirmation(self, tool: BaseTool) -> bool:
        tool_level = _normalize_safety_level(tool.safety_level)
        if tool.name in self._confirmation_bypass_tools:
            return False
        return tool_level in self._require_confirmation_levels

    def _expire_pending_approval(self) -> PendingToolApproval | None:
        pending = self._pending_approval
        if pending is None:
            return None
        if self._approval_timeout_seconds <= 0:
            return None
        if time.monotonic() - pending.requested_at <= self._approval_timeout_seconds:
            return None

        self._pending_approval = None
        logger.warning("Pending approval expired for tool: %s", pending.tool_name)
        return pending

    def _format_approval_required(self, tool: BaseTool, kwargs: dict[str, Any]) -> str:
        timeout_hint = (
            f"({int(self._approval_timeout_seconds)}s timeout, auto-cancelled)"
            if self._approval_timeout_seconds > 0
            else ""
        )
        return (
            f"{_APPROVAL_REQUIRED_PREFIX} 高风险操作待确认: "
            f"{tool.name}({self._format_kwargs(kwargs)})。"
            f" 请说[确认执行]继续，或说[取消]放弃。{timeout_hint}"
        )

    def _format_approval_expired(self, pending: PendingToolApproval) -> str:
        return (
            f"{_APPROVAL_EXPIRED_PREFIX} 待确认操作已过期: "
            f"{pending.tool_name}({self._format_kwargs(pending.kwargs)})。"
            " 如需继续，请重新发起操作。"
        )

    def _format_approval_pending(self, pending: PendingToolApproval) -> str:
        return (
            f"{_APPROVAL_PENDING_PREFIX} 高风险操作等待确认: "
            f"{pending.tool_name}({self._format_kwargs(pending.kwargs)})。"
            " 请先回复【确认执行】继续，或【取消】放弃，再发起新指令。"
        )

    def _is_tool_exposed(
        self,
        tool: BaseTool,
        *,
        allowed_names: set[str] | None,
        max_safety_level: str,
    ) -> bool:
        return (
            self._get_access_error(
                tool,
                allowed_names=allowed_names,
                max_safety_level=max_safety_level,
            )
            is None
        )

    def _cooldown_remaining(self, name: str) -> float:
        until = self._cooldown_until.get(name)
        if until is None:
            return 0.0
        remaining = until - time.monotonic()
        if remaining <= 0:
            self._cooldown_until.pop(name, None)
            return 0.0
        return remaining

    def _mark_timed_out(self, name: str) -> None:
        if self._timeout_cooldown <= 0:
            return
        self._cooldown_until[name] = time.monotonic() + self._timeout_cooldown

    def _resolve_timeout(self, tool: BaseTool, timeout: float | None) -> float:
        if timeout is not None:
            return float(timeout)
        return self._timeout_by_safety[_normalize_safety_level(tool.safety_level)]

    def _run_with_timeout(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
        *,
        timeout: float,
    ) -> str:
        if timeout <= 0:
            return str(tool.execute(**kwargs))

        # Use concurrent.futures so the result and exception live in the
        # Future object rather than shared mutable dicts, eliminating the
        # race condition where a timed-out thread writes to result_box
        # after the caller has already moved on.
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"askme-tool-{tool.name}",
        ) as executor:
            future = executor.submit(tool.execute, **kwargs)
            try:
                return str(future.result(timeout=timeout))
            except _FuturesTimeoutError:
                future.cancel()
                raise ToolExecutionTimeoutError(tool.name)

    @staticmethod
    def _format_kwargs(kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return ""
        return json.dumps(kwargs, ensure_ascii=False, separators=(", ", ": "))

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        compact = re.sub(r"[\s\.\,\!\?\;\:，。！？；：\"'`]+", "", text or "")
        return compact.lower()
