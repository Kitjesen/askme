"""
Tool registry system for askme.

Provides an abstract BaseTool class and a ToolRegistry that manages
tool registration, OpenAI-format definition export, and execution dispatch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from askme.config import get_config
from askme.conversation import ApprovalScope, InteractionTurnContext
from askme.tools.core.execution_control import (
    CircuitBreaker,
    ScheduledWork,
    ToolExecutionScheduler,
    ToolQueueFullError,
    WindowRateLimiter,
)

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
_DEFAULT_EXECUTOR_MAX_WORKERS = 4
_DEFAULT_QUEUE_MAX_SIZE = 256
_DEFAULT_JOB_HISTORY_LIMIT = 100
_DEFAULT_RATE_LIMIT_PER_MINUTE = 0.0
_DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 3
_DEFAULT_CIRCUIT_COOLDOWN_SECONDS = 30.0
_DEFAULT_PRIORITY_BY_SAFETY = {
    "critical": 0,
    "dangerous": 50,
    "normal": 100,
}
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
    approval_scope: ApprovalScope | None = None


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
      - name: unique tool identifier
      - description: human-readable description
      - parameters: JSON Schema dict for the tool's parameters
      - execute(**kwargs) -> str: the tool's implementation

    Optional class attributes:
      - dev_only: if True, the tool is skipped when production_mode is enabled
      - queue_priority: lower values run before lower-priority queued work
      - rate_limit_per_minute: per-tool override; None uses registry default
      - backgroundable: if True, can be submitted as a tracked background job
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    safety_level: str = "normal"  # normal | dangerous | critical
    dev_only: bool = False  # if True, excluded when production_mode=True
    agent_allowed: bool = False  # if True, available in ThunderAgentShell
    voice_label: str = ""  # Chinese TTS label (e.g. "观察环境"), empty = no announce
    queue_priority: int | None = None
    rate_limit_per_minute: float | None = None
    backgroundable: bool = False

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
        executor_max_workers = cfg.get(
            "executor_max_workers",
            _DEFAULT_EXECUTOR_MAX_WORKERS,
        )
        self._executor_max_workers = max(1, int(executor_max_workers))
        self._queue_max_size = max(
            1,
            int(cfg.get("queue_max_size", _DEFAULT_QUEUE_MAX_SIZE)),
        )
        priority_cfg = cfg.get("priority_by_safety", {})
        self._priority_by_safety = dict(_DEFAULT_PRIORITY_BY_SAFETY)
        if isinstance(priority_cfg, dict):
            for key, value in priority_cfg.items():
                level = _normalize_safety_level(str(key))
                self._priority_by_safety[level] = int(value)
        self._executor: ToolExecutionScheduler | None = None
        self._executor_lock = threading.Lock()
        self._job_history_limit = max(
            1,
            int(cfg.get("job_history_limit", _DEFAULT_JOB_HISTORY_LIMIT)),
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_order: deque[str] = deque()
        self._job_futures: dict[str, Any] = {}
        self._jobs_lock = threading.RLock()
        self._rate_limit_per_minute = max(
            0.0,
            float(cfg.get("rate_limit_per_minute", _DEFAULT_RATE_LIMIT_PER_MINUTE)),
        )
        self._rate_limiter = WindowRateLimiter(window_seconds=60.0)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=int(
                cfg.get(
                    "circuit_failure_threshold",
                    _DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
                )
            ),
            cooldown_seconds=float(
                cfg.get(
                    "circuit_cooldown_seconds",
                    _DEFAULT_CIRCUIT_COOLDOWN_SECONDS,
                )
            ),
        )
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
        self._approval_lock = threading.RLock()
        self._pending_approval: PendingToolApproval | None = None
        self._pending_approvals_by_id: dict[str, PendingToolApproval] = {}

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
        with self._approval_lock:
            if self._pending_approval and self._pending_approval.tool_name == name:
                self._pending_approval = None
            scoped_ids = [
                approval_id
                for approval_id, pending in self._pending_approvals_by_id.items()
                if pending.tool_name == name
            ]
            for approval_id in scoped_ids:
                self._pending_approvals_by_id.pop(approval_id, None)
        return removed is not None

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name, or None."""
        return self._tools.get(name)

    def get_agent_allowed_names(self) -> set[str]:
        """Return names of all tools with agent_allowed=True."""
        return {name for name, tool in self._tools.items() if tool.agent_allowed}

    def get_voice_labels(self) -> dict[str, str]:
        """Return {name: voice_label} for tools with non-empty voice_label."""
        return {name: tool.voice_label for name, tool in self._tools.items() if tool.voice_label}

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

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shut down the shared tool execution pool.

        The pool is recreated lazily if the registry is used again after
        shutdown, making this safe to call during tests or application teardown.
        """
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        if cancel_futures:
            with self._jobs_lock:
                for job_id, job in self._jobs.items():
                    if job.get("status") in {"queued", "running"}:
                        self._mark_job_cancelled_locked(job_id, "registry_shutdown")

    def close(self) -> None:
        """Close resources owned by this registry."""
        self.shutdown()

    def diagnostics(self) -> dict[str, Any]:
        """Return lightweight runtime diagnostics for health surfaces."""
        with self._executor_lock:
            executor = self._executor
            executor_active = executor is not None
            scheduler_diag = executor.diagnostics() if executor is not None else {}
        job_counts = self._job_counts()
        circuit_diag = self._circuit_breaker.diagnostics()
        return {
            "tool_count": len(self._tools),
            "executor": {
                "active": executor_active,
                "max_workers": self._executor_max_workers,
                "queue_max_size": self._queue_max_size,
                "queued": int(scheduler_diag.get("queued", 0)),
                "running": int(scheduler_diag.get("running", 0)),
                "completed": int(scheduler_diag.get("completed", 0)),
            },
            "cooldown_count": sum(
                1 for expires_at in self._cooldown_until.values() if expires_at > time.monotonic()
            ),
            "pending_approval": self._has_any_pending_approval(),
            "rate_limit": self._rate_limiter.diagnostics(),
            "circuit_breakers": circuit_diag,
            "background_jobs": {
                "history_limit": self._job_history_limit,
                "stored": len(self._jobs),
                **job_counts,
            },
        }

    def has_pending_approval(
        self,
        interaction_context: InteractionTurnContext | None = None,
    ) -> bool:
        """Whether the caller has a dangerous invocation waiting for approval.

        Calls without an interaction context intentionally see only legacy
        approvals. Scoped approvals require their bound context and exact
        challenge ID before they can be resolved.
        """
        if interaction_context is not None:
            return self.pending_approval_scope(interaction_context) is not None
        with self._approval_lock:
            self._expire_pending_approval_locked(time.monotonic())
            return self._pending_approval is not None

    def pending_approval_scope(
        self,
        interaction_context: InteractionTurnContext,
    ) -> ApprovalScope | None:
        """Return the live approval scope eligible for *interaction_context*."""
        if interaction_context is None:
            return None
        now = time.monotonic()
        with self._approval_lock:
            self._expire_scoped_approvals_locked(now)
            pending = self._find_scoped_pending_locked(interaction_context)
            return pending.approval_scope if pending is not None else None

    def matches_confirmation(
        self,
        text: str,
        interaction_context: InteractionTurnContext | None = None,
        *,
        approval_id: str | None = None,
    ) -> bool:
        """Return True when *text* confirms the pending dangerous action.

        Matching rules (ordered by specificity):
        1. Exact match after normalization (strips punctuation/spaces).
        2. Multi-character phrases (>=2 chars) may appear anywhere inside the
           normalized text — handles "好的，确认执行" matching "确认执行".
        Single-character phrases ("是", "好") require exact match to avoid
        false-positive on negations ("不是", "不好").
        """
        if interaction_context is None:
            if approval_id is not None or not self.has_pending_approval():
                return False
        else:
            with self._approval_lock:
                pending, _expired = self._scoped_pending_for_response_locked(
                    interaction_context,
                    approval_id,
                    now=time.monotonic(),
                )
            if pending is None:
                return False
        return self._phrase_set_matches(text, self._confirmation_phrases)

    def matches_rejection(
        self,
        text: str,
        interaction_context: InteractionTurnContext | None = None,
        *,
        approval_id: str | None = None,
    ) -> bool:
        """Return True when *text* rejects the pending dangerous action."""
        if interaction_context is None:
            if approval_id is not None or not self.has_pending_approval():
                return False
        else:
            with self._approval_lock:
                pending, _expired = self._scoped_pending_for_response_locked(
                    interaction_context,
                    approval_id,
                    now=time.monotonic(),
                )
            if pending is None:
                return False
        return self._phrase_set_matches(text, self._rejection_phrases)

    def handle_pending_input(
        self,
        text: str,
        interaction_context: InteractionTurnContext | None = None,
        *,
        approval_id: str | None = None,
    ) -> str | None:
        """Resolve or restate the pending high-risk action for arbitrary operator input."""
        if interaction_context is None:
            if approval_id is not None:
                return None
            with self._approval_lock:
                expired = self._expire_pending_approval_locked(time.monotonic())
                pending = self._pending_approval
            if expired is not None:
                return self._format_approval_expired(expired)
            if pending is None:
                return None
            if self._phrase_set_matches(text, self._confirmation_phrases):
                return self.approve_pending()
            if self._phrase_set_matches(text, self._rejection_phrases):
                return self.reject_pending()
            return self._format_approval_pending(pending)

        now = time.monotonic()
        with self._approval_lock:
            if not self._normalize_approval_id(approval_id):
                self._expire_scoped_approvals_locked(now)
                pending = self._find_scoped_pending_locked(interaction_context)
                expired = None
            else:
                pending, expired = self._scoped_pending_for_response_locked(
                    interaction_context,
                    approval_id,
                    now=now,
                )
        if expired is not None:
            return self._format_approval_expired(expired)
        if pending is None:
            return None
        if not self._normalize_approval_id(approval_id):
            return self._format_approval_pending(pending)
        if self._phrase_set_matches(text, self._confirmation_phrases):
            return self.approve_pending(
                interaction_context,
                approval_id=approval_id,
            )
        if self._phrase_set_matches(text, self._rejection_phrases):
            return self.reject_pending(
                interaction_context,
                approval_id=approval_id,
            )
        return self._format_approval_pending(pending)

    def approve_pending(
        self,
        interaction_context: InteractionTurnContext | None = None,
        *,
        approval_id: str | None = None,
    ) -> str:
        """Execute the currently pending dangerous tool invocation."""
        if interaction_context is None:
            if approval_id is not None:
                return "[Approval] No pending high-risk operation."
            with self._approval_lock:
                expired = self._expire_pending_approval_locked(time.monotonic())
                pending = self._pending_approval
                if pending is not None:
                    self._pending_approval = None
        else:
            with self._approval_lock:
                pending, expired = self._scoped_pending_for_response_locked(
                    interaction_context,
                    approval_id,
                    now=time.monotonic(),
                    remove=True,
                )
        if expired is not None:
            return self._format_approval_expired(expired)
        if pending is None:
            return "[Approval] No pending high-risk operation."

        tool = self._tools.get(pending.tool_name)
        if tool is None:
            logger.warning("Approved tool disappeared before execution: %s", pending.tool_name)
            return f"[Error] Tool not found: {pending.tool_name}"

        logger.warning(
            "Operator approved %s tool: %s(%s)",
            pending.safety_level,
            pending.tool_name,
            pending.kwargs,
        )
        guard_error = self._get_operational_guard_error(tool, consume_rate=True)
        if guard_error:
            return guard_error
        return self._execute_tool(tool, pending.kwargs, timeout=self._resolve_timeout(tool, None))

    def reject_pending(
        self,
        interaction_context: InteractionTurnContext | None = None,
        *,
        approval_id: str | None = None,
    ) -> str:
        """Cancel the currently pending dangerous tool invocation."""
        if interaction_context is None:
            if approval_id is not None:
                return "[Approval] No pending high-risk operation."
            with self._approval_lock:
                expired = self._expire_pending_approval_locked(time.monotonic())
                pending = self._pending_approval
                if pending is not None:
                    self._pending_approval = None
        else:
            with self._approval_lock:
                pending, expired = self._scoped_pending_for_response_locked(
                    interaction_context,
                    approval_id,
                    now=time.monotonic(),
                    remove=True,
                )
        if expired is not None:
            return self._format_approval_expired(expired)
        if pending is None:
            return "[Approval] No pending high-risk operation."
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
        interaction_context: InteractionTurnContext | None = None,
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
            guard_error = self._get_operational_guard_error(tool, consume_rate=False)
            if guard_error:
                return guard_error
            return self._queue_pending_approval(
                tool,
                name,
                kwargs,
                args_json,
                interaction_context=interaction_context,
            )

        guard_error = self._get_operational_guard_error(tool, consume_rate=True)
        if guard_error:
            return guard_error
        return self._execute_tool(
            tool,
            kwargs,
            timeout=self._resolve_timeout(tool, timeout),
        )

    def submit_background(
        self,
        name: str,
        args_json: str | None = None,
        *,
        allowed_names: Iterable[str] | None = None,
        max_safety_level: str = "critical",
        priority: int | None = None,
        interaction_context: InteractionTurnContext | None = None,
    ) -> dict[str, Any]:
        """Queue a background tool job and return its tracked state."""
        prepared = self._prepare_tool_call(
            name,
            args_json,
            allowed_names=allowed_names,
            max_safety_level=max_safety_level,
        )
        tool, kwargs, error = prepared
        if error is not None or tool is None:
            return {
                "job_id": "",
                "tool_name": name,
                "status": "rejected",
                "result": error or f"[Error] Tool not found: {name}",
            }
        if not bool(getattr(tool, "backgroundable", False)):
            return {
                "job_id": "",
                "tool_name": name,
                "status": "rejected",
                "result": f"[Error] Tool '{name}' is not backgroundable.",
            }
        if self._requires_confirmation(tool):
            guard_error = self._get_operational_guard_error(tool, consume_rate=False)
            if guard_error:
                return {
                    "job_id": "",
                    "tool_name": name,
                    "status": "rejected",
                    "result": guard_error,
                }
            return {
                "job_id": "",
                "tool_name": name,
                "status": "pending_approval",
                "result": self._queue_pending_approval(
                    tool,
                    name,
                    kwargs,
                    args_json,
                    interaction_context=interaction_context,
                ),
            }

        guard_error = self._get_operational_guard_error(tool, consume_rate=True)
        if guard_error:
            return {
                "job_id": "",
                "tool_name": name,
                "status": "rejected",
                "result": guard_error,
            }

        job_id = f"tool_{uuid4().hex}"
        resolved_priority = self._resolve_priority(tool, priority)
        self._record_job(
            job_id,
            tool=tool,
            args_json=args_json,
            kwargs=kwargs,
            priority=resolved_priority,
        )
        try:
            future = self._submit_tool_execution(
                tool,
                kwargs,
                priority=resolved_priority,
                job_id=job_id,
            )
        except ToolQueueFullError as exc:
            self._mark_job_failed(job_id, str(exc))
        else:
            with self._jobs_lock:
                self._job_futures[job_id] = future
        return self.get_job(job_id) or {"job_id": job_id, "status": "unknown"}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a copy of one background job state."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent background jobs, newest first."""
        with self._jobs_lock:
            ids = list(reversed(self._job_order))[: max(1, int(limit))]
            return [dict(self._jobs[job_id]) for job_id in ids if job_id in self._jobs]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued background job; running sync tools cannot be interrupted."""
        with self._executor_lock:
            executor = self._executor
        cancelled = executor.cancel(job_id) if executor is not None else False
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "not_found", "cancelled": False}
            if cancelled:
                self._mark_job_cancelled_locked(job_id, "operator_cancelled")
                self._job_futures.pop(job_id, None)
            elif job.get("status") == "running":
                job["cancel_requested"] = True
            result = dict(job)
        result["cancelled"] = bool(cancelled)
        return result

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
            raw_result = str(self._run_with_timeout(tool, kwargs, timeout=timeout))
            result = self._format_tool_result(raw_result)
            self._audit_tool_result(tool, kwargs, result)
            self._circuit_breaker.record_success(tool.name)
            return result
        except ToolQueueFullError:
            logger.warning("Tool execution queue is full: %s", tool.name)
            return "[Busy] Tool execution queue is full. Retry shortly."
        except ToolExecutionTimeoutError:
            self._mark_timed_out(tool.name)
            self._circuit_breaker.record_failure(tool.name)
            logger.error("Tool execution timed out: %s", tool.name)
            if self._timeout_cooldown > 0:
                return (
                    f"[Timeout] Tool '{tool.name}' exceeded {timeout:.1f}s and "
                    f"is unavailable for {self._timeout_cooldown:.0f}s."
                )
            return f"[Timeout] Tool '{tool.name}' exceeded {timeout:.1f}s."
        except Exception as exc:
            self._circuit_breaker.record_failure(tool.name)
            logger.exception("Tool execution failed: %s", tool.name)
            return f"[Error] Tool '{tool.name}' execution failed: {exc}"

    def _prepare_tool_call(
        self,
        name: str,
        args_json: str | None,
        *,
        allowed_names: Iterable[str] | None,
        max_safety_level: str,
    ) -> tuple[BaseTool | None, dict[str, Any], str | None]:
        tool = self._tools.get(name)
        if tool is None:
            return None, {}, f"[Error] Tool not found: {name}"

        allowed = set(allowed_names) if allowed_names is not None else None
        access_error = self._get_access_error(
            tool,
            allowed_names=allowed,
            max_safety_level=max_safety_level,
        )
        if access_error:
            return tool, {}, access_error

        try:
            kwargs = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return tool, {}, f"[Error] Invalid JSON arguments: {exc}"
        if not isinstance(kwargs, dict):
            return tool, {}, "[Error] Tool arguments must decode to an object."

        schema = getattr(tool, "parameters", None)
        if schema and isinstance(schema, dict):
            validation_error = self._validate_args(tool.name, kwargs, schema)
            if validation_error:
                logger.warning(
                    "Tool '%s' argument validation failed: %s",
                    tool.name,
                    validation_error,
                )
                return tool, kwargs, f"[Error] {validation_error}"

        return tool, kwargs, None

    def _queue_pending_approval(
        self,
        tool: BaseTool,
        name: str,
        kwargs: dict[str, Any],
        args_json: str | None,
        *,
        interaction_context: InteractionTurnContext | None = None,
    ) -> str:
        requested_at = time.monotonic()
        if interaction_context is not None:
            is_voice = str(interaction_context.channel or "").strip().lower() == "voice" or (
                str(interaction_context.source or "").strip().lower() == "voice"
            )
            person_id = str(interaction_context.person_id or "").strip()
            operator_id = str(interaction_context.operator_id or "").strip()
            trusted_operator = bool(
                operator_id and interaction_context.metadata.get("operator_authenticated") is True
            )
            if is_voice and not person_id and not trusted_operator:
                logger.warning(
                    "Blocked anonymous voice approval for %s; trusted operator required",
                    name,
                )
                return (
                    f"{_APPROVAL_REQUIRED_PREFIX} 高风险操作需要已认证操作员，"
                    "请在控制台登录后重新发起。"
                )
        safety_level = _normalize_safety_level(tool.safety_level)
        with self._approval_lock:
            self._expire_pending_approval_locked(requested_at)
            self._expire_scoped_approvals_locked(requested_at)
            if interaction_context is None:
                if self._pending_approval is not None:
                    return self._format_approval_pending(self._pending_approval)
                pending = PendingToolApproval(
                    tool_name=name,
                    kwargs=kwargs,
                    args_json=args_json,
                    safety_level=safety_level,
                    requested_at=requested_at,
                )
                self._pending_approval = pending
            else:
                existing = self._find_scoped_pending_locked(interaction_context)
                if existing is not None:
                    return self._format_approval_pending(existing)
                approval_id = uuid4().hex
                expires_at = (
                    requested_at + self._approval_timeout_seconds
                    if self._approval_timeout_seconds > 0
                    else math.inf
                )
                approval_scope = ApprovalScope.create(
                    interaction_context,
                    approval_id=approval_id,
                    subject=name,
                    risk_level=safety_level,
                    payload_digest=self._canonical_payload_digest(kwargs),
                    expires_at_monotonic=expires_at,
                    allows_short_reply=True,
                )
                pending = PendingToolApproval(
                    tool_name=name,
                    kwargs=kwargs,
                    args_json=args_json,
                    safety_level=safety_level,
                    requested_at=requested_at,
                    approval_scope=approval_scope,
                )
                self._pending_approvals_by_id[approval_id] = pending
        logger.warning(
            "Queued %s tool for operator approval: %s(%s)",
            safety_level,
            name,
            kwargs,
        )
        return self._format_approval_required(
            tool,
            kwargs,
            approval_scope=pending.approval_scope,
        )

    def _format_tool_result(self, result: str) -> str:
        if len(result) > self._RESULT_MAX_CHARS:
            return result[: self._RESULT_MAX_CHARS] + self._RESULT_TRUNCATION_SUFFIX
        return result

    def _audit_tool_result(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
        result: str,
    ) -> None:
        safety = _normalize_safety_level(tool.safety_level)
        if safety not in ("dangerous", "critical"):
            return
        args_preview = self._format_kwargs(kwargs)
        if len(args_preview) > 200:
            args_preview = args_preview[:200] + "..."
        logger.warning(
            "[AUDIT] tool_call tool=%s safety=%s args=%s result_len=%d",
            tool.name,
            safety,
            args_preview,
            len(result),
        )

    @staticmethod
    def _validate_args(
        tool_name: str, kwargs: dict[str, Any], schema: dict[str, Any]
    ) -> str | None:
        """Lightweight schema validation — checks required fields and basic types.

        Returns an error message string on failure, or None if valid.

        Uses the ``required`` and ``properties`` fields of the JSON Schema.
        Does NOT perform deep nested validation — that would require jsonschema.
        """
        required = schema.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            return f"Tool '{tool_name}' missing required argument(s): " + ", ".join(
                f"'{m}'" for m in missing
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

    def _get_operational_guard_error(
        self,
        tool: BaseTool,
        *,
        consume_rate: bool,
    ) -> str | None:
        cooldown_remaining = self._cooldown_remaining(tool.name)
        if cooldown_remaining > 0:
            return (
                f"[Error] Tool '{tool.name}' is temporarily unavailable after a timeout. "
                f"Retry in {math.ceil(cooldown_remaining)}s."
            )

        circuit_remaining = self._circuit_breaker.remaining_open_seconds(tool.name)
        if circuit_remaining > 0:
            return (
                f"[Circuit Open] Tool '{tool.name}' is temporarily disabled after "
                f"repeated failures. Retry in {math.ceil(circuit_remaining)}s."
            )

        limit = self._resolve_rate_limit(tool)
        if consume_rate:
            allowed, retry_after = self._rate_limiter.check_and_consume(
                tool.name,
                limit,
            )
            if not allowed:
                return (
                    f"[Rate Limited] Tool '{tool.name}' exceeded "
                    f"{limit:.0f}/min. Retry in {math.ceil(retry_after)}s."
                )
        return None

    def _requires_confirmation(self, tool: BaseTool) -> bool:
        tool_level = _normalize_safety_level(tool.safety_level)
        if tool.name in self._confirmation_bypass_tools:
            return False
        return tool_level in self._require_confirmation_levels

    def _has_any_pending_approval(self) -> bool:
        now = time.monotonic()
        with self._approval_lock:
            self._expire_pending_approval_locked(now)
            self._expire_scoped_approvals_locked(now)
            return bool(self._pending_approval or self._pending_approvals_by_id)

    def _expire_pending_approval(self) -> PendingToolApproval | None:
        with self._approval_lock:
            return self._expire_pending_approval_locked(time.monotonic())

    def _expire_pending_approval_locked(
        self,
        now: float,
    ) -> PendingToolApproval | None:
        pending = self._pending_approval
        if pending is None or self._approval_timeout_seconds <= 0:
            return None
        if now - pending.requested_at <= self._approval_timeout_seconds:
            return None

        self._pending_approval = None
        logger.warning("Pending approval expired for tool: %s", pending.tool_name)
        return pending

    def _expire_scoped_approvals_locked(
        self,
        now: float,
    ) -> dict[str, PendingToolApproval]:
        expired: dict[str, PendingToolApproval] = {}
        for approval_id, pending in tuple(self._pending_approvals_by_id.items()):
            scope = pending.approval_scope
            if scope is None or now < scope.expires_at_monotonic:
                continue
            expired[approval_id] = pending
            self._pending_approvals_by_id.pop(approval_id, None)
            logger.warning(
                "Scoped approval expired for tool: %s (%s)",
                pending.tool_name,
                approval_id,
            )
        return expired

    def _find_scoped_pending_locked(
        self,
        interaction_context: InteractionTurnContext,
    ) -> PendingToolApproval | None:
        for pending in reversed(tuple(self._pending_approvals_by_id.values())):
            scope = pending.approval_scope
            if scope is not None and self._scope_identity_matches(scope, interaction_context):
                return pending
        return None

    def _scoped_pending_for_response_locked(
        self,
        interaction_context: InteractionTurnContext,
        approval_id: str | None,
        *,
        now: float,
        remove: bool = False,
    ) -> tuple[PendingToolApproval | None, PendingToolApproval | None]:
        normalized_id = self._normalize_approval_id(approval_id)
        expired_by_id = self._expire_scoped_approvals_locked(now)
        if normalized_id is None:
            return None, None

        expired = expired_by_id.get(normalized_id)
        if expired is not None:
            scope = expired.approval_scope
            if scope is not None and self._scope_identity_matches(scope, interaction_context):
                return None, expired
            return None, None

        pending = self._pending_approvals_by_id.get(normalized_id)
        if pending is None or pending.approval_scope is None:
            return None, None
        if not pending.approval_scope.matches(
            interaction_context,
            approval_id=normalized_id,
            now_monotonic=now,
        ):
            return None, None
        if remove:
            self._pending_approvals_by_id.pop(normalized_id, None)
        return pending, None

    @staticmethod
    def _scope_identity_matches(
        scope: ApprovalScope,
        interaction_context: InteractionTurnContext,
    ) -> bool:
        if str(interaction_context.thread_id).strip() != scope.thread_id:
            return False
        person_id = str(interaction_context.person_id or "").strip() or None
        if scope.person_id is not None and person_id != scope.person_id:
            return False
        operator_id = str(interaction_context.operator_id or "").strip() or None
        return scope.operator_id is None or operator_id == scope.operator_id

    @staticmethod
    def _normalize_approval_id(approval_id: str | None) -> str | None:
        normalized = str(approval_id or "").strip()
        return normalized or None

    @staticmethod
    def _canonical_payload_digest(kwargs: dict[str, Any]) -> str:
        canonical = json.dumps(
            kwargs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _format_approval_required(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
        *,
        approval_scope: ApprovalScope | None = None,
    ) -> str:
        timeout_hint = (
            f"({int(self._approval_timeout_seconds)}s timeout, auto-cancelled)"
            if self._approval_timeout_seconds > 0
            else ""
        )
        approval_hint = (
            f" approval_id={approval_scope.approval_id}." if approval_scope is not None else ""
        )
        return (
            f"{_APPROVAL_REQUIRED_PREFIX} 高风险操作待确认: "
            f"{tool.name}({self._format_kwargs(kwargs)})。"
            f"{approval_hint}"
            f" 请说[确认执行]继续，或说[取消]放弃。{timeout_hint}"
        )

    @staticmethod
    def _approval_id_hint(pending: PendingToolApproval) -> str:
        scope = pending.approval_scope
        if scope is None:
            return ""
        return f" approval_id={scope.approval_id}."

    def _format_approval_expired(self, pending: PendingToolApproval) -> str:
        return (
            f"{_APPROVAL_EXPIRED_PREFIX} 待确认操作已过期: "
            f"{pending.tool_name}({self._format_kwargs(pending.kwargs)})。"
            f"{self._approval_id_hint(pending)}"
            " 如需继续，请重新发起操作。"
        )

    def _format_approval_pending(self, pending: PendingToolApproval) -> str:
        return (
            f"{_APPROVAL_PENDING_PREFIX} 高风险操作等待确认: "
            f"{pending.tool_name}({self._format_kwargs(pending.kwargs)})。"
            f"{self._approval_id_hint(pending)}"
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

    def _resolve_priority(self, tool: BaseTool, priority: int | None = None) -> int:
        if priority is not None:
            return int(priority)
        tool_priority = getattr(tool, "queue_priority", None)
        if tool_priority is not None:
            return int(tool_priority)
        return self._priority_by_safety[_normalize_safety_level(tool.safety_level)]

    def _resolve_rate_limit(self, tool: BaseTool) -> float:
        tool_limit = getattr(tool, "rate_limit_per_minute", None)
        if tool_limit is not None:
            return max(0.0, float(tool_limit))
        return self._rate_limit_per_minute

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
        future = self._submit_tool_execution(
            tool,
            kwargs,
            priority=self._resolve_priority(tool),
        )
        try:
            return str(future.result(timeout=timeout))
        except _FuturesTimeoutError:
            future.cancel()
            raise ToolExecutionTimeoutError(tool.name)

    def _submit_tool_execution(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
        *,
        priority: int,
        job_id: str | None = None,
    ):
        work_id = job_id or f"call_{uuid4().hex}"
        with self._executor_lock:
            if self._executor is None:
                self._executor = ToolExecutionScheduler(
                    max_workers=self._executor_max_workers,
                    max_queue_size=self._queue_max_size,
                    thread_name_prefix="askme-tool",
                    on_start=self._on_scheduled_work_started,
                    on_finish=self._on_scheduled_work_finished,
                )
            return self._executor.submit(
                lambda: tool.execute(**kwargs),
                priority=priority,
                work_id=work_id,
                metadata={
                    "tool_name": tool.name,
                    "safety": _normalize_safety_level(tool.safety_level),
                    "background": bool(job_id),
                },
            )

    def _record_job(
        self,
        job_id: str,
        *,
        tool: BaseTool,
        args_json: str | None,
        kwargs: dict[str, Any],
        priority: int,
    ) -> None:
        now = time.time()
        job = {
            "job_id": job_id,
            "tool_name": tool.name,
            "status": "queued",
            "priority": int(priority),
            "safety": _normalize_safety_level(tool.safety_level),
            "submitted_at": now,
            "started_at": None,
            "finished_at": None,
            "elapsed_ms": None,
            "args_json": args_json or "{}",
            "args_preview": self._format_kwargs(kwargs)[:500],
            "result": "",
            "error": "",
            "cancel_requested": False,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job
            self._job_order.append(job_id)
            while len(self._job_order) > self._job_history_limit:
                old_id = self._job_order.popleft()
                self._jobs.pop(old_id, None)
                self._job_futures.pop(old_id, None)

    def _on_scheduled_work_started(self, work: ScheduledWork) -> None:
        with self._jobs_lock:
            job = self._jobs.get(work.work_id)
            if job is not None and job.get("status") == "queued":
                job["status"] = "running"
                job["started_at"] = time.time()

    def _on_scheduled_work_finished(
        self,
        work: ScheduledWork,
        status: str,
        result: Any,
        error: BaseException | None,
        elapsed_ms: float,
    ) -> None:
        with self._jobs_lock:
            job = self._jobs.get(work.work_id)
            if job is None:
                return
            if job.get("status") in {"cancelled", "timeout"}:
                self._job_futures.pop(work.work_id, None)
                return
            job["finished_at"] = time.time()
            job["elapsed_ms"] = round(float(elapsed_ms), 3)
            if status == "succeeded":
                formatted = self._format_tool_result(str(result))
                job["status"] = "completed"
                job["result"] = formatted
                self._circuit_breaker.record_success(str(work.metadata.get("tool_name")))
            elif status == "cancelled":
                self._mark_job_cancelled_locked(work.work_id, "cancelled_before_start")
            else:
                tool_name = str(work.metadata.get("tool_name") or job["tool_name"])
                job["status"] = "failed"
                job["error"] = str(error or "tool execution failed")
                self._circuit_breaker.record_failure(tool_name)
            self._job_futures.pop(work.work_id, None)

    def _mark_job_failed(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = error
                job["finished_at"] = time.time()

    def _mark_job_cancelled_locked(self, job_id: str, reason: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job["status"] = "cancelled"
            job["error"] = reason
            job["finished_at"] = time.time()

    def _job_counts(self) -> dict[str, int]:
        with self._jobs_lock:
            counts = {
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "rejected": 0,
            }
            for job in self._jobs.values():
                status = str(job.get("status") or "unknown")
                if status in counts:
                    counts[status] += 1
            return counts

    @staticmethod
    def _format_kwargs(kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return ""
        return json.dumps(kwargs, ensure_ascii=False, separators=(", ", ": "))

    def _phrase_set_matches(self, text: str, phrase_set: set[str]) -> bool:
        """Return True if *text* matches any phrase in *phrase_set*.

        Rules (item 22 — word-boundary safe):
        - Exact match: normalized text equals any phrase (always checked).
        - Embedded match: a multi-char phrase (len >= 2) appears anywhere in the
          normalized text — "好的，确认执行" → finds "确认执行".
        - Single-char phrases ("是", "好") are EXACT-ONLY to prevent matching
          within negations like "不是" or "不好".
        """
        normalized = self._normalize_phrase(text)
        # 1. Exact match (covers all lengths)
        if normalized in phrase_set:
            return True
        # 2. Multi-char phrase embedded in input (graceful for verbose voice input)
        for phrase in phrase_set:
            if len(phrase) >= 2 and phrase in normalized:
                return True
        return False

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        compact = re.sub(r"[\s\.\,\!\?\;\:，。！？；：\"'`]+", "", text or "")
        return compact.lower()
