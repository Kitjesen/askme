"""Shared pipeline utilities — small helpers used by multiple sub-modules."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re

# ---------------------------------------------------------------------------
# Structured logging context (item 21)
# ---------------------------------------------------------------------------

# Context vars for structured log fields injected per-turn.
# Set at the start of TurnExecutor.process() so all log records within a turn
# automatically carry trace_id and session_id without manual thread-local tricks.
_log_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_trace_id", default="-"
)
_log_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_session_id", default="-"
)


class PipelineLogFilter(logging.Filter):
    """Inject ``trace_id`` and ``session_id`` from context vars into every record.

    Install once at startup::

        import logging
        from askme.pipeline.utils import PipelineLogFilter

        for handler in logging.root.handlers:
            handler.addFilter(PipelineLogFilter())
        # or add to specific logger:
        logging.getLogger("askme").addFilter(PipelineLogFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _log_trace_id.get()
        record.session_id = _log_session_id.get()
        return True


def set_log_context(*, trace_id: str = "-", session_id: str = "-") -> None:
    """Set structured logging context for the current asyncio task/thread."""
    _log_trace_id.set(trace_id)
    _log_session_id.set(session_id)

# Compiled once; shared across StreamProcessor, SkillGate, BrainPipeline, etc.
_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks from a complete string."""
    return _RE_THINK.sub("", text).strip()


# ---------------------------------------------------------------------------
# Error message factory (item 27) — consistent user-facing voice messages
# ---------------------------------------------------------------------------

def classify_llm_error(exc: Exception) -> str:
    """Return a user-facing voice message for an LLM pipeline error."""
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (asyncio.TimeoutError, APITimeoutError)):
            return "想了一下没想出来，你再说一遍？"
        if isinstance(exc, APIConnectionError):
            return "网络有点问题，基本功能还在。"
    except ImportError:
        pass
    s = str(exc).lower()
    if "timeout" in s:
        return "响应超时，请再说一遍。"
    if "connect" in s or "network" in s:
        return "网络连接异常，请稍候重试。"
    return "处理出错，请重试。"


def classify_skill_error(exc: Exception, skill_name: str) -> str:
    """Return a user-facing voice message for a skill execution error."""
    if isinstance(exc, asyncio.TimeoutError):
        return f"{skill_name}执行超时，跳过了。要不要换个方式？"
    s = str(exc).lower()
    if "connect" in s or "network" in s:
        return f"网络有问题，{skill_name}暂时做不了。"
    return f"{skill_name}执行失败，要不要重试？"
