"""Shared, conservative classification for operator runtime controls."""

from __future__ import annotations

import re
from typing import Any

_RUNTIME_CONTROL_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pause", ("pause", "hold", "stop for now", "暂停", "停一下", "先停", "等一下")),
    ("resume", ("resume", "continue", "go on", "继续", "恢复", "接着执行")),
    ("cancel", ("cancel task", "cancel run", "取消任务", "取消执行", "终止任务")),
    (
        "status",
        ("status", "progress", "where are we", "执行到哪", "现在状态", "任务状态", "到哪了"),
    ),
)

_RUNTIME_CONTROL_EXACT_COMMANDS: dict[str, tuple[str, ...]] = {
    "pause": (
        "pause",
        "pause task",
        "pause current task",
        "pause the task",
        "pause run",
        "pause current run",
        "pause the run",
        "pause runtime",
        "pause execution",
        "hold",
        "hold task",
        "hold current task",
        "hold run",
        "hold current run",
        "stop for now",
        "暂停",
        "暂停任务",
        "暂停当前任务",
        "暂停执行",
        "暂停运行",
        "停一下",
        "先停",
        "先停一下",
        "等一下",
    ),
    "resume": (
        "resume",
        "resume task",
        "resume current task",
        "resume the task",
        "resume run",
        "resume current run",
        "resume the run",
        "resume runtime",
        "resume execution",
        "continue",
        "continue task",
        "continue current task",
        "continue run",
        "continue current run",
        "continue execution",
        "go on",
        "继续",
        "继续任务",
        "继续当前任务",
        "继续执行",
        "继续运行",
        "继续巡检",
        "恢复",
        "恢复任务",
        "恢复执行",
        "恢复运行",
        "接着执行",
    ),
    "cancel": (
        "cancel task",
        "cancel current task",
        "cancel the task",
        "cancel run",
        "cancel current run",
        "cancel the run",
        "cancel runtime",
        "cancel execution",
        "取消任务",
        "取消当前任务",
        "取消执行",
        "取消运行",
        "终止任务",
        "终止当前任务",
        "终止执行",
    ),
    "status": (
        "status",
        "progress",
        "where are we",
        "task status",
        "run status",
        "runtime status",
        "task progress",
        "run progress",
        "执行到哪",
        "现在执行到哪了",
        "现在状态",
        "任务状态",
        "到哪了",
    ),
}

_RUNTIME_CONTROL_POLITE_PREFIXES = ("please ", "please, ", "请", "麻烦", "帮我")
_RUNTIME_MUTATIONS = frozenset({"pause", "resume", "cancel"})


def runtime_control_candidate_intent(text: Any) -> str | None:
    """Detect control-like prose without authorizing or executing it."""
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    for intent, tokens in _RUNTIME_CONTROL_TOKENS:
        if any(token in lowered for token in tokens):
            return intent
    return None


def runtime_control_intent(text: Any) -> str | None:
    """Classify an explicit command without matching ordinary prose."""
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None
    normalized = re.sub(r"[.!?。！？,，;；:：]+$", "", normalized).strip()
    for prefix in _RUNTIME_CONTROL_POLITE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    for intent, commands in _RUNTIME_CONTROL_EXACT_COMMANDS.items():
        if normalized in commands:
            return intent
    return None


def runtime_control_permission(
    text: Any,
    *,
    default: str | None = None,
) -> str | None:
    """Map an explicit command to the existing runtime RBAC permission."""
    intent = runtime_control_intent(text)
    if intent in _RUNTIME_MUTATIONS:
        return f"runtime:{intent}"
    if intent == "status":
        return "runtime:read"
    return default
