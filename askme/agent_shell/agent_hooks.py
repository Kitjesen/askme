"""Runtime hooks for product-level Agent Profiles.

Claude Code supports shell, HTTP, MCP, prompt, and agent hooks. For a robot
runtime we intentionally start narrower: profile hooks are declarative
tool-use rules that can allow, warn, or block a tool call. The hook runner does
not execute arbitrary shell commands from profile files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentHookDecision:
    """Decision returned by profile hook evaluation."""

    ok: bool = True
    blocked: bool = False
    event: str = ""
    tool_name: str = ""
    reason: str = ""
    matched_hooks: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def error_text(self) -> str:
        reason = self.reason or "profile hook blocked this tool call"
        return f"[Error] Agent hook blocked {self.tool_name}: {reason}"


class AgentHookRunner:
    """Evaluate declarative Agent Profile hooks around tool execution."""

    def __init__(self, hooks: dict[str, Any] | None = None) -> None:
        self._hooks = hooks if isinstance(hooks, dict) else {}

    def before_tool(self, *, tool_name: str, arguments: str = "") -> AgentHookDecision:
        return self._evaluate("PreToolUse", tool_name=tool_name, arguments=arguments)

    def after_tool(
        self,
        *,
        tool_name: str,
        arguments: str = "",
        result: str = "",
    ) -> AgentHookDecision:
        return self._evaluate(
            "PostToolUse",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    def _evaluate(
        self,
        event: str,
        *,
        tool_name: str,
        arguments: str = "",
        result: str = "",
    ) -> AgentHookDecision:
        matched: list[dict[str, Any]] = []
        searchable = f"{arguments}\n{result}" if result else arguments
        for rule in self._event_rules(event):
            if not _matches(rule.get("matcher"), tool_name):
                continue
            if not _contains(rule.get("if_contains", rule.get("contains")), searchable):
                continue
            matched.append(_public_rule(rule))
            hook_type = str(rule.get("type") or "declarative").strip().lower()
            if hook_type not in {"declarative", "policy", "rule"}:
                return AgentHookDecision(
                    ok=False,
                    blocked=True,
                    event=event,
                    tool_name=tool_name,
                    reason=f"unsupported executable hook type: {hook_type}",
                    matched_hooks=tuple(matched),
                )
            decision = _decision(rule)
            if decision in {"deny", "block", "refuse"}:
                return AgentHookDecision(
                    ok=False,
                    blocked=True,
                    event=event,
                    tool_name=tool_name,
                    reason=str(rule.get("reason") or rule.get("message") or decision),
                    matched_hooks=tuple(matched),
                )
            if decision == "warn":
                matched[-1] = {**matched[-1], "warning": True}
        return AgentHookDecision(
            ok=True,
            blocked=False,
            event=event,
            tool_name=tool_name,
            matched_hooks=tuple(matched),
        )

    def _event_rules(self, event: str) -> list[dict[str, Any]]:
        raw = self._hooks.get(event) or self._hooks.get(_camel_to_snake(event))
        if isinstance(raw, dict):
            raw_items: list[Any] = [raw]
        elif isinstance(raw, list):
            raw_items = raw
        else:
            raw_items = []

        rules: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            nested = item.get("hooks")
            if isinstance(nested, list):
                for child in nested:
                    if isinstance(child, dict):
                        rules.append(
                            {**child, "matcher": child.get("matcher", item.get("matcher"))}
                        )
            else:
                rules.append(item)
        return rules


def _matches(matcher: Any, tool_name: str) -> bool:
    pattern = str(matcher or "*").strip()
    if pattern in {"", "*"}:
        return True
    if pattern == tool_name:
        return True
    try:
        return re.search(pattern, tool_name) is not None
    except re.error:
        return pattern == tool_name


def _contains(expected: Any, text: str) -> bool:
    if expected in (None, "", []):
        return True
    if isinstance(expected, list):
        return all(str(item) in text for item in expected)
    return str(expected) in text


def _decision(rule: dict[str, Any]) -> str:
    for key in ("decision", "permissionDecision", "permission_decision", "action"):
        value = str(rule.get(key) or "").strip().lower()
        if value:
            return value
    return "allow"


def _public_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in rule.items()
        if key not in {"command", "args", "input", "secret", "token"}
    }


def _camel_to_snake(value: str) -> str:
    return re.sub(r"(?<!^)([A-Z])", r"_\1", value).lower()
