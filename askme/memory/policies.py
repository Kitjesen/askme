"""L6 Policies & Templates — declarative rules and response templates.

Policies are loaded from YAML files in ``data/memory/policies/``.
They provide:
  - **Response templates**: pre-written responses for common situations
  - **Behavior rules**: constraints on what the robot should/shouldn't do
  - **Escalation rules**: when to escalate to human operator

Usage::

    from askme.memory.policies import PolicyStore

    store = PolicyStore()
    template = store.get_template("greeting", language="zh")
    rules = store.get_rules("safety")
    prompt = store.get_policy_prompt()  # inject into system prompt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from askme.config import get_config, project_root

logger = logging.getLogger(__name__)


class PolicyStore:
    """L6 policy and template store backed by YAML files.

    Directory structure::

        data/memory/policies/
        ├── behavior.yaml     # behavior rules
        ├── safety.yaml       # safety constraints
        ├── escalation.yaml   # escalation rules
        └── templates.yaml    # response templates
    """

    def __init__(self) -> None:
        cfg = get_config()
        data_dir = cfg.get("app", {}).get("data_dir", "data")
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = project_root() / resolved
        self._policy_dir = resolved / "memory" / "policies"
        self._policy_dir.mkdir(parents=True, exist_ok=True)

        self._policies: dict[str, Any] = {}
        self._templates: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load all YAML policy files."""
        for yaml_file in self._policy_dir.glob("*.yaml"):
            try:
                content = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if not isinstance(content, dict):
                    continue
                name = yaml_file.stem
                if name == "templates":
                    self._templates = content
                else:
                    self._policies[name] = content
                logger.debug("[Policy] Loaded %s (%d entries)", name, len(content))
            except Exception as e:
                logger.warning("[Policy] Failed to load %s: %s", yaml_file.name, e)

        if not self._policies and not self._templates:
            self._create_defaults()
            self._load()

    def _create_defaults(self) -> None:
        """Create default policy files if none exist."""
        defaults = {
            "behavior.yaml": {
                "rules": [
                    {"id": "be-concise", "rule": "回复控制在80字以内，像对讲机值班员一样简洁"},
                    {"id": "speak-chinese", "rule": "默认用中文回复，除非用户用其他语言"},
                    {"id": "no-speculation", "rule": "不确定的信息不要猜测，直接说不知道"},
                    {"id": "confirm-actions", "rule": "执行动作前口头确认，急停除外"},
                ],
            },
            "safety.yaml": {
                "rules": [
                    {"id": "estop-immediate", "rule": "急停指令立即执行，不需要确认", "priority": "critical"},
                    {"id": "no-override-safety", "rule": "不允许通过语音覆盖安全服务的判断", "priority": "critical"},
                    {"id": "report-anomaly", "rule": "检测到异常立即上报操作员", "priority": "high"},
                    {"id": "battery-warning", "rule": "电量低于20%时主动提醒", "priority": "medium"},
                ],
            },
            "escalation.yaml": {
                "rules": [
                    {"trigger": "连续3次无法理解用户意图", "action": "suggest_human_help"},
                    {"trigger": "检测到安全隐患", "action": "alert_operator"},
                    {"trigger": "任务执行失败2次", "action": "pause_and_report"},
                    {"trigger": "用户情绪激动", "action": "calm_and_escalate"},
                ],
            },
            "templates.yaml": {
                "greeting": {
                    "zh": "你好，我是值班助手，有什么可以帮你的？",
                    "en": "Hello, I'm the duty assistant. How can I help?",
                },
                "farewell": {
                    "zh": "好的，有事再叫我。",
                    "en": "Sure, call me if you need anything.",
                },
                "not_understood": {
                    "zh": "没听清，能再说一遍吗？",
                    "en": "Sorry, could you repeat that?",
                },
                "task_confirm": {
                    "zh": "收到，{task}，现在执行。",
                    "en": "Got it, {task}, executing now.",
                },
                "task_done": {
                    "zh": "{task}完成。",
                    "en": "{task} done.",
                },
                "error_generic": {
                    "zh": "出了点问题，我再试一下。",
                    "en": "Something went wrong, let me try again.",
                },
                "estop": {
                    "zh": "急停！已停止所有动作。",
                    "en": "Emergency stop! All actions halted.",
                },
            },
        }
        for filename, content in defaults.items():
            path = self._policy_dir / filename
            if not path.exists():
                path.write_text(
                    yaml.dump(content, allow_unicode=True, default_flow_style=False),
                    encoding="utf-8",
                )
                logger.info("[Policy] Created default %s", filename)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_template(self, name: str, language: str = "zh", **kwargs: Any) -> str:
        """Get a response template by name, with optional variable substitution."""
        tpl = self._templates.get(name, {})
        text = tpl.get(language, tpl.get("zh", ""))
        if kwargs and text:
            try:
                text = text.format(**kwargs)
            except (KeyError, IndexError):
                pass
        return text

    def get_rules(self, category: str) -> list[dict[str, Any]]:
        """Get rules for a policy category (behavior, safety, escalation)."""
        policy = self._policies.get(category, {})
        return policy.get("rules", [])

    def get_all_rules(self) -> list[dict[str, Any]]:
        """Get all rules across all policy categories."""
        rules = []
        for category, policy in self._policies.items():
            for rule in policy.get("rules", []):
                rule_copy = dict(rule)
                rule_copy["category"] = category
                rules.append(rule_copy)
        return rules

    def get_policy_prompt(self, max_chars: int = 500) -> str:
        """Generate a system prompt section from all policies.

        Suitable for injection into the LLM system prompt.
        """
        parts = []
        for category in ("safety", "behavior", "escalation"):
            rules = self.get_rules(category)
            if not rules:
                continue
            lines = []
            for r in rules:
                rule_text = r.get("rule", r.get("trigger", ""))
                if rule_text:
                    priority = r.get("priority", "")
                    prefix = f"[{priority}] " if priority else ""
                    lines.append(f"- {prefix}{rule_text}")
            if lines:
                parts.append(f"{category}:\n" + "\n".join(lines))

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n..."
        return result

    def reload(self) -> None:
        """Reload all policy files from disk."""
        self._policies.clear()
        self._templates.clear()
        self._load()
