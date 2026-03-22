"""
Skill management tools for askme.

Enables the LLM to create new skills dynamically and hot-reload them at runtime
without restarting the assistant.  The LLM should use create_skill when a user
request cannot be satisfied by existing skills.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tool_registry import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from ..brain.intent_router import IntentRouter
    from ..skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [{tags}]
depends: []
conflicts: []
safety_level: {safety_level}
voice_trigger: {voice_trigger}
---

## Tools

{tools_section}

## Prompt

{prompt}
"""


class CreateSkillTool(BaseTool):
    """Dynamically create a new skill and hot-reload it immediately.

    The LLM should call this when the user requests something that no existing
    skill can handle.  The new skill is written to data/skills/ and becomes
    active within the same session — no restart required.
    """

    name = "create_skill"
    description = (
        "动态创建新技能并立即热加载生效——当用户需求超出现有技能范围时使用。"
        "新技能写入 data/skills/ 并实时激活，无需重启。"
        "\n\n可在 tools_section 中列出以下工具（每行一个）："
        "\n- bash：执行 shell 命令（工作区内）"
        "\n- write_file：创建/写入文件（工作区内）"
        "\n- web_search：搜索互联网"
        "\n- web_fetch：抓取网页内容"
        "\n- http_request：调用任意 REST API"
        "\n- robot_api：调用 Thunder runtime 服务"
        "\n- get_current_time：获取当前时间"
        "\n- read_file：读取 data/ 目录下的文件"
        "\n示例：要让技能每次查询天气，tools_section 填 'web_search'，"
        "prompt 里写明搜索什么关键词、如何解读结果、说什么给用户。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "技能唯一标识符（英文小写+下划线，如 check_battery）",
            },
            "description": {
                "type": "string",
                "description": "技能功能简述，一句话说清楚它做什么",
            },
            "voice_trigger": {
                "type": "string",
                "description": "语音触发词，逗号分隔（如 查电量,电池多少,剩余电量）",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "技能执行时给LLM的提示词，支持 {{user_input}} 占位符。"
                    "如果需要调用服务，在提示词里明确写出：用哪个工具、调用什么 URL、"
                    "传什么参数、如何解读响应、最终说什么给用户。"
                ),
            },
            "tools_section": {
                "type": "string",
                "description": (
                    "允许使用的工具名称，每行一个（如 'http_request\\nnav_status'）。"
                    "留空表示纯对话技能，不调用任何工具。"
                    "需要调用真实服务时必须填写。"
                ),
            },
            "tags": {
                "type": "string",
                "description": "标签，逗号分隔（可选，如 robot,sensor）",
            },
        },
        "required": ["name", "description", "prompt"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "创建新技能"  # visible to LLM in general chat; skill content is prompt-only

    def __init__(self) -> None:
        self._mgr: SkillManager | None = None
        self._router: IntentRouter | None = None

    def set_context(self, skill_manager: SkillManager, router: IntentRouter) -> None:
        """Wire skill manager and router after construction."""
        self._mgr = skill_manager
        self._router = router

    def execute(
        self,
        *,
        name: str = "",
        description: str = "",
        prompt: str = "",
        voice_trigger: str = "",
        tools_section: str = "",
        tags: str = "",
        **kwargs: Any,
    ) -> str:
        if self._mgr is None:
            return "[Error] SkillManager 未初始化"

        # Sanitize skill name
        name = re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())
        if not name:
            return "[Error] 技能名称无效（仅允许英文小写、数字、下划线）"

        skill_dir = self._mgr.generated_skills_dir / name
        skill_file = skill_dir / "SKILL.md"

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            content = _SKILL_TEMPLATE.format(
                name=name,
                description=description.replace("\n", " "),
                tags=tags or "generated",
                safety_level="normal",
                voice_trigger=voice_trigger or "",
                tools_section=tools_section or "",
                prompt=prompt,
            )
            skill_file.write_text(content, encoding="utf-8")
            logger.info("Created skill '%s' at %s", name, skill_file)
        except OSError as exc:
            return f"[Error] 写入技能文件失败: {exc}"

        # Hot-reload and update router
        n = self._mgr.hot_reload(self._router)
        new_triggers = [
            phrase for phrase, skill in self._mgr.get_voice_triggers().items()
            if skill == name
        ]

        result = f"技能 '{name}' 已创建并热加载（共 {n} 个技能）。"
        if new_triggers:
            result += f" 语音触发词：{', '.join(new_triggers)}。"
        result += f" 文件路径：{skill_file}"
        return result


def register_skill_tools(
    registry: ToolRegistry,
    skill_manager: SkillManager,
    router: IntentRouter,
) -> None:
    """Instantiate and register skill management tools."""
    tool = CreateSkillTool()
    tool.set_context(skill_manager, router)
    registry.register(tool)
