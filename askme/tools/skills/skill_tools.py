"""Skill management tools for askme.

The online-growth path lets the agent draft new SKILL.md capabilities at
runtime.  Generated skills are deliberately not auto-enabled: they enter the
generated-skill governance queue and need review before voice triggers become
active.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from askme.agent_shell.agent_profile import AgentProfileRegistry
from askme.tools.core.tool_registry import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from askme.robot_interaction import IntentRouter
    from askme.skills.core.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_AGENT_PROFILE_CORE_TOOLS = frozenset({
    "spawn_agent",
    "dispatch_skill",
    "create_skill",
    "create_agent_profile",
    "read_file",
    "list_directory",
    "web_search",
    "web_fetch",
    "http_request",
    "robot_api",
    "temporal_query",
    "speak_progress",
    "space_lookup_place",
    "space_recommend_route",
})


class CreateSkillTool(BaseTool):
    """Create a generated skill draft and refresh the in-memory catalog."""

    name = "create_skill"
    description = (
        "Create a new generated skill draft under data/skills. "
        "The skill is hot-loaded for review but remains disabled until an "
        "operator approves it from the generated-skill governance queue."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique lowercase skill id, e.g. check_battery.",
            },
            "description": {
                "type": "string",
                "description": "One sentence explaining what the skill does.",
            },
            "voice_trigger": {
                "type": "string",
                "description": "Comma-separated voice trigger phrases.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt template. Supports {{user_input}}.",
            },
            "tools_section": {
                "type": "string",
                "description": "Allowed tool names, one per line. Empty means prompt-only.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags, e.g. robot,sensor.",
            },
        },
        "required": ["name", "description", "prompt"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "创建新技能草稿"

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
            return "[Error] SkillManager is not initialized"

        result = self._mgr.create_generated_skill_draft(
            name=name,
            description=description,
            prompt=prompt,
            voice_trigger=voice_trigger,
            tools_section=tools_section,
            tags=tags or "generated",
            safety_level="normal",
            confirm_before_execute=False,
            source="agent_create_skill_tool",
            router=self._router,
        )
        if not result.get("ok"):
            return f"[Error] {result.get('error', 'Failed to create skill draft')}"

        skill_name = str(result.get("skill_name") or name)
        skill_file = str(result.get("path") or "")
        loaded_count = int(result.get("loaded_count") or 0)
        logger.info("Created generated skill draft '%s' at %s", skill_name, skill_file)
        result = (
            f"技能 '{skill_name}' 已创建并进入待审批（共 {loaded_count} 个技能）。"
            "审批通过后语音触发词才会生效。"
            f"文件路径：{skill_file}"
        )
        return result


class CreateAgentProfileTool(BaseTool):
    """Create a project-level agent profile draft with scoped tool access."""

    name = "create_agent_profile"
    description = (
        "Create or update a project-level agent profile under .askme/agents. "
        "Profiles define an agent role, instructions, allowed tools, spawnable "
        "profiles, preloaded skills, MCP servers, and risk level."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Agent profile id, e.g. parking_detector_pm.",
            },
            "display_name": {
                "type": "string",
                "description": "Customer-visible agent name.",
            },
            "description": {
                "type": "string",
                "description": "When this agent profile should be used.",
            },
            "instructions": {
                "type": "string",
                "description": "Detailed system instructions for the agent lane.",
            },
            "tools": {
                "type": "string",
                "description": "Comma-separated allowed tool names. Empty inherits all non-denied tools.",
            },
            "disallowed_tools": {
                "type": "string",
                "description": "Comma-separated denied tool names.",
            },
            "spawnable_profiles": {
                "type": "string",
                "description": "Comma-separated profiles this agent may spawn.",
            },
            "skills": {
                "type": "string",
                "description": "Comma-separated skill names preloaded into this profile.",
            },
            "risk_level": {
                "type": "string",
                "description": "low, medium, high, or critical.",
            },
        },
        "required": ["name", "description", "instructions"],
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "创建代理配置"

    def __init__(self, *, known_tools: set[str] | None = None) -> None:
        self._known_tools = set(known_tools or _AGENT_PROFILE_CORE_TOOLS)

    def execute(
        self,
        *,
        name: str = "",
        description: str = "",
        instructions: str = "",
        display_name: str = "",
        tools: str = "",
        disallowed_tools: str = "",
        spawnable_profiles: str = "",
        skills: str = "",
        risk_level: str = "medium",
        operator_id: str = "",
        **kwargs: Any,
    ) -> str:
        registry = AgentProfileRegistry()
        result = registry.write_project_profile(
            name=name,
            display_name=display_name,
            description=description,
            instructions=instructions,
            tools=tools,
            disallowed_tools=disallowed_tools,
            spawnable_profiles=spawnable_profiles,
            skills=skills,
            risk_level=risk_level,
            operator_id=operator_id or "agent",
            known_tools=self._known_tools,
            overwrite=True,
        )
        if not result.get("ok"):
            return f"[Error] {result.get('error', 'Failed to create agent profile')}"
        profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
        return (
            f"Agent profile '{profile.get('name') or name}' 已写入项目配置，"
            "它会按工具边界、派生边界和审计策略进入 Agent 体系。"
            f"文件路径：{result.get('path') or ''}"
        )


def register_skill_tools(
    registry: ToolRegistry,
    skill_manager: SkillManager,
    router: IntentRouter,
) -> None:
    """Instantiate and register skill management tools."""
    known_tools = set(_AGENT_PROFILE_CORE_TOOLS)
    get_allowed = getattr(registry, "get_agent_allowed_names", None)
    if callable(get_allowed):
        try:
            known_tools.update(str(name) for name in get_allowed() if name)
        except Exception as exc:
            logger.warning("Could not derive agent-profile tool allowlist: %s", exc)
    tool = CreateSkillTool()
    tool.set_context(skill_manager, router)
    registry.register(tool)
    known_tools.add(CreateSkillTool.name)
    known_tools.add(CreateAgentProfileTool.name)
    registry.register(CreateAgentProfileTool(known_tools=known_tools))
