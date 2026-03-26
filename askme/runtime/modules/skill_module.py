"""SkillModule — wraps SkillManager + SkillDispatcher + PlannerAgent.

Mirrors the skill wiring from ``assembly.py`` lines 450-546::

    skill_manager = SkillManager()
    skill_manager.load()
    skill_executor = SkillExecutor(llm, tools, ...)
    planner = PlannerAgent(llm_client=..., skill_manager=..., model=...)
    dispatcher = SkillDispatcher(pipeline=..., skill_manager=..., audio=..., planner=...)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.llm.client import LLMClient
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.planner_agent import PlannerAgent
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_manager import SkillManager
from askme.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillModule(Module):
    """Provides SkillManager, SkillExecutor, PlannerAgent, and SkillDispatcher."""

    name = "skill"
    depends_on = ("llm", "tools", "pipeline")
    provides = ("skills", "openapi", "mcp_catalog")

    dispatcher: Out[SkillDispatcher]
    skill_manager_out: Out[SkillManager]

    # In ports — auto-wired by runtime before build() is called
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    pipeline_in: In[BrainPipeline]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        llm_mod = getattr(self, "llm_in", None)
        llm = getattr(llm_mod, "client", None) if llm_mod else None
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None

        tools_mod = getattr(self, "tool_registry_in", None)
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        # In port set by auto-wire; fall back to registry for standalone test compat.
        pipeline_mod = getattr(self, "pipeline_in", None)
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        brain_cfg = cfg.get("brain", {})

        # Build skill manager
        self._skill_manager = SkillManager()
        self._skill_manager.load()

        # Build skill executor
        skill_model = brain_cfg.get("voice_model") or brain_cfg.get(
            "model", "claude-sonnet-4-5-20250929"
        )
        self._skill_executor = SkillExecutor(
            llm,
            tools,
            default_model=skill_model,
            metrics=ota_metrics,
        )

        # Cross-link: pipeline needs skill refs for tool-calling and skill dispatch.
        # SkillModule owns skill_manager/skill_executor, so it is responsible for
        # injecting them into the pipeline that was built earlier (pipeline cannot
        # depend on skill — that would create a cycle).
        if pipeline is not None:
            pipeline.set_skill_manager(self._skill_manager)
            pipeline.set_skill_executor(self._skill_executor)

        # Build planner
        self._planner = PlannerAgent(
            llm_client=llm,
            skill_manager=self._skill_manager,
            model=brain_cfg.get("plan_model"),
        )

        # Build dispatcher (audio=None until VoiceModule/TextModule sets it)
        self._dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=self._skill_manager,
            audio=None,
            planner=self._planner,
        )

        # Wire dispatch_skill tool
        if tools is not None:
            from askme.tools.builtin_tools import DispatchSkillTool
            from askme.tools.skill_tools import register_skill_tools
            from askme.llm.intent_router import IntentRouter

            dispatch_tool = DispatchSkillTool()
            dispatch_tool.set_dispatcher(self._dispatcher)
            tools.register(dispatch_tool)
            # Build a temporary router for skill_tools registration
            router = IntentRouter(
                voice_triggers=self._skill_manager.get_voice_triggers(),
            )
            register_skill_tools(tools, self._skill_manager, router)

        logger.info(
            "SkillModule: built (%d skills loaded)",
            len(self._skill_manager.get_all()),
        )

    # -- typed accessors ------------------------------------------------
    @property
    def skill_manager_out(self) -> SkillManager:
        """The SkillManager instance (Out port)."""
        return self._skill_manager

    @property
    def skill_manager(self) -> SkillManager:
        """The skill manager."""
        return self._skill_manager

    @property
    def skill_dispatcher(self) -> SkillDispatcher:
        """The skill dispatcher."""
        return self._dispatcher

    @property
    def skill_executor(self) -> SkillExecutor:
        """The skill executor."""
        return self._skill_executor

    @property
    def planner(self) -> PlannerAgent:
        """The planner agent."""
        return self._planner

    def health(self) -> dict[str, Any]:
        enabled = self._skill_manager.get_enabled()
        return {
            "status": "ok",
            "skill_count": len(self._skill_manager.get_all()),
            "enabled_count": len(enabled),
        }
