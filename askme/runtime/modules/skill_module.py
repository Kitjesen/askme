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

from askme.pipeline.planner_agent import PlannerAgent
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.runtime.module import Module, ModuleRegistry, Out
from askme.skills.skill_executor import SkillExecutor
from askme.skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)


class SkillModule(Module):
    """Provides SkillManager, SkillExecutor, PlannerAgent, and SkillDispatcher."""

    name = "skill"
    depends_on = ("pipeline", "llm", "tools")
    provides = ("skills", "openapi", "mcp_catalog")

    dispatcher: Out[SkillDispatcher]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        llm_mod = registry.get("llm")
        llm = getattr(llm_mod, "client", None) if llm_mod else None
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None

        tools_mod = registry.get("tools")
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        brain_cfg = cfg.get("brain", {})

        # Build skill manager
        self.skill_manager = SkillManager()
        self.skill_manager.load()

        # Build skill executor
        skill_model = brain_cfg.get("voice_model") or brain_cfg.get(
            "model", "claude-sonnet-4-5-20250929"
        )
        self.skill_executor = SkillExecutor(
            llm,
            tools,
            default_model=skill_model,
            metrics=ota_metrics,
        )

        # Wire skill_manager and skill_executor into the pipeline
        if pipeline is not None:
            pipeline._skill_manager = self.skill_manager
            pipeline._skill_executor = self.skill_executor

        # Build planner
        self.planner = PlannerAgent(
            llm_client=llm,
            skill_manager=self.skill_manager,
            model=brain_cfg.get("plan_model"),
        )

        # Build dispatcher (audio=None until VoiceModule/TextModule sets it)
        self.skill_dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=self.skill_manager,
            audio=None,
            planner=self.planner,
        )

        # Wire dispatch_skill tool
        if tools is not None:
            from askme.tools.builtin_tools import DispatchSkillTool
            from askme.tools.skill_tools import register_skill_tools
            from askme.llm.intent_router import IntentRouter

            dispatch_tool = DispatchSkillTool()
            dispatch_tool.set_dispatcher(self.skill_dispatcher)
            tools.register(dispatch_tool)
            # Build a temporary router for skill_tools registration
            router = IntentRouter(
                voice_triggers=self.skill_manager.get_voice_triggers(),
            )
            register_skill_tools(tools, self.skill_manager, router)

        logger.info(
            "SkillModule: built (%d skills loaded)",
            len(self.skill_manager.get_all()),
        )

    def health(self) -> dict[str, Any]:
        enabled = self.skill_manager.get_enabled()
        return {
            "status": "ok",
            "skill_count": len(self.skill_manager.get_all()),
            "enabled_count": len(enabled),
        }
