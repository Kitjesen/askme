"""ExecutorModule — wraps ThunderAgentShell as a declarative module.

Mirrors the agent shell creation from ``assembly.py`` lines 534-541::

    agent_shell = ThunderAgentShell(
        llm_client=llm, tool_registry=tools, audio=audio, model=...,
    )
    pipeline._agent_shell = agent_shell
"""

from __future__ import annotations

import logging
from typing import Any

from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
from askme.llm.client import LLMClient
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExecutorModule(Module):
    """Provides the ThunderAgentShell to the runtime."""

    name = "executor"
    depends_on = ("llm", "tools", "pipeline")
    provides = ("executor",)

    agent_shell: Out[ThunderAgentShell]

    # In ports — auto-wired by runtime before build() is called
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    pipeline_in: In[BrainPipeline]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        llm_mod = getattr(self, "llm_in", None)
        llm = getattr(llm_mod, "client", None) if llm_mod else None

        tools_mod = getattr(self, "tool_registry_in", None)
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        brain_cfg = cfg.get("brain", {})

        self.shell = ThunderAgentShell(
            llm_client=llm,
            tool_registry=tools,
            audio=None,  # set post-build by VoiceModule
            model=brain_cfg.get("agent_model"),
        )
        self.shell._default_timeout = float(brain_cfg.get("agent_timeout", 120.0))

        # Cross-link: pipeline needs the agent shell for agent_task skill dispatch.
        # ExecutorModule owns the shell, so it injects into the pipeline built earlier.
        # In port is set by auto-wire; fall back to registry for standalone test compat.
        pipeline_mod = getattr(self, "pipeline_in", None)
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        if pipeline is not None:
            pipeline._agent_shell = self.shell

        logger.info("ExecutorModule: built")

    # -- typed accessors ------------------------------------------------
    @property
    def agent_shell(self) -> ThunderAgentShell:
        """The ThunderAgentShell instance."""
        return self.shell

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
