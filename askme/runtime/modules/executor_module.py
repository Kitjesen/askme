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
from askme.runtime.module import Module, ModuleRegistry, Out

logger = logging.getLogger(__name__)


class ExecutorModule(Module):
    """Provides the ThunderAgentShell to the runtime."""

    name = "executor"
    depends_on = ("llm", "tools")
    provides = ("executor",)

    agent_shell: Out[ThunderAgentShell]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        llm_mod = registry.get("llm")
        llm = getattr(llm_mod, "client", None) if llm_mod else None

        tools_mod = registry.get("tools")
        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        brain_cfg = cfg.get("brain", {})

        self.shell = ThunderAgentShell(
            llm_client=llm,
            tool_registry=tools,
            audio=None,  # set post-build by VoiceModule
            model=brain_cfg.get("agent_model"),
        )
        self.shell._default_timeout = float(brain_cfg.get("agent_timeout", 120.0))

        # Wire agent_shell into pipeline if available
        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        if pipeline is not None:
            pipeline._agent_shell = self.shell

        logger.info("ExecutorModule: built")

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
