"""PipelineModule — wraps BrainPipeline as a declarative module.

Mirrors the pipeline creation from ``assembly.py`` lines 493-520::

    pipeline = BrainPipeline(llm=llm, conversation=conversation, ...)

BrainPipeline has many constructor args. This module pulls them from
the registry (LLM, Memory, Tools modules) and config.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from askme.pipeline.brain_pipeline import BrainPipeline
from askme.perception.vision_bridge import VisionBridge
from askme.robot.control_client import DogControlClient
from askme.robot.safety_client import DogSafetyClient
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.schemas.messages import MemoryContext
from askme.tools.tool_registry import ToolRegistry
from askme.voice.stream_splitter import StreamSplitter

# LLMClient imported lazily to avoid circular imports at module scan time
from askme.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_soul_seed(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Load SOUL.md and convert it into a prompt seed."""
    soul_file = cfg.get("brain", {}).get("soul_file", "SOUL.md")
    if not os.path.isabs(soul_file):
        soul_file = str(_project_root() / soul_file)
    if not os.path.isfile(soul_file):
        return []
    try:
        with open(soul_file, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return []
    brief = re.sub(r"^#+\s+.*$", "", raw, flags=re.MULTILINE)
    brief = re.sub(r"\n{3,}", "\n\n", brief).strip()
    if not brief:
        return []
    return [
        {"role": "user", "content": f"请读取这份角色定义，并在整个会话中保持一致。\n{brief}"},
        {"role": "assistant", "content": "已加载 Thunder 角色定义，将按该设定持续响应。"},
    ]


class PipelineModule(Module):
    """Provides the BrainPipeline to the runtime."""

    name = "pipeline"
    depends_on = ("llm", "memory", "tools")
    provides = ("pipeline",)

    pipeline: Out[BrainPipeline]

    # In ports — auto-wired by runtime before build() is called
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    memory_context: In[MemoryContext]
    safety_client: In[DogSafetyClient]
    vision: In[VisionBridge]
    control_in: In[DogControlClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # In ports are set by auto-wire; fall back to registry for compat
        llm_mod = getattr(self, "llm_in", None)
        mem_mod = getattr(self, "memory_context", None)
        tools_mod = getattr(self, "tool_registry_in", None)
        safety_mod = getattr(self, "safety_client", None)
        perception_mod = getattr(self, "vision", None)
        control_mod = getattr(self, "control_in", None)

        llm = getattr(llm_mod, "client", None) if llm_mod else None
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None

        conversation = getattr(mem_mod, "conversation", None) if mem_mod else None
        memory_bridge = getattr(mem_mod, "memory_bridge", None) if mem_mod else None
        session_memory = getattr(mem_mod, "session_memory", None) if mem_mod else None
        episodic = getattr(mem_mod, "episodic", None) if mem_mod else None
        memory_system = getattr(mem_mod, "memory_system", None) if mem_mod else None

        tools = getattr(tools_mod, "registry", None) if tools_mod else None

        dog_safety = getattr(safety_mod, "client", None) if safety_mod else None

        vision = getattr(perception_mod, "vision_bridge", None) if perception_mod else None

        brain_cfg = cfg.get("brain", {})

        # Skill manager + executor from skill module (may not be built yet)
        # These are set post-build via the registry in SkillModule
        skill_manager = None
        skill_executor = None

        # Control client
        dog_control = getattr(control_mod, "client", None) if control_mod else None

        prompt_seed = _load_soul_seed(cfg) or brain_cfg.get("prompt_seed", [])

        self._pipeline = BrainPipeline(
            llm=llm,
            conversation=conversation,
            memory=memory_bridge,
            tools=tools,
            skill_manager=skill_manager,
            skill_executor=skill_executor,
            audio=None,  # set post-build by VoiceModule/TextModule
            splitter=StreamSplitter(),
            arm_controller=None,
            dog_safety_client=dog_safety,
            dog_control_client=dog_control,
            vision=vision,
            session_memory=session_memory,
            episodic_memory=episodic,
            system_prompt=brain_cfg.get(
                "system_prompt",
                "You are Thunder, an industrial inspection AI.",
            ),
            prompt_seed=prompt_seed,
            user_prefix=brain_cfg.get("user_prefix", ""),
            voice_model=brain_cfg.get("voice_model"),
            general_tool_max_safety_level=cfg.get("tools", {}).get(
                "general_chat_max_safety_level", "normal"
            ),
            max_response_chars=int(brain_cfg.get("max_response_chars", 0)),
            memory_system=memory_system,
        )
        logger.info("PipelineModule: built")

    # -- typed accessors ------------------------------------------------
    @property
    def brain_pipeline(self) -> BrainPipeline:
        """The BrainPipeline instance."""
        return self._pipeline

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
