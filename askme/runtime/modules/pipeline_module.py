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
        # Helper: safely extract a named attribute from a wired In-port module.
        # In-ports are set to the provider Module by auto-wire; they may be None
        # if the optional dependency wasn't wired.
        def _from(mod: Any, attr: str) -> Any:
            return getattr(mod, attr, None) if mod is not None else None

        llm_mod: Any = getattr(self, "llm_in", None)
        mem_mod: Any = getattr(self, "memory_context", None)
        tools_mod: Any = getattr(self, "tool_registry_in", None)
        safety_mod: Any = getattr(self, "safety_client", None)
        perception_mod: Any = getattr(self, "vision", None)
        control_mod: Any = getattr(self, "control_in", None)

        # Unpack wired module attributes into typed locals.
        llm = _from(llm_mod, "client")
        conversation = _from(mem_mod, "conversation")
        memory_bridge = _from(mem_mod, "memory_bridge")
        session_memory = _from(mem_mod, "session_memory")
        episodic = _from(mem_mod, "episodic")
        memory_system = _from(mem_mod, "memory_system")
        tools = _from(tools_mod, "registry")
        dog_safety = _from(safety_mod, "client")
        vision = _from(perception_mod, "vision_bridge")
        dog_control = _from(control_mod, "client")

        brain_cfg = cfg.get("brain", {})
        prompt_seed = _load_soul_seed(cfg) or brain_cfg.get("prompt_seed", [])

        self._pipeline = BrainPipeline(
            llm=llm,
            conversation=conversation,
            memory=memory_bridge,
            tools=tools,
            # Skill manager + executor set post-build by SkillModule.
            skill_manager=None,
            skill_executor=None,
            audio=None,  # set post-build by VoiceModule/TextModule
            splitter=StreamSplitter(),
            dog_safety_client=dog_safety,
            dog_control_client=dog_control,
            vision=vision,
            session_memory=session_memory,
            episodic_memory=episodic,
            system_prompt=brain_cfg.get("system_prompt", ""),
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
