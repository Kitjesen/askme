"""PipelineModule - wraps BrainPipeline as a declarative module.

Canonical wiring::

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

# LLMClient imported lazily to avoid circular imports at module scan time
from askme.conversation import VoiceTurnLedger
from askme.conversation.paths import resolve_turn_ledger_path
from askme.llm.core.client import LLMClient
from askme.memory.core.conversation_consumer import ConversationMemoryConsumer
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.core.persona import persona_from_brain_config
from askme.ports import RobotControlPort, SafetyPort, VisionPort
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
from askme.schemas.messages import MemoryContext
from askme.tools.core.tool_registry import ToolRegistry
from askme.voice.core.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _turn_ledger_path(cfg: dict[str, Any]) -> Path:
    """Resolve the centralized Conversation Core event-log path."""

    return resolve_turn_ledger_path(cfg, project_root=_project_root())


def _legacy_conversation_history_path(cfg: dict[str, Any]) -> Path:
    configured = (
        cfg.get("conversation", {}).get("history_file")
        or cfg.get("memory", {}).get("history_file")
        or "data/conversation_history.json"
    )
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else _project_root() / path


def _load_soul_seed(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Load prompts/SOUL.md and convert it into a prompt seed."""
    soul_file = cfg.get("brain", {}).get("soul_file", "prompts/SOUL.md")
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
        {"role": "assistant", "content": "已加载当前项目的角色定义，将按该设定持续响应。"},
    ]


class PipelineModule(Module):
    """Provides the BrainPipeline to the runtime."""

    name = "pipeline"
    depends_on = ("llm", "memory", "tools", "perception")
    provides = ("pipeline",)

    pipeline: Out[BrainPipeline]

    # In ports - auto-wired by runtime before build() is called
    llm_in: In[LLMClient]
    tool_registry_in: In[ToolRegistry]
    memory_context: In[MemoryContext]
    safety_client: In[SafetyPort]
    vision: In[VisionPort]
    control_in: In[RobotControlPort]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # Helper: safely extract a named attribute from a wired In-port module.
        # In-ports are set to the provider Module by auto-wire; they may be None
        # if the optional dependency wasn't wired.
        def _from(mod: Any, attr: str) -> Any:
            return getattr(mod, attr, None) if mod is not None else None

        llm_mod: Any = self.llm_in
        mem_mod: Any = self.memory_context
        tools_mod: Any = self.tool_registry_in
        safety_mod: Any = self.safety_client
        perception_mod: Any = self.vision
        control_mod: Any = self.control_in

        # Unpack wired module attributes into typed locals.
        llm = _from(llm_mod, "llm_client")
        conversation = _from(mem_mod, "conversation")
        memory_bridge = _from(mem_mod, "memory_bridge")
        session_memory = _from(mem_mod, "session_memory")
        episodic = _from(mem_mod, "episodic")
        memory_system = _from(mem_mod, "memory_system")
        tools = _from(tools_mod, "registry")
        dog_safety = _from(safety_mod, "client")
        # Depending on runtime auto-wiring, this input is either the
        # PerceptionModule (which exposes ``vision_bridge``) or the VisionPort
        # itself.  Preserve the actual port in the latter case so TurnExecutor
        # can capture frames and call the configured VLM.
        vision = _from(perception_mod, "vision_bridge")
        if vision is None and callable(
            getattr(perception_mod, "describe_scene_with_question", None)
        ):
            vision = perception_mod
        dog_control = _from(control_mod, "client")

        brain_cfg = cfg.get("brain", {})
        persona = persona_from_brain_config(brain_cfg)
        system_prompt = brain_cfg.get("system_prompt") or persona.build_system_prompt()
        prompt_seed = (
            _load_soul_seed(cfg) or brain_cfg.get("prompt_seed", []) or persona.build_prompt_seed()
        )
        user_prefix = brain_cfg.get("user_prefix") or persona.build_user_prefix()
        ledger_path = _turn_ledger_path(cfg)
        self._turn_ledger = VoiceTurnLedger(ledger_path)
        self._memory_consumer = (
            ConversationMemoryConsumer(
                source=self._turn_ledger,
                sink=memory_bridge,
                checkpoint_path=ledger_path.with_name("memory_consumer_checkpoint.json"),
                source_id=f"voice-turn-ledger-v1:{ledger_path.resolve()}",
                # Conversation Core's current committed-turn projection omits
                # erased threads but cannot notify Memory to delete an event it
                # already projected. Keep processing fail-closed until that
                # deletion contract exists end to end.
                erasure_deletion_supported=False,
            )
            if memory_bridge is not None
            else None
        )
        conversation_cfg = cfg.get("conversation", {})
        legacy_history_path = _legacy_conversation_history_path(cfg)
        migrate_legacy = getattr(self._turn_ledger, "migrate_legacy_history", None)
        if (
            bool(conversation_cfg.get("migrate_legacy_history", True))
            and legacy_history_path.is_file()
            and callable(migrate_legacy)
        ):
            try:
                result = migrate_legacy(legacy_history_path)
                if getattr(result, "turn_count", 0):
                    logger.info(
                        "Conversation Core: migrated %d legacy turns from %s",
                        result.turn_count,
                        legacy_history_path,
                    )
            except Exception as exc:
                # The source stays untouched and ordinary turn handling remains
                # available; operators can retry the deterministic import.
                logger.warning(
                    "Conversation Core legacy history migration failed: %s",
                    exc,
                )

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
            system_prompt=system_prompt,
            prompt_seed=prompt_seed,
            user_prefix=user_prefix,
            voice_model=brain_cfg.get("voice_model"),
            voice_memory_retrieval_deadline_s=brain_cfg.get(
                "voice_memory_retrieval_deadline_s",
                0.25,
            ),
            voice_llm_latency_budget_ms=brain_cfg.get(
                "voice_llm_latency_budget_ms",
                1500,
            ),
            general_tool_max_safety_level=cfg.get("tools", {}).get(
                "general_chat_max_safety_level", "normal"
            ),
            max_response_chars=int(brain_cfg.get("max_response_chars", 0)),
            voice_tts_coalesce=bool(brain_cfg.get("voice_tts_coalesce", False)),
            memory_system=memory_system,
            rag_policy_templates=brain_cfg.get("rag_policy_templates", {}),
            relay_compat_mode=bool(brain_cfg.get("relay_compat_mode", False)),
            turn_ledger=self._turn_ledger,
        )
        logger.info("PipelineModule: built")

    # -- typed accessors ------------------------------------------------
    @property
    def brain_pipeline(self) -> BrainPipeline:
        """The BrainPipeline instance."""
        return self._pipeline

    @property
    def turn_ledger(self) -> VoiceTurnLedger:
        """Runtime-owned authoritative conversation event ledger."""

        return self._turn_ledger

    @property
    def memory_consumer(self) -> ConversationMemoryConsumer:
        """Committed-event Memory projection, currently privacy-gated off."""

        consumer = self._memory_consumer
        if consumer is None:
            raise RuntimeError("memory consumer is unavailable without a memory bridge")
        return consumer

    def health(self) -> dict[str, Any]:
        ledger = getattr(self, "_turn_ledger", None)
        pipeline = getattr(self, "_pipeline", None)
        health_snapshot = getattr(pipeline, "conversation_core_health", None)
        conversation_health = dict(health_snapshot()) if callable(health_snapshot) else {}
        conversation_health.update(
            {
                "enabled": ledger is not None,
                "event_count": int(getattr(ledger, "event_count", 0) or 0),
                "path": str(getattr(ledger, "path", "")),
            }
        )
        consumer = getattr(self, "_memory_consumer", None)
        consumer_status = consumer.status() if consumer is not None else None
        memory_projection = {
            "configured": consumer is not None,
            "processing_allowed": bool(
                consumer_status is not None and consumer_status.processing_allowed
            ),
            "erasure_deletion_supported": bool(
                consumer_status is not None and consumer_status.erasure_deletion_supported
            ),
            "blocked_reason": (
                consumer_status.blocked_reason
                if consumer_status is not None
                else "memory_bridge_unavailable"
            ),
        }
        return {
            "status": conversation_health.get("status", "ok"),
            "conversation_core": conversation_health,
            "memory_committed_event_consumer": memory_projection,
        }
