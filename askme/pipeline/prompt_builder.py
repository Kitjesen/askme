"""Prompt construction helpers for BrainPipeline."""

from __future__ import annotations

import datetime
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.memory.session import SessionMemory
    from askme.memory.system import MemorySystem
    from askme.perception.vision_bridge import VisionBridge
    from askme.robot.safety_client import DogSafetyClient
    from askme.skills.skill_manager import SkillManager
    from askme.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Assembles system prompts and prepares message lists for LLM calls."""

    def __init__(
        self,
        *,
        base_prompt: str,
        prompt_seed: list[dict[str, str]],
        user_prefix: str,
        tools: ToolRegistry,
        skill_manager: SkillManager,
        general_tool_max_safety_level: str,
        dog_safety: DogSafetyClient | None,
        episodic: EpisodicMemory | None,
        session_memory: SessionMemory | None,
        vision: VisionBridge | None,
        qp_memory: Any,
        memory_system: MemorySystem | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._prompt_seed = prompt_seed
        self._user_prefix = user_prefix
        self._tools = tools
        self._skill_manager = skill_manager
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._dog_safety = dog_safety
        self._episodic = episodic
        self._session_memory = session_memory
        self._vision = vision
        self._qp_memory = qp_memory
        self._memory_system = memory_system

    def build_l0_runtime_block(self) -> str:
        """Return a compact L0 runtime truth block from authoritative services.

        Non-blocking: reads only the in-memory cache on DogSafetyClient.
        Returns an empty string when no services are configured.
        """
        if not (self._dog_safety and self._dog_safety.is_configured()):
            return ""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        estop = self._dog_safety.is_estop_active()
        estop_str = "⚠️ 已激活 — 禁止运动指令" if estop else "正常"
        return f"[运行时状态 {ts}]\n急停: {estop_str}"

    def build_system_prompt(
        self,
        context_str: str | None,
        *,
        scene_desc: str = "",
        user_text: str = "",
    ) -> str:
        """Assemble system prompt with episodic knowledge, session summaries, and memory context."""
        l0 = self.build_l0_runtime_block()
        prompt = (l0 + "\n") if l0 else ""
        prompt += self._base_prompt

        if self._episodic:
            world_ctx = self._episodic.get_knowledge_context()
            if world_ctx:
                prompt += f"\n{world_ctx}"
            digest_ctx = self._episodic.get_recent_digest()
            if digest_ctx:
                prompt += f"\n{digest_ctx}"
            relevant_ctx = self._episodic.get_relevant_context(user_text)
            if relevant_ctx:
                prompt += f"\n{relevant_ctx}"

        if self._session_memory:
            session_ctx = self._session_memory.get_recent_summaries()
            if session_ctx:
                prompt += f"\n{session_ctx}"

        # Trends from barrier capability (TrendAnalyzer)
        if self._memory_system:
            try:
                trends_text = self._memory_system.get_trends()
                if trends_text:
                    prompt += f"\n[趋势]\n{trends_text}"
            except Exception as _e:
                logger.debug("Trend injection failed: %s", _e)

        # qp_memory: spatial/procedural/markdown context (optional, additive)
        # NOTE: WebChat path injects into user message directly (app.py).
        # Voice path injects here into system prompt. Both use get_context_smart().
        # Only inject if not already injected by the caller (avoid double-injection).
        if self._qp_memory is not None and "[站点记忆]" not in (user_text or ""):
            try:
                qp_ctx = self._qp_memory.get_context_smart(query=user_text, max_chars=800)
                if qp_ctx:
                    prompt += f"\n[站点记忆]\n{qp_ctx}"
            except Exception as _e:
                logger.debug("qp_memory context retrieval failed: %s", _e)

        if context_str:
            prompt += f"\nRelevant memory:\n{context_str}"

        if self._vision and self._vision.available:
            prompt += "\n视觉能力: 已启用"
            if scene_desc:
                prompt += f"\n当前视野: {scene_desc}"

        tool_defs = (
            self._tools.get_definitions(max_safety_level=self._general_tool_max_safety_level)
            if self._tools else []
        )
        if tool_defs:
            tool_names = [
                td.get("function", {}).get("name", "") for td in tool_defs
            ]
            prompt += (
                f"\n你可以主动调用以下工具: {', '.join(tool_names)}。"
                "当用户提问涉及时间、文件、目录等信息时，主动调用对应工具获取真实数据再回答。"
            )

        if self._skill_manager:
            skill_catalog = self._skill_manager.get_skill_catalog()
            if skill_catalog != "none":
                prompt += f"\n可用技能: {skill_catalog}"

        return prompt

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject prompt seed and user format prefix for relay compatibility.

        The relay service overrides our system prompt with its own developer
        assistant identity. To counter this:
          1. DROP the system message when prompt_seed is present — the seed
             establishes identity via fake user/assistant turns, which the
             relay cannot override.
          2. Prepend a TTS format tag to the latest user message to enforce
             output constraints (no markdown, short, Chinese).

        Original conversation history is NOT modified — transformations are
        applied only to the copy sent to the LLM.
        """
        if not self._prompt_seed and not self._user_prefix:
            return messages

        # When seed is present, skip system message to avoid relay conflict.
        # The relay injects its own system prompt regardless; including ours
        # creates competing identities. Seed messages work better alone.
        #
        # BUT: the dropped system prompt contained tool instructions.
        # Inject tool awareness as a seed exchange so the LLM knows it
        # CAN and SHOULD call tools for factual queries.
        if self._prompt_seed:
            result = list(self._prompt_seed)

            tool_defs = self._tools.get_definitions(
                max_safety_level=self._general_tool_max_safety_level
            )
            if tool_defs:
                tool_names = [
                    td.get("function", {}).get("name", "")
                    for td in tool_defs
                ]
                result.append({
                    "role": "user",
                    "content": (
                        f"你有以下工具可用: {', '.join(tool_names)}。"
                        "涉及时间、文件等真实数据时必须调用工具，不要编造。"
                    ),
                })
                result.append({
                    "role": "assistant",
                    "content": "明白，需要真实数据时我会调用工具获取。",
                })

            rest = [m for m in messages if m.get("role") != "system"]
        else:
            result = [messages[0]] if messages else []
            rest = list(messages[1:])

        if self._user_prefix:
            for i in range(len(rest) - 1, -1, -1):
                if rest[i].get("role") == "user":
                    rest[i] = {
                        **rest[i],
                        "content": f"{self._user_prefix}\n{rest[i]['content']}",
                    }
                    break

        result.extend(rest)
        return result
