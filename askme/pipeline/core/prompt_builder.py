"""Prompt construction helpers for BrainPipeline."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any

from askme.pipeline.core.rag_policy import forced_rag_reply

if TYPE_CHECKING:
    from askme.memory.core.episodic_memory import EpisodicMemory
    from askme.memory.core.session import SessionMemory
    from askme.memory.core.system import MemorySystem
    from askme.ports import SafetyPort, VisionPort
    from askme.skills.core.skill_manager import SkillManager
    from askme.tools.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Assembles system prompts and prepares message lists for LLM calls."""

    _TEXT_CHANNEL_HINT = "文本/API通道：当前用户消息已明确发给你；除非消息为空或噪声，不要回复[SILENT]。"

    _DEFAULT_RAG_POLICY_REPLIES = {
        "filtered": "我找到了相关资料，但还不能作为可靠依据。请管理员先复核知识。",
        "stale": "这条知识已经过期，我不能按旧信息回答。请先刷新问答可用性。",
        "conflict": "我查到的信息互相冲突，不能给出确定结论。请管理员确认保留哪一条。",
        "unapproved": "相关知识还没有审批发布，不能用于回答。请先完成审批。",
        "no_evidence": "我这里没有可靠依据，不能直接回答。请补充位置或让管理员上传知识。",
    }

    def __init__(
        self,
        *,
        base_prompt: str,
        prompt_seed: list[dict[str, str]],
        user_prefix: str,
        tools: ToolRegistry,
        skill_manager: SkillManager,
        general_tool_max_safety_level: str,
        dog_safety: SafetyPort | None,
        episodic: EpisodicMemory | None,
        session_memory: SessionMemory | None,
        vision: VisionPort | None,
        qp_memory: Any,
        memory_system: MemorySystem | None = None,
        rag_policy_templates: dict[str, str] | None = None,
        relay_compat_mode: bool = False,
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
        self._relay_compat_mode = bool(relay_compat_mode)
        self._rag_policy_templates = {
            str(key).strip(): str(value).strip()
            for key, value in (rag_policy_templates or {}).items()
            if str(key).strip() and str(value).strip()
        }

    def build_l0_runtime_block(self) -> str:
        """Return a compact L0 runtime truth block from authoritative services.

        Non-blocking: reads only the in-memory cache on the configured safety port.
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
        behavior_context: str = "",
        scene_desc: str = "",
        user_text: str = "",
        rag_policy: dict[str, Any] | None = None,
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

        if behavior_context:
            prompt += (
                "\n[\u957f\u671f\u4ea4\u4e92\u8bb0\u5fc6]\n"
                "\u4ec5\u7528\u4e8e\u79f0\u547c\u3001\u504f\u597d\u548c\u4ea4\u4e92\u65b9\u5f0f\uff1b\u4e0d\u5f97\u4f5c\u4e3a\u8def\u7ebf\u3001SOP\u3001\u8bbe\u5907\u6216 FAQ \u4e8b\u5b9e\u4f9d\u636e\u3002\n"
                f"{behavior_context}"
            )

        policy_block = self._format_rag_policy_block(rag_policy)
        if policy_block:
            prompt += f"\n{policy_block}"

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

    def build_forced_rag_reply(self, policy: dict[str, Any] | None) -> str:
        """Return a deterministic customer-facing reply when RAG policy forbids answering."""
        return forced_rag_reply(policy, templates=self._rag_policy_templates)

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        source: str = "voice",
    ) -> list[dict[str, Any]]:
        """Inject prompt seed and user format prefix.

        Direct providers retain the authoritative system message. Legacy relay
        deployments can explicitly enable ``relay_compat_mode`` to drop it and
        establish identity through seed turns when the relay overwrites system
        messages upstream.

        Original conversation history is NOT modified — transformations are
        applied only to the copy sent to the LLM.
        """
        source = str(source or "voice").strip().lower()
        text_channel_hint = self._TEXT_CHANNEL_HINT if source != "voice" else ""

        if not self._prompt_seed and not self._user_prefix and not text_channel_hint:
            return messages

        if self._prompt_seed:
            system_messages = [m for m in messages if m.get("role") == "system"]
            system_prompt = "\n".join(
                str(message.get("content", "")) for message in system_messages
            )
            result = [] if self._relay_compat_mode else list(system_messages)
            result.extend(self._prompt_seed)

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

            memory_block = self._extract_system_section(system_prompt, "Relevant memory:")
            if memory_block:
                result.append({
                    "role": "user",
                    "content": (
                        "当前回合可引用的记忆证据如下。回答必须优先贴合这些证据，"
                        f"不要说没有记录。\n{memory_block}"
                    ),
                })
                result.append({
                    "role": "assistant",
                    "content": "明白，我会使用当前回合提供的记忆证据回答，不会忽略这些记录。",
                })

            behavior_block = self._extract_system_section(
                system_prompt,
                "[\u957f\u671f\u4ea4\u4e92\u8bb0\u5fc6]",
            )
            if behavior_block:
                result.append({
                    "role": "user",
                    "content": (
                        "\u4e0b\u5217\u662f\u4ea4\u4e92\u504f\u597d\u8bb0\u5fc6\uff0c\u53ea\u7528\u4e8e\u79f0\u547c\u548c\u4ea4\u4e92\u65b9\u5f0f\uff0c"
                        f"\u4e0d\u5f97\u5f53\u4f5c\u5ba2\u6237\u77e5\u8bc6\u8bc1\u636e\uff1a\n{behavior_block}"
                    ),
                })
                result.append({
                    "role": "assistant",
                    "content": "\u660e\u767d\uff0c\u6211\u53ea\u4f1a\u5c06\u5b83\u7528\u4e8e\u4e2a\u6027\u5316\u4ea4\u4e92\u3002",
                })

            rag_policy_block = self._extract_system_section(system_prompt, "[知识回答策略]")
            if rag_policy_block:
                result.append({
                    "role": "user",
                    "content": (
                        "当前回合必须遵守以下知识回答策略，尤其不能把未通过审核、过期、"
                        f"冲突或无证据的内容当作事实回答:\n{rag_policy_block}"
                    ),
                })
                result.append({
                    "role": "assistant",
                    "content": "明白，我会按知识回答策略处理证据不足、冲突或过期的情况。",
                })

            rest = [m for m in messages if m.get("role") != "system"]
        else:
            result = [messages[0]] if messages else []
            rest = list(messages[1:])

        if self._user_prefix or text_channel_hint:
            prefix = "\n".join(
                part for part in (self._user_prefix, text_channel_hint) if part
            )
            for i in range(len(rest) - 1, -1, -1):
                if rest[i].get("role") == "user":
                    rest[i] = {
                        **rest[i],
                        "content": f"{prefix}\n{rest[i]['content']}",
                    }
                    break

        result.extend(rest)
        return result

    def reconfigure(
        self,
        *,
        base_prompt: str | None = None,
        prompt_seed: list[dict[str, str]] | None = None,
        user_prefix: str | None = None,
        rag_policy_templates: dict[str, str] | None = None,
        relay_compat_mode: bool | None = None,
    ) -> dict[str, Any]:
        """Apply prompt settings for subsequent turns without rebuilding the pipeline."""

        if base_prompt is not None:
            self._base_prompt = str(base_prompt).strip()
        if prompt_seed is not None:
            self._prompt_seed = [
                dict(item) for item in prompt_seed if isinstance(item, dict)
            ]
        if user_prefix is not None:
            self._user_prefix = str(user_prefix)
        if rag_policy_templates is not None:
            self._rag_policy_templates = {
                str(key).strip(): str(value).strip()
                for key, value in rag_policy_templates.items()
                if str(key).strip() and str(value).strip()
            }
        if relay_compat_mode is not None:
            self._relay_compat_mode = bool(relay_compat_mode)
        return self.runtime_settings()

    def runtime_settings(self) -> dict[str, Any]:
        """Return the active non-secret prompt settings for the control plane."""

        return {
            "system_prompt": self._base_prompt,
            "prompt_seed": [dict(item) for item in self._prompt_seed],
            "user_prefix": self._user_prefix,
            "rag_policy_templates": dict(self._rag_policy_templates),
            "relay_compat_mode": self._relay_compat_mode,
        }

    def _format_rag_policy_block(self, policy: dict[str, Any] | None) -> str:
        if not isinstance(policy, dict) or not policy:
            return ""
        state = str(policy.get("state") or "").strip()
        action = str(policy.get("action") or "").strip()
        reason = str(policy.get("reason") or "").strip()
        message = str(policy.get("message") or "").strip()
        required_operator_action = str(policy.get("required_operator_action") or "").strip()
        if not state and not action:
            return ""

        rule = (
            "只能把 Relevant memory 中通过审核且未过期的内容当作可引用事实；"
            "不能用被过滤、过期、冲突或未审批的知识补全答案。"
        )
        if state == "grounded":
            rule += " 当前有可用证据，回答必须贴合证据；证据不足的部分要说明不确定。"
        elif state == "conflict":
            rule += " 当前相关知识存在冲突，必须说明无法给出确定结论，并请用户确认或更新知识。"
        elif state == "stale":
            rule += " 当前相关知识过期或版本不匹配，必须拒绝确定性回答，并请求重新同步或更新知识。"
        elif state == "unapproved":
            rule += " 当前相关知识未审批或已删除，必须拒绝把它作为事实回答。"
        elif state in {"no_evidence", "filtered"}:
            rule += (
                " 当前没有可用证据；如果用户询问站点、设备、路线、SOP、巡检或客户业务事实，"
                "必须说明没有可靠依据，不能编造。"
            )

        reply_template = self._rag_policy_templates.get(state, "")
        if reply_template:
            rule += (
                " 对外回复优先使用 customer_reply_template 的含义，保持口语、简短、可执行；"
                "不要把 state、action、reason 等内部字段念给用户。"
            )

        lines = [
            "[知识回答策略]",
            f"state: {state or 'unknown'}",
            f"action: {action or 'unknown'}",
            f"reason: {reason or '-'}",
        ]
        if required_operator_action:
            lines.append(f"required_operator_action: {required_operator_action}")
        if message:
            lines.append(f"message: {message}")
        if reply_template:
            lines.append(f"customer_reply_template: {reply_template}")
        lines.append(f"rule: {rule}")
        return "\n".join(lines)

    @staticmethod
    def _extract_system_section(system_prompt: str, heading: str) -> str:
        start = system_prompt.find(heading)
        if start < 0:
            return ""
        section = system_prompt[start:].strip()
        next_heading = section.find("\n[", len(heading))
        if next_heading > 0:
            section = section[:next_heading].strip()
        return section
