"""PlannerAgent — turns high-level user goals into ordered skill sequences.

When a user says "帮我完成巡逻任务" or "先去仓库取货然后送到会议室",
a keyword-based IntentRouter cannot match these to a single skill.
PlannerAgent closes that gap: it uses the LLM to decompose the intent into
an ordered list of (skill, sub-intent) pairs, which SkillDispatcher then
executes step by step.

Design constraints:
- Only fires from handle_general() — never from keyword-triggered paths
- Returns None for single-skill intents or conversational turns (no-op)
- Max 5 steps — prevents runaway multi-step chains
- Uses temperature=0 for deterministic JSON output
- Silently falls back to None on any LLM or parse error
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.brain.llm_client import LLMClient
    from askme.skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)

MAX_STEPS = 5

_SYSTEM_PROMPT = """\
你是机器人任务规划器。分析用户请求，判断是否需要按顺序执行多个技能。

规则：
1. 如果用户请求涉及多个步骤或高层目标，输出执行计划（最多5步）。
   触发条件（任意一条满足）：
   - 含关键词：然后/再/先...再/之后/完成后/接着/最后
   - 隐含先后顺序：取货后送达、巡逻+汇报、导航+执行动作
   - 复合目标："完成巡逻任务"（需要多步才能完成）
2. 如果只需要单个技能，或是普通问题/对话，输出 {"plan": null}
3. 每一步的 skill 必须是可用技能列表中存在的名称，不可自造
4. intent 字段写该步骤的完整执行指令，保留原始请求中的地点/对象/参数细节

示例（可用技能假设有 navigate, robot_grab, patrol_report）：
用户：先去东区仓库取货然后送到会议室
→ {"plan": [{"skill": "navigate", "intent": "前往东区仓库"}, {"skill": "robot_grab", "intent": "抓取仓库中的货物"}, {"skill": "navigate", "intent": "导航到会议室"}]}

用户：完成A区巡逻并汇报
→ {"plan": [{"skill": "navigate", "intent": "导航至A区开始巡逻"}, {"skill": "patrol_report", "intent": "汇报A区巡逻结果"}]}

用户：现在几点了
→ {"plan": null}

用户：去仓库
→ {"plan": null}

输出格式（严格JSON，不要有任何其他文字）：
{"plan": [{"skill": "技能名", "intent": "该步骤的完整指令"}]}
或
{"plan": null}
"""


@dataclass
class PlanStep:
    """One step in a multi-skill execution plan."""

    skill_name: str
    intent: str


class PlannerAgent:
    """Decompose high-level goals into ordered PlanStep lists.

    Usage::

        planner = PlannerAgent(llm_client=..., skill_manager=...)
        steps = await planner.plan("先导航到仓库，再抓取货物")
        # steps = [PlanStep("navigate", "导航到仓库"), PlanStep("robot_grab", "抓取货物")]
        # steps = None  ← for single-skill or conversational input
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        skill_manager: "SkillManager",
        *,
        model: str | None = None,
    ) -> None:
        self._llm = llm_client
        self._skill_manager = skill_manager
        # Use a fast/cheap model for planning — haiku is ideal since the task
        # is simple JSON generation.  Falls back to the LLM client's default
        # when None (e.g. no voice_model configured).
        self._model = model

    async def plan(self, user_text: str) -> list[PlanStep] | None:
        """Return an ordered plan, or None if the intent is single-step/conversational.

        Always returns None on LLM/parse errors — callers fall back to normal LLM processing.
        """
        skills = self._skill_manager.get_enabled()
        if not skills:
            return None

        catalog = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        user_msg = f"可用技能：\n{catalog}\n\n用户请求：{user_text}"

        try:
            raw = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                model=self._model,
            )
            # Strip markdown code fences if model wraps in ```json
            text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(text)
            steps_data = parsed.get("plan")
            if not steps_data:
                return None

            steps: list[PlanStep] = []
            for item in steps_data[:MAX_STEPS]:
                skill_name = str(item.get("skill", "")).strip()
                intent = str(item.get("intent", user_text)).strip()
                if skill_name and self._skill_manager.get(skill_name):
                    steps.append(PlanStep(skill_name=skill_name, intent=intent))
                else:
                    logger.warning("PlannerAgent: unknown skill in plan: %r", skill_name)

            # A single-step plan adds no value over normal routing — return None
            if len(steps) <= 1:
                return None

            logger.info(
                "PlannerAgent: %d-step plan for %r: %s",
                len(steps),
                user_text[:40],
                [s.skill_name for s in steps],
            )
            return steps

        except Exception as exc:
            logger.debug("PlannerAgent: planning failed (%s), falling back to LLM", exc)
            return None
