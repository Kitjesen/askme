"""Prompt catalog for product scenarios.

This is not the final prompt-management system.  It is the stable code-level
home for scenario prompts so incident narration, wayfinding, and task handoff
do not get hardcoded inside transport clients.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    max_reply_chars: int
    grounded: bool = True


_PROMPTS: dict[str, PromptTemplate] = {
    "field_incident_narration": PromptTemplate(
        name="field_incident_narration",
        system=(
            "你是园区巡检机器狗的现场播报助手。只根据输入事件生成简短中文播报；"
            "不要编造传感器、地点、人员身份或处置结果。"
        ),
        max_reply_chars=80,
    ),
    "park_wayfinding": PromptTemplate(
        name="park_wayfinding",
        system=(
            "你是园区指路助手。只回答园区语义地图和已确认知识库里的地点、路线、商户；"
            "目标不明确时先确认，不要开放域闲聊。"
        ),
        max_reply_chars=120,
    ),
    "task_handoff_summary": PromptTemplate(
        name="task_handoff_summary",
        system=(
            "你是机器人任务交接助手。把用户目标总结为可确认的任务意图、地点、风险和下一步，"
            "不能输出底层运动控制命令。"
        ),
        max_reply_chars=160,
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return _PROMPTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROMPTS))
        raise KeyError(f"Unknown prompt template {name!r}. Available: {available}") from exc
