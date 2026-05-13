"""Customer-configurable assistant persona helpers.

The runtime must not hardcode vendor ownership into customer-facing prompts.
This module turns deployment config into system prompts and seed turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssistantPersona:
    robot_name: str = "Thunder"
    product_name: str = "现场任务平台"
    customer_name: str = ""
    operator_audience: str = "现场运营人员、安保和交付工程师"
    role: str = "园区巡检与服务机器人"
    ownership_label: str = ""
    speaking_style: str = "中文口语，简洁汇报，像现场值班员"
    max_reply_chars: int = 80

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> AssistantPersona:
        data = raw if isinstance(raw, dict) else {}
        return cls(
            robot_name=str(data.get("robot_name") or cls.robot_name).strip(),
            product_name=str(data.get("product_name") or cls.product_name).strip(),
            customer_name=str(data.get("customer_name") or "").strip(),
            operator_audience=str(data.get("operator_audience") or cls.operator_audience).strip(),
            role=str(data.get("role") or cls.role).strip(),
            ownership_label=str(data.get("ownership_label") or "").strip(),
            speaking_style=str(data.get("speaking_style") or cls.speaking_style).strip(),
            max_reply_chars=int(data.get("max_reply_chars") or cls.max_reply_chars),
        )

    def build_system_prompt(self) -> str:
        owner = f"归属口径：{self.ownership_label}。" if self.ownership_label else "不要主动声明厂商或归属。"
        customer = f"当前服务项目：{self.customer_name}。" if self.customer_name else "当前服务项目由部署配置决定。"
        return (
            f"你是{self.robot_name}，{self.product_name}中的{self.role}。"
            f"{customer}"
            f"服务对象是{self.operator_audience}。"
            f"{owner}"
            f"说话风格：{self.speaking_style}，短句为主，不超过{self.max_reply_chars}字。"
            "不用 markdown、emoji、英文。"
            "如果判断用户不是在跟你说话，只回复[SILENT]。"
            "没有真实传感器、地图、任务或知识库证据时，不编造结论。"
            "不确定时说不确定，并要求确认或补充信息。"
            "不要说自己是 AI 助手或语言模型。"
        )

    def build_prompt_seed(self) -> list[dict[str, str]]:
        customer = f"，当前项目是{self.customer_name}" if self.customer_name else ""
        owner = f"。对外归属口径：{self.ownership_label}" if self.ownership_label else "。不主动声明厂商或归属"
        return [
            {
                "role": "user",
                "content": (
                    f"你是{self.robot_name}，{self.product_name}里的{self.role}{customer}{owner}。"
                    f"请用中文口语简洁回复，{self.max_reply_chars}字以内。"
                    "没有证据不要编造；不是对你说话时只回复[SILENT]。"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"收到。我是{self.robot_name}，会按当前项目配置回答，"
                    "不主动声明厂商归属，没有证据不编造。"
                ),
            },
        ]

    def build_user_prefix(self) -> str:
        return (
            f"[{self.role}模式：中文口语，{self.max_reply_chars}字以内，"
            "简洁汇报，不用markdown，不说英文]"
        )


def persona_from_brain_config(brain_cfg: dict[str, Any]) -> AssistantPersona:
    """Build persona from ``brain.persona`` or legacy ``brain.identity`` config."""
    raw = brain_cfg.get("persona")
    if not isinstance(raw, dict):
        raw = brain_cfg.get("identity")
    return AssistantPersona.from_config(raw if isinstance(raw, dict) else {})
