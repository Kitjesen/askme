"""Auditable scene-intent routing for customer field scenarios.

This layer deliberately stays deterministic. It catches common customer
utterance variants that are not exact ``voice_trigger`` strings, then records
the evidence that caused the route. High-risk actions still go through the
normal skill gate, confirmation, safety preflight, and runtime boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[\s,，。.!！?？:：;；、\"'“”‘’（）()\[\]{}<>《》]+")


@dataclass(frozen=True)
class ScenarioIntentDecision:
    """One deterministic scene-intent decision."""

    skill_name: str
    scenario_id: str
    confidence: float
    matched_terms: tuple[str, ...]
    rule_id: str
    risk_level: str
    evidence: str


@dataclass(frozen=True)
class ScenarioIntentRule:
    """A conservative lexical rule for one product scenario."""

    rule_id: str
    skill_name: str
    scenario_id: str
    any_terms: tuple[str, ...]
    all_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    confidence: float = 0.74
    risk_level: str = "normal"
    evidence: str = ""

    def match(self, text: str) -> ScenarioIntentDecision | None:
        normalized = normalize_intent_text(text)
        if not normalized:
            return None

        excluded = [term for term in self.exclude_terms if _has_term(normalized, term)]
        if excluded:
            return None

        missing_required = [term for term in self.all_terms if not _has_term(normalized, term)]
        if missing_required:
            return None

        matched = tuple(term for term in self.any_terms if _has_term(normalized, term))
        if not matched:
            return None

        return ScenarioIntentDecision(
            skill_name=self.skill_name,
            scenario_id=self.scenario_id,
            confidence=self.confidence,
            matched_terms=matched,
            rule_id=self.rule_id,
            risk_level=self.risk_level,
            evidence=self.evidence or self.rule_id,
        )


def normalize_intent_text(text: str) -> str:
    """Normalize ASR text for deterministic Chinese/English term matching."""

    return _PUNCT_RE.sub("", str(text or "").strip().lower())


def classify_scenario_intent(
    text: str,
    *,
    available_skills: Iterable[str] | None = None,
) -> ScenarioIntentDecision | None:
    """Return the first matching scenario decision allowed by installed skills."""

    allowed = {str(skill).strip() for skill in (available_skills or []) if str(skill).strip()}
    if not allowed:
        return None

    for rule in SCENARIO_INTENT_RULES:
        if rule.skill_name not in allowed:
            continue
        decision = rule.match(text)
        if decision is not None:
            return decision
    return None


def _has_term(normalized_text: str, term: str) -> bool:
    return normalize_intent_text(term) in normalized_text


SCENARIO_INTENT_RULES: tuple[ScenarioIntentRule, ...] = (
    # Visitor spatial service. These are safe question/answer routes and should
    # not create robot movement by themselves.
    ScenarioIntentRule(
        rule_id="wayfinding_lookup",
        skill_name="lookup_place",
        scenario_id="wayfinding",
        any_terms=(
            "在哪里",
            "在哪",
            "哪儿",
            "哪里",
            "找厕所",
            "找卫生间",
            "厕所在哪",
            "卫生间在哪",
            "咖啡店在哪",
            "停车场在哪",
            "西门在哪",
            "问路",
        ),
        exclude_terms=("违停", "乱停", "堵路", "挡路"),
        confidence=0.78,
        risk_level="visitor_service",
        evidence="visitor destination lookup wording",
    ),
    ScenarioIntentRule(
        rule_id="wayfinding_voice_route",
        skill_name="answer_wayfinding",
        scenario_id="wayfinding",
        any_terms=("怎么走", "怎么去", "路线怎么走", "给我指路", "帮我指路", "指一下路"),
        confidence=0.8,
        risk_level="visitor_service",
        evidence="visitor route guidance wording",
    ),
    ScenarioIntentRule(
        rule_id="visitor_escort_request",
        skill_name="escort_visitor",
        scenario_id="visitor_escort",
        any_terms=("带我去", "请带路", "机器狗带路", "送我去", "带路去", "领我去"),
        confidence=0.82,
        risk_level="dangerous",
        evidence="visitor requested robot escort",
    ),
    ScenarioIntentRule(
        rule_id="urgent_patrol_dispatch_request",
        skill_name="patrol_scan",
        scenario_id="urgent_patrol_dispatch",
        any_terms=(
            "突发巡检",
            "临时巡检",
            "派遣巡检",
            "派机器狗",
            "派遣机器狗",
            "去a区巡检",
            "去a区北门巡检",
            "前往a区巡检",
            "前往北门巡检",
            "去北门巡检",
            "去三号楼巡检",
            "打开相机看一下",
            "去现场查看",
        ),
        exclude_terms=("在哪", "哪里", "哪儿", "怎么走", "带我去", "请带路"),
        confidence=0.82,
        risk_level="dangerous",
        evidence="operator requested urgent patrol dispatch",
    ),
    # Field incident scenarios.
    ScenarioIntentRule(
        rule_id="illegal_parking_report",
        skill_name="detect_illegal_parking",
        scenario_id="illegal_parking",
        any_terms=(
            "违停",
            "乱停",
            "车停路中间",
            "车停在路上",
            "车停在主通道",
            "停在主通道",
            "车堵住",
            "堵住主通道",
            "主通道有车",
            "消防通道有车",
            "道路上有车",
        ),
        exclude_terms=("停车场在哪", "去停车场", "停车场怎么走"),
        confidence=0.84,
        risk_level="dangerous",
        evidence="vehicle appears to block a road or forbidden area",
    ),
    ScenarioIntentRule(
        rule_id="fire_or_smoke_report",
        skill_name="detect_fire_smoke",
        scenario_id="fire_or_smoke",
        any_terms=(
            "冒烟",
            "有烟",
            "烟味",
            "烟感",
            "着火",
            "起火",
            "火苗",
            "火警",
            "温度太高",
            "高温报警",
        ),
        confidence=0.86,
        risk_level="dangerous",
        evidence="fire, smoke, or high-temperature wording",
    ),
    ScenarioIntentRule(
        rule_id="trash_bin_full_report",
        skill_name="inspect_trash_bin",
        scenario_id="trash_bin_full",
        any_terms=("垃圾桶满", "垃圾桶快满", "垃圾溢出来", "桶满了", "清理垃圾桶", "通知保洁"),
        confidence=0.82,
        risk_level="dangerous",
        evidence="trash-bin overflow wording",
    ),
    ScenarioIntentRule(
        rule_id="night_stranger_photo_report",
        skill_name="detect_night_intruder",
        scenario_id="night_stranger_photo",
        any_terms=(
            "陌生人拍照",
            "有人在窗户拍照",
            "窗户旁边有人",
            "角落有人拍照",
            "围栏边有人拍照",
            "夜间有人停留",
            "夜里有人拍照",
        ),
        confidence=0.83,
        risk_level="dangerous",
        evidence="night stranger or sensitive-area photo wording",
    ),
    ScenarioIntentRule(
        rule_id="robot_fall_report",
        skill_name="report_fall_unrecoverable",
        scenario_id="robot_abnormal_incident",
        any_terms=("摔倒", "倒地", "翻倒", "起不来", "爬不起来"),
        confidence=0.86,
        risk_level="dangerous",
        evidence="robot fall/unrecoverable wording",
    ),
    ScenarioIntentRule(
        rule_id="robot_motor_fault_report",
        skill_name="report_motor_fault",
        scenario_id="robot_abnormal_incident",
        any_terms=("电机故障", "关节故障", "关节报警", "电机坏", "腿部故障", "执行器故障"),
        confidence=0.86,
        risk_level="dangerous",
        evidence="robot actuator or joint-motor fault wording",
    ),
    ScenarioIntentRule(
        rule_id="robot_malicious_blocking_report",
        skill_name="report_malicious_blocking",
        scenario_id="robot_abnormal_incident",
        any_terms=(
            "恶意挡路",
            "人为挡路",
            "有人挡路",
            "有人恶意挡住",
            "有人恶意挡住机器狗",
            "有人拦住机器狗",
            "有人故意挡住",
            "被人挡住",
        ),
        confidence=0.86,
        risk_level="dangerous",
        evidence="human intentionally blocking the robot",
    ),
    ScenarioIntentRule(
        rule_id="robot_stuck_report",
        skill_name="report_stuck",
        scenario_id="robot_abnormal_incident",
        any_terms=("卡住", "动不了", "无法移动", "走不动", "困住"),
        confidence=0.84,
        risk_level="dangerous",
        evidence="robot immobilized or blocked wording",
    ),
    ScenarioIntentRule(
        rule_id="crowd_gathering_report",
        skill_name="detect_crowd_gathering",
        scenario_id="crowd_gathering",
        any_terms=("人群聚集", "人太多", "多人聚集", "停留太久", "聚了一群人"),
        confidence=0.8,
        risk_level="dangerous",
        evidence="crowd-gathering wording",
    ),
)


__all__ = [
    "SCENARIO_INTENT_RULES",
    "ScenarioIntentDecision",
    "ScenarioIntentRule",
    "classify_scenario_intent",
    "normalize_intent_text",
]
