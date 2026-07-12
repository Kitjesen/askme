"""Mission-state voice command policy.

This module stays inside ``robot_interaction`` so voice channels can ask a
single question before forwarding speech to the agent/runtime:

    "Is this utterance allowed in the current field state?"

It is intentionally deterministic. The LLM may interpret allowed commands
later, but it does not decide whether free-form chat is allowed during patrol,
pause, or emergency states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class MissionMode(StrEnum):
    """User-facing operating state for voice admission."""

    SETUP = "setup"
    IDLE = "idle"
    MISSION_ACTIVE = "mission_active"
    PAUSED = "paused"
    EMERGENCY = "emergency"
    REVIEW = "review"

    @classmethod
    def coerce(cls, value: object) -> MissionMode:
        normalized = str(value or cls.IDLE.value).strip().lower().replace("-", "_")
        aliases = {
            "active": cls.MISSION_ACTIVE.value,
            "mission": cls.MISSION_ACTIVE.value,
            "running": cls.MISSION_ACTIVE.value,
            "patrol": cls.MISSION_ACTIVE.value,
            "巡检中": cls.MISSION_ACTIVE.value,
            "任务中": cls.MISSION_ACTIVE.value,
            "暂停": cls.PAUSED.value,
            "急停": cls.EMERGENCY.value,
            "复盘": cls.REVIEW.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError:
            return cls.IDLE


class MissionActorRole(StrEnum):
    """Coarse role used before RBAC/runtime authority checks."""

    VISITOR = "visitor"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"

    @classmethod
    def coerce(cls, value: object) -> MissionActorRole:
        normalized = str(value or cls.OPERATOR.value).strip().lower().replace("-", "_")
        aliases = {
            "guest": cls.VISITOR.value,
            "public": cls.VISITOR.value,
            "viewer": cls.VISITOR.value,
            "ops": cls.OPERATOR.value,
            "driver": cls.OPERATOR.value,
            "owner": cls.ADMIN.value,
            "访客": cls.VISITOR.value,
            "操作员": cls.OPERATOR.value,
            "安全员": cls.SUPERVISOR.value,
            "管理员": cls.ADMIN.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError:
            return cls.OPERATOR


class MissionCommandCategory(StrEnum):
    CHAT = "chat"
    STATUS = "status"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    EMERGENCY_STOP = "emergency_stop"
    REPORT_ANOMALY = "report_anomaly"
    CALL_OPERATOR = "call_operator"
    START_MISSION = "start_mission"
    ROUTE_CHANGE = "route_change"
    REVIEW_RESULT = "review_result"
    WAYFINDING = "wayfinding"


@dataclass(frozen=True)
class MissionModeDecision:
    action: str
    reason: str
    confidence: float
    reply: str = ""
    should_record_environment: bool = False
    category: MissionCommandCategory = MissionCommandCategory.CHAT
    mode: MissionMode = MissionMode.IDLE
    actor_role: MissionActorRole = MissionActorRole.OPERATOR


_ROLE_LEVEL = {
    MissionActorRole.VISITOR: 0,
    MissionActorRole.OPERATOR: 1,
    MissionActorRole.SUPERVISOR: 2,
    MissionActorRole.ADMIN: 3,
}

_DEFAULT_REPLIES = {
    MissionMode.SETUP: "正在配置中，请由操作员完成设置。",
    MissionMode.MISSION_ACTIVE: "巡检中，只接受任务相关指令。",
    MissionMode.PAUSED: "任务已暂停，只接受继续、停止、状态查询或安全相关指令。",
    MissionMode.EMERGENCY: "急停状态，只接受安全相关指令。",
    MissionMode.REVIEW: "任务已结束，可以查看结果或报告。",
}

_STATUS_TERMS = (
    "状态",
    "进度",
    "电量",
    "位置",
    "到哪",
    "在哪里",
    "在哪",
    "还有多久",
    "完成了吗",
    "结果",
    "报告",
    "status",
    "progress",
    "battery",
    "where are you",
    "result",
    "report",
)

_PAUSE_TERMS = (
    "暂停",
    "停一下",
    "先停",
    "等一下",
    "hold",
    "pause",
)

_RESUME_TERMS = (
    "继续",
    "恢复",
    "接着",
    "resume",
    "continue",
)

_CANCEL_TERMS = (
    "取消",
    "结束任务",
    "终止",
    "停止任务",
    "返回",
    "回去",
    "返航",
    "cancel",
    "abort",
    "return",
)

_EMERGENCY_TERMS = (
    "急停",
    "紧急停止",
    "立刻停止",
    "马上停止",
    "危险停止",
    "e-stop",
    "estop",
    "emergency stop",
)

_ANOMALY_TERMS = (
    "异常",
    "故障",
    "报警",
    "着火",
    "烟",
    "漏水",
    "摔倒",
    "有人受伤",
    "危险",
    "report issue",
    "anomaly",
    "fault",
)

_CALL_OPERATOR_TERMS = (
    "人工",
    "管理员",
    "安全员",
    "联系",
    "呼叫",
    "求助",
    "help",
    "operator",
)

_START_TERMS = (
    "开始巡检",
    "启动巡检",
    "开始任务",
    "启动任务",
    "开始重建",
    "启动重建",
    "开始扫描",
    "开始建图",
    "start patrol",
    "start mission",
    "start reconstruction",
)

_ROUTE_TERMS = (
    "导航到",
    "前往",
    "带我去",
    "去",
    "改路线",
    "换路线",
    "route",
    "navigate",
)

_REVIEW_TERMS = (
    "查看结果",
    "打开结果",
    "打开报告",
    "复盘",
    "总结",
    "summary",
    "review",
)

_WAYFINDING_TERMS = (
    "厕所",
    "卫生间",
    "出口",
    "前台",
    "电梯",
    "展区",
    "怎么走",
    "问路",
)


def classify_mission_command(text: str) -> MissionCommandCategory:
    """Classify command category with safety-sensitive terms first."""

    clean = " ".join(str(text or "").strip().lower().split())
    if not clean:
        return MissionCommandCategory.CHAT
    ordered: tuple[tuple[MissionCommandCategory, tuple[str, ...]], ...] = (
        (MissionCommandCategory.EMERGENCY_STOP, _EMERGENCY_TERMS),
        (MissionCommandCategory.PAUSE, _PAUSE_TERMS),
        (MissionCommandCategory.RESUME, _RESUME_TERMS),
        (MissionCommandCategory.CANCEL, _CANCEL_TERMS),
        (MissionCommandCategory.REPORT_ANOMALY, _ANOMALY_TERMS),
        (MissionCommandCategory.CALL_OPERATOR, _CALL_OPERATOR_TERMS),
        (MissionCommandCategory.REVIEW_RESULT, _REVIEW_TERMS),
        (MissionCommandCategory.START_MISSION, _START_TERMS),
        (MissionCommandCategory.ROUTE_CHANGE, _ROUTE_TERMS),
        (MissionCommandCategory.WAYFINDING, _WAYFINDING_TERMS),
        (MissionCommandCategory.STATUS, _STATUS_TERMS),
    )
    for category, terms in ordered:
        if any(term and term in clean for term in terms):
            return category
    return MissionCommandCategory.CHAT


def evaluate_mission_mode(
    text: str,
    *,
    mission_mode: object = MissionMode.IDLE,
    actor_role: object = MissionActorRole.OPERATOR,
    addressed: bool = True,
    replies: Mapping[str, str] | None = None,
) -> MissionModeDecision | None:
    """Return a deterministic gate decision for active field states.

    ``None`` means the normal interaction gate should continue evaluating the
    turn. This keeps idle/setup behavior compatible with the existing voice
    assistant while tightening patrol-time admission.
    """

    mode = MissionMode.coerce(mission_mode)
    role = MissionActorRole.coerce(actor_role)
    category = classify_mission_command(text)

    if mode == MissionMode.IDLE:
        return None
    if mode == MissionMode.SETUP:
        if role == MissionActorRole.VISITOR and addressed:
            return _deny(mode, role, category, replies, "mission_setup_requires_operator")
        return None

    if mode == MissionMode.MISSION_ACTIVE:
        return _active_decision(mode, role, category, addressed, replies)

    if mode == MissionMode.PAUSED:
        return _paused_decision(mode, role, category, addressed, replies)

    if mode == MissionMode.EMERGENCY:
        return _emergency_decision(mode, role, category, addressed, replies)

    if mode == MissionMode.REVIEW:
        return _review_decision(mode, role, category, addressed, replies)

    return None


def _active_decision(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    addressed: bool,
    replies: Mapping[str, str] | None,
) -> MissionModeDecision | None:
    allowed = {
        MissionCommandCategory.STATUS,
        MissionCommandCategory.PAUSE,
        MissionCommandCategory.EMERGENCY_STOP,
        MissionCommandCategory.REPORT_ANOMALY,
        MissionCommandCategory.CALL_OPERATOR,
    }
    if _ROLE_LEVEL[role] >= _ROLE_LEVEL[MissionActorRole.OPERATOR]:
        allowed.update({MissionCommandCategory.CANCEL, MissionCommandCategory.RESUME})
    if category in allowed:
        return _allow(mode, role, category, "mission_active_command_allowed")
    if not addressed and category == MissionCommandCategory.CHAT:
        return None
    return _deny(mode, role, category, replies, "mission_active_chat_blocked")


def _paused_decision(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    addressed: bool,
    replies: Mapping[str, str] | None,
) -> MissionModeDecision | None:
    allowed = {
        MissionCommandCategory.STATUS,
        MissionCommandCategory.EMERGENCY_STOP,
        MissionCommandCategory.REPORT_ANOMALY,
        MissionCommandCategory.CALL_OPERATOR,
        MissionCommandCategory.CANCEL,
    }
    if _ROLE_LEVEL[role] >= _ROLE_LEVEL[MissionActorRole.OPERATOR]:
        allowed.add(MissionCommandCategory.RESUME)
    if _ROLE_LEVEL[role] >= _ROLE_LEVEL[MissionActorRole.SUPERVISOR]:
        allowed.update({MissionCommandCategory.ROUTE_CHANGE, MissionCommandCategory.START_MISSION})
    if category in allowed:
        return _allow(mode, role, category, "mission_paused_command_allowed")
    if not addressed and category == MissionCommandCategory.CHAT:
        return None
    return _deny(mode, role, category, replies, "mission_paused_chat_blocked")


def _emergency_decision(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    addressed: bool,
    replies: Mapping[str, str] | None,
) -> MissionModeDecision | None:
    allowed = {
        MissionCommandCategory.STATUS,
        MissionCommandCategory.EMERGENCY_STOP,
        MissionCommandCategory.REPORT_ANOMALY,
        MissionCommandCategory.CALL_OPERATOR,
    }
    if _ROLE_LEVEL[role] >= _ROLE_LEVEL[MissionActorRole.SUPERVISOR]:
        allowed.add(MissionCommandCategory.RESUME)
    if category in allowed:
        return _allow(mode, role, category, "mission_emergency_command_allowed")
    if not addressed and category == MissionCommandCategory.CHAT:
        return None
    return _deny(mode, role, category, replies, "mission_emergency_chat_blocked")


def _review_decision(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    addressed: bool,
    replies: Mapping[str, str] | None,
) -> MissionModeDecision | None:
    allowed = {
        MissionCommandCategory.STATUS,
        MissionCommandCategory.REVIEW_RESULT,
        MissionCommandCategory.REPORT_ANOMALY,
        MissionCommandCategory.CALL_OPERATOR,
    }
    if _ROLE_LEVEL[role] >= _ROLE_LEVEL[MissionActorRole.OPERATOR]:
        allowed.add(MissionCommandCategory.START_MISSION)
    if category in allowed:
        return _allow(mode, role, category, "mission_review_command_allowed")
    if not addressed and category == MissionCommandCategory.CHAT:
        return None
    return _deny(mode, role, category, replies, "mission_review_chat_blocked")


def _allow(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    reason: str,
) -> MissionModeDecision:
    return MissionModeDecision(
        action="respond",
        reason=f"{reason}:{category.value}",
        confidence=0.9,
        category=category,
        mode=mode,
        actor_role=role,
    )


def _deny(
    mode: MissionMode,
    role: MissionActorRole,
    category: MissionCommandCategory,
    replies: Mapping[str, str] | None,
    reason: str,
) -> MissionModeDecision:
    reply = _reply_for(mode, replies)
    return MissionModeDecision(
        action="defer" if mode != MissionMode.EMERGENCY else "refuse",
        reason=f"{reason}:{category.value}",
        confidence=0.86,
        reply=reply,
        should_record_environment=True,
        category=category,
        mode=mode,
        actor_role=role,
    )


def _reply_for(mode: MissionMode, replies: Mapping[str, str] | None) -> str:
    if replies:
        configured = replies.get(mode.value)
        if configured:
            return configured
    return _DEFAULT_REPLIES.get(mode, _DEFAULT_REPLIES[MissionMode.MISSION_ACTIVE])
