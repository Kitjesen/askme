"""Interaction gate for real-world robot voice turns.

The gate decides whether recognized speech should enter the conversation
pipeline. It separates ambient/environment speech from intentional human-robot
interaction before LLM planning or task execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from askme.robot_interaction.mission.mode import (
    MissionActorRole,
    MissionCommandCategory,
    MissionMode,
    evaluate_mission_mode,
)
from askme.robot_interaction.perception_context import InteractionPerceptionSnapshot


class InteractionAction(StrEnum):
    """Decision returned by :class:`InteractionGate`."""

    IGNORE = "ignore"
    RECORD_ONLY = "record_only"
    CLARIFY = "clarify"
    RESPOND = "respond"
    DEFER = "defer"
    REFUSE = "refuse"


@dataclass(frozen=True)
class InteractionDecision:
    action: InteractionAction
    reason: str
    confidence: float
    reply: str = ""
    should_record_environment: bool = False

    @property
    def should_continue_to_brain(self) -> bool:
        return self.action == InteractionAction.RESPOND


_DEFAULT_WAKE_TERMS = (
    "thunder",
    "\u673a\u5668\u4eba",  # robot
    "\u673a\u5668\u72d7",  # robot dog
    "\u5c0f\u96f7",
    "\u96f7\u9706",
)

_HELP_TERMS = (
    "\u8bf7\u95ee",
    "\u5e2e\u6211",
    "\u5e26\u6211",
    "\u600e\u4e48\u8d70",
    "\u5728\u54ea",
    "\u5728\u54ea\u91cc",
    "\u51fa\u53e3",
    "\u5395\u6240",
    "\u536b\u751f\u95f4",
    "\u524d\u53f0",
    "\u7535\u68af",
    "\u5c55\u533a",
    "\u95ee\u8def",
)

_ROBOT_TASK_TERMS = (
    "\u5de1\u68c0",
    "\u68c0\u67e5",
    "\u5bfc\u822a",
    "\u8fd4\u56de",
    "\u56de\u53bb",
    "\u62cd\u7167",
    "\u626b\u63cf",
    "\u751f\u6210\u62a5\u544a",
    "\u6682\u505c",
    "\u7ee7\u7eed",
    "\u53d6\u6d88",
    "\u786e\u8ba4",
    "\u505c\u4e0b",
    "\u6025\u505c",
    "\u7d27\u6025\u505c\u6b62",
)

_SAFETY_TERMS = (
    "暂停",
    "停下",
    "急停",
    "紧急停止",
    "别动",
)

_TOOL_ROUTE_TERMS = (
    "查一下",
    "查一查",
    "帮我查",
    "查询",
    "搜索",
    "搜一下",
    "联网",
    "最新",
    "实时",
    "打开文件",
    "读取文件",
    "文件",
    "目录",
    "发送",
    "保存",
    "下载",
    "上传",
    "创建",
    "删除",
    "调用工具",
    "运行脚本",
    "执行脚本",
    "计算",
)


def contains_robot_task_intent(text: str) -> bool:
    """Return whether an utterance asks the physical robot to act.

    This check intentionally stays independent from :meth:`InteractionGate.evaluate`.
    An explicit wake word is strong evidence that speech is addressed to the robot,
    but it must not turn a command such as ``小算，暂停一下`` into ordinary chat.
    Realtime S2S admission therefore reuses this vocabulary as a separate safety
    dimension after the normal interaction gate has admitted the speaker.
    """

    clean = " ".join(str(text or "").strip().lower().split())
    return bool(clean and InteractionGate._contains_any(clean, _ROBOT_TASK_TERMS))


def contains_emergency_intent(text: str) -> bool:
    """Return whether an utterance contains a local safety-stop instruction."""

    clean = " ".join(str(text or "").strip().lower().split())
    return bool(clean and InteractionGate._contains_any(clean, _SAFETY_TERMS))


def contains_tool_route_intent(text: str) -> bool:
    """Fail closed when an utterance explicitly asks for external/tool data."""

    clean = " ".join(str(text or "").strip().lower().split())
    return bool(clean and InteractionGate._contains_any(clean, _TOOL_ROUTE_TERMS))

_CASUAL_TERMS = (
    "\u6211\u4eec",
    "\u54c8\u54c8",
    "\u597d\u53ef\u7231",
    "\u597d\u73a9",
    "\u53bb\u90a3\u8fb9\u770b\u770b",
    "\u5403\u996d",
    "\u4e0b\u73ed",
    "\u56de\u5bb6",
    "\u62cd\u4e2a\u89c6\u9891",
)

_PRIVACY_OR_UNSAFE_TERMS = (
    "\u62cd\u4e00\u4e0b\u90a3\u4e2a\u4eba",
    "\u8ddf\u8e2a\u90a3\u4e2a\u4eba",
    "\u8ba4\u51fa\u90a3\u4e2a\u4eba",
    "\u4ed6\u662f\u8c01",
    "\u5979\u662f\u8c01",
    "\u76f4\u63a5\u63a7\u5236\u7535\u673a",
    "\u7ed5\u8fc7\u5b89\u5168",
)

_SHORT_GREETING = ("\u4f60\u597d", "hi", "hello", "\u55e8")

_ATTENTION_GESTURES = {
    "wave",
    "waving",
    "raise_hand",
    "raised_hand",
    "pointing",
    "\u6325\u624b",
    "\u4e3e\u624b",
    "\u6307\u5411",
}

_STOP_GESTURES = {"stop", "halt", "\u505c", "\u505c\u4e0b"}

_ENGAGED_POSTURES = {
    "standing",
    "leaning_forward",
    "approaching",
    "squatting_near",
    "\u7ad9\u7acb",
    "\u9760\u8fd1",
    "\u524d\u503e",
}

_DISENGAGED_POSTURES = {
    "walking_away",
    "back_turned",
    "sitting_away",
    "lying_down",
    "\u8d70\u5f00",
    "\u80cc\u5bf9",
    "\u8eba\u4e0b",
}


class InteractionGate:
    """Decide if a voice transcript should trigger a robot response.

    Config keys under ``voice.interaction_gate``::

        enabled: bool
        min_asr_confidence: float
        max_perception_age_s: float
        max_interaction_distance_m: float
        clarify_reply: str
        refresh_perception_reply: str
        defer_reply: str
        refuse_reply: str
        default_mission_mode: str
        default_actor_role: str
        wake_terms: list[str]
        allow_unaddressed_public_help: bool
        allow_unaddressed_robot_tasks: bool
        silent_on_ambiguous: bool
        mission_modes: dict
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_asr_confidence = float(cfg.get("min_asr_confidence", 0.45))
        self.max_perception_age_s = float(cfg.get("max_perception_age_s", 2.0))
        self.max_interaction_distance_m = float(
            cfg.get("max_interaction_distance_m", 4.0)
        )
        self.sound_angle_tolerance_deg = float(
            cfg.get("sound_angle_tolerance_deg", 35.0)
        )
        self.allow_unaddressed_public_help = bool(
            cfg.get("allow_unaddressed_public_help", True)
        )
        self.allow_unaddressed_robot_tasks = bool(
            cfg.get("allow_unaddressed_robot_tasks", True)
        )
        self.silent_on_ambiguous = bool(cfg.get("silent_on_ambiguous", False))
        configured_wake_terms = cfg.get("wake_terms")
        if configured_wake_terms is None:
            self.wake_terms: tuple[str, ...] = _DEFAULT_WAKE_TERMS
        else:
            if isinstance(configured_wake_terms, str):
                configured_wake_terms = [configured_wake_terms]
            self.wake_terms = tuple(
                str(term).strip().lower()
                for term in configured_wake_terms
                if str(term).strip()
            )
        self.clarify_reply = str(
            cfg.get("clarify_reply")
            or "\u9700\u8981\u6211\u5e2e\u4f60\u95ee\u8def\u6216\u5904\u7406\u4efb\u52a1\u5417\uff1f"
        )
        self.refresh_perception_reply = str(
            cfg.get("refresh_perception_reply")
            or "\u6211\u9700\u8981\u91cd\u65b0\u786e\u8ba4\u4f60\u7684\u4f4d\u7f6e\uff0c\u8bf7\u9760\u8fd1\u6211\u6216\u770b\u5411\u6211\u518d\u8bf4\u4e00\u904d\u3002"
        )
        self.defer_reply = str(
            cfg.get("defer_reply")
            or "\u6211\u6b63\u5728\u6267\u884c\u4efb\u52a1\uff0c\u8bf7\u7a0d\u7b49\u3002"
        )
        self.refuse_reply = str(
            cfg.get("refuse_reply")
            or "\u8fd9\u4e2a\u64cd\u4f5c\u6d89\u53ca\u5b89\u5168\u6216\u9690\u79c1\uff0c\u6211\u4e0d\u80fd\u6267\u884c\u3002"
        )
        self.default_mission_mode = MissionMode.coerce(
            cfg.get("default_mission_mode", cfg.get("mission_mode", MissionMode.IDLE.value))
        ).value
        self.default_actor_role = MissionActorRole.coerce(
            cfg.get("default_actor_role", cfg.get("actor_role", MissionActorRole.OPERATOR.value))
        ).value
        self.mission_mode_replies = self._mission_mode_replies(cfg.get("mission_modes"))

    def set_mission_context(
        self,
        *,
        mission_mode: str | MissionMode | None = None,
        actor_role: str | MissionActorRole | None = None,
    ) -> None:
        """Update the default mission context used by future voice turns."""

        if mission_mode is not None:
            self.default_mission_mode = MissionMode.coerce(mission_mode).value
        if actor_role is not None:
            self.default_actor_role = MissionActorRole.coerce(actor_role).value

    def evaluate(
        self,
        text: str,
        *,
        addressed: bool = True,
        asr_confidence: float | None = None,
        visual_attention: bool | None = None,
        sound_source_matches_person: bool | None = None,
        perception: InteractionPerceptionSnapshot | dict[str, Any] | None = None,
        task_interruptible: bool = True,
        mission_mode: str | MissionMode | None = None,
        actor_role: str | MissionActorRole | None = None,
        wake_authorized: bool = False,
        wake_source: str = "none",
        followup_active: bool = False,
        awaiting_confirmation: bool = False,
    ) -> InteractionDecision:
        """Classify a recognized utterance before it reaches the brain."""

        clean = " ".join(str(text or "").strip().lower().split())
        if not clean:
            return InteractionDecision(
                InteractionAction.IGNORE,
                "empty_transcript",
                1.0,
            )
        if not self.enabled:
            return InteractionDecision(InteractionAction.RESPOND, "gate_disabled", 1.0)

        normalized_wake_source = str(wake_source or "none").strip().lower()
        explicit_wake = bool(wake_authorized) or normalized_wake_source == "keyword"
        strong_address = explicit_wake or self._contains_any(
            clean,
            self.wake_terms,
        )
        public_help = self._contains_any(clean, _HELP_TERMS)
        robot_task = self._contains_any(clean, _ROBOT_TASK_TERMS)
        safety_command = self._contains_any(clean, _SAFETY_TERMS)
        casual = self._contains_any(clean, _CASUAL_TERMS)
        unsafe = self._contains_any(clean, _PRIVACY_OR_UNSAFE_TERMS)
        short_greeting = clean in _SHORT_GREETING

        perception_snapshot = (
            InteractionPerceptionSnapshot.from_payload(
                perception,
                max_age_s=self.max_perception_age_s,
                max_interaction_distance_m=self.max_interaction_distance_m,
                sound_angle_tolerance_deg=self.sound_angle_tolerance_deg,
            )
            if perception is not None
            else None
        )
        if perception_snapshot is not None:
            if visual_attention is None:
                visual_attention = perception_snapshot.visual_attention
            if sound_source_matches_person is None:
                sound_source_matches_person = perception_snapshot.sound_source_matches_person

        if asr_confidence is not None and asr_confidence < self.min_asr_confidence:
            if strong_address or public_help or robot_task:
                return InteractionDecision(
                    InteractionAction.CLARIFY,
                    "low_asr_confidence_but_potentially_addressed",
                    0.55,
                    reply=self.clarify_reply,
                    should_record_environment=True,
                )
            return InteractionDecision(
                InteractionAction.RECORD_ONLY,
                "low_asr_confidence_background",
                0.8,
                should_record_environment=True,
            )

        if unsafe:
            return InteractionDecision(
                InteractionAction.REFUSE,
                "unsafe_or_privacy_intent",
                0.9,
                reply=self.refuse_reply,
                should_record_environment=True,
            )

        mission_decision = evaluate_mission_mode(
            clean,
            mission_mode=mission_mode or self.default_mission_mode,
            actor_role=actor_role or self.default_actor_role,
            addressed=addressed,
            replies=self.mission_mode_replies,
        )
        if mission_decision is not None:
            return InteractionDecision(
                InteractionAction(mission_decision.action),
                mission_decision.reason,
                mission_decision.confidence,
                reply=mission_decision.reply,
                should_record_environment=mission_decision.should_record_environment,
            )

        if followup_active and awaiting_confirmation:
            return InteractionDecision(
                InteractionAction.RESPOND,
                "expected_followup_answer",
                0.9,
            )

        if casual and not addressed and not public_help and not robot_task:
            return InteractionDecision(
                InteractionAction.RECORD_ONLY,
                "not_addressed_casual_robot_mention" if strong_address else "not_addressed_casual",
                0.82,
                should_record_environment=True,
            )

        if perception_snapshot is not None:
            public_help_lock = public_help and (
                addressed or self.allow_unaddressed_public_help
            )
            robot_task_lock = robot_task and (
                addressed or self.allow_unaddressed_robot_tasks or safety_command
            )
            sensory_lock = strong_address or public_help_lock or robot_task_lock
            gesture = perception_snapshot.gesture
            posture = perception_snapshot.posture
            visually_engaged = perception_snapshot.visual_attention is True
            attention_gesture = gesture in _ATTENTION_GESTURES
            stop_gesture = gesture in _STOP_GESTURES
            engaged_posture = posture in _ENGAGED_POSTURES
            disengaged_posture = posture in _DISENGAGED_POSTURES
            facing_robot = perception_snapshot.person_facing_robot is True
            not_facing_robot = perception_snapshot.person_facing_robot is False
            ambiguous_crowd = perception_snapshot.person_count > 1 and (
                perception_snapshot.sound_source_matches_person is not True
            )
            if not perception_snapshot.fresh and not sensory_lock:
                if (addressed or short_greeting) and not self.silent_on_ambiguous:
                    return InteractionDecision(
                        InteractionAction.CLARIFY,
                        "stale_perception_needs_refresh",
                        0.62,
                        reply=self.refresh_perception_reply,
                        should_record_environment=True,
                    )
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "stale_perception_background",
                    0.82,
                    should_record_environment=True,
                )
            if perception_snapshot.person_detected is False and not sensory_lock:
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "no_person_lock",
                    0.82,
                    should_record_environment=True,
                )
            if (
                perception_snapshot.nearest_person_distance_m is not None
                and perception_snapshot.nearest_person_distance_m
                > self.max_interaction_distance_m
                and not sensory_lock
            ):
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "person_too_far",
                    0.8,
                    should_record_environment=True,
                )
            if (
                ambiguous_crowd
                and not strong_address
                and not robot_task
            ):
                if self.silent_on_ambiguous:
                    return InteractionDecision(
                        InteractionAction.RECORD_ONLY,
                        "multi_person_ambiguous_speaker",
                        0.82,
                        should_record_environment=True,
                    )
                return InteractionDecision(
                    InteractionAction.CLARIFY,
                    "multi_person_ambiguous_speaker",
                    0.62,
                    reply=self.clarify_reply,
                    should_record_environment=True,
                )
            if (
                perception_snapshot.sound_source_matches_person is False
                and not strong_address
                and not robot_task
            ):
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "audio_visual_mismatch",
                    0.85,
                    should_record_environment=True,
                )
            if stop_gesture:
                return InteractionDecision(
                    InteractionAction.RESPOND,
                    "safety_stop_gesture",
                    0.9,
                )
            if attention_gesture:
                if short_greeting:
                    return InteractionDecision(
                        InteractionAction.CLARIFY,
                        "weak_greeting_with_attention_gesture",
                        0.72,
                        reply=self.clarify_reply,
                        should_record_environment=True,
                    )
                if addressed:
                    return InteractionDecision(
                        InteractionAction.RESPOND,
                        "attention_gesture",
                        0.78,
                    )
            if not_facing_robot and not attention_gesture and not sensory_lock:
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "person_not_facing_robot",
                    0.74,
                    should_record_environment=True,
                )
            if disengaged_posture and not attention_gesture and not sensory_lock:
                return InteractionDecision(
                    InteractionAction.RECORD_ONLY,
                    "disengaged_posture",
                    0.74,
                    should_record_environment=True,
                )
            if short_greeting and (visually_engaged or facing_robot or engaged_posture):
                return InteractionDecision(
                    InteractionAction.CLARIFY,
                    "weak_greeting_with_visual_attention",
                    0.7,
                    reply=self.clarify_reply,
                    should_record_environment=True,
                )
            if public_help and (visually_engaged or facing_robot or attention_gesture):
                return InteractionDecision(
                    InteractionAction.RESPOND,
                    "public_help_with_attention",
                    0.82,
                )
            if robot_task and (visually_engaged or facing_robot or attention_gesture):
                return InteractionDecision(
                    InteractionAction.RESPOND,
                    "robot_task_with_attention",
                    0.84,
                )
            if addressed and engaged_posture and not public_help:
                return InteractionDecision(
                    InteractionAction.RESPOND,
                    "engaged_posture",
                    0.66,
                )

        if not task_interruptible and not robot_task:
            return InteractionDecision(
                InteractionAction.DEFER,
                "active_task_not_interruptible",
                0.75,
                reply=self.defer_reply,
                should_record_environment=True,
            )

        if visual_attention is False and sound_source_matches_person is False:
            if strong_address or robot_task:
                return InteractionDecision(InteractionAction.RESPOND, "explicit_address_without_visual_lock", 0.7)
            return InteractionDecision(
                InteractionAction.RECORD_ONLY,
                "audio_visual_mismatch",
                0.85,
                should_record_environment=True,
            )

        if explicit_wake:
            return InteractionDecision(
                InteractionAction.RESPOND,
                "wake_word_authorized",
                0.98,
            )

        if strong_address:
            return InteractionDecision(InteractionAction.RESPOND, "explicit_robot_address", 0.95)

        if robot_task and (
            addressed or self.allow_unaddressed_robot_tasks or safety_command
        ):
            return InteractionDecision(InteractionAction.RESPOND, "robot_task_intent", 0.88)

        if robot_task:
            return InteractionDecision(
                InteractionAction.IGNORE,
                "unaddressed_robot_task",
                0.8,
            )

        if public_help and (addressed or self.allow_unaddressed_public_help):
            return InteractionDecision(InteractionAction.RESPOND, "public_help_or_wayfinding", 0.82)

        if public_help:
            return InteractionDecision(
                InteractionAction.IGNORE,
                "unaddressed_public_help",
                0.8,
            )

        if short_greeting:
            if not addressed and self.silent_on_ambiguous:
                return InteractionDecision(
                    InteractionAction.IGNORE,
                    "unaddressed_greeting",
                    0.8,
                )
            return InteractionDecision(
                InteractionAction.CLARIFY,
                "weak_greeting",
                0.6,
                reply=self.clarify_reply,
                should_record_environment=True,
            )

        if followup_active and not addressed:
            return InteractionDecision(
                InteractionAction.IGNORE,
                "followup_not_addressed",
                0.82,
            )

        if not addressed:
            return InteractionDecision(
                InteractionAction.RECORD_ONLY if casual else InteractionAction.IGNORE,
                "not_addressed_casual" if casual else "not_addressed",
                0.78,
                should_record_environment=casual,
            )

        if casual and not visual_attention:
            return InteractionDecision(
                InteractionAction.RECORD_ONLY,
                "casual_bystander_speech",
                0.7,
                should_record_environment=True,
            )

        return InteractionDecision(InteractionAction.RESPOND, "addressed_or_uncertain", 0.55)

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term and term in text for term in terms)

    @staticmethod
    def _mission_mode_replies(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        replies: dict[str, str] = {}
        for mode, value in raw.items():
            normalized = MissionMode.coerce(mode).value
            if isinstance(value, dict):
                reply = value.get("reply")
            else:
                reply = value
            if reply:
                replies[normalized] = str(reply)
        return replies


__all__ = [
    "InteractionAction",
    "InteractionDecision",
    "InteractionGate",
    "MissionActorRole",
    "MissionCommandCategory",
    "MissionMode",
]
