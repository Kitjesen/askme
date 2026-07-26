"""Structured skill readiness and execution outcomes.

The legacy skill APIs still return strings.  This contract keeps those strings
available for callers and audit logs while giving voice callers a separate,
customer-safe sentence.  Internal ``[Skill]``/``[Error]`` markers must never be
queued for TTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

NAV_LOCATION_UNAVAILABLE_MESSAGE = "当前定位服务未就绪，请稍后再试。"
GENERIC_SKILL_UNAVAILABLE_MESSAGE = "这个功能暂时不可用，请稍后再试。"
GENERIC_SKILL_FAILURE_MESSAGE = "这个功能暂时执行失败，请稍后再试。"
GENERIC_SKILL_TIMEOUT_MESSAGE = "处理超时了，请稍后再试。"

_INTERNAL_TTS_PREFIXES = (
    "[Skill]",
    "[Skill Error]",
    "[AgentShell Error]",
    "[Error]",
    "[Timeout]",
    "[超时]",
    "[错误]",
)


class SkillOutcomeStatus(StrEnum):
    """Stable states shared by preflight and execution callers."""

    READY = "ready"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SkillOutcome:
    """A machine-facing result plus an optional customer-facing sentence."""

    status: SkillOutcomeStatus
    code: str
    result: str = ""
    user_message: str = ""
    should_speak: bool = False

    @property
    def can_execute(self) -> bool:
        return self.status is SkillOutcomeStatus.READY

    @property
    def legacy_result(self) -> str:
        return self.result

    @classmethod
    def ready(cls) -> SkillOutcome:
        return cls(SkillOutcomeStatus.READY, "ready")

    @classmethod
    def blocked(
        cls,
        *,
        code: str,
        result: str,
        user_message: str,
        should_speak: bool = True,
    ) -> SkillOutcome:
        return cls(
            SkillOutcomeStatus.BLOCKED,
            code,
            result=result,
            user_message=user_message,
            should_speak=bool(should_speak and user_message),
        )

    @classmethod
    def from_legacy_result(
        cls,
        result: str,
        *,
        skill_name: str = "",
    ) -> SkillOutcome:
        """Classify a legacy string without exposing its marker to TTS."""

        text = str(result or "")
        stripped = text.lstrip()
        if stripped.startswith(("[Timeout]", "[超时]")):
            return cls(
                SkillOutcomeStatus.TIMED_OUT,
                "execution_timeout",
                result=text,
                user_message=GENERIC_SKILL_TIMEOUT_MESSAGE,
                should_speak=True,
            )
        if is_internal_skill_text(stripped):
            message = (
                NAV_LOCATION_UNAVAILABLE_MESSAGE
                if skill_name == "nav_query"
                else GENERIC_SKILL_FAILURE_MESSAGE
            )
            return cls(
                SkillOutcomeStatus.FAILED,
                "internal_error_result",
                result=text,
                user_message=message,
                should_speak=True,
            )
        return cls(
            SkillOutcomeStatus.SUCCEEDED,
            "succeeded",
            result=text,
            user_message=text,
            should_speak=bool(text),
        )


def is_internal_skill_text(text: str) -> bool:
    """Return whether *text* is a legacy machine marker unsafe for TTS."""

    return str(text or "").lstrip().startswith(_INTERNAL_TTS_PREFIXES)


__all__ = [
    "GENERIC_SKILL_FAILURE_MESSAGE",
    "GENERIC_SKILL_TIMEOUT_MESSAGE",
    "GENERIC_SKILL_UNAVAILABLE_MESSAGE",
    "NAV_LOCATION_UNAVAILABLE_MESSAGE",
    "SkillOutcome",
    "SkillOutcomeStatus",
    "is_internal_skill_text",
]
