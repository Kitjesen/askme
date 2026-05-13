"""Validation for generated skills before they can be approved."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from askme.skills.skill_model import SkillDefinition

_LOW_RISK_GENERATED_TOOLS = {
    "get_current_time",
    "read_file",
    "list_directory",
    "http_request",
    "robot_api",
    "nav_status",
    "temporal_query",
    "look_around",
    "find_target",
    "scan_around",
    "web_search",
    "web_fetch",
    "speak_progress",
}
_HIGH_RISK_GENERATED_TOOLS = {
    "bash",
    "run_command",
    "write_file",
    "edit_file",
    "move_robot",
    "nav_dispatch",
    "dog_control_dispatch",
    "robot_move",
    "robot_grab",
    "robot_release",
    "robot_emergency_stop",
    "dispatch_skill",
}
_BLOCKED_PROMPT_PATTERNS = (
    "ignore safety",
    "ignore previous instructions",
    "bypass safety",
    "disable safety",
    "绕过安全",
    "忽略安全",
    "无需确认",
)


@dataclass(frozen=True)
class SkillValidationIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


def validate_generated_skill(
    skill: SkillDefinition,
    *,
    all_skills: Iterable[SkillDefinition] = (),
) -> dict[str, object]:
    """Return a deterministic preflight result for generated skill approval."""
    issues: list[SkillValidationIssue] = []

    if skill.source != "generated":
        issues.append(
            SkillValidationIssue("error", "not_generated", "Only generated skills use this approval preflight.")
        )

    if not skill.description.strip():
        issues.append(SkillValidationIssue("error", "missing_description", "Skill description is required."))

    prompt = skill.prompt_template.strip()
    if len(prompt) < 10:
        issues.append(SkillValidationIssue("error", "prompt_too_short", "Skill prompt is too short to review."))

    lowered_prompt = prompt.lower()
    for pattern in _BLOCKED_PROMPT_PATTERNS:
        if pattern in lowered_prompt or pattern in prompt:
            issues.append(
                SkillValidationIssue(
                    "error",
                    "unsafe_prompt_instruction",
                    f"Prompt contains a blocked safety-bypass phrase: {pattern}",
                )
            )

    trigger_phrases = _trigger_phrases(skill.voice_trigger)
    if not trigger_phrases:
        issues.append(SkillValidationIssue("error", "missing_voice_trigger", "Generated voice skills need a trigger phrase."))

    for phrase in trigger_phrases:
        if len(phrase) < 2:
            issues.append(
                SkillValidationIssue("error", "trigger_too_short", f"Trigger phrase is too short: {phrase!r}")
            )

    collisions = _trigger_collisions(skill, trigger_phrases, all_skills)
    for phrase, other_name in collisions:
        issues.append(
            SkillValidationIssue(
                "error",
                "trigger_collision",
                f"Trigger phrase {phrase!r} already belongs to skill {other_name}.",
            )
        )

    requested_tools = _tools_from_section(skill.tools_section)
    unknown_tools = requested_tools - _LOW_RISK_GENERATED_TOOLS - _HIGH_RISK_GENERATED_TOOLS
    for tool in sorted(unknown_tools):
        issues.append(
            SkillValidationIssue("error", "unknown_tool", f"Unknown or unsupported generated-skill tool: {tool}.")
        )
    high_risk_tools = requested_tools & _HIGH_RISK_GENERATED_TOOLS
    for tool in sorted(high_risk_tools):
        issues.append(
            SkillValidationIssue(
                "error",
                "high_risk_tool",
                f"Generated skill requests high-risk tool {tool}; use a code-reviewed built-in skill instead.",
            )
        )
    if not requested_tools:
        issues.append(SkillValidationIssue("warning", "prompt_only", "Skill has no tools and will be prompt-only."))

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return {
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": [issue.to_dict() for issue in issues],
        "allowed_tools": sorted(requested_tools - high_risk_tools - unknown_tools),
        "blocked_tools": sorted(high_risk_tools | unknown_tools),
        "voice_triggers": trigger_phrases,
    }


def _trigger_phrases(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [phrase.strip() for phrase in raw.split(",") if phrase.strip()]


def _tools_from_section(raw: str) -> set[str]:
    tools: set[str] = set()
    for line in raw.splitlines():
        item = line.strip().strip("-*` ")
        if not item or item.startswith("#"):
            continue
        for part in item.split(","):
            tool = part.strip().strip("` ")
            if tool:
                tools.add(tool)
    return tools


def _trigger_collisions(
    skill: SkillDefinition,
    trigger_phrases: list[str],
    all_skills: Iterable[SkillDefinition],
) -> list[tuple[str, str]]:
    phrases = set(trigger_phrases)
    collisions: list[tuple[str, str]] = []
    for other in all_skills:
        if other.name == skill.name:
            continue
        for phrase in _trigger_phrases(other.voice_trigger):
            if phrase in phrases:
                collisions.append((phrase, other.name))
    return collisions
