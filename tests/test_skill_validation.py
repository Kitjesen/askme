from __future__ import annotations

from askme.skills.skill_model import SkillDefinition
from askme.skills.validation import validate_generated_skill


def _skill(**overrides) -> SkillDefinition:
    base = {
        "name": "generated_help",
        "description": "Offer help",
        "voice_trigger": "help-trigger",
        "prompt_template": "Answer the visitor with approved information.",
        "tools_section": "web_search",
        "source": "generated",
    }
    base.update(overrides)
    return SkillDefinition(**base)


def test_generated_skill_validation_passes_low_risk_tool() -> None:
    result = validate_generated_skill(_skill())

    assert result["ok"] is True
    assert result["error_count"] == 0
    assert result["allowed_tools"] == ["web_search"]


def test_generated_skill_validation_blocks_high_risk_tool() -> None:
    result = validate_generated_skill(_skill(tools_section="bash\nwrite_file"))

    assert result["ok"] is False
    codes = {issue["code"] for issue in result["issues"]}
    assert "high_risk_tool" in codes
    assert result["blocked_tools"] == ["bash", "write_file"]


def test_generated_skill_validation_blocks_trigger_collision() -> None:
    other = SkillDefinition(
        name="existing",
        voice_trigger="help-trigger",
        prompt_template="Existing prompt.",
        source="builtin",
    )

    result = validate_generated_skill(_skill(), all_skills=[other])

    assert result["ok"] is False
    assert any(issue["code"] == "trigger_collision" for issue in result["issues"])


def test_generated_skill_validation_blocks_missing_prompt_and_trigger() -> None:
    result = validate_generated_skill(_skill(prompt_template="", voice_trigger=""))

    assert result["ok"] is False
    codes = {issue["code"] for issue in result["issues"]}
    assert "prompt_too_short" in codes
    assert "missing_voice_trigger" in codes
