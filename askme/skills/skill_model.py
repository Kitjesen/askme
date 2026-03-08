"""
Skill definition data model for askme.

A SkillDefinition captures all metadata parsed from a SKILL.md frontmatter,
plus the prompt template body.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillDefinition:
    """Represents a single skill parsed from a SKILL.md file."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    trigger: str = "manual"  # manual | auto | voice | schedule
    model: str = "deepseek-chat"
    timeout: int = 30
    tags: list[str] = field(default_factory=list)
    depends: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    safety_level: str = "normal"  # normal | dangerous | critical
    voice_trigger: str | None = None
    schedule: str | None = None  # cron expression
    prompt_template: str = ""
    tools_section: str = ""
    source: str = "builtin"  # builtin | user | project
    path: str = ""
    enabled: bool = True

    def build_prompt(self, context: dict[str, str] | None = None) -> str:
        """Substitute {{key}} placeholders in the prompt template.

        Args:
            context: A dict of variable names to values.

        Returns:
            The prompt with all known placeholders replaced.
        """
        prompt = self.prompt_template
        if context:
            for key, value in context.items():
                prompt = prompt.replace("{{" + key + "}}", str(value))
        return prompt
