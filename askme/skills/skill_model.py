"""
Skill definition data model for askme.

A SkillDefinition captures all metadata parsed from a SKILL.md frontmatter,
plus the prompt template body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import SkillContract, SkillParameter


@dataclass
class SlotSpec:
    """A single typed slot required by a skill."""

    name: str
    type: str = "text"       # text | location | referent | datetime | enum
    prompt: str = ""         # question to ask when this slot is missing
    optional: bool = False   # if True, missing value does not block execution
    default: str = ""        # default value to use when slot is optional + absent


@dataclass
class SkillDefinition:
    """Represents a single skill parsed from a SKILL.md file."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    trigger: str = "manual"  # manual | auto | voice | schedule
    model: str = ""
    timeout: int = 30
    tags: list[str] = field(default_factory=list)
    depends: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    safety_level: str = "normal"  # normal | dangerous | critical
    voice_trigger: str | None = None
    required_prompt: str = ""   # Question to ask if required slot is missing, e.g. "导航去哪里？"
    confirm_before_execute: bool = False  # True → ConfirmationAgent runs before dispatch
    required_slots: list["SlotSpec"] = field(default_factory=list)  # typed slot schema
    schedule: str | None = None  # cron expression
    prompt_template: str = ""
    tools_section: str = ""
    execution: str = "skill_executor"  # skill_executor | agent_shell
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
        # Replace any remaining unresolved {{placeholders}} with empty string
        # so the LLM doesn't see confusing template syntax
        prompt = re.sub(r"\{\{[^}]+\}\}", "", prompt)
        return prompt

    def to_contract(self) -> SkillContract:
        """Build a contract view for legacy markdown-defined skills."""
        return SkillContract(
            name=self.name,
            description=self.description,
            version=self.version,
            safety_level=self.safety_level,
            execution=self.execution,
            tags=tuple(self.tags),
            parameters=tuple(
                SkillParameter(
                    name=slot.name,
                    type=_slot_type_to_json_type(slot.type),
                    description=slot.prompt,
                    required=not slot.optional,
                    default=slot.default or None,
                )
                for slot in self.required_slots
            ),
            confirm_before_execute=self.confirm_before_execute,
            source="legacy_markdown",
        )


def _slot_type_to_json_type(slot_type: str) -> str:
    return {
        "text": "string",
        "location": "string",
        "referent": "string",
        "datetime": "string",
        "enum": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }.get(slot_type, "string")
