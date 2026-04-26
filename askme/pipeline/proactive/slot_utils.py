"""Utility: detect whether a required slot is already present in user_text.

Extracted from VoiceLoop._slot_present so it can be shared by SlotCollectorAgent
and tested independently without constructing a full VoiceLoop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition


def slot_present(
    skill: SkillDefinition, user_text: str, pipeline: Any = None
) -> bool:
    """Return True if *user_text* already contains the required slot content.

    For ``navigate``: uses pipeline.extract_semantic_target() to check whether
    a concrete destination is embedded (e.g. "去厨房" has "厨房").

    For other skills: finds the longest matching voice trigger inside user_text,
    strips it out, and checks whether the remainder has ≥ 2 meaningful characters.
    """
    if skill.name == "navigate":
        if pipeline is None:
            # Without a pipeline we can't extract — assume slot is missing.
            return False
        target = pipeline.extract_semantic_target(user_text)
        return bool(target) and target.strip() != user_text.strip()

    # Generic trigger-based detection.
    if skill.voice_trigger:
        triggers = sorted(
            [t.strip() for t in skill.voice_trigger.split(",") if t.strip()],
            key=len,
            reverse=True,
        )
        for trigger in triggers:
            pos = user_text.find(trigger)
            if pos >= 0:
                remainder = (user_text[:pos] + user_text[pos + len(trigger) :]).strip()
                if len(remainder) >= 2:
                    # Also reject if remainder is a vague placeholder word
                    from askme.pipeline.proactive.slot_analyst import is_vague
                    return not is_vague(remainder)
                return False

    # No trigger matched or no triggers defined — fall back to length check.
    return len(user_text.strip()) > len(skill.voice_trigger or "")
