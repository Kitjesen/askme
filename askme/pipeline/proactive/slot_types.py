"""Analysis result types for the typed slot system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.skills.skill_model import SlotSpec


@dataclass
class SlotFill:
    """Analysis result for a single slot of a skill."""

    spec: SlotSpec
    value: str | None = None    # extracted value (may be vague)
    status: str = "missing"     # "filled" | "missing" | "vague"

    @property
    def is_ok(self) -> bool:
        return self.status == "filled"


@dataclass
class SlotAnalysis:
    """Full slot analysis result for a skill + user_text pair."""

    skill_name: str
    slots: list[SlotFill] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when all required slots are filled with non-vague values."""
        return all(f.is_ok for f in self.slots)

    @property
    def missing_required(self) -> list[SlotFill]:
        """Slots that are missing or vague (and need to be collected)."""
        return [f for f in self.slots if not f.is_ok]
