"""Reaction data types -- scene context and decision structures.

SceneContext captures all signals available for reaction decisions.
ReactionDecision records what rule fired and what reaction was chosen.
Both are frozen dataclasses for immutability and testability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from askme.schemas.events import ChangeEvent


class ReactionType(str, Enum):
    """How the robot should react to a scene event."""

    IGNORE = "ignore"
    OBSERVE = "observe"
    GREET = "greet"
    INFORM = "inform"
    WARN = "warn"
    ASSIST = "assist"
    ALERT = "alert"
    ACT = "act"


@dataclass(frozen=True)
class SceneContext:
    """All signals available for reaction decisions.

    Every field is derived from existing components (WorldState, SiteKnowledge,
    EpisodicMemory, time, robot state). No new sensors needed.
    """

    event: ChangeEvent
    person_count: int = 0
    person_distance_m: float | None = None
    person_duration_s: float = 0.0
    person_approaching: bool = False
    person_stationary: bool = False
    hour: int = 0
    is_business_hours: bool = True
    zone_name: str = ""
    zone_tags: list[str] = field(default_factory=list)
    seen_person_recently: bool = False
    minutes_since_last_person: float = 999.0
    robot_busy: bool = False
    wake_word_heard: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "event_type": self.event.event_type.value,
            "person_count": self.person_count,
            "person_distance_m": self.person_distance_m,
            "person_duration_s": round(self.person_duration_s, 1),
            "person_approaching": self.person_approaching,
            "person_stationary": self.person_stationary,
            "hour": self.hour,
            "is_business_hours": self.is_business_hours,
            "zone_name": self.zone_name,
            "zone_tags": self.zone_tags,
            "seen_person_recently": self.seen_person_recently,
            "minutes_since_last_person": round(self.minutes_since_last_person, 1),
            "robot_busy": self.robot_busy,
            "wake_word_heard": self.wake_word_heard,
        }


@dataclass(frozen=True)
class ReactionDecision:
    """The outcome of a reaction rule evaluation."""

    rule_name: str
    reaction_type: ReactionType
    metadata: dict[str, Any] = field(default_factory=dict)
    context: SceneContext | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "rule_name": self.rule_name,
            "reaction_type": self.reaction_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
