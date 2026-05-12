"""In-memory planning sessions for multi-turn cognition loops."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class PlanningSession:
    """One short-lived planning loop owned by the cognition adapter."""

    session_id: str
    goal: str = ""
    intent: str = "operator_assist"
    stage: str = "observing"
    last_plan_id: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)
    mission: dict[str, Any] | None = None
    confirmation_status: str = "unconfirmed"
    operator_id: str | None = None
    robot_id: str | None = None
    site_id: str | None = None
    channel: str = "cognition"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, session_id: str | None = None) -> PlanningSession:
        return cls(session_id=session_id or f"cog-session-{uuid4().hex[:12]}")

    def update(
        self,
        *,
        plan_id: str,
        goal: str,
        intent: str,
        stage: str,
        missing_inputs: list[str],
        reference: dict[str, Any],
        mission: dict[str, Any] | None,
        confirmation_status: str,
        operator_id: str | None,
        robot_id: str | None,
        site_id: str | None,
        channel: str,
    ) -> None:
        self.last_plan_id = plan_id
        self.goal = goal
        self.intent = intent
        self.stage = stage
        self.missing_inputs = list(missing_inputs)
        self.reference = dict(reference)
        self.mission = mission
        self.confirmation_status = confirmation_status
        self.operator_id = operator_id
        self.robot_id = robot_id
        self.site_id = site_id
        self.channel = channel
        self.updated_at = time.time()
        self.history.append({
            "plan_id": plan_id,
            "goal": goal,
            "intent": intent,
            "stage": stage,
            "missing_inputs": list(missing_inputs),
            "confirmation_status": confirmation_status,
            "updated_at": self.updated_at,
        })
        if len(self.history) > 20:
            del self.history[: len(self.history) - 20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "intent": self.intent,
            "stage": self.stage,
            "last_plan_id": self.last_plan_id,
            "missing_inputs": list(self.missing_inputs),
            "reference": dict(self.reference),
            "mission": self.mission,
            "confirmation_status": self.confirmation_status,
            "operator_id": self.operator_id,
            "robot_id": self.robot_id,
            "site_id": self.site_id,
            "channel": self.channel,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [dict(item) for item in self.history],
        }
