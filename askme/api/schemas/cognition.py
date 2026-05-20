"""Cognition planning API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CognitionContextResponse(BaseModel):
    """World state and working memory context used before planning."""

    model_config = ConfigDict(extra="allow")

    world_state: dict[str, Any] = Field(default_factory=dict)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    perception: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class CognitionPlanResponse(BaseModel):
    """Cognitive planner result for a natural-language task request."""

    model_config = ConfigDict(extra="allow")

    planned: bool | None = None
    handled: bool | None = None
    status: str = ""
    reason: str = ""
    plan: dict[str, Any] = Field(default_factory=dict)
    planning_session_id: str = ""
    confirmation_status: str = ""
    runtime: dict[str, Any] = Field(default_factory=dict)
