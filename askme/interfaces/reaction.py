"""Reaction backend interface -- pluggable scene-aware decision engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from askme.runtime.registry import BackendRegistry
from askme.schemas.reaction import ReactionDecision, SceneContext


class ReactionBackend(ABC):
    """Abstract reaction decision engine.

    Implementations decide how the robot reacts to scene events.
    Three phases: decide (fast) -> generate content (optional LLM) -> execute.
    """

    @abstractmethod
    async def decide(self, context: SceneContext) -> ReactionDecision:
        """Decide how to react to a scene context. Must be fast (<100ms)."""

    @abstractmethod
    async def generate_content(
        self, decision: ReactionDecision, context: SceneContext
    ) -> str:
        """Generate spoken content for decisions that need speech.

        May call LLM (~2s) for GREET/ASSIST, or return a template (<1ms).
        Returns empty string for IGNORE/OBSERVE.
        """

    @abstractmethod
    async def execute(self, decision: ReactionDecision, content: str) -> None:
        """Execute the reaction (speak, alert, dispatch skill, etc.)."""


reaction_registry = BackendRegistry("reaction", ReactionBackend, default="hybrid")
