"""Stable robot interaction service facade."""

from __future__ import annotations

from typing import Any

from askme.robot_interaction.intent_router import Intent, IntentRouter
from askme.robot_interaction.observability import (
    attach_intent_route_trace,
    intent_route_payload,
)
from askme.robot_interaction.routing_policy import RoutingPolicy


class RobotInteractionService:
    """Intent routing boundary used by voice and text channels."""

    def __init__(self, router: IntentRouter) -> None:
        self._router = router

    @classmethod
    def from_policy(
        cls,
        *,
        voice_triggers: dict[str, str] | None = None,
        safety_checker: Any | None = None,
        policy: RoutingPolicy | None = None,
    ) -> "RobotInteractionService":
        router = IntentRouter(
            voice_triggers=voice_triggers,
            safety_checker=safety_checker,
            policy=policy,
        )
        return cls(router)

    @property
    def router(self) -> IntentRouter:
        """Underlying router, exposed for incremental migration."""
        return self._router

    def route(self, text: str) -> Intent:
        return self._router.route(text)

    def route_payload(self, text: str, *, source: str = "") -> dict[str, Any]:
        return intent_route_payload(self.route(text), source=source)

    def attach_trace(
        self,
        trace: dict[str, Any] | None,
        text: str,
        *,
        source: str = "",
        stage: str = "intent_route",
    ) -> tuple[Intent, dict[str, Any]]:
        intent = self.route(text)
        payload = attach_intent_route_trace(
            trace,
            intent,
            source=source,
            stage=stage,
        )
        return intent, payload
