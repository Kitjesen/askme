"""Robot interaction layer.

This package owns intent routing, interaction observability, and interaction
service APIs. Legacy ``askme.interaction`` imports remain as facades.
"""

from __future__ import annotations

from askme.robot_interaction.address_detector import AddressDetector
from askme.robot_interaction.intent_router import Intent, IntentRouter, IntentType
from askme.robot_interaction.interaction_gate import (
    InteractionAction,
    InteractionDecision,
    InteractionGate,
    MissionActorRole,
    MissionCommandCategory,
    MissionMode,
)
from askme.robot_interaction.observability import (
    attach_intent_route_trace,
    intent_route_payload,
)
from askme.robot_interaction.perception_context import InteractionPerceptionSnapshot
from askme.robot_interaction.routing_policy import RoutingPolicy
from askme.robot_interaction.scenario_intents import (
    ScenarioIntentDecision,
    classify_scenario_intent,
)
from askme.robot_interaction.service import RobotInteractionService

__all__ = [
    "AddressDetector",
    "InteractionAction",
    "InteractionDecision",
    "InteractionGate",
    "InteractionPerceptionSnapshot",
    "Intent",
    "IntentRouter",
    "IntentType",
    "MissionActorRole",
    "MissionCommandCategory",
    "MissionMode",
    "RobotInteractionService",
    "RoutingPolicy",
    "ScenarioIntentDecision",
    "attach_intent_route_trace",
    "classify_scenario_intent",
    "intent_route_payload",
]
