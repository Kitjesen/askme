"""Compatibility facade for :mod:`askme.robot_interaction.scenario_intents`."""

from __future__ import annotations

from askme.robot_interaction.scenario_intents import (
    SCENARIO_INTENT_RULES,
    ScenarioIntentDecision,
    ScenarioIntentRule,
    classify_scenario_intent,
    normalize_intent_text,
)

__all__ = [
    "SCENARIO_INTENT_RULES",
    "ScenarioIntentDecision",
    "ScenarioIntentRule",
    "classify_scenario_intent",
    "normalize_intent_text",
]
