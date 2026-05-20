from __future__ import annotations

from askme.api.services.scenario_intent_payloads import (
    capabilities_payload,
    default_product_skill_names,
    enabled_skill_names,
    requested_or_runtime_skills,
    scenario_intent_decision_payload,
    scenario_intent_rule_payload,
)
from askme.robot_interaction.scenario_intents import (
    SCENARIO_INTENT_RULES,
    classify_scenario_intent,
)


def test_capabilities_payload_is_defensive() -> None:
    assert capabilities_payload(None) == {}
    assert capabilities_payload(lambda: ["bad"]) == {}
    assert capabilities_payload(lambda: {"ok": True}) == {"ok": True}


def test_enabled_skill_names_derives_from_runtime_inventory() -> None:
    payload = {
        "skills": {"catalog": [{"skill_name": "answer_wayfinding", "enabled": True}]},
    }

    assert enabled_skill_names(payload) == {"answer_wayfinding"}


def test_requested_or_runtime_skills_prefers_preview_body() -> None:
    skills = requested_or_runtime_skills(
        {"available_skills": [" answer_wayfinding ", "", "detect_parking_violation"]},
        lambda: {"skills": {"catalog": [{"skill_name": "ignored", "enabled": True}]}},
    )

    assert skills == {"answer_wayfinding", "detect_parking_violation"}


def test_requested_or_runtime_skills_falls_back_to_product_catalog() -> None:
    skills = requested_or_runtime_skills({}, None)

    assert default_product_skill_names()
    assert "lookup_place" in skills
    assert "answer_wayfinding" in skills


def test_scenario_intent_rule_payload_is_auditable() -> None:
    rule = SCENARIO_INTENT_RULES[0]
    payload = scenario_intent_rule_payload(rule, {rule.skill_name})

    assert payload["rule_id"] == rule.rule_id
    assert payload["skill_name"] == rule.skill_name
    assert payload["enabled"] is True
    assert isinstance(payload["match_terms"], list)
    assert payload["evidence"]


def test_scenario_intent_decision_payload_is_safe_for_preview() -> None:
    rule = SCENARIO_INTENT_RULES[0]
    decision = classify_scenario_intent(
        rule.any_terms[0],
        available_skills={rule.skill_name},
    )

    payload = scenario_intent_decision_payload(decision)

    assert payload is not None
    assert payload["skill_name"] == rule.skill_name
    assert payload["scenario_id"] == rule.scenario_id
    assert payload["matched_terms"]
    assert scenario_intent_decision_payload(None) is None
