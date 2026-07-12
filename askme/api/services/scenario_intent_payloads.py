"""Payload helpers for scenario-intent routing preview APIs."""

from __future__ import annotations

from typing import Any

from askme.api.services.capability_package_payloads import (
    CapabilitiesProvider,
    default_product_capability_center,
    inventory_from_capabilities_payload,
)


def capabilities_payload(provider: CapabilitiesProvider | None) -> dict[str, Any]:
    """Return a defensive capabilities payload for scenario routing checks."""

    if provider is None:
        return {}
    payload = provider()
    return payload if isinstance(payload, dict) else {}


def enabled_skill_names(payload: dict[str, Any]) -> set[str]:
    """Return enabled skills from the runtime capability inventory."""

    return set(inventory_from_capabilities_payload(payload).skills)


def default_product_skill_names() -> set[str]:
    """Return enabled skills from the local product catalog for dashboard-only mode."""

    center = default_product_capability_center()
    if not center:
        return set()
    return enabled_skill_names({"skills": {"capability_center": center}})


def requested_or_runtime_skills(
    body: dict[str, Any],
    provider: CapabilitiesProvider | None,
) -> set[str]:
    """Prefer caller-provided preview skills; otherwise derive from runtime state."""

    requested = body.get("available_skills")
    if isinstance(requested, list):
        return {str(item).strip() for item in requested if str(item).strip()}
    runtime_skills = enabled_skill_names(capabilities_payload(provider))
    return runtime_skills or default_product_skill_names()


def scenario_intent_rule_payload(rule: Any, available_skills: set[str]) -> dict[str, Any]:
    """Serialize one deterministic scenario-intent rule for audit display."""

    return {
        "rule_id": rule.rule_id,
        "skill_name": rule.skill_name,
        "scenario_id": rule.scenario_id,
        "enabled": rule.skill_name in available_skills,
        "risk_level": rule.risk_level,
        "confidence": rule.confidence,
        "match_terms": list(rule.any_terms),
        "required_terms": list(rule.all_terms),
        "exclude_terms": list(rule.exclude_terms),
        "evidence": rule.evidence,
    }


def scenario_intent_decision_payload(decision: Any | None) -> dict[str, Any] | None:
    """Serialize a preview decision without exposing internal objects."""

    if decision is None:
        return None
    return {
        "skill_name": decision.skill_name,
        "scenario_id": decision.scenario_id,
        "confidence": decision.confidence,
        "matched_terms": list(decision.matched_terms),
        "rule_id": decision.rule_id,
        "risk_level": decision.risk_level,
        "evidence": decision.evidence,
    }


__all__ = [
    "capabilities_payload",
    "default_product_skill_names",
    "enabled_skill_names",
    "requested_or_runtime_skills",
    "scenario_intent_decision_payload",
    "scenario_intent_rule_payload",
]
