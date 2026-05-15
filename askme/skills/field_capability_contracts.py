"""Field capability execution contracts.

Customer project bindings use product package ids such as
``capability.detect_illegal_parking``.  This module translates those package
ids into executable skill contract metadata that delivery plans, field events,
and dashboards can audit without importing field-operation services.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import contracts_builtin as _contracts_builtin  # noqa: F401 - registers contracts
from .contracts import SkillContract, registered_skill_contracts

_DEFAULT_ROUTE = {
    "route": "manual_review",
    "tool": "not_bound",
    "output_contract": "unbound_skill_contract",
    "hardware_boundary": "This capability package is not executable until a code-defined skill contract is registered.",
}

_FIELD_EVENT_BOUNDARY = (
    "The skill may create or enrich a field event only. Notifications, robot motion, "
    "and hardware actions still pass incident workflow, runtime arbiter, and safety preflight."
)
_VOICE_BOUNDARY = (
    "The skill may produce grounded visitor-facing speech only. It cannot move hardware "
    "or start escort without destination confirmation and a runtime handoff."
)
_RUNTIME_BOUNDARY = (
    "The skill may request a TaskHandoff only. Robot movement requires destination confirmation, "
    "runtime arbiter acceptance, safety preflight, and trusted runtime callbacks."
)

_FIELD_CAPABILITY_ROUTES: dict[str, dict[str, str]] = {
    "detect_night_intruder": {
        "route": "field_event.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "field_event",
        "hardware_boundary": _FIELD_EVENT_BOUNDARY,
    },
    "detect_illegal_parking": {
        "route": "field_event.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "field_event",
        "hardware_boundary": _FIELD_EVENT_BOUNDARY,
    },
    "detect_fire_smoke": {
        "route": "field_event.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "field_event",
        "hardware_boundary": _FIELD_EVENT_BOUNDARY,
    },
    "inspect_trash_bin": {
        "route": "field_event.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "field_event",
        "hardware_boundary": _FIELD_EVENT_BOUNDARY,
    },
    "detect_crowd_gathering": {
        "route": "field_event.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "field_event",
        "hardware_boundary": _FIELD_EVENT_BOUNDARY,
    },
    "offer_wayfinding_help": {
        "route": "visitor_interaction.skill_package",
        "tool": "field_event_trigger",
        "output_contract": "visitor_interaction_event",
        "hardware_boundary": _VOICE_BOUNDARY,
    },
    "answer_wayfinding": {
        "route": "visitor_interaction.skill_package",
        "tool": "answer_wayfinding",
        "output_contract": "grounded_voice_answer",
        "hardware_boundary": _VOICE_BOUNDARY,
    },
    "lookup_place": {
        "route": "space_semantic_map.skill_package",
        "tool": "lookup_place",
        "output_contract": "destination_resolution",
        "hardware_boundary": _VOICE_BOUNDARY,
    },
    "recommend_route": {
        "route": "space_semantic_map.skill_package",
        "tool": "recommend_route",
        "output_contract": "route_recommendation",
        "hardware_boundary": _VOICE_BOUNDARY,
    },
    "escort_visitor": {
        "route": "runtime_handoff.skill_package",
        "tool": "task_handoff",
        "output_contract": "task_handoff",
        "hardware_boundary": _RUNTIME_BOUNDARY,
    },
    "navigate": {
        "route": "runtime_handoff.skill_package",
        "tool": "task_handoff",
        "output_contract": "task_handoff",
        "hardware_boundary": _RUNTIME_BOUNDARY,
    },
}


def normalize_capability_package(package_id: str) -> str:
    """Return the executable capability name from a delivery package id."""
    text = str(package_id or "").strip()
    for prefix in ("capability.", "skill."):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def field_capability_route(
    package_id: str,
    *,
    scenario_id: str = "",
    required_inputs: Iterable[str] = (),
) -> dict[str, Any]:
    """Return auditable execution metadata for one field capability package."""
    capability = normalize_capability_package(package_id)
    contracts = registered_skill_contracts()
    contract = contracts.get(capability)
    route = dict(_DEFAULT_ROUTE)
    route.update(_FIELD_CAPABILITY_ROUTES.get(capability, {}))
    safety_level = contract.safety_level if contract else "unknown"
    confirm_before_execute = bool(contract and contract.confirm_before_execute)
    approval_required = (
        not contract
        or safety_level in {"dangerous", "critical"}
        or confirm_before_execute
    )
    parameter_contract = _parameter_contract(contract)
    contract_required = [item["name"] for item in parameter_contract if item["required"]]
    required = _unique_strings([*required_inputs, *contract_required])
    return {
        "package_id": str(package_id or ""),
        "capability": capability,
        "scenario_id": str(scenario_id or ""),
        "installed_contract": contract is not None,
        "contract_source": contract.source if contract else "",
        "description": contract.description if contract else "",
        "safety_level": safety_level,
        "confirm_before_execute": confirm_before_execute,
        "approval_required": approval_required,
        "approval_policy": "supervisor_required" if approval_required else "operator_allowed",
        "execution": contract.execution if contract else "not_bound",
        "tags": list(contract.tags) if contract else [],
        "required_inputs": required,
        "parameter_contract": parameter_contract,
        **route,
    }


def field_capability_routes(
    package_ids: Iterable[str],
    *,
    scenario_id: str = "",
    required_inputs: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return route metadata for a customer object's configured skill packages."""
    return [
        field_capability_route(
            package_id,
            scenario_id=scenario_id,
            required_inputs=required_inputs,
        )
        for package_id in _unique_strings(package_ids)
    ]


def _parameter_contract(contract: SkillContract | None) -> list[dict[str, Any]]:
    if contract is None:
        return []
    return [
        {
            "name": parameter.name,
            "type": parameter.type,
            "required": parameter.required,
            "description": parameter.description,
        }
        for parameter in contract.parameters
    ]


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
