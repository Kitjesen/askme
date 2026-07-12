"""Code-defined contracts for the core askme skills.

These contracts are the authoritative structured metadata used for
capabilities, MCP catalogs, and generated OpenAPI. Legacy ``SKILL.md``
files still provide prompt bodies during the migration period.
"""

from __future__ import annotations

from .contracts import SkillParameter, skill_contract


@skill_contract(
    name="agent_task",
    description=(
        "Deprecated local AgentShell compatibility skill; active multi-step "
        "agent execution belongs to the ZeroClaw/MCP path."
    ),
    execution="agent_shell",
    tags=("agent", "automation", "deprecated"),
)
def _agent_task_contract() -> None:
    return None


@skill_contract(
    name="navigate",
    description="Navigate the robot to a named destination or landmark.",
    safety_level="dangerous",
    tags=("robot", "navigation"),
    parameters=(
        SkillParameter(
            name="destination",
            type="string",
            description="Target location, zone, or landmark name.",
            required=True,
        ),
    ),
)
def _navigate_contract() -> None:
    return None


@skill_contract(
    name="find_object",
    description="Search the current environment for a named object.",
    tags=("vision", "search"),
    parameters=(
        SkillParameter(
            name="object_name",
            type="string",
            description="Object category or description to look for.",
            required=True,
        ),
    ),
)
def _find_object_contract() -> None:
    return None


@skill_contract(
    name="find_person",
    description="Search the current environment for a person.",
    tags=("vision", "people"),
    parameters=(
        SkillParameter(
            name="person_name",
            type="string",
            description="Optional person identifier, role, or appearance hint.",
            required=False,
        ),
    ),
)
def _find_person_contract() -> None:
    return None


@skill_contract(
    name="recall_memory",
    description="Retrieve relevant session, episodic, or long-term memory.",
    tags=("memory", "retrieval"),
)
def _recall_memory_contract() -> None:
    return None


@skill_contract(
    name="solve_problem",
    description="Run a structured problem-solving pass over a user issue or anomaly.",
    tags=("reasoning", "analysis"),
)
def _solve_problem_contract() -> None:
    return None


@skill_contract(
    name="web_search",
    description="Search the web and summarize relevant findings.",
    tags=("search", "web"),
    parameters=(
        SkillParameter(
            name="query",
            type="string",
            description="Search query or question to investigate.",
            required=True,
        ),
    ),
)
def _web_search_contract() -> None:
    return None


_FIELD_EVENT_PARAMETERS = (
    SkillParameter(
        name="location",
        type="string",
        description="Site location where the field event is observed.",
        required=True,
    ),
    SkillParameter(
        name="evidence",
        type="string",
        description="Optional image path, sensor evidence, or operator note.",
        required=False,
    ),
)


@skill_contract(
    name="report_fall_unrecoverable",
    description="Record an unrecoverable robot fall incident, notify security, and archive evidence.",
    safety_level="dangerous",
    tags=("field", "incident", "robot_fault"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _report_fall_unrecoverable_contract() -> None:
    return None


@skill_contract(
    name="report_stuck",
    description="Record an immobilized robot incident, notify responders, and archive evidence.",
    safety_level="dangerous",
    tags=("field", "incident", "robot_fault"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _report_stuck_contract() -> None:
    return None


@skill_contract(
    name="report_motor_fault",
    description="Record a joint motor fault, notify maintenance/security, and archive diagnostics.",
    safety_level="dangerous",
    tags=("field", "incident", "robot_fault"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _report_motor_fault_contract() -> None:
    return None


@skill_contract(
    name="detect_night_intruder",
    description="Create a night stranger photo/security event with location and evidence.",
    safety_level="dangerous",
    tags=("field", "security", "vision"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _detect_night_intruder_contract() -> None:
    return None


@skill_contract(
    name="detect_illegal_parking",
    description="Create an illegal parking event with vehicle evidence and zone policy.",
    safety_level="dangerous",
    tags=("field", "security", "parking"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _detect_illegal_parking_contract() -> None:
    return None


@skill_contract(
    name="detect_fire_smoke",
    description="Create a fire or smoke safety event from sensor or image evidence.",
    safety_level="dangerous",
    tags=("field", "safety", "sensor"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _detect_fire_smoke_contract() -> None:
    return None


@skill_contract(
    name="inspect_trash_bin",
    description="Create a trash-bin-full service event for cleaning responders.",
    safety_level="dangerous",
    tags=("field", "facility", "cleaning"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _inspect_trash_bin_contract() -> None:
    return None


@skill_contract(
    name="detect_crowd_gathering",
    description="Create a crowd-gathering security event when person count and dwell time exceed policy.",
    safety_level="dangerous",
    tags=("field", "security", "vision", "crowd"),
    parameters=(
        SkillParameter(
            name="location",
            type="string",
            description="Site location where the gathering is observed.",
            required=True,
        ),
        SkillParameter(
            name="person_count",
            type="integer",
            description="Observed person count.",
            required=False,
        ),
        SkillParameter(
            name="duration_min",
            type="number",
            description="Observed dwell duration in minutes.",
            required=False,
        ),
        SkillParameter(
            name="evidence",
            type="string",
            description="Optional image path or operator note.",
            required=False,
        ),
    ),
    confirm_before_execute=True,
)
def _detect_crowd_gathering_contract() -> None:
    return None


@skill_contract(
    name="offer_wayfinding_help",
    description="Record a visitor wayfinding help-point interaction without starting a robot task.",
    tags=("field", "visitor", "wayfinding"),
    parameters=_FIELD_EVENT_PARAMETERS,
)
def _offer_wayfinding_help_contract() -> None:
    return None


@skill_contract(
    name="escort_visitor",
    description="Request a visitor escort handoff for an approved destination.",
    safety_level="dangerous",
    tags=("field", "visitor", "navigation"),
    parameters=_FIELD_EVENT_PARAMETERS,
    confirm_before_execute=True,
)
def _escort_visitor_contract() -> None:
    return None


@skill_contract(
    name="lookup_place",
    description="Resolve a visitor's destination against the park semantic map and ask for confirmation.",
    tags=("field", "visitor", "wayfinding", "space"),
    parameters=(
        SkillParameter(
            name="query",
            type="string",
            description="Destination query such as 咖啡店在哪 or 最近的厕所。",
            required=True,
        ),
        SkillParameter(
            name="current_point_id",
            type="string",
            description="Current service point id or map point id.",
            required=False,
        ),
    ),
)
def _lookup_place_contract() -> None:
    return None


@skill_contract(
    name="recommend_route",
    description="Generate park route speech and optional escort handoff payload from semantic-map data.",
    tags=("field", "visitor", "wayfinding", "space", "route"),
    parameters=(
        SkillParameter(
            name="query",
            type="string",
            description="Destination query or confirmed place name.",
            required=True,
        ),
        SkillParameter(
            name="current_point_id",
            type="string",
            description="Current service point id or map point id.",
            required=False,
        ),
        SkillParameter(
            name="guide_mode",
            type="string",
            description="voice or escort.",
            required=False,
        ),
    ),
)
def _recommend_route_contract() -> None:
    return None


@skill_contract(
    name="answer_wayfinding",
    description="Answer visitor wayfinding questions with destination confirmation and route speech.",
    tags=("field", "visitor", "wayfinding", "space", "route"),
    parameters=(
        SkillParameter(
            name="query",
            type="string",
            description="Visitor question or destination query.",
            required=True,
        ),
        SkillParameter(
            name="current_point_id",
            type="string",
            description="Current service point id or map point id.",
            required=False,
        ),
    ),
)
def _answer_wayfinding_contract() -> None:
    return None
