"""Code-defined contracts for the core askme skills.

These contracts are the authoritative structured metadata used for
capabilities, MCP catalogs, and generated OpenAPI. Legacy ``SKILL.md``
files still provide prompt bodies during the migration period.
"""

from __future__ import annotations

from .contracts import SkillParameter, skill_contract


@skill_contract(
    name="agent_task",
    description="Delegate a multi-step task to the Thunder agent shell.",
    execution="agent_shell",
    tags=("agent", "automation"),
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
