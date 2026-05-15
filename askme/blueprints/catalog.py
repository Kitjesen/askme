"""Product-facing catalog for askme runtime blueprints.

Blueprint files assemble runtime modules. This catalog adds the missing product
contract around them: who should use each runtime, what it can do, what external
systems it needs, and what safety boundary it must keep.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any

from askme.runtime.module import Runtime


@dataclass(frozen=True)
class BlueprintSpec:
    """Customer and delivery metadata for one runtime blueprint."""

    name: str
    title: str
    description: str
    import_path: str
    object_name: str
    startup_command: str
    product_stage: str
    primary_loop: str
    customer_visible: bool
    deployment_targets: tuple[str, ...]
    modules: tuple[str, ...]
    capabilities: tuple[str, ...]
    scenarios: tuple[str, ...]
    required_config: tuple[str, ...]
    external_services: tuple[str, ...]
    safety_boundaries: tuple[str, ...]
    validation_commands: tuple[str, ...]
    config_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
            if key == "config_aliases" and isinstance(value, dict):
                payload[key] = {
                    alias: list(paths) if isinstance(paths, tuple) else paths
                    for alias, paths in value.items()
                }
        return payload


TEXT_MODULES = (
    "llm",
    "tools",
    "memory",
    "mission",
    "cognition",
    "runtime_handoff",
    "pipeline",
    "skill",
    "executor",
    "text",
    "health",
)

VOICE_MODULES = (
    "llm",
    "tools",
    "memory",
    "mission",
    "cognition",
    "runtime_handoff",
    "pipeline",
    "skill",
    "executor",
    "voice",
    "text",
    "health",
)

VOICE_PERCEPTION_MODULES = VOICE_MODULES + (
    "pulse",
    "perception",
    "safety",
    "reaction",
)

EDGE_ROBOT_MODULES = VOICE_PERCEPTION_MODULES + (
    "control",
    "led",
    "proactive",
)

MCP_MODULES = (
    "llm",
    "tools",
    "pulse",
    "memory",
    "mission",
    "cognition",
    "runtime_handoff",
    "safety",
    "pipeline",
    "skill",
    "executor",
    "voice",
    "control",
    "health",
)

LINGTU_VOICE_MODULES = (
    "llm",
    "tools",
    "memory",
    "pipeline",
    "skill",
    "voice",
    "text",
    "telegram",
)


BLUEPRINTS: tuple[BlueprintSpec, ...] = (
    BlueprintSpec(
        name="text",
        title="Text Operations Console",
        description="Keyboard-first runtime for development, knowledge work, and task dry runs.",
        import_path="askme.blueprints.text",
        object_name="text",
        startup_command="python -m askme.blueprints.text",
        product_stage="dev_or_ops",
        primary_loop="text",
        customer_visible=False,
        deployment_targets=("developer_laptop", "ops_terminal", "ci_smoke"),
        modules=TEXT_MODULES,
        capabilities=(
            "text task intake",
            "RAG memory retrieval",
            "skill dispatch",
            "cognition planning",
            "fake or shadow runtime handoff",
            "health HTTP dashboard",
        ),
        scenarios=(
            "knowledge upload and answer verification",
            "operator confirms a robot task draft",
            "runtime handoff smoke without microphone hardware",
        ),
        required_config=("llm", "memory", "runtime_handoff", "dashboard"),
        external_services=("LLM provider",),
        safety_boundaries=(
            "No microphone or speaker dependency.",
            "No direct hardware control from cognition or LLM output.",
            "Physical tasks still require runtime handoff and safety preflight.",
        ),
        validation_commands=(
            "python -m pytest tests/test_cognition.py tests/test_text_loop.py -q",
            "python -m askme runtime capabilities --profile text --json",
        ),
        config_aliases={
            "llm": ("brain",),
            "dashboard": ("health_server",),
        },
    ),
    BlueprintSpec(
        name="voice",
        title="Voice Task Center",
        description="Voice-first runtime for operator dialogue, task planning, and customer demos without robot IO.",
        import_path="askme.blueprints.voice",
        object_name="voice",
        startup_command="python -m askme.blueprints.voice",
        product_stage="demo_or_lab",
        primary_loop="voice",
        customer_visible=True,
        deployment_targets=("demo_laptop", "lab_pc", "s100p_voice_box"),
        modules=VOICE_MODULES,
        capabilities=(
            "real microphone intake",
            "ASR to cognition planning",
            "MiniMax or configured TTS playback",
            "text fallback",
            "runtime handoff after operator confirmation",
        ),
        scenarios=(
            "operator says patrol area A and confirms the plan",
            "visitor asks a park knowledge question",
            "voice interruption and cancellation",
        ),
        required_config=("voice.asr", "voice.tts", "llm", "memory", "dashboard"),
        external_services=("ASR provider", "LLM provider", "TTS provider"),
        safety_boundaries=(
            "Voice input can request a task but cannot dispatch hardware directly.",
            "Every physical task needs explicit confirmation before handoff.",
            "The runtime arbiter owns execution and interruption.",
        ),
        validation_commands=(
            "python -m askme runtime voice-health --json",
            "python -m pytest tests/test_voice_loop.py tests/test_cognition.py -q",
        ),
        config_aliases={
            "llm": ("brain",),
            "dashboard": ("health_server",),
        },
    ),
    BlueprintSpec(
        name="voice_perception",
        title="Voice Plus Perception Runtime",
        description="Voice runtime with scene facts, interaction gating, safety state, and reactive perception.",
        import_path="askme.blueprints.voice_perception",
        object_name="voice_perception",
        startup_command="python -m askme.blueprints.voice_perception",
        product_stage="lab",
        primary_loop="voice",
        customer_visible=True,
        deployment_targets=("lab_pc_with_camera", "robot_companion_pc"),
        modules=VOICE_PERCEPTION_MODULES,
        capabilities=(
            "voice task intake",
            "fresh perception sync",
            "interaction gate evidence",
            "active perception refresh",
            "safety state awareness",
        ),
        scenarios=(
            "visitor speaks near a configured service point",
            "operator says inspect this and the system refreshes perception",
            "no response when the person is not addressing the robot",
        ),
        required_config=("voice", "perception", "interaction_gate", "cognition", "runtime_handoff"),
        external_services=("ASR provider", "LLM provider", "TTS provider", "camera or perception bridge"),
        safety_boundaries=(
            "Perception facts are evidence, not motor commands.",
            "Stale perception blocks deictic tasks until refreshed.",
            "Safety state must be visible before runtime handoff.",
        ),
        validation_commands=(
            "python -m pytest tests/test_interaction_gate.py tests/test_active_perception_resolver.py -q",
            "python -m askme runtime capabilities --profile voice --json",
        ),
        config_aliases={
            "interaction_gate": ("voice.interaction_gate",),
        },
    ),
    BlueprintSpec(
        name="edge_robot",
        title="Park Patrol Robot Runtime",
        description="Full edge runtime for park patrol, field events, voice interaction, and robot-side IO.",
        import_path="askme.blueprints.edge_robot",
        object_name="edge_robot",
        startup_command="python -m askme.blueprints.edge_robot",
        product_stage="pilot",
        primary_loop="voice",
        customer_visible=True,
        deployment_targets=("robot_edge_pc", "customer_pilot_site", "lab_robot"),
        modules=EDGE_ROBOT_MODULES,
        capabilities=(
            "voice interaction",
            "field event intake",
            "scene perception",
            "runtime handoff",
            "robot control adapter",
            "LED or status indicators",
            "proactive monitoring",
        ),
        scenarios=(
            "fall unrecoverable announcement and DingTalk security notice",
            "robot stuck or motor fault event archive",
            "night intruder near window or corner",
            "illegal parking evidence capture",
            "fire or smoke alert",
            "trash bin full inspection",
            "visitor wayfinding and escort handoff",
            "urgent patrol interrupts routine patrol",
        ),
        required_config=(
            "voice",
            "perception",
            "field_operations",
            "runtime_handoff",
            "dingding",
            "robot_control",
        ),
        external_services=(
            "ASR provider",
            "LLM provider",
            "TTS provider",
            "DingTalk webhook",
            "robot control or nav gateway",
            "camera and sensor bridges",
        ),
        safety_boundaries=(
            "LLM and voice layers do not control hardware directly.",
            "Field events enter through the controlled field ingress path.",
            "Runtime arbiter and safety preflight decide physical execution.",
            "High-risk event closure requires governed operator approval.",
        ),
        validation_commands=(
            "python -m askme runtime field-readiness --json",
            "python -m pytest tests/test_field_operations.py tests/test_runtime_handoff.py -q",
        ),
        config_aliases={
            "dingding": (
                "field_operations.dingtalk_webhooks",
                "field_operations.dingtalk_security_webhook",
                "field_operations.dingtalk_webhook",
                "alert.dingtalk_webhook",
                "ASKME_DINGTALK_SECURITY_WEBHOOK",
            ),
            "robot_control": (
                "runtime.dog_control",
                "DOG_CONTROL_SERVICE_URL",
            ),
        },
        notes=(
            "This is the primary customer pilot blueprint.",
            "Production claim still depends on site readiness evidence and real hardware smoke tests.",
        ),
    ),
    BlueprintSpec(
        name="mcp",
        title="MCP Tool Provider",
        description="Runtime for exposing askme tools, resources, and controlled robot capabilities to MCP clients.",
        import_path="askme.blueprints.mcp",
        object_name="mcp",
        startup_command="python -m askme.blueprints.mcp",
        product_stage="integration",
        primary_loop="mcp",
        customer_visible=False,
        deployment_targets=("developer_laptop", "integration_server"),
        modules=MCP_MODULES,
        capabilities=(
            "MCP tool serving",
            "memory and skill resources",
            "controlled cognition planning",
            "runtime handoff tool surface",
            "robot API bridge where configured",
        ),
        scenarios=(
            "Claude or another MCP client queries robot memory",
            "external agent drafts a task without direct hardware dispatch",
            "operator reviews controlled tool calls",
        ),
        required_config=("mcp", "tools", "memory", "skills", "runtime_handoff"),
        external_services=("MCP client", "optional robot gateway"),
        safety_boundaries=(
            "MCP tools must keep the same SkillGate and runtime handoff constraints.",
            "No client receives raw direct motor authority through this blueprint.",
        ),
        validation_commands=(
            "python -m askme mcp serve --transport stdio",
            "python -m pytest tests/test_mcp_server.py tests/test_tool_registry_extended.py -q",
        ),
    ),
    BlueprintSpec(
        name="lingtu_voice",
        title="LingTu Voice Navigation Adapter",
        description="Voice runtime for LingTu navigation deployment with Telegram support and no Thunder control plane.",
        import_path="askme.blueprints.lingtu_voice",
        object_name="lingtu_voice",
        startup_command="python -m askme.blueprints.lingtu_voice",
        product_stage="site_specific",
        primary_loop="voice",
        customer_visible=True,
        deployment_targets=("lingtu_navigation_host", "s100p_voice_box"),
        modules=LINGTU_VOICE_MODULES,
        capabilities=(
            "voice navigation request intake",
            "Telegram operator channel",
            "RAG answer support",
            "skill dispatch to LingTu REST adapter",
        ),
        scenarios=(
            "visitor asks for a known destination",
            "operator receives a Telegram-side service note",
            "navigation request is routed to LingTu REST service",
        ),
        required_config=("voice", "llm", "telegram", "NAV_GATEWAY_URL"),
        external_services=("LingTu REST API", "Telegram Bot", "ASR provider", "TTS provider"),
        safety_boundaries=(
            "Does not include Thunder control, safety, or LED modules.",
            "Navigation authority stays behind the LingTu gateway.",
        ),
        validation_commands=(
            "python -m askme runtime voice-health --json",
            "python -m pytest tests/test_voice_loop.py -q",
        ),
        config_aliases={
            "llm": ("brain",),
            "NAV_GATEWAY_URL": ("platforms.telegram.lingtu_url",),
        },
        notes=("Site-specific blueprint; do not present as the default park patrol runtime.",),
    ),
)

ALIASES = {
    "edge": "edge_robot",
    "robot": "edge_robot",
    "park": "edge_robot",
    "voice-plus-perception": "voice_perception",
    "voice_perception": "voice_perception",
    "lingtu": "lingtu_voice",
}


def list_blueprints(*, customer_visible: bool | None = None) -> list[BlueprintSpec]:
    """Return blueprint specs, optionally filtered by customer visibility."""
    items = list(BLUEPRINTS)
    if customer_visible is not None:
        items = [item for item in items if item.customer_visible is customer_visible]
    return items


def get_blueprint_spec(name: str) -> BlueprintSpec:
    """Return one blueprint spec by name or alias."""
    key = ALIASES.get(str(name or "").strip(), str(name or "").strip())
    for spec in BLUEPRINTS:
        if spec.name == key:
            return spec
    raise KeyError(f"unknown blueprint: {name}")


def load_blueprint_runtime(name: str) -> Runtime:
    """Import and return the Runtime object for a blueprint."""
    spec = get_blueprint_spec(name)
    module = importlib.import_module(spec.import_path)
    runtime = getattr(module, spec.object_name)
    if not isinstance(runtime, Runtime):
        raise TypeError(f"{spec.import_path}.{spec.object_name} is not a Runtime")
    return runtime


def inspect_blueprint(name: str) -> dict[str, Any]:
    """Inspect a blueprint composition without starting services."""
    spec = get_blueprint_spec(name)
    runtime = load_blueprint_runtime(spec.name)
    module_names = tuple(module_class.name for module_class in runtime._module_classes)
    duplicate_modules = _duplicates(module_names)
    missing_declared = [item for item in spec.modules if item not in module_names]
    undeclared_runtime_modules = [item for item in module_names if item not in spec.modules]
    order_findings = _order_findings(module_names)
    valid = not duplicate_modules and not missing_declared and not undeclared_runtime_modules
    return {
        "name": spec.name,
        "title": spec.title,
        "valid": valid,
        "module_count": len(module_names),
        "modules": list(module_names),
        "declared_modules": list(spec.modules),
        "duplicate_modules": duplicate_modules,
        "missing_declared_modules": missing_declared,
        "undeclared_runtime_modules": undeclared_runtime_modules,
        "order_findings": order_findings,
        "startup_command": spec.startup_command,
        "product_stage": spec.product_stage,
        "customer_visible": spec.customer_visible,
        "safety_boundaries": list(spec.safety_boundaries),
    }


def blueprint_readiness(
    name: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return delivery readiness gates for one product runtime blueprint.

    This is intentionally a static preflight. It checks the blueprint contract
    and the current configuration shape; live credentials, hardware, and site
    smoke tests must still be verified by the validation commands.
    """
    spec = get_blueprint_spec(name)
    inspection = inspect_blueprint(spec.name)
    cfg = config if isinstance(config, dict) else {}
    config_evidence = [
        _required_config_evidence(cfg, item, aliases=spec.config_aliases)
        for item in spec.required_config
    ]
    missing_config = [
        item["requirement"] for item in config_evidence if not item["present"]
    ]
    gates = [
        {
            "gate_id": "runtime_composition",
            "status": "pass" if inspection["valid"] else "fail",
            "message": "Runtime modules match the product blueprint contract.",
            "evidence": {
                "module_count": inspection["module_count"],
                "missing_declared_modules": inspection["missing_declared_modules"],
                "undeclared_runtime_modules": inspection["undeclared_runtime_modules"],
                "duplicate_modules": inspection["duplicate_modules"],
            },
        },
        {
            "gate_id": "required_config",
            "status": "pass" if not missing_config else "fail",
            "message": (
                "All required configuration sections are present."
                if not missing_config
                else "Required configuration is missing before this blueprint can be delivered."
            ),
            "missing": missing_config,
            "required": list(spec.required_config),
            "evidence": config_evidence,
        },
        {
            "gate_id": "external_services",
            "status": "manual_check" if spec.external_services else "pass",
            "message": "External services must be credentialed and smoke-tested on site.",
            "services": list(spec.external_services),
        },
        {
            "gate_id": "validation_commands",
            "status": "manual_check" if spec.validation_commands else "fail",
            "message": "Run these commands before claiming the blueprint is ready.",
            "commands": list(spec.validation_commands),
        },
    ]
    if not inspection["valid"]:
        status = "blocked"
        claim = "Cannot be delivered until the runtime module contract is repaired."
    elif missing_config:
        status = "configuration_incomplete"
        claim = "Can be discussed as a product package, but cannot be deployed until required config is filled."
    else:
        status = "ready_for_validation"
        claim = "Ready for lab or site validation; production claim still requires live service and hardware evidence."
    return {
        "name": spec.name,
        "status": status,
        "customer_visible": spec.customer_visible,
        "product_stage": spec.product_stage,
        "production_ready": False,
        "customer_claim": claim,
        "missing_config": missing_config,
        "config_evidence": config_evidence,
        "external_services": list(spec.external_services),
        "validation_commands": list(spec.validation_commands),
        "gates": gates,
    }


def blueprint_delivery_package(
    name: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a concrete delivery package for one product runtime blueprint.

    The package is the product handoff surface: product can explain it, QA can
    test it, delivery can deploy it, and a customer pilot can be accepted or
    rejected against the listed evidence.
    """
    spec = get_blueprint_spec(name)
    inspection = inspect_blueprint(spec.name)
    readiness = blueprint_readiness(spec.name, config=config)
    config_items = [
        {
            "requirement": item["requirement"],
            "status": "ready" if item["present"] else "missing",
            "matched_path": item.get("matched_path") or "",
            "paths_checked": item.get("paths_checked") or [],
        }
        for item in readiness.get("config_evidence", [])
        if isinstance(item, dict)
    ]
    external_items = [
        {
            "service": service,
            "status": "manual_check",
            "evidence_required": f"Credential and smoke-test evidence for {service}",
        }
        for service in spec.external_services
    ]
    scenario_items = [
        {
            "scenario_id": _scenario_id(spec.name, scenario),
            "customer_scenario": scenario,
            "acceptance": (
                "Run the scenario in the selected deployment target, capture the "
                "operator-visible result, and attach audit or event evidence."
            ),
        }
        for scenario in spec.scenarios
    ]
    package_status = _delivery_package_status(readiness)
    return {
        "package_id": f"blueprint.{spec.name}",
        "blueprint": spec.name,
        "title": spec.title,
        "product_stage": spec.product_stage,
        "customer_visible": spec.customer_visible,
        "status": package_status,
        "customer_claim": readiness["customer_claim"],
        "release_boundary": _release_boundary(package_status, spec.product_stage),
        "startup_command": spec.startup_command,
        "deployment_targets": list(spec.deployment_targets),
        "deliverables": {
            "runtime_composition": {
                "status": "ready" if inspection["valid"] else "blocked",
                "module_count": inspection["module_count"],
                "modules": inspection["modules"],
                "findings": {
                    "missing_declared_modules": inspection["missing_declared_modules"],
                    "undeclared_runtime_modules": inspection["undeclared_runtime_modules"],
                    "duplicate_modules": inspection["duplicate_modules"],
                    "order_findings": inspection["order_findings"],
                },
            },
            "configuration_checklist": config_items,
            "external_service_checklist": external_items,
            "scenario_acceptance": scenario_items,
            "validation_commands": list(spec.validation_commands),
            "safety_boundaries": list(spec.safety_boundaries),
        },
        "handoff_steps": _handoff_steps(spec, readiness),
        "stop_conditions": _stop_conditions(readiness),
        "operator_runbook": _operator_runbook(spec),
        "customer_questions_to_answer": _customer_questions_to_answer(spec),
        "notes": list(spec.notes),
    }


def catalog_payload(*, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a serializable catalog for UI, CLI, and documentation surfaces."""
    inspections = [inspect_blueprint(spec.name) for spec in BLUEPRINTS]
    readiness = [blueprint_readiness(spec.name, config=config) for spec in BLUEPRINTS]
    return {
        "summary": {
            "blueprint_count": len(BLUEPRINTS),
            "customer_visible_count": sum(1 for item in BLUEPRINTS if item.customer_visible),
            "valid_count": sum(1 for item in inspections if item["valid"]),
            "ready_for_validation_count": sum(
                1 for item in readiness if item["status"] == "ready_for_validation"
            ),
            "configuration_incomplete_count": sum(
                1 for item in readiness if item["status"] == "configuration_incomplete"
            ),
            "pilot_blueprints": [
                item.name for item in BLUEPRINTS if item.product_stage in {"pilot", "lab"}
            ],
        },
        "items": [
            {
                **spec.to_dict(),
                "inspection": next(item for item in inspections if item["name"] == spec.name),
                "readiness": next(item for item in readiness if item["name"] == spec.name),
                "delivery_package": blueprint_delivery_package(spec.name, config=config),
            }
            for spec in BLUEPRINTS
        ],
    }


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _order_findings(module_names: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    if "cognition" in module_names and "runtime_handoff" in module_names:
        if module_names.index("runtime_handoff") < module_names.index("cognition"):
            findings.append("runtime_handoff_before_cognition")
    if "executor" in module_names and "voice" in module_names:
        if module_names.index("voice") < module_names.index("executor"):
            findings.append("voice_before_executor")
    if "executor" in module_names and "text" in module_names:
        if module_names.index("text") < module_names.index("executor"):
            findings.append("text_before_executor")
    return findings


def _config_path_present(config: dict[str, Any], path: str) -> bool:
    if not path:
        return False
    if path.isupper():
        import os

        return bool(os.getenv(path))
    current: Any = config
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    if current is None:
        return False
    if isinstance(current, dict):
        return True
    if isinstance(current, (list, tuple, set)):
        return bool(current)
    return str(current).strip() != ""


def _required_config_evidence(
    config: dict[str, Any],
    requirement: str,
    *,
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    paths = _candidate_config_paths(requirement, aliases=aliases)
    for path in paths:
        if _config_path_present(config, path):
            return {
                "requirement": requirement,
                "present": True,
                "matched_path": path,
                "paths_checked": list(paths),
                "matched_by": "primary" if path == requirement else "alias",
            }
    return {
        "requirement": requirement,
        "present": False,
        "matched_path": "",
        "paths_checked": list(paths),
        "matched_by": "",
    }


def _candidate_config_paths(
    requirement: str,
    *,
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in (requirement, *aliases.get(requirement, ())):
        text = str(item or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    return tuple(ordered)


def _delivery_package_status(readiness: dict[str, Any]) -> str:
    status = str(readiness.get("status") or "")
    if status == "ready_for_validation":
        return "ready_for_site_validation"
    if status == "configuration_incomplete":
        return "missing_configuration"
    if status == "blocked":
        return "blocked"
    return "needs_review"


def _release_boundary(package_status: str, product_stage: str) -> str:
    if package_status == "ready_for_site_validation":
        return (
            "Can be used for lab or customer pilot validation. Do not claim "
            "unattended production operation until live credentials, hardware, "
            "scenario evidence, and customer acceptance are attached."
        )
    if package_status == "missing_configuration":
        return (
            "Can be sold or discussed as a product package, but cannot be deployed "
            f"as {product_stage} until required configuration is filled."
        )
    return "Cannot be delivered until blocking runtime or contract findings are resolved."


def _handoff_steps(spec: BlueprintSpec, readiness: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [
        {
            "step": "select_deployment_target",
            "owner": "delivery",
            "action": f"Choose one target from: {', '.join(spec.deployment_targets)}",
        },
        {
            "step": "fill_required_config",
            "owner": "delivery",
            "action": "Fill every missing required configuration item before launch.",
            "missing": readiness.get("missing_config") or [],
        },
    ]
    if spec.name == "edge_robot":
        steps.append({
            "step": "generate_site_env_template",
            "owner": "delivery",
            "action": (
                "Run python -m askme runtime field-site-env-template --site-profile "
                "deploy/site-profiles/park-demo.yaml --output .env.site"
            ),
        })
    steps.extend(
        [
            {
                "step": "run_validation_commands",
                "owner": "qa",
                "action": "Run every validation command and archive the output.",
                "commands": list(spec.validation_commands),
            },
            {
                "step": "start_runtime",
                "owner": "delivery",
                "action": spec.startup_command,
            },
            {
                "step": "customer_acceptance",
                "owner": "product_and_delivery",
                "action": "Demonstrate selected customer scenarios and attach audit evidence.",
            },
        ]
    )
    return steps


def _stop_conditions(readiness: dict[str, Any]) -> list[str]:
    conditions = [
        "Stop if the runtime composition gate fails.",
        "Stop if a physical robot task would bypass runtime handoff or safety preflight.",
        "Stop if validation evidence cannot be attached for the selected customer scenario.",
    ]
    missing = readiness.get("missing_config") or []
    if missing:
        conditions.append(f"Stop until required config is filled: {', '.join(missing)}.")
    return conditions


def _operator_runbook(spec: BlueprintSpec) -> dict[str, Any]:
    return {
        "start": spec.startup_command,
        "health": "Open the dashboard health page or call /health after startup.",
        "validate": list(spec.validation_commands),
        "rollback": "Stop this runtime and return to the previous approved blueprint/package.",
        "audit": "Export unified audit records after pilot or incident closure.",
    }


def _customer_questions_to_answer(spec: BlueprintSpec) -> list[str]:
    questions = [
        "Which customer scenario is being accepted in this pilot?",
        "Which deployment target and site profile are in scope?",
        "Which external service credentials are configured and smoke-tested?",
        "Which audit evidence proves the scenario was completed?",
    ]
    if spec.name == "edge_robot":
        questions.append("Which real robot, camera, sensor, DingTalk, and navigation gateways are bound?")
    return questions


def _scenario_id(blueprint: str, scenario: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in scenario)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{blueprint}.{safe[:48]}"
