# Module Ownership

This table is the handoff map for human developers and parallel agents. Use it
with `docs/MULTI_AGENT_WORKFLOW.md` before assigning work.

For root-directory placement rules, read `docs/REPOSITORY_LAYOUT.md` first. The
repository root is a project workspace; the Python product architecture lives
inside `askme/`. For bounded contexts, API surface rules, and architecture
decision gates, read `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` before changing
ownership lanes.

Before assigning a lane, check `docs/PRODUCT_REQUIREMENTS.md`,
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, and `docs/DEMAND_EVIDENCE_LEDGER.md`.
Field Delivery Domain work must preserve Product/Admin/Platform/Internal vs
Runtime / Safety / Hardware ownership; customer signoff != production readiness.

## Ownership Lanes

| Lane | Owns | Does not own | Required verification |
| --- | --- | --- | --- |
| Runtime Graph | `askme/runtime/core`, `askme/runtime/modules`, `askme/runtime/task` | Product preset choice, concrete provider implementation | `pytest tests/test_runtime_modules.py tests/test_all_modules.py tests/test_runtime_profiles.py tests/test_runtime_planes.py -q` |
| Runtime / blueprints | `askme/runtime`, `askme/blueprints` | Concrete provider implementation, product API routes | `pytest tests/test_runtime_modules.py tests/test_all_modules.py tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py -q` |
| Blueprint Catalog | `askme/blueprints/presets`, `askme/blueprints/catalog`, `askme/blueprints/runner` | Runtime module behavior, provider selection | `pytest tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py tests/test_runtime_modules.py -q` |
| Ports Contracts | `askme/ports`, narrow shared contracts when approved by the lead | Providers, runtime modules, SDK clients | `pytest tests/test_six_layer_package_boundaries.py tests/test_contract_voice_gate.py tests/test_product_contracts.py -q` |
| Providers / ports | `askme/providers`, `askme/ports` | Product orchestration, runtime policy, API route behavior | `pytest tests/test_six_layer_package_boundaries.py tests/test_register_defaults.py tests/test_registry.py tests/test_robot_tools_ext.py -q` |
| Pipeline Orchestration | `askme/pipeline/core`, `askme/pipeline/channels`, `askme/pipeline/skills`, `askme/pipeline/proactive`, `askme/pipeline/reactions` | Concrete hardware/provider clients, API transport contracts | `pytest tests/test_voice_loop.py tests/test_text_loop.py tests/test_brain_pipeline_estop.py tests/test_state_led_bridge.py -q` |
| Field Delivery Domain | `askme/pipeline/field`, `askme/api/routes/field_*`, `askme/api/services/field_*`, field/customer-project schemas | Core runtime graph, low-level providers, API transport composition, Dashboard layout/assets | `pytest tests/test_field_operations.py tests/test_field_ingest_adapters.py tests/test_field_contracts.py tests/test_dashboard_customer_project_contract.py tests/test_field_customer_project_acceptance_routes.py -q` |
| Voice gateway / interaction | `askme/voice_gateway`, `askme/robot_interaction` | ASR/TTS implementation, robot hardware clients, tool dispatch | `pytest tests/test_voice_loop.py tests/test_text_loop.py tests/test_voice_runtime_bridge.py tests/test_interaction_gate.py tests/test_contract_voice_gate.py -q` |
| Voice Gateway | `askme/voice_gateway` | ASR/TTS implementation, robot interaction policy, provider construction | `pytest tests/test_voice_runtime_bridge.py tests/test_voice_loop.py tests/test_contract_voice_gate.py tests/test_six_layer_package_boundaries.py -q` |
| Robot Interaction | `askme/robot_interaction` | Runtime execution, tool dispatch, robot hardware clients | `pytest tests/test_interaction_gate.py tests/test_address_detector.py tests/test_contract_voice_gate.py tests/test_six_layer_package_boundaries.py -q` |
| API / MCP / tools | `askme/api`, `askme/mcp`, `askme/tools` | Runtime module lifecycle, provider internals | `pytest tests/test_api_route_dependency_injection.py tests/test_mcp_tools.py tests/test_mcp_memory_tools.py tests/test_mcp_misc_resources.py tests/test_builtin_tools.py tests/test_tool_registry.py -q` |
| API HTTP Surface | `askme/api`, route schemas, route services | MCP protocol, tool implementations, hardware clients | `pytest tests/test_api_route_dependency_injection.py tests/test_health.py tests/test_blueprint_api_payloads.py tests/test_knowledge_route_payloads.py -q` |
| CLI Surface | `askme/cli`, compatibility facade `askme/cli.py` | Product business logic, hardware/provider implementation | `pytest tests/test_cli.py tests/test_cli_helpers.py tests/test_cli_agent_speak.py -q` |
| MCP + Tools Surface | `askme/mcp`, `askme/tools` | Runtime module lifecycle, provider internals | `pytest tests/test_mcp_tools.py tests/test_mcp_memory_tools.py tests/test_mcp_misc_resources.py tests/test_builtin_tools.py tests/test_tool_registry.py -q` |
| Memory / RAG | `askme/memory`, memory routes and knowledge payload services | Voice/robot runtime behavior, provider internals | `pytest tests/test_memory_bridge.py tests/test_memory_importer.py tests/test_memory_system.py tests/test_knowledge_route_payloads.py -q` |
| Migration compatibility | `askme/compat`, package `__init__.py` facades | New business behavior | `pytest tests/test_package_migration_compat.py tests/test_six_layer_package_boundaries.py -q` |
| Test hardening | `tests` only | Product code, unless explicitly reassigned by the lead | `pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q` plus changed test targets |

## Current Owner Package Matrix

Use this table when splitting work across agents. The `askme/` package is
organized by owner package first; the six-layer voice/robot diagram is the
target dependency direction, not a command to collapse everything into one
folder.

Field delivery is the product-domain lane for the 现场运营交付中台. Its business
rules include customer projects, managed objects, site profiles, field events,
delivery resources, acceptance dossiers, onsite evidence, customer signoff, and
solution/product launch readiness. Field HTTP routes and route services are
thin transport surfaces around that domain; they may validate request/response
contracts, choose Product/Admin/Internal exposure, and call field-domain
builders, but they must not become separate sources of truth for acceptance,
production-readiness, or hardware-control claims.

| Package | Owns | Does not own | Default verification |
| --- | --- | --- | --- |
| `agent_shell` | Agent Profile governance, declarative hooks, and deprecated AgentShell compatibility stubs | active agent decision loop, runtime graph, MCP/API transport | `pytest tests/test_agent_profiles.py tests/test_agent_hooks.py tests/test_thunder_agent_shell.py -q` |
| `api` | HTTP route composition, schemas, route services | MCP protocol, provider construction | `pytest tests/test_api_route_dependency_injection.py tests/test_*_http.py -q` |
| `audit` | audit queries and audit payload helpers | route ownership, provider clients | `pytest tests/test_audit_query.py tests/test_audit_http.py -q` |
| `blueprints` | product presets, catalog data, startup metadata | module internals, providers | `pytest tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py -q` |
| `cli` | operator CLI parser, command dispatch, diagnostics, script-facing entrypoints | product business logic, provider implementation | `pytest tests/test_cli.py tests/test_cli_helpers.py tests/test_cli_agent_speak.py -q` |
| `cognition` | world state, working memory, planning, perception sync | direct robot dispatch, provider clients | `pytest tests/test_cognition.py tests/test_working_memory.py tests/test_space_cognition.py -q` |
| `compat` | legacy import facades and migration warnings | new business behavior | `pytest tests/test_package_migration_compat.py -q` |
| `contracts` | shared cross-module contracts and adapters | concrete provider behavior | `pytest tests/test_contract_catalog_mcp.py tests/test_six_layer_package_boundaries.py -q` |
| `data` | package-local static data parking | runtime/customer data, Python modules | `pytest tests/test_repository_layout.py -q` |
| `interfaces` | legacy ABCs and backend registry facade | concrete adapter implementations | `pytest tests/test_six_layer_package_boundaries.py tests/test_register_defaults.py -q` |
| `interaction` | legacy compatibility imports for interaction APIs | new interaction logic | `pytest tests/test_package_migration_compat.py tests/test_six_layer_package_boundaries.py -q` |
| `llm` | LLM gateway, providers, routing, conversation client | runtime lifecycle, tool dispatch | `pytest tests/test_conversation.py tests/test_runtime_modules.py -q` |
| `mcp` | MCP server, AppContext transport view, tools/resources | runtime ownership, provider internals | `pytest tests/test_mcp_runtime_adapter.py tests/test_mcp_tools.py tests/test_mcp_memory_tools.py -q` |
| `memory` | conversation, session, episodic, knowledge/RAG backends | voice/robot runtime behavior | `pytest tests/test_memory_bridge.py tests/test_memory_system.py tests/test_memory_importer.py -q` |
| `pipeline` | turn orchestration, channels, skills, reactions, field workflows | provider construction, transport routes | `pytest tests/test_text_loop.py tests/test_voice_loop.py tests/test_turn_executor.py -q` |
| `ports` | stable application-facing protocols | providers, SDK clients, runtime modules | `pytest tests/test_six_layer_package_boundaries.py -q` |
| `providers` | bottom-layer factories and concrete adapter selection | API/MCP/routes/runtime policy | `pytest tests/test_six_layer_package_boundaries.py tests/test_register_defaults.py -q` |
| `robot` | concrete robot, arm, dog, telemetry, hardware/service clients | user intent, product orchestration | `pytest tests/test_robot_tools_ext.py tests/test_robot_api_tool.py -q` |
| `robot_interaction` | address detection, interaction gate, intent routing, observability | ASR/TTS, hardware clients, tool execution | `pytest tests/test_interaction_gate.py tests/test_address_detector.py tests/test_six_layer_package_boundaries.py -q` |
| `runtime` | module graph, module lifecycle, runtime task services | blueprint choice, provider internals | `pytest tests/test_runtime_modules.py tests/test_all_modules.py tests/test_runtime_planes.py -q` |
| `skills` | skill catalog, packages, contracts, governance, execution model | transport routes, hardware clients | `pytest tests/test_skill_governance.py tests/test_capability_center.py tests/test_create_skill_e2e.py -q` |
| `space` | spatial/site domain helpers | vision provider implementation | `pytest tests/test_space_cognition.py tests/test_space_tools.py -q` |
| `static` | bundled dashboard assets | API behavior, runtime services | `pytest tests/test_dashboard_http.py tests/test_dashboard_customer_project_contract.py -q` |
| `telemetry` | shared OTA/runtime metrics | robot implementation packages | `pytest tests/test_six_layer_package_boundaries.py tests/test_runtime_modules.py -q` |
| `tools` | callable tool implementations and registry | runtime lifecycle, provider internals | `pytest tests/test_builtin_tools.py tests/test_tool_registry.py tests/test_mcp_tools.py -q` |
| `voice` | audio input/output, ASR/TTS, diagnostics, compatibility facades | voice gateway policy, robot interaction decisions | `pytest tests/test_voice_health.py tests/test_voice_profiles.py tests/test_voice_loop.py -q` |
| `voice_gateway` | voice turn facade and gateway service | ASR/TTS implementation, provider construction | `pytest tests/test_voice_gateway_session.py tests/test_voice_runtime_bridge.py -q` |

## Dependency Direction

Keep the default direction:

```text
blueprints
  -> runtime / pipeline / voice_gateway / robot_interaction / api / mcp / tools
  -> ports / interfaces
  -> providers
  -> robot / perception / voice / external SDKs
```

Important ownership rules:

- `ports` defines protocols and must not import providers or concrete clients.
- `providers` may import concrete adapters but must not import product,
  runtime, pipeline, API, MCP, tools, voice gateway, or robot interaction code.
- Robot-facing tools should consume ports such as `RobotControlPort`,
  `NavigationPort`, and `TemporalMemoryPort`; HTTP/env-var access belongs in
  provider adapters, with legacy fallback paths kept isolated and tested.
- Runtime, pipeline, MCP, and API code should import provider factories from
  `askme.providers`, not from provider submodules.
- MCP tools/resources should consume `AppContext`; when MCP needs runtime-built
  services, use `askme.mcp.runtime_adapter.app_context_from_runtime_app()` to
  adapt an already-built `RuntimeApp` instead of rebuilding providers.
- `voice_gateway` owns turn facade behavior; `robot_interaction` owns address,
  gate, and intent decisions.
- Compatibility facades remain importable but must not grow new implementation
  classes or functions.

## Shared Control Files

Only the lead agent, or one explicitly assigned worker, should edit these in a
round:

- `askme/BOUNDARIES.md`
- `askme/CODE_MAP.md`
- `docs/MODULE_OWNERSHIP.md`
- `docs/MULTI_AGENT_WORKFLOW.md`
- package root `__init__.py` files
- `askme/compat/legacy_facades.py`
- boundary and migration tests

If an optimization needs one of these files, reserve it before starting
parallel implementation.
