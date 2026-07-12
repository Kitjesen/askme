# Multi-Agent Workflow

This project can be advanced by several agents at once, but only if each agent
owns a clear module boundary and the lead agent integrates the result.

Before lane assignment, read `docs/PRODUCT_REQUIREMENTS.md`,
`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`,
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, and `docs/DEMAND_EVIDENCE_LEDGER.md`.
Field Delivery Domain work must preserve Product/Admin/Platform/Internal vs
Runtime / Safety / Hardware ownership; customer signoff != production readiness.

## Lead Agent Responsibilities

- Pick the work lane before spawning agents.
- Assign each agent a non-overlapping write scope.
- Keep boundary-sensitive files under one owner per round.
- Review returned diffs before accepting them.
- Run the final target tests and boundary tests.
- Reject work that crosses the assigned module boundary without explanation.

The lead agent owns final correctness. Worker agents only own their assigned
slice.

## Safe Parallel Lanes

| Lane | Typical write scope | Required verification |
| --- | --- | --- |
| Runtime / blueprints | `askme/runtime`, `askme/blueprints` | `pytest tests/test_runtime_modules.py tests/test_all_modules.py tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py -q` |
| Voice gateway / interaction | `askme/voice_gateway`, `askme/robot_interaction` | `pytest tests/test_voice_loop.py tests/test_text_loop.py tests/test_voice_runtime_bridge.py tests/test_interaction_gate.py tests/test_contract_voice_gate.py -q` |
| API / MCP / tools | `askme/api`, `askme/mcp`, `askme/tools` | `pytest tests/test_api_route_dependency_injection.py tests/test_mcp_tools.py tests/test_mcp_memory_tools.py tests/test_mcp_misc_resources.py tests/test_builtin_tools.py tests/test_tool_registry.py -q` |
| CLI surface | `askme/cli`, compatibility facade `askme/cli.py` | `pytest tests/test_cli.py tests/test_cli_helpers.py tests/test_cli_agent_speak.py -q` |
| Providers / ports | `askme/providers`, `askme/ports` | `pytest tests/test_six_layer_package_boundaries.py tests/test_register_defaults.py tests/test_registry.py tests/test_robot_tools_ext.py tests/test_arm_controller.py -q` |
| Field Delivery Domain | `askme/pipeline/field`, `askme/api/routes/field_*`, `askme/api/services/field_*` | `pytest tests/test_field_operations.py tests/test_field_ingest_adapters.py tests/test_field_contracts.py tests/test_dashboard_customer_project_contract.py tests/test_field_customer_project_acceptance_routes.py -q` |
| Memory / RAG | `askme/memory`, `askme/api/routes/memory.py`, `askme/api/services/knowledge_route_payloads.py` | `pytest tests/test_memory_bridge.py tests/test_memory_importer.py tests/test_memory_system.py tests/test_knowledge_route_payloads.py -q` |
| Migration compatibility | `askme/compat`, package `__init__.py` facades | `pytest tests/test_package_migration_compat.py tests/test_six_layer_package_boundaries.py -q` |
| Test hardening | `tests` only | `pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q` plus the lane-specific target under change |

Avoid assigning two agents to the same lane in one round unless their exact
file lists are disjoint.

## Shared File Reservation

Reserve these files for the lead agent or exactly one named worker per round:

- `askme/BOUNDARIES.md`
- `askme/CODE_MAP.md`
- `docs/MODULE_OWNERSHIP.md`
- `docs/MULTI_AGENT_WORKFLOW.md`
- package root `__init__.py` files that define `_LEGACY_MODULE_ALIASES`
- `askme/compat/legacy_facades.py`
- `tests/test_six_layer_package_boundaries.py`
- `tests/test_package_migration_compat.py`

If two workers need one of these files, stop and merge through the lead.

## Worker Prompt Template

```text
You are Agent <N>.
Working directory: <repo-root>.
Write scope: <exact files or directories>.
Do not modify: <explicit exclusions>.
Goal: <bounded outcome>.
Constraints:
- Do not revert edits you did not make.
- Do not cross your write scope.
- Follow existing package boundaries.
- Use provider/port boundaries instead of concrete lower-layer imports.
Verification:
- Run <target tests>.
Final report:
- Files changed.
- Tests run and result.
- Any blocker or boundary concern.
```

## Integration Rules

- Merge lower-layer contracts first: `ports`, then `providers`, then
  runtime/pipeline/API users.
- Keep compatibility facades thin. If a file is listed in
  `askme.compat.legacy_facades`, it should not grow new implementation logic.
- New runtime, pipeline, or MCP code should import provider factories from
  `askme.providers`, not from provider submodules.
- If an agent needs to change a shared boundary file such as
  `askme/BOUNDARIES.md`, `askme/CODE_MAP.md`, or a package `__init__.py`, the
  lead agent should own that edit or reserve it for one worker only.

## Stop Conditions

Stop a worker and route back to the lead when:

- The worker needs to edit outside its assigned write scope.
- Two agents need the same file.
- A change requires a new port or provider contract.
- A test failure appears outside the worker's lane.
- The worker finds legacy paths that are not listed in
  `askme.compat.legacy_facades`.

## Required Final Verification

For architecture or boundary work, the lead should run:

```powershell
pytest tests\test_six_layer_package_boundaries.py tests\test_package_migration_compat.py -q
```

For broad integration, run the relevant module tests first, then the main
regression set used by the current change.
