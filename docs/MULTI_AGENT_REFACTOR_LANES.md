# Multi-Agent Refactor Lanes

Use this playbook when several agents refactor the repo in one round. The lead
agent assigns one lane per worker, reserves shared files, and owns final
integration.

## Lane Map

| Lane | Owns | Allowed write scope | Required tests |
| --- | --- | --- | --- |
| 1. Blueprints / runtime | Product startup shape, runtime module wiring, profile and preset composition | `askme/blueprints`, `askme/runtime` | `tests/test_runtime_modules.py`, `tests/test_all_modules.py`, `tests/test_blueprints_catalog.py`, `tests/test_blueprint_api_payloads.py` |
| 2. Voice / interaction | Voice turn gateway, text/voice loop contracts, robot interaction gate and address decisions | `askme/voice_gateway`, `askme/robot_interaction` | `tests/test_voice_loop.py`, `tests/test_text_loop.py`, `tests/test_voice_runtime_bridge.py`, `tests/test_interaction_gate.py`, `tests/test_contract_voice_gate.py` |
| 3. Cognition | Cognitive planning, world state, working memory, perception sync contracts | `askme/cognition` | `tests/test_six_layer_package_boundaries.py` plus any cognition-specific tests touched by the change |
| 4. Providers / robot | Provider registry/factories, concrete adapters, robot-facing implementation behind ports | `askme/providers`, `askme/robot` | `tests/test_register_defaults.py`, `tests/test_registry.py`, `tests/test_robot_tools_ext.py`, `tests/test_arm_controller.py`, `tests/test_six_layer_package_boundaries.py` |
| 5. Tools / API / MCP | FastAPI surface, MCP resources/tools, builtin tool registry and dispatch glue | `askme/api`, `askme/mcp`, `askme/tools` | `tests/test_api_route_dependency_injection.py`, `tests/test_mcp_tools.py`, `tests/test_mcp_memory_tools.py`, `tests/test_mcp_misc_resources.py`, `tests/test_builtin_tools.py`, `tests/test_tool_registry.py` |
| 6. Boundary / test hardening | Boundary assertions, compatibility coverage, tests that lock refactor behavior | `tests` only unless the lead explicitly reserves a boundary doc or facade | `tests/test_six_layer_package_boundaries.py`, `tests/test_package_migration_compat.py`, plus the lane-specific target under change |

## Boundary Rules

- Keep lane ownership exclusive for a round. Do not assign two agents to the
  same lane unless the lead lists non-overlapping files.
- `providers` and `robot` may implement concrete behavior, but upper layers
  must cross through ports, registries, or provider factories.
- `blueprints` and `runtime` choose composition and lifecycle. They must not
  grow concrete provider, robot, API, MCP, or tool implementation logic.
- `voice_gateway` owns voice turn facades. `robot_interaction` owns addressing,
  gating, and intent decisions. Neither lane owns hardware clients.
- `api`, `mcp`, and `tools` expose capabilities. They must not own runtime
  module lifecycle or provider internals.
- Tool changes that touch robot, navigation, or scene-memory execution must
  inject `askme.ports` contracts first, then keep provider HTTP details behind
  `askme.providers` factories.
- `cognition` owns planning and state interpretation. It must not dispatch
  robot hardware directly.

## Conflict Rules

- Stop and route to the lead if a worker needs a file outside its write scope.
- Reserve shared boundary files for exactly one owner before editing them.
- Merge contract or boundary changes before dependent implementation changes.
- If two workers need the same file, the lead integrates that file or splits the
  work into separate rounds.
- A worker must report test failures outside its lane instead of fixing them
  opportunistically.
- Final integration must run the lane tests for every changed lane and the
  boundary tests when imports, facades, or dependency direction changed.
