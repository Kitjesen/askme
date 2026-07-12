# Documentation And Multi-Agent Coordination

Use this page as the first handoff note for a 6-agent parallel round. It points
to the source-of-truth docs, assigns non-overlapping directory ownership, lists
shared files that need reservation, and gives the verification command each
agent should run before returning work.

## Read First

Read these in order before assigning agents:

1. `docs/MULTI_AGENT_WORKFLOW.md` - lead/worker protocol, shared-file rules,
   worker prompt template, and final integration requirements.
2. `docs/MULTI_AGENT_REFACTOR_LANES.md` - 6-lane refactor split for a parallel
   round.
3. `docs/MODULE_OWNERSHIP.md` - current package owner matrix, exclusions, and
   default verification commands.
4. `docs/MISSION_VOICE_STATES.md` - voice admission states for setup, idle,
   active mission, pause, emergency, and review.
5. `askme/README.md` - package map for top-level `askme/` directories.
6. `docs/REPOSITORY_LAYOUT.md` - root directory placement rules.
7. `docs/PRODUCT_REQUIREMENTS.md` - PRD-level product requirements spine connecting demand evidence to architecture contracts.
8. `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` - bounded contexts, package ownership, API surface rules, and architecture decision gates.
9. `docs/SOLUTION_PROVIDER_ICP.md` - product-demand source for the P0 solution-provider delivery lane.
10. `docs/PRODUCT_ARCHITECTURE_TRACE.md` - R1-R7 demand-to-architecture ownership and verification map.
11. `docs/GITHUB_SKILL_RESEARCH.md` - external GitHub skill candidates for market, product, and architecture research.
12. `docs/DEMAND_EVIDENCE_LEDGER.md` - research evidence ledger separating hypotheses from validated product and architecture claims.
13. `docs/COMPETITIVE_REPLACEMENT_MATRIX.md` - competitive and replacement boundaries for the solution-provider delivery product.
14. `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md` - interview plan for validating ICP, replacement boundaries, R1-R7 demand, and architecture assumptions.
15. `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` - four-industry scenario cards constrained to three high-value scenes each.
16. `docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md` - pilot acceptance dossier product surface and signoff/readiness boundary.
17. `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` - minimum fields, failure states, and audit boundaries for VMS, CMMS, IAM, map, OEM fleet, notification, and SIEM/WORM connections.
18. `docs/SITE_LAUNCH_READINESS_CHECKLIST.md` - site acceptance checklist, launch readiness, and hardware acceptance boundary.
19. `docs/PRICING_PACKAGING_HYPOTHESES.md` - pricing and packaging hypotheses bound to Field Delivery Domain facts and usage evidence.
20. `docs/SCENARIO_ROI_MODEL.md` - scenario ROI model tying industry cards to baseline, value metrics, payback signals, and usage evidence.

## 6-Agent Round Map

Assign exactly one owner per lane unless the lead lists exact disjoint files.
Workers must not edit outside their lane without routing back to the lead.

| Agent | Lane | Primary write scope | Must not own | Verification before report |
| --- | --- | --- | --- | --- |
| 1 | Blueprints / runtime | `askme/blueprints`, `askme/runtime` | Provider internals, API route behavior, shared boundary docs | `python -m pytest tests/test_runtime_modules.py tests/test_all_modules.py tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py -q` |
| 2 | Voice / interaction | `askme/voice_gateway`, `askme/robot_interaction` | ASR/TTS implementation, robot hardware clients, tool dispatch | `python -m pytest tests/test_voice_loop.py tests/test_text_loop.py tests/test_voice_runtime_bridge.py tests/test_interaction_gate.py tests/test_contract_voice_gate.py -q` |
| 3 | Cognition / pipeline core | `askme/cognition`, `askme/pipeline/core`, `askme/pipeline/channels`, `askme/pipeline/proactive`, `askme/pipeline/reactions`, `askme/pipeline/skills` | Concrete providers, HTTP routes, hardware clients | `python -m pytest tests/test_cognition.py tests/test_working_memory.py tests/test_text_loop.py tests/test_voice_loop.py tests/test_turn_executor.py -q` |
| 4 | Providers / robot / ports | `askme/ports`, `askme/providers`, `askme/robot`, `askme/perception` | Product orchestration, API/MCP/routes/runtime policy | `python -m pytest tests/test_six_layer_package_boundaries.py tests/test_register_defaults.py tests/test_registry.py tests/test_robot_tools_ext.py tests/test_arm_controller.py -q` |
| 5 | API / MCP / tools | `askme/api`, `askme/mcp`, `askme/tools` | Runtime lifecycle, provider internals, field workflow ownership outside route/service glue | `python -m pytest tests/test_api_route_dependency_injection.py tests/test_mcp_tools.py tests/test_mcp_memory_tools.py tests/test_mcp_misc_resources.py tests/test_builtin_tools.py tests/test_tool_registry.py -q` |
| 6 | Field Delivery Domain / tests / compatibility | `askme/pipeline/field`, `askme/api/routes/field_*`, `askme/api/services/field_*`, `askme/compat`, `tests` | Core runtime graph, low-level providers, API transport composition outside field route/service glue, unrelated package reshapes | `python -m pytest tests/test_field_operations.py tests/test_field_ingest_adapters.py tests/test_field_contracts.py tests/test_dashboard_customer_project_contract.py tests/test_field_customer_project_acceptance_routes.py tests/test_package_migration_compat.py -q` |

The lead owns `docs/`, integration order, final conflict resolution, and the
final cross-lane verification run.

## Shared File Reservation

Reserve these files before any worker edits them. A reservation means exactly
one named owner for the current round; if two workers need the same file, the
lead merges that file or splits the work into another round.

| Shared file or group | Default owner | Why it is shared |
| --- | --- | --- |
| `docs/README.md` | Lead | Round brief, task queue, and documentation index. |
| `docs/MULTI_AGENT_WORKFLOW.md` | Lead | Lead/worker protocol and shared-file policy. |
| `docs/MULTI_AGENT_REFACTOR_LANES.md` | Lead | 6-agent lane map. |
| `docs/MODULE_OWNERSHIP.md` | Lead | Package owner matrix and lane verification commands. |
| `askme/BOUNDARIES.md` | Lead or one reserved boundary worker | Dependency direction and architecture rules. |
| `askme/CODE_MAP.md` | Lead or one reserved boundary worker | Package map used by multiple agents. |
| package root `__init__.py` files | Lead or owning package worker | Compatibility aliases and public exports. |
| `askme/compat/legacy_facades.py` | Agent 6 when reserved | Legacy import registry. |
| `tests/test_six_layer_package_boundaries.py` | Agent 6 when reserved | Cross-lane import guard. |
| `tests/test_package_migration_compat.py` | Agent 6 when reserved | Compatibility and package layout guard. |

Reservation note template:

```text
Reserved for this round:
- <file>: Agent <N>, reason <short reason>, expected tests <pytest command>
```

## Lead Integration Checklist

1. Confirm each worker has a non-overlapping write scope.
2. Record shared-file reservations before implementation starts.
3. Merge contract and boundary changes before dependent implementation changes.
4. Read each worker report for files changed, tests run, failures, and boundary
   concerns.
5. Run lane tests for every lane that changed.
6. Run boundary verification when imports, compatibility facades, ports, or
   provider direction changed:

```powershell
python -m pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q
```

7. For docs-only coordination changes, run:

```powershell
python -m pytest tests/test_repository_layout.py tests/test_package_migration_compat.py::test_multi_agent_docs_reference_existing_verification_targets -q
```

## Pytest Partitioning

The default pytest command is the fast shard because `pyproject.toml` sets
`-m "not slow"`. `tests/conftest.py` automatically marks scenario, e2e, and
benchmark-style tests as slow so workers can run focused shards without moving
tests between folders.

```powershell
python -m pytest tests -q
python -m pytest tests -m "slow" -q
python -m pytest tests -m "scenario" -q
python -m pytest tests -m "e2e" -q
python -m pytest tests -m "benchmark" -q
python -m pytest tests -m "e2e or benchmark" -q
```

## Next Batch Task Queue

Use this queue for the next 6-agent round. The lead should copy one row into
each worker prompt and add exact file exclusions if any shared file is reserved.

| Agent | Next task | Stop condition |
| --- | --- | --- |
| 1 | Verify runtime and blueprint package ownership after current package reshapes; keep startup composition separate from providers. | Needs provider internals, API route changes, or shared boundary files. |
| 2 | Verify voice gateway and robot interaction imports use middle-layer contracts and do not pull hardware clients directly. | Needs ASR/TTS implementation or robot adapter changes. |
| 3 | Verify cognition and pipeline orchestration stay above provider/robot layers after moved modules. | Needs API transport changes, provider construction, or field workflow ownership. |
| 4 | Verify ports/providers/robot dependency direction and keep concrete adapter details below provider factories. | Needs runtime, pipeline, API, or MCP policy changes. |
| 5 | Verify API, MCP, and tool surfaces consume runtime context and provider factories without rebuilding lower layers. | Needs runtime lifecycle ownership or provider internals. |
| 6 | Harden compatibility and package-boundary tests for moved modules; update only tests and reserved compatibility files. | Needs product behavior changes outside compatibility coverage. |

## Worker Final Report Contract

Each worker should return:

```text
Agent <N> / <lane>
Files changed:
- <path>
Tests run:
- <command>: <pass/fail/blocked>
Boundary concerns:
- <none or exact issue>
Shared files touched:
- <none or reservation name>
```
