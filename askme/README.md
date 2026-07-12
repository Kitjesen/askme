# askme Package Map

`askme/` is organized by owner package, not by visual folder grouping. Do not
move these directories into category folders unless the import migration is
planned and compatibility facades are in place.

## Product And Architecture Spine

Before changing package ownership, read the PRD and architecture spine:
`docs/PRODUCT_REQUIREMENTS.md`, `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`,
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, and `docs/DEMAND_EVIDENCE_LEDGER.md`.
This package map is not the product authority by itself.

AskMe is the field delivery center for robot solution providers/integrators.
Field Delivery Domain owns customer projects, field events, evidence,
acceptance dossiers, customer signoff, and readiness gaps.
Product/Admin/Platform/Internal surfaces expose Dashboard/API/admin workflows
around those facts. Runtime / Safety / Hardware owns real execution, safety
state, hardware adaptation, runtime roundtrip, takeover, and rollback.
customer signoff != production readiness.

## Product Composition

These decide what product is being assembled and how it runs.

| Folder | Status | Owns |
| --- | --- | --- |
| `blueprints/` | active | Product presets, catalog metadata, readiness, delivery packages. |
| `runtime/` | active | Module graph, lifecycle wiring, task runtime, handoff. |

## Voice And Interaction

These map to the voice gateway and robot interaction layers.

| Folder | Status | Owns |
| --- | --- | --- |
| `voice_gateway/` | active | Unified voice middle-layer facade and runtime bridge contract. |
| `robot_interaction/` | active | Address detection, interaction gate, intent routing, observability. |
| `voice/` | active | Audio input/output, ASR/TTS helpers, diagnostics, voice orchestration. |
| `pipeline/` | active | Turn orchestration, skill routing, field workflows, reactions. |
| `cognition/` | active | World state, planning, perception sync, task context. |

## Field Delivery Domain

This is the product-domain core for the 现场运营交付中台.
`pipeline/field` owns customer projects, managed objects, site profiles, field
events, delivery resources, acceptance dossiers, onsite evidence, customer
signoff, and readiness gates. Field HTTP routes and services under
`api/routes/field_*` and `api/services/field_*` are transport surfaces around
that domain; Dashboard assets in `static/` render the resulting contracts but
do not own acceptance or production-readiness rules. Route/service changes in
this lane must preserve the `docs/PRODUCT_ARCHITECTURE_TRACE.md` mapping.

## Capabilities And Tools

These expose what the system can do.

| Folder | Status | Owns |
| --- | --- | --- |
| `skills/` | active | Customer-visible skills, capability packages, governance. |
| `tools/` | active | Callable tool implementations for runtime, MCP, and agents. |
| `agent_shell/` | governance + compat | Agent Profile registry, declarative hooks, and the deprecated local AgentShell compatibility facade. Active agent decisions belong to ZeroClaw/MCP plus TaskHandoff. |

## Contracts And Boundaries

These define the interfaces between layers.

| Folder | Status | Owns |
| --- | --- | --- |
| `ports/` | active boundary | Stable application-facing protocols such as robot, voice, safety, perception. |
| `contracts/` | active boundary | Shared cross-module contracts and package descriptors. |
| `schemas/` | active boundary | Typed messages and domain/API schemas. |
| `interfaces/` | legacy boundary | Legacy backend ABCs and registries; prefer `ports/` for new app-facing contracts. |

## Provider And Edge Implementations

These are lower-layer adapters and concrete implementations.

| Folder | Status | Owns |
| --- | --- | --- |
| `providers/` | active edge | Provider factories and deployment-specific adapter assembly. |
| `robot/` | active edge | Concrete robot, arm, LED, safety, and robot-service clients. |
| `perception/` | active edge | Camera, vision, scene intelligence, and image/archive adapters. |
| `telemetry/` | active edge | OTA bridge, runtime metrics, telemetry upload helpers. |
| `llm/` | active edge | LLM provider access, gateway, routing policy, streaming, audit. |
| `memory/` | active edge | Knowledge, evidence, retrieval, memory stores and backends. |
| `space/` | active data model | Spatial places, routes, and park-space semantics. |
| `errors/` | active cross-cutting | Enterprise error taxonomy, codes, severity, and structured responses. |

## External Surfaces

These expose product capabilities to outside clients or operators.

| Folder | Status | Owns |
| --- | --- | --- |
| `api/` | active surface | HTTP routes, services, schemas, route composition. |
| `cli/` | active surface | Operator CLI parser, command dispatch, diagnostics, and script-facing entrypoints. |
| `mcp/` | active surface | MCP server, tools, resources. |
| `static/` | active surface | Dashboard static HTML/CSS/JS assets. |
| `audit/` | active surface | Audit query, review, and export helpers. |

`askme/cli.py` remains a compatibility facade; new CLI implementation belongs
under `askme/cli/`.

## Registries, Assets, And Compatibility

These are useful but should stay small.

| Folder | Status | Owns |
| --- | --- | --- |
| `prompts/` | active registry | Code-level prompt templates. Runtime prompt files stay in root `prompts/`. |
| `compat/` | compatibility only | Legacy import registry and migration notes. Do not add business logic. |
| `interaction/` | compatibility only | Legacy facade for `robot_interaction`. New code must not import it. |
| `data/` | parking only | Package-local data parking. Runtime data belongs in root `data/`; no Python modules. |

## Usefulness Summary

- Keep: all active folders above; they currently own real product behavior or
  public surfaces.
- Keep but shrink over time: `interfaces/`, `compat/`, `interaction/`.
- Do not grow: `data/`; move runtime state to the repository-root `data/`.
- Do not physically group folders yet: preserving import paths matters more
  than visual nesting.

## First File To Read

- Contributor entry: root `README.md`.
- Package-level orientation: `askme/CODE_MAP.md`.
- Boundary rule: `askme/BOUNDARIES.md`.
- Multi-agent ownership: `docs/MODULE_OWNERSHIP.md`.
- Parallel-work workflow: `docs/MULTI_AGENT_WORKFLOW.md`.
