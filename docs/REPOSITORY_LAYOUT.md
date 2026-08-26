# Repository Layout

The repository root is a workspace. The product code lives in `askme/`.
Same-level folders do not mean same-level architecture authority.

Product and architecture authority lives in `docs/PRODUCT_REQUIREMENTS.md`,
`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`,
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, and
`docs/DEMAND_EVIDENCE_LEDGER.md`. This layout file only explains where code and
documents live.

Field Delivery Domain owns customer projects, field events, evidence,
acceptance dossiers, customer signoff, and readiness gaps.
Product/Admin/Platform/Internal surfaces expose Dashboard/API/admin workflows.
Runtime / Safety / Hardware owns real execution, safety state, hardware
adaptation, runtime roundtrip, takeover, and rollback.
customer signoff != production readiness.

For a new contributor, read the root `README.md` first for runnable entry
points, then use `askme/README.md` for package ownership and
`docs/MULTI_AGENT_WORKFLOW.md` for parallel work rules.
Use `CONTEXT.md` for the accepted domain language and `docs/adr/` for durable
architecture decisions.

## Root Map

| Path | Purpose |
| --- | --- |
| `askme/` | Product Python package and runtime implementation. |
| `tests/` | Automated tests and architecture guards. |
| `docs/` | Architecture, operations, ownership, and workflow docs. |
| `CONTEXT.md` | Accepted domain vocabulary shared by product and engineering. |
| `scripts/` | Operator scripts, smoke checks, demos, benchmarks, maintenance. |
| `deploy/` | Install files, service units, site profiles, delivery templates. |
| `docker/` | Dockerfiles, Compose stacks, and container entrypoints. |
| `config/` | Environment-specific config templates such as production defaults. |
| `pyproject.toml` / `uv.lock` | Dependency declarations and the reproducible dependency lock. |
| `prompts/` | Runtime prompt assets such as `SOUL.md`. |
| `data/` | Local runtime state, customer knowledge, captures, sessions. |
| `models/` | Local model files for ASR, TTS, VAD, KWS, vision, policy. |
| `artifacts/` | Generated reports, evidence, screenshots, benchmark output. |
| `output/` | Temporary tool output. |
| `archive/` | Deprecated or parked historical material. |
| `video-lab/` | Separate video/Remotion workspace. |

## Package Direction

Inside `askme/`, use owner packages and keep this dependency direction:

```text
blueprints
  -> runtime / pipeline / voice_gateway / robot_interaction / api / mcp
  -> conversation
  -> ports / interfaces / contracts / schemas
  -> providers
  -> robot / perception / voice / llm / memory backends / external services
```

## Move Rule

Do not start with a large root-folder move. Many paths intentionally point to
`data/`, `models/`, `artifacts/`, and `prompts/`.

Move only real implementation code inside `askme/`, one owner package at a
time, and keep legacy imports working until the migration is verified.

## Multi-Agent Split

Use disjoint write scopes:

| Lane | Scope |
| --- | --- |
| Runtime composition | `askme/runtime/`, `askme/blueprints/` |
| Voice middle layer | `askme/voice_gateway/`, `askme/robot_interaction/` |
| Conversation Domain | `askme/conversation/` |
| Provider boundary | `askme/ports/`, `askme/providers/` |
| Robot/perception adapters | `askme/robot/`, `askme/perception/` |
| API/MCP/tools | `askme/api/`, `askme/mcp/`, `askme/tools/` |
| Field Delivery Domain | `askme/pipeline/field/`, `askme/api/routes/field_*`, `askme/api/services/field_*` |
| Docs/layout | `README.md`, `docs/` |
| Tests | `tests/` |

Shared boundary files such as `askme/CODE_MAP.md`, `askme/BOUNDARIES.md`, and
`docs/MODULE_OWNERSHIP.md` should have one owner per round.

## Verify

```powershell
pytest tests\test_six_layer_package_boundaries.py tests\test_package_migration_compat.py -q
```
