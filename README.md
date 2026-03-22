# Askme

Thunder industrial inspection assistant runtime.

`askme` is not a generic transport framework. Its value is the runtime composition around voice I/O, memory, vision, robot APIs, agent execution, and skill dispatch for the dog platform.

## Quick Start

```bash
# Full-screen terminal UI
python -m askme

# Plain text runtime
python -m askme --text

# Voice runtime
python -m askme --voice

# Voice + robot edge runtime
python -m askme runtime run --profile edge_robot

# Text runtime with robot APIs enabled
python -m askme runtime run --text --robot

# One-shot agent CLI
python -m askme agent send "帮我看看当前有哪些技能"
```

Legacy flags are still accepted for compatibility, including `python -m askme --legacy --text --robot`.

## Runtime Direction

The current architecture follows four rules:

1. `voice_io`, `memory`, `vision`, `robot_api`, `agent_shell`, and `skill_runtime` are assembled as composable runtime components instead of being hard-wired inside `app.py`.
2. A named profile layer controls assembly. The main profiles are `voice`, `text`, `mcp`, and `edge_robot`.
3. Skill contracts are defined in code first. MCP catalogs and OpenAPI are generated from those contracts, while legacy `SKILL.md` files still provide prompt bodies during migration.
4. Every runtime component exposes a consistent lifecycle and introspection surface: `start()`, `stop()`, `health()`, and `capabilities()`.

This deliberately does not copy the `dimos` transport stack (`In/Out/LCM/Transport`). `askme` does not currently need a bigger message bus as much as it needs a smaller composition root and clearer boundaries between skill, tool, and agent execution.

## CLI

The CLI now follows a command-tree shape closer to `dimos`, but it stays local to `askme`'s runtime model.

If you just want to tell askme what to do, running `askme` with no arguments now opens the full-screen terminal UI directly.

```bash
# Full-screen terminal UI
askme
askme tui --robot

# Plain interactive runtime
askme --text
askme runtime run --text --robot

# Inspect capabilities
askme runtime capabilities --profile voice --json
askme runtime status --server http://127.0.0.1:8765

# Inspect skills
askme skills list
askme skills show navigate --json
askme skills openapi

# One-shot agent interaction
askme agent send "去机房巡检一圈"

# MCP serving
askme mcp serve --transport sse --host 0.0.0.0 --port 8080
```

`askme agent send` prefers a running local runtime at `http://127.0.0.1:8765` and falls back to local one-shot execution when that endpoint is unavailable.

## Runtime Profiles

Profiles live in [askme/runtime/profiles.py](/D:/inovxio/tools/askme/askme/runtime/profiles.py).

| Profile | Primary loop | Purpose |
| --- | --- | --- |
| `voice` | `voice` | Interactive voice runtime with health HTTP endpoints and proactive services |
| `text` | `text` | Interactive text runtime with the same shared memory/skill stack |
| `mcp` | `mcp` | MCP-only serving profile for tools/resources |
| `edge_robot` | `voice` | Voice-first edge runtime with robot API, LED bridge, and event-driven perception |

The legacy CLI now resolves flags into profiles instead of rebuilding the app by hand.

## Composition Layout

The actual composition root is [askme/runtime/assembly.py](/D:/inovxio/tools/askme/askme/runtime/assembly.py).

Key files:

- [askme/app.py](/D:/inovxio/tools/askme/askme/app.py): thin legacy facade over the assembled runtime
- [askme/runtime/components.py](/D:/inovxio/tools/askme/askme/runtime/components.py): uniform component lifecycle and introspection
- [askme/runtime/profiles.py](/D:/inovxio/tools/askme/askme/runtime/profiles.py): named runtime profiles
- [askme/skills/contracts.py](/D:/inovxio/tools/askme/askme/skills/contracts.py): code-defined skill contracts and OpenAPI generation
- [askme/skills/contracts_builtin.py](/D:/inovxio/tools/askme/askme/skills/contracts_builtin.py): authoritative contracts for core built-in skills
- [askme/skills/skill_manager.py](/D:/inovxio/tools/askme/askme/skills/skill_manager.py): merges code contracts with legacy markdown skill metadata

High-level runtime view:

```text
profile
  -> runtime assembly
      -> voice_io
      -> memory
      -> vision
      -> robot_api
      -> agent_shell
      -> skill_runtime
      -> optional control_plane / proactive_runtime / perception_runtime / signal_runtime
```

## Skill Contracts

Legacy `SKILL.md` is no longer the only source of truth.

- Contract metadata comes from code via `SkillContract` and `@skill_contract`.
- `SkillManager.get_contracts()` produces the structured catalog used by HTTP and MCP surfaces.
- Markdown skills still provide prompt templates and fallback metadata while the migration is in progress.
- OpenAPI is generated from loaded contracts through `SkillManager.openapi_document()`.

Generated surfaces:

- MCP catalog: `askme://skills`
- MCP OpenAPI: `askme://skills/openapi`
- HTTP capabilities: `/api/capabilities`

## Health And Introspection

The embedded health server lives in [askme/health_server.py](/D:/inovxio/tools/askme/askme/health_server.py).

Available endpoints:

- `/health`: compact runtime health snapshot
- `/metrics`: Prometheus-style metrics output
- `/api/status`: broader runtime status for monitoring UI
- `/api/capabilities`: profile, component, skill contract, and OpenAPI summary
- `/api/chat`: web chat entrypoint wired through the same dispatcher/runtime stack

## Repository Layout

```text
askme/
  app.py
  health_server.py
  mcp_server.py
  runtime/
    assembly.py
    components.py
    profiles.py
  brain/
  pipeline/
  agent_shell/
  skills/
    builtin/
    contracts.py
    contracts_builtin.py
    skill_manager.py
  tools/
  voice/
  perception/
tests/
```

There is no active `core/Module/Orchestrator/EventBus` layer in the current codebase. Older references to that structure were stale and have been removed.
