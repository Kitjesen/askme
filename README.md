# Askme

Thunder industrial inspection assistant runtime.

Architecture reference: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

1. `operator_io`, `memory`, `vision`, `robot_io`, `executor`, and `skills` are assembled as composable runtime components instead of being hard-wired inside `app.py`.
2. A named mode layer controls assembly. The main modes are `voice`, `text`, `mcp`, and `edge_robot`.
3. Skill contracts are defined in code first. MCP catalogs and OpenAPI are generated from those contracts, while legacy `SKILL.md` files still provide prompt bodies during migration.
4. Every runtime component exposes a consistent lifecycle and introspection surface: `start()`, `stop()`, `health()`, and `capabilities()`.

This deliberately does not copy the `dimos` transport stack (`In/Out/LCM/Transport`). `askme` does not currently need a bigger message bus as much as it needs a smaller composition root and clearer boundaries between skill, tool, and agent execution.

## Platform Boundary

Inside `askme`, the runtime still assembles local components such as `vision`,
`robot_io`, and `skills`. That is an internal composition concern, not a
statement that `askme` owns the full robot runtime.

For the NOVA Dog product boundary:

- `askme` is the operator AI and channel entry layer
- `voice` is the production service wrapper around `askme`
- `arbiter` owns mission lifecycle
- `safety` and `control` own action gating and actuation
- `nav` owns the product navigation contract
- `sense` owns product-readable perception snapshots
- `lingtu` remains the autonomy and perception provider behind `nav` and `sense`

The deeper rationale and migration direction are documented in
[`products/nova-dog/runtime/docs/ASKME_LINGTU_DECOUPLING.md`](/D:/inovxio/products/nova-dog/runtime/docs/ASKME_LINGTU_DECOUPLING.md).

This means local `VisionBridge` and direct runtime clients inside `askme` are
valid for development and fallback profiles, but they should not be described as
the production system of record.

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
mode
  -> runtime assembly
      -> operator_io
      -> memory
      -> vision
      -> robot_io
      -> executor
      -> skills
      -> optional diagnostics / supervisor / change_monitor / indicators
```

Runtime planes:

- `executive`: operator IO, memory, executor, skills
- `platform`: telemetry, robot IO, vision, indicators
- `diagnostics`: health, metrics, capabilities, HTTP endpoints

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

## Event-Driven Perception (V2)

The perception pipeline transforms raw YOLO detections into structured events and scene state.

```text
frame_daemon (5Hz BPU YOLO)
  → /tmp/askme_frame_detections.json
    → ChangeDetector (1Hz IoU matching + N-frame debounce)
      → ChangeEvent (person_appeared, person_left, object_appeared, ...)
        → WorldState (live scene snapshot: tracked objects + Chinese summary)
        → AttentionManager (cooldowns + importance thresholds → alert/investigate)
        → ProactiveAgent._change_event_loop() (TTS alerts, episodic logging)
```

Key files:

- `askme/perception/change_detector.py`: greedy IoU matching, configurable debounce (3 frames appear, 5 frames disappear)
- `askme/perception/world_state.py`: `TrackedObject` dict, `get_summary_sync()` → "当前场景：1名人（距离2.3米）、2个椅子。"
- `askme/perception/attention_manager.py`: per-event-type cooldowns, `should_alert()` / `should_investigate()`
- `askme/schemas/observation.py`: `Detection`, `Observation` dataclasses
- `askme/schemas/events.py`: `ChangeEvent`, `ChangeEventType` enum

## Terminal UI (TUI)

Full-screen terminal dashboard: `python -m askme` or `askme tui --robot`

Features:
- Header bar: THUNDER name, ESTOP state (red when active), profile, clock
- Context bar: scene summary, mission state, LLM latency
- Left panel: color-coded chat transcript (`[你]` cyan, `[askme]` green, `[系统]` yellow)
- Right panel: component health (colored ● dots), event feed
- `Ctrl+C` triggers ESTOP (not exit), `/estop` command
- CJK-aware display width calculation

## Repository Layout

```text
askme/
  app.py                    # thin facade over runtime assembly
  cli.py                    # CLI command tree (runtime/skills/agent/mcp/tui)
  tui.py                    # full-screen terminal UI with color + ESTOP
  health_server.py          # HTTP health/metrics/capabilities/chat
  mcp_server.py             # MCP tool/resource server
  runtime/
    assembly.py             # DI composition root + component lifecycle
    components.py           # CallableComponent lifecycle abstraction
    profiles.py             # voice/text/mcp/edge_robot profiles
  brain/
    llm_client.py           # LLM with retry/timeout/fallback
    conversation.py         # L1 sliding window, compression
    intent_router.py        # ESTOP → quick_reply → voice_trigger → general
    vision_bridge.py        # ROS2 camera + BPU YOLO + VLM (DashScope)
    episodic_memory.py      # L3 robot experience + reflection
  pipeline/
    brain_pipeline.py       # streaming LLM → TTS orchestration
    prompt_builder.py       # system prompt assembly + seed injection
    tool_executor.py        # LLM tool call execution + approval flow
    skill_dispatcher.py     # mission tracking + multi-step orchestration
    voice_loop.py           # mic → intent → dispatch → TTS
    text_loop.py            # stdin → intent → dispatch → stdout
    proactive_agent.py      # autonomous patrol + anomaly detection
  perception/
    change_detector.py      # YOLO frame diff → debounced events
    world_state.py          # live scene snapshot (tracked objects)
    attention_manager.py    # alert fatigue prevention (cooldowns)
  agent_shell/
    thunder_agent_shell.py  # autonomous task execution (5 iterations, 120s)
  skills/
    builtin/                # SKILL.md definitions (41 skills)
    contracts.py            # code-defined skill contracts + OpenAPI
    skill_manager.py        # merge contracts + markdown metadata
  tools/                    # LLM tool-calling (24 tools)
  voice/                    # ASR/VAD/KWS/TTS/noise reduction
  schemas/                  # Detection, Observation, ChangeEvent
tests/                      # 1431 tests, 0 failures
```

## Test Suite

```bash
python -m pytest tests/ -q    # ~2 min, 1431 tests
```
