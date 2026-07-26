# AskMe Code Map

This file is the working map for changing `askme` without starting from a
blank directory tree. The repository is currently organized by owner package
and compatibility facades, not by the six-layer voice-robot diagram. Use this
map to choose the first file to read before editing.

For a compact folder-by-folder classification, start with `askme/README.md`.

## Product And Architecture Spine

Before choosing an owner package, read `docs/PRODUCT_REQUIREMENTS.md`,
`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`,
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, and
`docs/DEMAND_EVIDENCE_LEDGER.md`. Code ownership follows the PRD and bounded
contexts; this map is navigation, not product authority.

Field Delivery Domain owns customer projects, field events, evidence,
acceptance dossiers, customer signoff, and readiness gaps.
Product/Admin/Platform/Internal surfaces expose Dashboard/API/admin workflows.
Runtime / Safety / Hardware owns real execution, safety state, hardware
adaptation, runtime roundtrip, takeover, and rollback.
customer signoff != production readiness.

## Current Rule

The current architecture is owner-package based:

- `blueprints`: product runtime packages and readiness gates.
- `runtime`: module graph, dependency wiring, task runtime, and handoff.
- `conversation`: authoritative conversation threads, turns, provider
  generations, lifecycle events, and compatibility identity aliases.
- `pipeline`: user turn orchestration, skill routing, and field workflows.
- `voice_gateway`: unified voice middle-layer service and runtime bridge.
- `robot_interaction`: address detection, interaction gate, user intent routing,
  and interaction observability.
- `voice`: audio input/output, ASR/TTS helpers, voice orchestration, diagnostics.
- `ports`: stable application-facing contracts such as `RobotControlPort`.
- `interfaces`: legacy backend ABCs and `interfaces/core/registry.py`.
- `providers`: bottom-layer concrete provider/hardware adapter factories.
- `api`: HTTP surfaces, route composition, request/response schemas.
- `cli`: operator CLI parser, command dispatch, diagnostics, and script-facing
  entrypoints. `askme/cli.py` is only the compatibility facade.
- `robot`: hardware and robot-control adapters.
- `perception`: camera/sensor bridges and environment snapshots.
- `cognition`: world state, planning, working memory, perception sync.
- `memory`: knowledge, evidence, RAG, memory backends.
- `skills`: customer-visible capability packages and governance.
- `tools`: callable tool implementations exposed to agents and runtimes.
- `llm`: model providers, gateway, routing policy, streaming, audit.
- `telemetry`: OTA bridge, runtime metrics, and telemetry upload helpers.
- `contracts`: shared cross-module contracts.
- `schemas`: typed messages and domain/API schemas shared across modules.
- `prompts`: code-level prompt template registry. Runtime prompt assets stay in
  repository-root `prompts/`.
- `compat`: staged migration helpers and legacy import aliases.
- `data`: package-local data parking only. Runtime state and customer data
  belong in repository-root `data/`; do not add Python modules here.

Many root imports are compatibility aliases. New code should import from the
owner subpackage documented in that package's `README.md`.
Confusing compatibility modules are tracked in
`askme.compat.legacy_facades.LEGACY_FACADES`; if an old path is in that list,
it is not a new-code entrypoint.

If the boundary is unclear, read `askme/BOUNDARIES.md` first. It gives the
short rule for `robot_interaction`, `ports`, `providers`, and `robot`.

## Six-Layer Mapping

| Product layer | Current code owners | What belongs here |
| --- | --- | --- |
| 1. Audio preprocessing | `voice/input`, `voice/diagnostics` | mic input, VAD, KWS, noise reduction, device checks, calibration |
| 2. Voice capabilities | `voice/input/asr*`, `voice/output/tts*`, `interfaces/asr.py`, `interfaces/tts.py`, `llm` | ASR, TTS, NLU/intent, translation/provider routing, speech capability metadata |
| 3. Voice Gateway | `conversation`, `voice_gateway`, `runtime/modules/voice_module.py`, `ports/voice.py`, `providers/voice.py`, `providers/voice_runtime.py`, `api/routes/voice.py`, parts of `pipeline/core` | authoritative thread/turn/generation lifecycle, unified ASR/TTS access, context, routing, logging, quality/cost boundaries |
| 4. Robot interaction | `robot_interaction`, `pipeline/channels`, `pipeline/skills`, `api/routes/conversation.py`, `api/routes/field_events.py` | address detection, interaction gate, perception snapshot normalization, natural dialogue, task dialogue, status broadcast, customer/operator interaction |
| 5. Robot execution system | `runtime/core`, `runtime/task`, `runtime/modules`, `tools/robot`, `skills/builtin` | task state machine, handoff, safety preflight, command dispatch, runtime callbacks |
| 6. Providers and hardware | `providers`, `robot`, `perception`, `llm/providers`, `memory/backends` | hardware clients, cloud SDKs, local services, serial bridges, WebRTC/sounddevice/cv2 adapters |

The target direction is that layers 1-5 depend on stable interfaces, while
layer 6 supplies concrete implementations.
`providers` is the lowest application adapter package: it can implement ports
and talk to SDKs or hardware clients, but it must not import product,
runtime, pipeline, API, or interaction layers.

## Start Here By Change Type

| If you want to change... | Start with | Then inspect | Verification |
| --- | --- | --- | --- |
| Product runtime composition | `blueprints/presets/*.py` | `blueprints/catalog/catalog.py`, `runtime/modules` | `python -m askme.cli runtime blueprints --customer-visible --json` |
| Whether a blueprint is customer-ready | `blueprints/catalog/catalog.py` | package README and related tests | `pytest tests/test_blueprints_catalog.py tests/test_blueprint_api_payloads.py -q` |
| API ownership or route placement | `api/composition.py` | `api/README.md`, `api/routes/*`, `api/schemas/*` | route classification and OpenAPI schema tests |
| A Dashboard/customer HTTP endpoint | `api/routes/*` | matching `api/services/*` and `api/schemas/*` | endpoint test plus OpenAPI schema guard |
| Voice input, ASR, VAD, KWS | `ports/voice.py`, `providers/voice.py` | `voice/input`, `voice/orchestration/audio_agent.py`, `voice/core` | voice/input unit tests and voice loop tests |
| TTS or audio output | `ports/voice.py`, `providers/voice.py` | `voice/output`, `voice/orchestration/audio_agent.py` | TTS/audio router tests |
| Addressing, wake/ignore/refuse, interaction gating | `robot_interaction` | `pipeline/channels/voice_loop.py`, `contracts/adapters.py` | interaction gate and voice loop tests |
| Conversation identity, turn settlement, provider generation recovery | `conversation` | `pipeline/core/brain_pipeline.py`, `pipeline/channels/external_turns.py`, `runtime/modules/pipeline_module.py` | `tests/test_voice_turn_ledger.py` and conversation integration tests |
| One user turn flow | `pipeline/core/brain_pipeline.py` | `turn_executor.py`, `stream_processor.py`, `tool_executor.py`, `pipeline/skills/skill_gate.py` | conversation/voice loop tests |
| Skill execution or tool dispatch | `pipeline/skills` | `skills/core`, `tools/core`, `tools/robot` | skill gate/tool tests |
| Runtime module wiring | `runtime/modules/*.py` | `runtime/core/module.py`, relevant blueprint preset | runtime module tests |
| Robot control | `ports/robot_control.py` | `providers/robot_control.py`, `runtime/modules/control_module.py`, `robot/dog/control_client.py`, `tools/robot`, `mcp/server.py` | control client, boundary, and robot tool tests |
| Mechanical arm direct control/defaults | `ports/arm_control.py` | `providers/arm_control.py`, `robot/arm`, `tools/robot/robot_tools.py`, `mcp/server.py`, `mcp/resources/robot_resources.py` | arm/robot tool tests and boundary tests |
| Data bus / robot telemetry contract | `interfaces/bus.py` | `providers/telemetry/bus.py`, `providers/register_defaults.py`, `robot/telemetry`, `runtime/modules/pulse_module.py` | registry, pubsub, messages, and boundary tests |
| Vision/camera bridge and scene intelligence | `ports/perception.py` | `providers/perception.py`, `perception/vision_bridge.py`, `perception/scene_intelligence.py`, `runtime/modules/perception_module.py`, `mcp/server.py` | vision/perception tests and boundary tests |
| Customer field operations | `pipeline/field` | `api/routes/field_*`, `api/services/field_*`, `providers/voice.py` for voice-profile resolution | field operation route and service tests |
| Memory/RAG/knowledge answers | `memory` | `api/routes/memory.py`, `api/services/knowledge_route_payloads.py` | memory and knowledge route tests |

## How A Typical Voice Robot Request Flows

```text
blueprint preset
      -> Runtime.use(...) modules
    -> VoiceModule / TextModule / PipelineModule
      -> AudioFrontendPort or text loop
        -> AddressDetector / InteractionGate
        -> providers.build_audio_frontend()
        -> BrainPipeline
          -> Conversation Core (Thread / Turn / Generation ledger)
          -> PromptBuilder / StreamProcessor / SkillGate / TurnExecutor
            -> tools or runtime handoff
              -> RuntimeHandoffService / robot control service / field event APIs
```

For customer-facing HTTP flows:

```text
health_server.create_health_app
  -> api.composition.register_api_routes
    -> product/admin/internal/platform surface registrar
      -> api.routes.<module>.register_*_routes
        -> api.services.<owner>
          -> pipeline/runtime/memory/skills services
```

## Current Pain Points

- The code still has compatibility facades, so old imports can hide the real
  owner package.
- `memory/core/conversation.py` remains a prompt-context compatibility
  projection during migration; new lifecycle truth belongs to `conversation`.
- Some route modules still use direct `app.*` decorators instead of APIRouter
  factories.
- Some upper layers still reference low-level audio/camera packages through
  legacy facades; `sounddevice` and `cv2` should remain behind provider or
  voice/perception adapter boundaries.
- Voice diagnostics still contain direct concrete imports because diagnostics
  intentionally exercise device-level adapters. MCP server voice/perception
  startup, CLI voice playback, health-server image codec calls, and default
  provider-backed registry setup have been moved behind providers; treat
  diagnostics as an edge validation surface, not the pattern for runtime or
  pipeline code.
- `robot` currently contains both simulated/local hardware code and HTTP
  service clients. Treat it as layer 6 until ports/adapters are introduced.

## Rules Before Editing

- Do not start from the package root. Pick the owner package from the table.
- Do not add new implementation files to a crowded package root. Add or use an
  owner subpackage and update its `README.md`.
- Keep legacy imports working unless the change explicitly removes them.
- Customer/product API routes are not hardware authority.
- Runtime and pipeline should move toward interface ports, not concrete
  hardware/provider imports.
- Hardware, local audio, camera, cloud SDK, and robot-service clients belong at
  the provider/adapter edge.
- Runtime, pipeline, and MCP code should import provider factories from the
  `askme.providers` package facade, not from provider submodules.
- Runtime turn handoff uses `VoiceTurnBridgePort`; construct the HTTP bridge
  with `providers.build_voice_runtime_bridge()` rather than importing
  `VoiceRuntimeBridge` in runtime or pipeline code.
- Runtime and pipeline interaction logic should import `AddressDetector`,
  `InteractionGate`, and `InteractionPerceptionSnapshot` from
  `robot_interaction`, not from legacy `voice.*` facades.

## Target Refactor Track

The clean migration toward the six-layer diagram should happen in this order:

1. Add stable ports under a small owner package, for example:
   `RobotControlPort`, `AudioFrontendPort`, `VisionCapturePort`, `ASRProvider`,
   and `TTSProvider`.
2. Wrap current concrete classes as adapters:
   `DogControlClient`, `ArmController`, `sounddevice`, `cv2`, cloud ASR/TTS.
3. Change `runtime` and `pipeline` constructors to depend on ports instead of
   concrete hardware/provider classes.
4. Add import-boundary tests so upper layers cannot import layer-6 adapters
   directly.
5. Continue directory migration from the new owner packages:
   `voice_gateway`, `robot_interaction`, and `providers`. Create audio
   frontend or voice capability packages only when moving real implementations.

## Minimal Safe First Slice

For hardware decoupling, continue from the existing provider slices:

1. `RobotControlPort` now lives in `ports/robot_control.py`.
2. `providers/robot_control.py` builds the current `DogControlClient` adapter.
3. `BusBackend` now lives in `interfaces/bus.py`, with `providers/telemetry/bus.py`
   building the current Pulse or mock telemetry bus.
4. `SafetyPort`, LED ports, perception ports, and voice ports now live under
   `ports/`.
5. `ArmControlPort` now lives in `ports/arm_control.py`, with
   `providers/arm_control.py` building the current arm adapter.
6. `providers/safety.py`, `providers/led.py`, `providers/perception.py`,
   `providers/voice.py`, and `providers/voice_runtime.py` build the concrete
   adapters, including MCP edge voice I/O.
7. `ControlModule`, `PulseModule`, `SafetyModule`, `LEDModule`, `PerceptionModule`,
   `VoiceModule`, `TextModule`, `PipelineModule`, `BrainPipeline`,
   `SkillGate`, and proactive flows depend on port types.
8. `tests/test_six_layer_package_boundaries.py` guards provider and port import
   boundaries, including the audio frontend boundary.
9. `AddressDetector`, `InteractionGate`, and `InteractionPerceptionSnapshot`
   now live in `robot_interaction`; legacy `voice.*` imports remain only as
   compatibility facades.

That slice is small enough to verify without moving the whole repository.
