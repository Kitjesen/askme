# Askme Layer Gap Analysis

> Generated: 2026-03-24
> Purpose: Actionable work plan to close architectural gaps across all askme layers.
> Format: Per-layer audit with file:line references, specific tasks, and priorities.

---

## Layer 1: blueprints/ — Product Definitions

### What's Done
- 3 blueprints: `voice.py`, `text.py`, `edge_robot.py`
- Declarative `Runtime.use()` composition (e.g., `voice.py:28-42`)
- `edge_robot.py:16-20` extends `voice` via `+`, demonstrating composition
- Blueprints serve as `__main__` entry points with `asyncio.run()`

### What's Missing

#### GAP-BP-1: No MCP blueprint (P1)
**File**: `askme/blueprints/` — missing `mcp.py`
**Evidence**: `askme/mcp/server.py:58-136` constructs all objects imperatively
(LLMClient, ConversationManager, MemoryBridge, SkillManager, etc.) — completely
bypasses the Module system. The `MCP_PROFILE` in `profiles.py:128-142` exists
but nothing uses it with `Runtime.use()`.
**Task**: Create `askme/blueprints/mcp.py` that composes modules for MCP mode.
Rewrite `mcp/server.py:app_lifespan()` to build from the blueprint instead of
manual construction. This is the only entry point that still uses imperative
assembly.

#### GAP-BP-2: Duplicated `__main__` boilerplate (P2)
**Files**: `voice.py:46-66`, `text.py:41-60`, `edge_robot.py:24-44`
**Evidence**: All three files copy identical signal-handling + build + start + stop
code (20 lines each, ~95% identical). Only the blueprint variable name differs.
**Task**: Extract a shared `askme/blueprints/_runner.py:run_blueprint(blueprint)`
helper. Each `__main__` block becomes 2 lines.

#### GAP-BP-3: `add_signal_handler` crashes on Windows (P1)
**Files**: `voice.py:58`, `text.py:53`, `edge_robot.py:36`
**Evidence**: `asyncio.get_running_loop().add_signal_handler(sig, stop.set)` —
this API raises `NotImplementedError` on Windows. The dev machine runs Windows 11.
`main.py` does NOT use signal handlers (it relies on KeyboardInterrupt), so this
is only broken in the blueprint `__main__` paths.
**Task**: Wrap in `try/except NotImplementedError` or use `signal.signal()` fallback
on Windows, matching the pattern in `main.py`.

#### GAP-BP-4: No test blueprint (P2)
**Evidence**: `tests/test_all_modules.py:22-73` creates ad-hoc mock modules for
each test. No reusable `test` blueprint with all mocks pre-wired.
**Task**: Create `askme/blueprints/test.py` that composes MockPulseModule,
StubLLMModule, etc. from `tests/test_foundation_modules.py:58-80`. Tests import
the pre-built test runtime instead of re-creating mocks.

---

## Layer 2: interfaces/ — ABC + Registries

### What's Done
- 6 ABC interfaces: `LLMBackend`, `ASRBackend`, `TTSBackend`, `DetectorBackend`,
  `NavigatorBackend`, `BusBackend`
- 6 registries with `BackendRegistry` (`registry.py:38-110`)
- `register_defaults.py` registers 5 implementations (LLM, ASR×2, TTS, Bus×2)
- Soft type check in `registry.py:64` — warns if class doesn't inherit ABC

### What's Missing

#### GAP-IF-1: No existing class inherits its ABC (P1)
**Evidence**:
- `LLMClient` at `askme/llm/client.py:38`: `class LLMClient:` — no parent
- `ASREngine` at `askme/voice/asr.py:14`: `class ASREngine:` — no parent
- `CloudASR` at `askme/voice/cloud_asr.py:31`: `class CloudASR:` — no parent
- `TTSEngine` at `askme/voice/tts.py:27`: `class TTSEngine:` — no parent
- `Pulse` at `askme/robot/pulse.py:68`: inherits `PubSubBase`, NOT `BusBackend`
- `MockPulse` at `askme/robot/mock_pulse.py:14`: inherits `PubSubBase`, NOT `BusBackend`
- Zero classes matching `class.*\(.*Backend\)` in the codebase (grep confirmed)

The TODO comments at `llm_module.py:15` and `pulse_module.py:14` explicitly call
this out: "TODO: migrate to xxx_registry.create(cfg) when X inherits XBackend"

**Tasks** (one per class):
1. `LLMClient(LLMBackend)` — add abstract method stubs, map `chat()`→`chat()`,
   add `model_name` property, `supports_tools() → True`
2. `ASREngine(ASRBackend)` — map `recognize()` signature
3. `CloudASR(ASRBackend)` — same
4. `TTSEngine(TTSBackend)` — map `synthesize()`/`stream()`, add `voice_name` property
5. `Pulse(BusBackend, PubSubBase)` — dual inheritance or make `BusBackend` extend `PubSubBase`
6. `MockPulse(BusBackend, PubSubBase)` — same

**Decision needed**: `BusBackend` and `PubSubBase` have overlapping APIs. Options:
(a) make `BusBackend` extend `PubSubBase` (less code), or (b) keep them separate
and have `Pulse` inherit both (more explicit). Recommend (a).

#### GAP-IF-2: DetectorBackend and NavigatorBackend registries are empty (P2)
**Files**: `register_defaults.py:66-74` — comments say "no implementations yet"
**Evidence**: `detector_registry` default is `"bpu_yolo"` (`detector.py:34`) but
no class is registered. `navigator_registry` default is `"lingtu"` (`navigator.py:38`)
but no class is registered.
**Tasks**:
1. Wrap `ChangeDetector` or create a `BPUYoloDetector(DetectorBackend)` adapter
2. Wrap LingTu gRPC bridge as `LingTuNavigator(NavigatorBackend)`

#### GAP-IF-3: `registry.create(cfg)` incompatible with keyword-only constructors (P1)
**File**: `registry.py:93`: `return cls(cfg)` — passes `cfg` as positional arg
**Evidence**: `LLMClient.__init__` at `client.py:41-48` uses `*, api_key=None, ...`
(keyword-only). `MockPulse.__init__` at `mock_pulse.py:17` takes no args at all.
Calling `llm_registry.create(cfg)` would fail with `TypeError`.
**Task**: Change `registry.py:93` to `return cls(**cfg)` or add a `from_config(cfg)`
classmethod protocol. The `cls(cfg)` convention only works for classes that accept
a single positional dict, which none of the existing implementations do.

#### GAP-IF-4: `register_defaults.py` is manual, not auto-discovery (P2)
**File**: `register_defaults.py:1-74` — explicit imports
**Evidence**: Adding a new backend requires editing this file. No entry_points or
`pkgutil.iter_modules` scanning.
**Task**: Consider `importlib.metadata.entry_points(group="askme.backends")` for
plugin-style discovery. Low priority — the manual approach works for 5 backends.

---

## Layer 3: runtime/module.py — Framework

### What's Done
- `Module` ABC with `build()/start()/stop()/health()/capabilities()`
- `In[T]/Out[T]/Required[T]` port descriptors (`module.py:52-82`)
- Auto-wiring: exact match + semantic match + orphan detection (`module.py:134-224`)
- Topology validation (`module.py:227-249`)
- Topological sort by `depends_on` (`module.py:441-465`)
- `Runtime` class with `use()/+/replace()/without()/build()` (`module.py:321-395`)
- `RuntimeApp` with ordered start/stop and health aggregation (`module.py:398-438`)
- 35 framework tests in `tests/test_module_system.py`

### What's Missing

#### GAP-FW-1: In/Out ports declared but not used for actual wiring (P0)
**Evidence**: Ports are declared in modules (16 total across all modules, see
`pulse_module.py:32-36`, `llm_module.py:27`, etc.) but only ONE module actually
uses its wired In port: `MemoryModule` at `memory_module.py:43` (`llm_client: In[LLMClient]`)
which reads `self.llm_client` at line 48.

All other modules (9 of 10 that have dependencies) bypass ports entirely and use
`registry.get()` + `getattr()`:
- `pipeline_module.py:62-81`: 6 `registry.get()` calls + 11 `getattr()` calls
- `voice_module.py:39-53`: 5 `registry.get()` calls + 6 `getattr()` calls
- `text_module.py:37-52`: 5 `registry.get()` calls + 8 `getattr()` calls
- `skill_module.py:36-44`: 3 `registry.get()` calls + 4 `getattr()` calls
- `proactive_module.py:33-46`: 5 `registry.get()` calls + 5 `getattr()` calls

Total: **32 `registry.get()` calls** and **44 `getattr()` calls** across modules.

This means the auto-wiring system is effectively dead code in production. The `Out`
ports serve only as documentation markers.

**Task**: Migrate modules one-by-one to use `In[T]`/`Required[T]` ports instead of
`registry.get()`. Start with leaf modules (SafetyModule, ControlModule) and work
inward. Each migration removes `registry.get()` calls and replaces them with typed
port references. The biggest challenge is `PipelineModule` which needs 6 different
module references.

**Trade-off**: The `registry.get()` pattern is simpler when a module needs access
to specific *attributes* of another module (e.g., `llm_mod.client`), not the module
itself. Wiring gives you the module instance, and you still need `getattr()` to
reach the inner object. A middle ground: add typed accessor properties to modules
(e.g., `LLMModule.llm_client -> LLMClient`) so port consumers get clean API.

#### GAP-FW-2: Hot-swap not implemented (P2)
**File**: `module.py` — no `RuntimeApp.replace()` or `RuntimeApp.hot_swap()` method
**Evidence**: `Runtime.replace()` at `module.py:346-352` works pre-build only.
Once `RuntimeApp` is built, you cannot swap a module at runtime.
**Task**: Add `RuntimeApp.hot_swap(old_name, new_module)` that:
1. Calls `old.stop()`
2. Re-wires In ports pointing to old module
3. Calls `new.build(cfg, registry)` + `new.start()`
Low priority — not needed until hardware module hot-plugging is required.

#### GAP-FW-3: Flow monitoring not implemented (P2)
**Evidence**: The `WireResult` at `module.py:125-131` tracks wired ports statically
at build time. There is no runtime monitoring of data flow through ports (e.g.,
message counts, latency between Out→In).
**Task**: Add optional `FlowMonitor` that wraps Out ports with counters. Expose
via `RuntimeApp.flow_stats()` for the health endpoint.

---

## Layer 4: runtime/modules/ — 15 Modules

### What's Done
- All 15 modules implemented and exported (`__init__.py:1-40`)
- Each module has `build()/start()/stop()/health()`
- Composition tests exist (`test_all_modules.py`, `test_foundation_modules.py`)

### What's Missing

#### GAP-MOD-1: Modules not self-contained — leak internals via getattr (P1)
**Evidence**: Consumers access internal attributes directly:
- `pipeline_module.py:64`: `getattr(llm_mod, "client", None)` — reaches into LLMModule
- `voice_module.py:53`: `getattr(executor_mod, "shell", None)` — reaches into ExecutorModule
- `skill_module.py:65`: `pipeline._skill_manager = self.skill_manager` — mutates
  PipelineModule's internal `BrainPipeline` instance

This makes module boundaries porous. Renaming an internal attribute in one module
breaks another module silently.

**Task**: Add typed accessor methods/properties to each module. Example:
```python
class LLMModule(Module):
    @property
    def llm_client(self) -> LLMClient: return self.client
    @property
    def metrics(self) -> OTABridgeMetrics: return self.ota_metrics
```

#### GAP-MOD-2: Post-build mutation pattern is fragile (P1)
**Evidence**: Multiple modules mutate other modules after build:
- `skill_module.py:64-66`: `pipeline._skill_manager = ...`, `pipeline._skill_executor = ...`
- `voice_module.py:72-77`: `pipeline._audio = ...`, `agent_shell._audio = ...`,
  `dispatcher._audio = ...`
- `executor_module.py:51-52`: `pipeline._agent_shell = self.shell`

These back-patches depend on build order and bypass the port system entirely.
**Task**: Use a two-phase init pattern: `build()` creates objects, then a
`wire(registry)` phase connects cross-references. Or use `Required[T]` ports
properly so the framework handles these connections.

#### GAP-MOD-3: HealthModule has no-op health provider (P1)
**File**: `health_module.py:22-24`: `_noop_health_provider` returns static `{"status": "ok"}`
**Evidence**: Line 41: `snapshot_provider=_noop_health_provider`. The real module-level
health is never wired into the HTTP health endpoint. The `build_health_snapshot()`
function in `health_server.py` is called separately by `tui.py:538` but not by the
HealthModule itself.
**Task**: Wire `RuntimeApp.health()` as the snapshot provider in HealthModule.build()
or in a post-build callback.

---

## Layer 5: robot/ — Hardware Abstraction

### What's Done
- `PubSubBase` ABC with typed convenience methods (`pubsub.py:47-189`)
- `Pulse` CycloneDDS implementation (`pulse.py:68-212`)
- `MockPulse` in-memory test double (`mock_pulse.py:14-59`)
- Per-topic tracking with staleness/rate metrics (`pubsub.py:77-123`)
- 5 typed message classes in `schemas/messages.py`
- Both registered in `bus_registry` (`register_defaults.py:57-63`)

### What's Missing

#### GAP-RB-1: Pulse publish whitelist removed in CycloneDDS migration (P1)
**Evidence**: The old rclpy Pulse had a publish whitelist (per git history: commit
`495d0b8` "rename: ThunderBus -> Pulse"). The current `pulse.py:193-200` creates
writers on-demand for ANY topic string — no validation.
**Task**: Add a `_PUBLISH_WHITELIST` set to `pulse.py`. `publish()` should reject
topics not in the whitelist. The subscribe side already has `_TOPIC_CONFIG` (`pulse.py:58-65`)
which acts as a read whitelist.

#### GAP-RB-2: `MockPulse.__init__` signature incompatible with `bus_registry.create()` (P1)
**File**: `mock_pulse.py:17`: `def __init__(self) -> None:` — takes no args
**File**: `registry.py:93`: `return cls(cfg)` — passes cfg dict as first arg
**Evidence**: `bus_registry.create({"backend": "mock"})` would fail with `TypeError:
MockPulse.__init__() takes 1 positional argument but 2 were given`
**Task**: Change `MockPulse.__init__` to accept optional `cfg: dict | None = None`.

#### GAP-RB-3: Stale rclpy references in comments (P2)
**Files**:
- `pulse.py:71`: "Same API as Pulse (rclpy version)"
- `bus.py:14`: "Implementations: Pulse (rclpy DDS)"
- `pubsub.py`: comments reference rclpy patterns
**Task**: Update all comments to say "CycloneDDS" instead of "rclpy".

---

## Layer 6: schemas/ — Message Types

### What's Done
- 5 frozen dataclasses: `EstopState`, `DetectionFrame`, `JointStateSnapshot`,
  `ImuSnapshot`, `CmsState` (`messages.py:22-178`)
- All have `from_dict()` / `to_dict()` methods
- `Detection` and `Observation` in `observation.py`
- `ChangeEvent` and `ChangeEventType` in `events.py`
- PubSubBase typed convenience methods use them (`pubsub.py:156-189`)

### What's Missing

#### GAP-SC-1: Core pipeline/voice/memory layers never use typed messages (P1)
**Evidence**: Grep for `EstopState|DetectionFrame|JointStateSnapshot|ImuSnapshot|CmsState`
in `askme/pipeline/`, `askme/voice/`, `askme/memory/` returns **zero results**.

The typed messages are only used in:
- `askme/robot/pubsub.py` (convenience methods)
- `askme/runtime/modules/pulse_module.py` (Out port declarations)
- `askme/schemas/messages.py` (definitions)
- `tests/` (test assertions)

`brain_pipeline.py` (1093 lines) reads safety state via `DogSafetyClient` which
internally calls `pulse.get_latest("/thunder/estop")` and returns raw dicts. The
pipeline never sees `EstopState` or `DetectionFrame` objects.

**Task**: Have `DogSafetyClient` return `EstopState` instead of raw dicts. Have
`ChangeDetector` consume `DetectionFrame` instead of raw JSON. Propagate typed
messages through the pipeline layer.

#### GAP-SC-2: Missing message types (P2)
- No `NavigationState` message for nav task status updates
- No `AudioState` message for voice pipeline state (IDLE/LISTENING/PROCESSING/SPEAKING)
- No `MissionState` message for mission lifecycle events
**Task**: Add these as frozen dataclasses in `schemas/messages.py` when the
corresponding Pulse topics are defined.

#### GAP-SC-3: `MemoryContext` is a marker class, not a real type (P2)
**File**: `memory_module.py:32-34`: `class MemoryContext: pass`
**Evidence**: Used only as an Out port type marker. No fields, no `from_dict()`.
The `MODULE_MIGRATION_PLAN.md:317-321` defines what it should contain:
`episodic_context`, `session_context`, `vector_context`, `timestamp`.
**Task**: Implement the full `MemoryContext` dataclass per the migration plan spec.

---

## Layer 7: pipeline/ + memory/ + voice/ — Core Business Logic

### What's Done
- These layers are the original pre-migration code and they work
- BrainPipeline handles LLM streaming, tool calling, TTS, memory prefetch
- VoiceLoop handles the full mic→VAD→ASR→intent→dispatch→TTS cycle
- SkillDispatcher manages multi-step missions
- 4-layer memory stack is operational

### What's Missing

#### GAP-CORE-1: These layers were never refactored for the Module system (P1)
**Evidence**: `BrainPipeline.__init__` still takes 20+ constructor arguments
(`pipeline_module.py:96-123` enumerates them all). The Module system wraps these
objects but does not simplify their interfaces.

Key coupling points:
- `BrainPipeline._audio` is mutated post-build by VoiceModule (`voice_module.py:73`)
- `BrainPipeline._skill_manager` is mutated post-build by SkillModule (`skill_module.py:65`)
- `BrainPipeline._skill_executor` is mutated post-build by SkillModule (`skill_module.py:66`)
- `BrainPipeline._agent_shell` is mutated post-build by ExecutorModule (`executor_module.py:52`)

**Task**: Decompose BrainPipeline into smaller classes:
1. `PromptAssembler` — system prompt + L0 block + seed (already partially in `prompt_builder.py`)
2. `LLMTurnExecutor` — single turn: messages → LLM → tool calls → response
3. `SkillGate` — estop check + safety level filter before skill execution
4. `StreamingTTSBridge` — LLM stream → sentence splitter → TTS

This is the largest refactoring task and should be done after the port wiring
is cleaned up.

#### GAP-CORE-2: `brain_pipeline.py` is 1093 lines (P2)
**Evidence**: Per MEMORY.md: "brain_pipeline.py — 1093 lines, decomposition pending"
**Task**: After GAP-CORE-1 decomposition, the file should be <400 lines.

---

## Layer 8: main.py + cli.py + tui.py — Entry Points

### What's Done
- `__main__.py`: clean 7-line entry point
- `cli.py`: full-featured CLI with subcommands (runtime, skills, agent, mcp, tui)
- `main.py`: `run_legacy_app()` selects blueprint and runs
- `tui.py`: full-screen terminal UI with 2Hz refresh, ESTOP, status panel

### What's Missing

#### GAP-EP-1: `main.py` still named "legacy" (P2)
**File**: `main.py:44`: `async def run_legacy_app`
**Evidence**: The function name suggests it's temporary, but it's the production
entry point used by `cli.py:399-401`. The blueprint-based build IS the new system.
**Task**: Rename to `run_app()`. Update `cli.py:399` import.

#### GAP-EP-2: `_assembly_legacy.py` is dead code (P1)
**File**: `askme/runtime/_assembly_legacy.py`
**Evidence**: No imports of `assembly` found in the codebase (grep confirmed).
It imports `components.py:40` which is the only remaining consumer of
`RuntimeComponent` and `resolve_start_order`.
**Task**: Delete `_assembly_legacy.py`. Then check if `components.py` has other
consumers; if not, delete it too.

#### GAP-EP-3: `components.py` — check for remaining consumers (P1)
**File**: `askme/runtime/components.py`
**Evidence**: Only imported by `_assembly_legacy.py:40`. The `CallableComponent`
and `RuntimeComponent` classes are from the pre-Module era. The Module system has
its own `Module` ABC.
**Task**: After deleting `_assembly_legacy.py`, delete `components.py` if unused.

#### GAP-EP-4: `profiles.py` has legacy aliases (P2)
**File**: `profiles.py:177-186`: `RuntimeMode = RuntimeProfile`, `VOICE_MODE = VOICE_PROFILE`, etc.
**Evidence**: These aliases exist for backward compatibility. `tui.py:22` imports
`RuntimeApp` from `module.py`, not profiles. `cli.py:430-435` uses `legacy_profile_for()`.
**Task**: Grep for `RuntimeMode|VOICE_MODE|TEXT_MODE|MCP_MODE|EDGE_ROBOT_MODE`
consumers. If none, remove the aliases.

#### GAP-EP-5: TUI duplicates capability/health logic from cli.py (P2)
**Files**: `tui.py:549-594` and `cli.py:425-480` have nearly identical
`_capabilities_snapshot()` / `_load_local_capabilities_async()` functions.
**Task**: Extract shared `build_capabilities_snapshot(app, profile)` function.

---

## Layer 9: docs/ — Documentation

### What's Done
- `ARCHITECTURE.md`: comprehensive 340-line doc with diagrams
- `MODULE_MIGRATION_PLAN.md`: detailed 514-line migration plan
- `PRODUCTION_GAPS.md`: P0-P2 production readiness gaps

### What's Missing

#### GAP-DOC-1: ARCHITECTURE.md says "rclpy" 11 times — should say CycloneDDS (P0)
**File**: `docs/ARCHITECTURE.md`
**Evidence**: Lines 13, 21, 31, 65, 70, 144, 209, 212, 224 reference rclpy.
The actual Pulse implementation at `pulse.py:26-39` imports `cyclonedds`, not rclpy.
Commit `092a2d2` ("feat: ThunderBus — CycloneDDS") made this change.

Specific fixes needed:
- Line 13: "in-process rclpy node" → "CycloneDDS direct (no ROS2)"
- Line 21: "Python 3.10, rclpy" → "Python 3.10" (frame_daemon may still use rclpy, verify)
- Line 31: "Pulse (in-process rclpy)" → "Pulse (CycloneDDS)"
- Line 65: "in-process rclpy node" → "CycloneDDS reader"
- Line 70: "rclpy node in background thread" → "CycloneDDS poll thread"
- Line 144: "Pulse: in-process DDS via rclpy" → "Pulse: CycloneDDS direct"
- Section 4 title: "In-process DDS via rclpy" → "In-process DDS via CycloneDDS"
- Lines 212, 224: "rclpy" → "CycloneDDS"
- System diagram: update the askme box label

#### GAP-DOC-2: MODULE_MIGRATION_PLAN.md is outdated (P1)
**File**: `docs/MODULE_MIGRATION_PLAN.md`
**Evidence**: Line 4 says "Status: PLAN — not yet started." But ALL 15 modules
are implemented, all compositions exist in `blueprints/`, and `assembly.py` is
deleted. The plan's success criteria (`line 507-514`) should be checked off.

Stale sections:
- Section 5.2 "Coexistence Bridge" — no longer needed, assembly.py is gone
- Section 5.4 "Cutover Sequence" step 4-7 — partially done (assembly deleted, planes deleted,
  but `components.py` still exists)
- Section 10 success criteria — most are met

**Task**: Update status to "COMPLETE (with gaps)" and add a "Remaining Gaps" section
pointing to this document. Or archive it and make this document the new canonical plan.

#### GAP-DOC-3: ARCHITECTURE.md module structure listing is stale (P1)
**File**: `ARCHITECTURE.md:119-203`
**Evidence**: Lists `runtime/assembly.py` (deleted), `runtime/components.py` (dead code),
`runtime/profiles.py` (exists but partially legacy). Does not list
`runtime/modules/` directory with 15 module files. Does not list `blueprints/`.
**Task**: Update the module tree to reflect current state.

---

## Priority Summary

### P0 — Must fix (correctness/safety)
| ID | Task | File(s) | Effort |
|----|------|---------|--------|
| GAP-DOC-1 | Fix 11 stale rclpy references in ARCHITECTURE.md | `docs/ARCHITECTURE.md` | 30min |
| GAP-FW-1 | Begin migrating modules from registry.get() to In/Out ports | `askme/runtime/modules/*.py` | 3 days |

### P1 — Should fix (architecture integrity)
| ID | Task | File(s) | Effort |
|----|------|---------|--------|
| GAP-IF-1 | Make 6 classes inherit their ABC interfaces | `llm/client.py`, `voice/asr.py`, `voice/tts.py`, `robot/pulse.py`, `robot/mock_pulse.py`, `voice/cloud_asr.py` | 1 day |
| GAP-IF-3 | Fix `registry.create(cfg)` for keyword-only constructors | `runtime/registry.py:93` | 1hr |
| GAP-BP-1 | Create MCP blueprint, rewrite mcp/server.py | `blueprints/mcp.py`, `mcp/server.py` | 1 day |
| GAP-BP-3 | Fix Windows signal handler crash in blueprints | `blueprints/voice.py:58`, `text.py:53`, `edge_robot.py:36` | 30min |
| GAP-MOD-1 | Add typed accessor properties to modules | `askme/runtime/modules/*.py` | 0.5 day |
| GAP-MOD-2 | Replace post-build mutation with wire() phase | `askme/runtime/modules/*.py` | 1 day |
| GAP-MOD-3 | Wire real health provider into HealthModule | `health_module.py` | 1hr |
| GAP-RB-1 | Add publish whitelist to Pulse | `robot/pulse.py` | 1hr |
| GAP-RB-2 | Fix MockPulse constructor for registry.create() | `robot/mock_pulse.py` | 15min |
| GAP-SC-1 | Propagate typed messages through pipeline layer | `pipeline/brain_pipeline.py`, `robot/safety_client.py` | 1 day |
| GAP-EP-2 | Delete dead `_assembly_legacy.py` | `runtime/_assembly_legacy.py` | 15min |
| GAP-EP-3 | Delete dead `components.py` | `runtime/components.py` | 15min |
| GAP-DOC-2 | Update MODULE_MIGRATION_PLAN.md status | `docs/MODULE_MIGRATION_PLAN.md` | 30min |
| GAP-DOC-3 | Update ARCHITECTURE.md module tree | `docs/ARCHITECTURE.md` | 1hr |
| GAP-CORE-1 | Decompose BrainPipeline (design phase) | `pipeline/brain_pipeline.py` | Design: 1hr, Impl: 3 days |

### P2 — Nice to have (polish/completeness)
| ID | Task | File(s) | Effort |
|----|------|---------|--------|
| GAP-BP-2 | Extract shared blueprint runner | `blueprints/_runner.py` | 30min |
| GAP-BP-4 | Create test blueprint with pre-wired mocks | `blueprints/test.py` | 1hr |
| GAP-IF-2 | Implement Detector and Navigator backends | `register_defaults.py` | 2 days |
| GAP-IF-4 | Auto-discovery for backend registration | `register_defaults.py` | 2hr |
| GAP-FW-2 | Hot-swap support in RuntimeApp | `runtime/module.py` | 1 day |
| GAP-FW-3 | Flow monitoring for wired ports | `runtime/module.py` | 1 day |
| GAP-RB-3 | Fix stale rclpy comments in robot/ | `robot/pulse.py`, `interfaces/bus.py` | 15min |
| GAP-SC-2 | Add NavigationState, AudioState, MissionState messages | `schemas/messages.py` | 1hr |
| GAP-SC-3 | Implement full MemoryContext dataclass | `memory_module.py` | 30min |
| GAP-CORE-2 | Reduce brain_pipeline.py below 400 lines | `pipeline/brain_pipeline.py` | 2 days |
| GAP-EP-1 | Rename run_legacy_app → run_app | `main.py`, `cli.py` | 15min |
| GAP-EP-4 | Remove legacy profile aliases | `runtime/profiles.py` | 15min |
| GAP-EP-5 | Extract shared capabilities builder | `tui.py`, `cli.py` | 30min |

---

## Recommended Execution Order

1. **Quick wins (1 day)**: GAP-DOC-1, GAP-EP-2, GAP-EP-3, GAP-RB-2, GAP-RB-3,
   GAP-BP-3, GAP-EP-1, GAP-MOD-3, GAP-IF-3
2. **Interface alignment (2 days)**: GAP-IF-1, GAP-SC-1, GAP-MOD-1
3. **Port wiring (3 days)**: GAP-FW-1, GAP-MOD-2
4. **MCP blueprint (1 day)**: GAP-BP-1
5. **Documentation (1 day)**: GAP-DOC-2, GAP-DOC-3
6. **Pipeline decomposition (3 days)**: GAP-CORE-1, GAP-CORE-2

Total estimated effort: ~12 working days for P0+P1, ~17 days including P2.