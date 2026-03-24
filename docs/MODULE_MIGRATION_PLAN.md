# Module System Migration Plan

> Migrate askme from imperative `assembly.py` to declarative `Module` composition.
> Status: COMPLETE — all 15 modules migrated, assembly.py deleted.
>
> **Remaining gaps**: See [`LAYER_GAPS.md`](./LAYER_GAPS.md) for outstanding
> architectural gaps (port wiring, typed message propagation, BrainPipeline
> decomposition, etc.).

## Current State

| System | Location | Role |
|--------|----------|------|
| Old (production) | `askme/runtime/assembly.py` (722 lines) | `build_runtime()` manually constructs ~30 objects, wires via constructor args |
| Old (planes) | `askme/runtime/planes/{platform,executive,diagnostics}_plane.py` | Wrap services in `CallableComponent` for lifecycle/health |
| Old (profiles) | `askme/runtime/profiles.py` | `RuntimeProfile` with component bundles (`frozenset[str]`) |
| New (framework) | `askme/runtime/module.py` (476 lines) | `Module` base, `In[T]/Out[T]/Required[T]` ports, `Runtime.use()`, auto-wiring, topo sort |
| New (tests) | `tests/test_module_system.py` (552 lines, 35 tests) | Full coverage of wiring, replace, cycles, semantic match, topology |
| Typed messages | `askme/schemas/messages.py` | `EstopState`, `DetectionFrame`, `JointStateSnapshot`, `ImuSnapshot`, `CmsState` |

---

## 1. Migration Phases

### Phase 1: Foundation (no dependencies)

#### PulseModule
- **File**: `askme/runtime/modules/pulse_module.py`
- **Wraps**: `Pulse` (`askme/robot/pulse.py`)
- **Ports**:
  - `detections: Out[DetectionFrame]`
  - `estop: Out[EstopState]`
  - `joints: Out[JointStateSnapshot]`
  - `imu: Out[ImuSnapshot]`
  - `cms_state: Out[CmsState]`
- **depends_on**: `()`
- **provides**: `("telemetry", "dds")`
- **build()**: Reads `cfg["pulse"]`, creates `Pulse(cfg)`, stores as `self.bus`
- **start()**: `await self.bus.start()`
- **stop()**: `await self.bus.stop()`
- **health()**: delegates to `self.bus.health()`
- **How it works**: The `Out` ports on this module are markers. Consumers that declare `In[DetectionFrame]` get wired to this module instance. They then access `self.detections.bus.get_detection_frame()`. Alternatively (cleaner): PulseModule exposes typed accessor methods directly.
- **Profiles**: all (`*`)

#### MemoryModule
- **File**: `askme/runtime/modules/memory_module.py`
- **Wraps**: `ConversationManager`, `SessionMemory`, `EpisodicMemory`, `MemoryBridge`, `MemorySystem`
- **Ports**:
  - `memory_context: Out[MemoryContext]` (new type)
- **depends_on**: `()`
- **provides**: `("conversation", "episodic", "vector_memory", "session_memory")`
- **build()**: Creates `LLMClient` → `SessionMemory` → `ConversationManager` → `MemoryBridge` → `EpisodicMemory` → `MemorySystem`. Stores all as attributes.
- **start()**: Kicks off `memory.warmup()` background task
- **stop()**: Cancels warmup task
- **Profiles**: all (`*`)

#### LLMModule
- **File**: `askme/runtime/modules/llm_module.py`
- **Wraps**: `LLMClient` (`askme/llm/client.py`)
- **Ports**:
  - `llm_client: Out[LLMClient]`
- **depends_on**: `()`
- **provides**: `("llm",)`
- **build()**: Creates `LLMClient(metrics=ota_metrics)`, also creates `OTABridgeMetrics`
- **Note**: LLMClient is a dependency of both MemoryModule and PipelineModule. To avoid duplication, LLMModule should be a separate Phase 1 module, and MemoryModule takes `In[LLMClient]`.
- **Profiles**: all (`*`)

#### ToolsModule
- **File**: `askme/runtime/modules/tools_module.py`
- **Wraps**: `ToolRegistry` + all `register_*_tools()` calls
- **Ports**:
  - `tool_registry: Out[ToolRegistry]`
- **depends_on**: `()`
- **provides**: `("tools",)`
- **build()**: Creates `ToolRegistry`, calls `register_builtin_tools`, `register_vision_tools`, `register_move_tools`, `register_scan_tools`
- **Profiles**: all (`*`)

### Phase 2: Perception + Safety (depend on Phase 1)

#### PerceptionModule
- **File**: `askme/runtime/modules/perception_module.py`
- **Wraps**: `ChangeDetector` (`askme/perception/change_detector.py`), `VisionBridge`
- **Ports**:
  - `detections: In[DetectionFrame]` — wired to PulseModule
  - `vision: Out[VisionBridge]` (new type or use class directly)
  - `change_events: Out[ChangeEvent]` (new type)
- **depends_on**: `("pulse_module",)`
- **provides**: `("vision", "change_monitor")`
- **build()**: Creates `VisionBridge()`, `ChangeDetector(cfg, pulse=self.detections.bus)`
- **start()**: Starts change detector as background task
- **stop()**: Cancels background task
- **Profiles**: `voice`, `edge_robot`

#### SafetyModule
- **File**: `askme/runtime/modules/safety_module.py`
- **Wraps**: `DogSafetyClient` (`askme/robot/safety_client.py`)
- **Ports**:
  - `estop: In[EstopState]` — wired to PulseModule
  - `safety_gate: Out[SafetyGate]` (new type)
- **depends_on**: `("pulse_module",)`
- **provides**: `("dog_safety",)`
- **build()**: Creates `DogSafetyClient(cfg, pulse=self.estop.bus)`
- **Profiles**: `voice`, `text`, `edge_robot`

### Phase 3: Agent Modules (depend on Phase 1 + 2)

#### PipelineModule
- **File**: `askme/runtime/modules/pipeline_module.py`
- **Wraps**: `BrainPipeline` (`askme/pipeline/brain_pipeline.py`)
- **Ports**:
  - `llm_client: Required[LLMClient]`
  - `memory_context: Required[MemoryContext]`
  - `safety_gate: In[SafetyGate]` — optional, None if SafetyModule absent
  - `tool_registry: Required[ToolRegistry]`
  - `pipeline: Out[BrainPipeline]`
- **depends_on**: `("llm_module", "memory_module", "tools_module")`
- **provides**: `("pipeline",)`
- **build()**: Creates `BrainPipeline` with all dependencies from wired ports
- **Profiles**: all (`*`)

#### SkillModule
- **File**: `askme/runtime/modules/skill_module.py`
- **Wraps**: `SkillManager`, `SkillExecutor`, `SkillDispatcher`, `PlannerAgent`
- **Ports**:
  - `pipeline: Required[BrainPipeline]`
  - `llm_client: Required[LLMClient]`
  - `tool_registry: Required[ToolRegistry]`
  - `skill_dispatch: Out[SkillDispatcher]`
- **depends_on**: `("pipeline_module", "llm_module", "tools_module")`
- **provides**: `("skills", "openapi", "mcp_catalog")`
- **build()**: Creates `SkillManager` → `SkillExecutor` → `PlannerAgent` → `SkillDispatcher`
- **Profiles**: all (`*`)

#### ExecutorModule
- **File**: `askme/runtime/modules/executor_module.py`
- **Wraps**: `ThunderAgentShell`
- **Ports**:
  - `llm_client: Required[LLMClient]`
  - `tool_registry: Required[ToolRegistry]`
  - `executor: Out[ThunderAgentShell]`
- **depends_on**: `("llm_module", "tools_module")`
- **provides**: `("executor",)`
- **Profiles**: all (`*`)

### Phase 4: IO Modules

#### VoiceModule
- **File**: `askme/runtime/modules/voice_module.py`
- **Wraps**: `AudioAgent`, `VoiceLoop`, `IntentRouter`, `AudioRouter`, `AddressDetector`
- **Ports**:
  - `pipeline: Required[BrainPipeline]`
  - `skill_dispatch: Required[SkillDispatcher]`
- **depends_on**: `("pipeline_module", "skill_module")`
- **provides**: `("voice", "tts", "asr")`
- **build()**: Creates `AudioAgent(cfg)` → `IntentRouter` → `VoiceLoop`
- **start()**: No-op (VoiceLoop.run() is called by the app entry point)
- **Profiles**: `voice`, `mcp`, `edge_robot`

#### TextModule
- **File**: `askme/runtime/modules/text_module.py`
- **Wraps**: `TextLoop`, `CommandHandler`
- **Ports**:
  - `pipeline: Required[BrainPipeline]`
  - `skill_dispatch: Required[SkillDispatcher]`
- **depends_on**: `("pipeline_module", "skill_module")`
- **provides**: `("text_io",)`
- **build()**: Creates `CommandHandler` → `TextLoop`
- **Profiles**: `text`, `voice` (text fallback), `edge_robot`

### Phase 5: Optional/Robot Modules

#### ControlModule
- **File**: `askme/runtime/modules/control_module.py`
- **Wraps**: `DogControlClient`
- **Ports**:
  - `control: Out[DogControlClient]`
- **depends_on**: `("pulse_module",)`
- **provides**: `("dog_control",)`
- **Profiles**: `mcp`, `edge_robot`

#### LEDModule
- **File**: `askme/runtime/modules/led_module.py`
- **Wraps**: `StateLedBridge`, `LedController`
- **Ports**:
  - `safety_gate: In[SafetyGate]`
  - `skill_dispatch: In[SkillDispatcher]`
- **depends_on**: `("safety_module", "skill_module")`
- **provides**: `("indicators",)`
- **start()**: Starts bridge as background task
- **Profiles**: `edge_robot`

#### ProactiveModule
- **File**: `askme/runtime/modules/proactive_module.py`
- **Wraps**: `ProactiveAgent`
- **Ports**:
  - `pipeline: In[BrainPipeline]`
  - `llm_client: In[LLMClient]`
- **depends_on**: `("pipeline_module", "memory_module")`
- **provides**: `("supervision",)`
- **start()**: Starts `proactive.run(stop_event)` background task
- **Profiles**: `voice`, `text`, `edge_robot`

#### HealthModule
- **File**: `askme/runtime/modules/health_module.py`
- **Wraps**: `AskmeHealthServer`
- **Ports**: none (leaf node, reads from RuntimeApp.health())
- **depends_on**: `()`
- **provides**: `("health_http", "http_chat", "capabilities")`
- **start()**: `await self.server.start()`
- **stop()**: `await self.server.stop()`
- **Profiles**: `voice`, `text`, `edge_robot`

---

## 2. Target Compositions

### VOICE_RUNTIME

```python
VOICE_RUNTIME = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)
```

### TEXT_RUNTIME

```python
TEXT_RUNTIME = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(TextModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)
```

### EDGE_ROBOT_RUNTIME

```python
EDGE_ROBOT_RUNTIME = (
    VOICE_RUNTIME
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
)
```

### MCP_RUNTIME

```python
MCP_RUNTIME = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(ControlModule)
)
```

### Test Override Example

```python
# Swap Pulse for a mock in tests:
TEST_RUNTIME = VOICE_RUNTIME.replace(PulseModule, MockPulseModule)
```

---

## 3. Communication Map

| Data Type | Producer (Out) | Consumer(s) (In/Required) | Transport |
|-----------|---------------|---------------------------|-----------|
| `DetectionFrame` | PulseModule | PerceptionModule | Pulse DDS topic `/thunder/detections` |
| `EstopState` | PulseModule | SafetyModule | Pulse DDS topic `/thunder/estop` |
| `JointStateSnapshot` | PulseModule | (future ControlModule) | Pulse DDS topic `/thunder/joint_states` |
| `ImuSnapshot` | PulseModule | (future) | Pulse DDS topic `/thunder/imu` |
| `CmsState` | PulseModule | (future) | Pulse DDS topic `/thunder/cms_state` |
| `LLMClient` | LLMModule | MemoryModule, PipelineModule, SkillModule, ExecutorModule, ProactiveModule | Direct reference (auto-wired) |
| `ToolRegistry` | ToolsModule | PipelineModule, SkillModule, ExecutorModule | Direct reference |
| `MemoryContext` | MemoryModule | PipelineModule | Direct reference (new type) |
| `SafetyGate` | SafetyModule | PipelineModule, LEDModule | Direct reference (new type) |
| `BrainPipeline` | PipelineModule | SkillModule, VoiceModule, TextModule, ProactiveModule | Direct reference |
| `SkillDispatcher` | SkillModule | VoiceModule, TextModule, LEDModule | Direct reference |
| `ThunderAgentShell` | ExecutorModule | PipelineModule (back-link) | Direct reference |
| `VisionBridge` | PerceptionModule | HealthModule (image archive) | Direct reference |
| `ChangeEvent` | PerceptionModule | ProactiveModule | JSONL file or Pulse topic (future) |

**Transport key**:
- **Pulse DDS topic**: Real-time sensor data flows through `Pulse.on()` callbacks. Modules access via `self.<port>.bus.get_latest()`.
- **Direct reference**: Auto-wired module reference. Consumer calls methods on the producer module directly.
- **JSONL file**: Legacy file-based event passing (ChangeDetector → ProactiveAgent). Future: migrate to Pulse topic.

---

## 4. New Types to Create

Add to `askme/schemas/messages.py`:

```python
@dataclass(frozen=True)
class MemoryContext:
    """Assembled memory context for a single LLM turn."""
    episodic_context: str      # L3 knowledge + digest
    session_context: str       # L2 recent summaries
    vector_context: str        # L4 retrieval result
    timestamp: float = 0.0

@dataclass(frozen=True)
class SafetyGate:
    """Safety gate state — pipeline checks before skill execution."""
    estop_active: bool
    source: str               # "pulse" | "http_cache" | "unknown"
    timestamp: float = 0.0
    configured: bool = False  # True when dog-safety-service URL is set
```

Existing types that are already sufficient (no changes needed):
- `EstopState` — `askme/schemas/messages.py:23`
- `DetectionFrame` — `askme/schemas/messages.py:41`
- `JointStateSnapshot` — `askme/schemas/messages.py:84`
- `ImuSnapshot` — `askme/schemas/messages.py:113`
- `CmsState` — `askme/schemas/messages.py:157`

---

## 5. Migration Strategy

### 5.1 One Module at a Time (Strangler Fig Pattern)

Each module is migrated independently. The old `build_runtime()` and new `Runtime.use()` coexist:

1. **Create the Module class** in `askme/runtime/modules/<name>_module.py`
2. **Write tests** using `MockPulse` / mock dependencies — verify `build()`, `start()`, `stop()`, `health()`, port wiring
3. **Wire into assembly.py** — the Module's `build()` method can receive the same config dict and construct the same objects that `build_runtime()` currently creates
4. **Verify** — run full test suite (`python -m pytest tests/ -q`), confirm 1367+ tests still pass
5. **Remove the old code path** from `build_runtime()` once the Module handles that service

### 5.2 Coexistence Bridge

During migration, `AskmeApp.__init__` (in `askme/app.py:28`) continues calling `build_runtime()`. A thin adapter converts `RuntimeAssembly` → `RuntimeApp` so both old and new consumers work:

```python
# In assembly.py, after build_runtime():
# Optionally build the new Module runtime for migrated modules
if _NEW_MODULES_ENABLED:
    new_rt = Runtime.use(PulseModule) + ...
    new_app = await new_rt.build(cfg)
    # Inject new_app modules into services for migrated paths
```

### 5.3 Testing Strategy

Each module gets a dedicated test file:

```
tests/
  test_module_system.py          # existing (35 tests, framework validation)
  test_pulse_module.py           # Phase 1
  test_memory_module.py          # Phase 1
  test_llm_module.py             # Phase 1
  test_tools_module.py           # Phase 1
  test_perception_module.py      # Phase 2
  test_safety_module.py          # Phase 2
  test_pipeline_module.py        # Phase 3
  test_skill_module.py           # Phase 3
  test_voice_module.py           # Phase 4
  test_text_module.py            # Phase 4
  test_runtime_compositions.py   # Integration: full VOICE/TEXT/EDGE compositions
```

**MockPulse pattern** (already proven in `test_module_system.py:36`):

```python
class MockPulseModule(Module):
    name = "pulse_module"
    detections: Out[DetectionFrame]
    estop: Out[EstopState]

    def build(self, cfg, registry):
        self.bus = MockPubSub()  # in-memory, no DDS
```

### 5.4 Cutover Sequence

1. All module tests pass independently
2. Integration test: `VOICE_RUNTIME` builds and starts with all real modules
3. Smoke test on S100P robot (voice mode, text mode)
4. Delete `askme/runtime/assembly.py` build_runtime()
5. Delete `askme/runtime/planes/` directory
6. Delete `askme/runtime/components.py` (CallableComponent)
7. Update `askme/runtime/__init__.py` to export Module-based API
8. Update `askme/app.py` to use `Runtime.use()` instead of `build_runtime()`

---

## 6. Dependency Graph

```
Phase 1 (foundation, no deps):
  LLMModule ──────────────────────────┐
  ToolsModule ────────────────────────┤
  PulseModule ────────────────────────┤
                                      │
Phase 2 (platform):                   │
  PerceptionModule ← PulseModule      │
  SafetyModule     ← PulseModule      │
                                      │
Phase 1 (continued):                  │
  MemoryModule ← LLMModule            │
                                      │
Phase 3 (agent):                      │
  PipelineModule ← LLMModule + MemoryModule + ToolsModule + SafetyModule
  SkillModule    ← PipelineModule + LLMModule + ToolsModule
  ExecutorModule ← LLMModule + ToolsModule
                                      │
Phase 4 (IO):                         │
  VoiceModule ← PipelineModule + SkillModule
  TextModule  ← PipelineModule + SkillModule
                                      │
Phase 5 (optional):                   │
  ControlModule   ← PulseModule
  LEDModule       ← SafetyModule + SkillModule
  ProactiveModule ← PipelineModule + MemoryModule
  HealthModule    ← (standalone)
```

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **BrainPipeline constructor takes 20 args** | Hardest module to wrap — high coupling | PipelineModule.build() receives all deps from wired ports; no change to BrainPipeline internals. Refactor BrainPipeline constructor later (separate task). |
| **Circular: PipelineModule needs AgentShell, AgentShell needs Tools** | Auto-wiring fails on cycles | ExecutorModule is separate; PipelineModule sets `_agent_shell` post-build via `registry.get("executor_module")`. This mirrors the current `pipeline._agent_shell = agent_shell` pattern at `assembly.py:541`. |
| **DispatchSkillTool needs SkillDispatcher reference** | Late-binding after build | Use `set_dispatcher()` in SkillModule.build() — same pattern as today (`assembly.py:544`). |
| **Health server needs RuntimeAssembly.health_snapshot()** | HealthModule can't access the full RuntimeApp | HealthModule.build() receives a `snapshot_provider` callback, same as today (`assembly.py:661`). |
| **Profile-based module inclusion** | Some modules are conditional (LED, Proactive, ChangeDetector) | Compose per-profile `Runtime` objects (Section 2). `Runtime.without()` removes modules. Profile bundles become `Runtime` constants. |
| **Pulse Out ports are markers, not data channels** | `In[DetectionFrame]` doesn't mean "stream of frames" | Document convention: wired In port gives access to the producer Module, which exposes typed accessor methods. The `Out/In` types are contracts, not message queues. |
| **Assembly.py has side-effects** (sys.path manipulation, file loading, SOUL.md) | Module.build() must be pure-ish | Move side effects to `_load_soul_seed()` and `_init_qp_memory()` helper functions, called from PipelineModule.build() and MemoryModule.build() respectively. |
| **Test count regression** | Breaking existing 1367 tests | Run full suite after each Phase. CI gate: no test count decrease. |

---

## 8. Estimated Effort

| Phase | Modules | Est. Lines | Est. Time |
|-------|---------|------------|-----------|
| Phase 1 | LLMModule, ToolsModule, PulseModule, MemoryModule | ~400 | 1 day |
| Phase 2 | PerceptionModule, SafetyModule | ~200 | 0.5 day |
| Phase 3 | PipelineModule, SkillModule, ExecutorModule | ~500 | 1.5 days |
| Phase 4 | VoiceModule, TextModule | ~300 | 1 day |
| Phase 5 | ControlModule, LEDModule, ProactiveModule, HealthModule | ~300 | 1 day |
| Integration | Composition constants, AskmeApp rewrite, cleanup | ~200 | 1 day |
| **Total** | **15 modules** | **~1900 lines** | **~6 days** |

---

## 9. File Layout (Target)

```
askme/runtime/
  module.py                    # Framework (exists, no changes needed)
  modules/
    __init__.py               # Re-exports all modules
    llm_module.py             # Phase 1
    tools_module.py           # Phase 1
    pulse_module.py           # Phase 1
    memory_module.py          # Phase 1
    perception_module.py      # Phase 2
    safety_module.py          # Phase 2
    pipeline_module.py        # Phase 3
    skill_module.py           # Phase 3
    executor_module.py        # Phase 3
    voice_module.py           # Phase 4
    text_module.py            # Phase 4
    control_module.py         # Phase 5
    led_module.py             # Phase 5
    proactive_module.py       # Phase 5
    health_module.py          # Phase 5
  compositions.py             # VOICE_RUNTIME, TEXT_RUNTIME, etc.
  assembly.py                 # DELETED after cutover
  components.py               # DELETED after cutover
  planes/                     # DELETED after cutover
```

---

## 10. Success Criteria

- [ ] All 15 modules have `build()`, `start()`, `stop()`, `health()` implemented
- [ ] Each module has a dedicated test file with >=5 tests
- [ ] `VOICE_RUNTIME.build(cfg)` produces a working RuntimeApp identical to old `build_runtime()`
- [ ] Full test suite passes (1367+ tests, zero regressions)
- [ ] Smoke test on S100P: voice mode conversation works end-to-end
- [ ] `assembly.py`, `components.py`, `planes/` deleted
- [ ] Adding a new module = one class + one `Runtime.use()` line
- [ ] Swapping implementation = one `Runtime.replace()` call