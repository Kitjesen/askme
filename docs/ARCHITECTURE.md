# Askme Architecture

Reference document for the multi-process runtime on S100P and the communication
patterns between processes. See [README.md](../README.md) for quick-start and
module layout.

---

## 1. System Overview

Five processes run on the S100P robot. Each owns a distinct concern; none share
memory space.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  S100P  (aarch64, Ubuntu 22.04, ROS2 Humble, Nash BPU 128 TOPS)        │
│                                                                          │
│  ┌──────────────────┐   /camera/color     ┌──────────────────────────┐  │
│  │  LingTu          │   /camera/depth     │  frame_daemon            │  │
│  │  (ROS2 native)   │◄────────────────────│  Python 3.10, rclpy      │  │
│  │  Navigation+SLAM │                     │  Camera sub + BPU YOLO   │  │
│  │  306 packages    │                     │  5Hz, publishes:         │  │
│  │  gRPC server     │                     │  /thunder/detections     │  │
│  └──────┬───────────┘                     │  /thunder/heartbeat      │  │
│         │ gRPC                            └──────────┬───────────────┘  │
│         ▼                                            │ DDS              │
│  ┌──────────────────┐   DDS topics        ┌──────────▼───────────────┐  │
│  │  nav-gateway     │                     │  askme_dds_bridge        │  │
│  │  (REST :8088)    │                     │  Python 3.10, rclpy      │  │
│  │  Navigation      │                     │  Subscribes DDS topics   │  │
│  │  contract layer  │                     │  → Unix socket           │  │
│  └──────────────────┘                     │  /tmp/askme_dds_bridge   │  │
│                                           │  .sock (NDJSON)          │  │
│  ┌──────────────────┐                     └──────────┬───────────────┘  │
│  │  brainstem       │                                │ Unix socket      │
│  │  (Dart)          │                                │                  │
│  │  Motor ctrl      │◄───────────────────────────────┤                  │
│  │  CAN bus         │   gRPC :13145                  │                  │
│  │  gRPC server     │                     ┌──────────▼───────────────┐  │
│  └──────────────────┘                     │  askme                   │  │
│                                           │  Python 3.13, asyncio    │  │
│  ┌──────────────────┐                     │  AI assistant runtime    │  │
│  │  dog-safety-svc  │◄────────────────────│  Health HTTP :8765       │  │
│  │  (REST :5070)    │   HTTP REST         │  MCP server              │  │
│  └──────────────────┘                     └──────────────────────────┘  │
│  ┌──────────────────┐                                │                  │
│  │  dog-control-svc │◄───────────────────────────────┘                  │
│  │  (REST :5080)    │   HTTP REST                                        │
│  └──────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                               HTTPS/WSS
                                    │
                    ┌───────────────▼──────────────┐
                    │  Cloud / Relay               │
                    │  LLM relay (OpenAI-compat)   │
                    │  OTA platform                │
                    └──────────────────────────────┘
```

---

## 2. Communication Architecture

### Data Plane — CycloneDDS topics → Unix socket

High-frequency sensor and safety data flows over DDS and is forwarded to askme
via a Unix domain socket. This keeps Python 3.13 (askme) and Python 3.10 (rclpy)
in separate processes.

```
DDS publisher            Bridge process              askme (consumer)
(frame_daemon /          (askme_dds_bridge)          (Python 3.13 asyncio)
 brainstem)
     │                         │                           │
     │  DDS QoS BEST_EFFORT    │                           │
     │────────────────────────►│                           │
     │                         │  Unix socket NDJSON       │
     │                         │──────────────────────────►│
     │                         │  {"topic":"/thunder/…",   │
     │                         │   "data":{…}, "ts":…}     │
```

**Wire format** — newline-delimited JSON per message:
```json
{"topic": "/thunder/detections", "data": {"detections": [...], "frame_id": 42}, "ts": 1234567890.123}
{"topic": "/thunder/estop",      "data": {"active": false},                      "ts": 1234567890.456}
```

**Socket path**: `/tmp/askme_dds_bridge.sock`

**Topics implemented** (PoC, 2026-03):

| Topic | Msg type | Hz | QoS | Publisher | Status |
|---|---|---|---|---|---|
| `/thunder/detections` | `std_msgs/String` (JSON) | 5 | BEST_EFFORT / VOLATILE / KEEP_LAST(1) | frame_daemon | Live |
| `/thunder/estop` | `std_msgs/Bool` | event | RELIABLE / TRANSIENT_LOCAL / KEEP_LAST(1) | brainstem safety bridge | Live |
| `/thunder/heartbeat` | `std_msgs/Bool` | 5 | BEST_EFFORT / VOLATILE / KEEP_LAST(1) | frame_daemon | Live |

**Topics planned** (not yet published):

| Topic | Msg type | Hz | QoS | Publisher | Status |
|---|---|---|---|---|---|
| `/thunder/joint_states` | `sensor_msgs/JointState` | 50 | BEST_EFFORT / VOLATILE / KEEP_LAST(1) | brainstem | Planned |
| `/thunder/imu` | `sensor_msgs/Imu` | 100 | BEST_EFFORT / VOLATILE / KEEP_LAST(1) | brainstem | Planned |
| `/thunder/cms_state` | `std_msgs/String` (JSON) | 2 | RELIABLE / TRANSIENT_LOCAL / KEEP_LAST(1) | brainstem | Planned |

### Control Plane — gRPC + REST

Low-frequency management and command dispatch. All paths are optional; missing
`*_URL` env vars disable the corresponding client silently.

| Service | Protocol | Default port | Env var | Owned by | Purpose |
|---|---|---|---|---|---|
| brainstem | gRPC | 13145 | — | brainstem | Motor commands, FSM control, IMU/joint stream |
| LingTu | gRPC | configured in LingTu | `LINGTU_GRPC_ADDR` | LingTu | Navigation tasks, mapping, follow-person |
| nav-gateway | REST | 8088 | `NAV_GATEWAY_URL` | nav-gateway | Navigation contract layer over LingTu gRPC |
| dog-safety-service | REST | 5070 | `DOG_SAFETY_SERVICE_URL` | NOVA Dog runtime | E-STOP gate: read state + notify |
| dog-control-service | REST | 5080 | `DOG_CONTROL_SERVICE_URL` | NOVA Dog runtime | Capability dispatch (stand, sit, patrol) |
| askme health | HTTP | 8765 | — | askme | Health, metrics, capabilities, web chat |
| OTA platform | HTTPS | configured | `OTA_*` | infra/ota | Firmware + app update delivery |
| LLM relay | HTTPS | 443 | `LLM_BASE_URL` | cloud | OpenAI-compatible Claude relay |

**E-STOP path** — designed to never block the local stop:
```
Voice "停！" → IntentRouter (tier-1, zero LLM) → DogSafetyClient.notify_estop()
                                                    └─ daemon thread, fire-and-forget
                                                       POST /api/v1/safety/modes/estop
                                                       connect_timeout=0.3s, read_timeout=0.5s
```

---

## 3. Module Structure

```
askme/
├── llm/
│   ├── client.py          — LLM HTTP client, retry/timeout/fallback chain
│   ├── conversation.py    — L1 sliding window (40 msg), compression
│   └── intent_router.py   — ESTOP (tier-0) → quick_reply → voice_trigger → general
│
├── memory/
│   ├── bridge.py          — MemoryBridge: L4 MemU vector DB read+write
│   ├── episodic_memory.py — L3 robot experience + reflection + knowledge
│   ├── session.py         — L2 session summary .md files
│   ├── system.py          — MemorySystem: unified L1–L4 access
│   ├── procedural.py      — skill/procedure knowledge
│   └── site_knowledge.py  — site-specific facts
│
├── perception/
│   ├── change_detector.py — IoU matching + N-frame debounce → ChangeEvents
│   ├── world_state.py     — TrackedObject dict, get_summary_sync()
│   ├── attention_manager.py — per-type cooldowns, should_alert()
│   └── vision_bridge.py   — reads /tmp/askme_frame_*.bin|json from frame_daemon
│
├── robot/
│   ├── safety_client.py   — DogSafetyClient: E-STOP notify + state query (30s TTL cache)
│   ├── control_client.py  — DogControlClient: capability dispatch
│   ├── ota_bridge.py      — OTA platform registration + telemetry
│   ├── led_controller.py  — HttpLedController / NullLedController
│   └── state_led_bridge.py — StateLedBridge: FSM state → LED color
│
├── pipeline/
│   ├── brain_pipeline.py  — streaming LLM → TTS, L0 runtime block, estop gate
│   ├── prompt_builder.py  — system prompt assembly + fake turn seed
│   ├── tool_executor.py   — LLM tool call execution + approval flow
│   ├── skill_dispatcher.py — MissionContext lifecycle, multi-step orchestration
│   ├── planner_agent.py   — LLM decomposes goals into ordered PlanStep list
│   ├── voice_loop.py      — mic → VAD → ASR → intent → dispatch → TTS
│   ├── text_loop.py       — stdin → intent → dispatch → stdout
│   ├── proactive_agent.py — autonomous patrol + ChangeEvent → TTS alerts
│   ├── frames.py          — typed Frame hierarchy (Pipecat-inspired)
│   └── trace.py           — span-based pipeline timing instrumentation
│
├── voice/
│   ├── audio_agent.py     — AudioAgent: mic input, VAD, noise gate, AgentState enum
│   ├── asr.py / asr_manager.py — ASR (sherpa-onnx local + cloud fallback)
│   ├── tts.py             — TTS: MiniMax SSE → sherpa-onnx → edge-tts fallback chain
│   ├── vad.py / vad_controller.py — WebRTC VAD + 30s max-speech guard
│   ├── kws.py             — keyword wake-word spotting
│   └── noise_reduction.py — SpectralSubtractor (FFT-based, optional)
│
├── skills/
│   ├── builtin/           — 41 SKILL.md definitions
│   ├── contracts.py       — SkillContract + @skill_contract decorator
│   ├── contracts_builtin.py — authoritative contracts for core built-ins
│   ├── skill_manager.py   — merge code contracts with markdown metadata
│   └── skill_executor.py  — executes a skill by name
│
├── tools/                 — 24 LLM tool-calling implementations
│   ├── builtin_tools.py   — DispatchSkillTool, SpeakProgressTool
│   ├── robot_api_tool.py  — RobotApiTool: 7 runtime service endpoints
│   ├── move_tool.py       — robot motion commands
│   ├── scan_tool.py       — environment scan
│   ├── vision_tool.py     — vision query
│   └── tool_registry.py   — ToolRegistry: name → callable
│
├── mcp/
│   ├── server.py          — MCP tool/resource server
│   └── tools/             — MCP-exposed tool implementations
│
├── agent_shell/
│   └── thunder_agent_shell.py — ThunderAgentShell: autonomous execution (5 iter, 120s)
│
├── runtime/
│   ├── assembly.py        — DI composition root, component lifecycle orchestration
│   ├── components.py      — CallableComponent: start/stop/health/capabilities
│   └── profiles.py        — voice / text / mcp / edge_robot profiles
│
└── schemas/
    ├── observation.py     — Detection, Observation dataclasses
    └── events.py          — ChangeEvent, ChangeEventType enum
```

---

## 4. DDS Bridge Architecture

### Why a separate process

askme runs Python 3.13 (asyncio). ROS2 Humble's `rclpy` requires Python 3.10 and
installs its own event loop executor that conflicts with asyncio. Bridging via a
Unix socket keeps the two runtimes fully isolated.

```
Process 1: frame_daemon (Python 3.10 + rclpy)
  - Subscribes /camera/color/image_raw + /camera/depth/image_raw
  - Runs BPU YOLO11s (~3ms per frame on Nash BPU)
  - Publishes /thunder/detections at 5Hz (JSON string)
  - Writes /tmp/askme_frame_color.bin + depth.bin for direct VisionBridge reads

Process 2: askme_dds_bridge (Python 3.10 + rclpy)
  - Subscribes /thunder/detections + /thunder/estop
  - Accepts Unix socket connections from N clients (fan-out)
  - Broadcasts each DDS message as NDJSON: {topic, data, ts}
  - Socket server runs in background thread; ROS2 executor spins in main thread

Process 3: askme (Python 3.13 + asyncio)
  - Connects to /tmp/askme_dds_bridge.sock via asyncio.open_unix_connection()
  - Reads NDJSON messages, dispatches to topic callbacks
  - DdsBridgeClient auto-reconnects every 2s on disconnect
  - No rclpy dependency anywhere in askme
```

### Latency

End-to-end DDS → Unix socket → askme callback: **~0.4ms** measured in PoC
(`poc_dds_client.py` logs per-message latency from `ts` field to receipt time).

### File-based perception channel

`frame_daemon` also writes frames atomically via `rename()` to:
- `/tmp/askme_frame_color.bin` — raw RGB, width+height prepended as two uint32
- `/tmp/askme_frame_depth.bin` — raw depth
- `/tmp/askme_frame_detections.json` — latest YOLO detections
- `/tmp/askme_frame_daemon.heartbeat` — timestamp of last write

`VisionBridge` in askme reads these files directly at 1Hz for the
`ChangeDetector` pipeline, bypassing the DDS socket.

---

## 5. Topic Specification

Full topic inventory. Status: **Live** = implemented and tested on S100P PoC;
**Planned** = listed in architecture, not yet published.

| Topic | Msg type | Hz | Reliability | Durability | Depth | Publisher | askme consumer | Status |
|---|---|---|---|---|---|---|---|---|
| `/thunder/detections` | `std_msgs/String` (JSON) | 5 | BEST_EFFORT | VOLATILE | 1 | frame_daemon | VisionBridge (via file), DdsBridgeClient | Live |
| `/thunder/estop` | `std_msgs/Bool` | event | RELIABLE | TRANSIENT_LOCAL | 1 | brainstem safety bridge | DdsBridgeClient → IntentRouter | Live |
| `/thunder/heartbeat` | `std_msgs/Bool` | 5 | BEST_EFFORT | VOLATILE | 1 | frame_daemon | DdsBridgeClient | Live |
| `/thunder/joint_states` | `sensor_msgs/JointState` | 50 | BEST_EFFORT | VOLATILE | 1 | brainstem | DdsBridgeClient → telemetry | Planned |
| `/thunder/imu` | `sensor_msgs/Imu` | 100 | BEST_EFFORT | VOLATILE | 1 | brainstem | DdsBridgeClient → telemetry | Planned |
| `/thunder/cms_state` | `std_msgs/String` (JSON) | 2 | RELIABLE | TRANSIENT_LOCAL | 1 | brainstem | DdsBridgeClient → health block | Planned |
| `/camera/color/image_raw` | `sensor_msgs/Image` | 30 | BEST_EFFORT | VOLATILE | 1 | camera driver | frame_daemon (subscribe) | Live |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | 30 | BEST_EFFORT | VOLATILE | 1 | camera driver | frame_daemon (subscribe) | Live |

**Detection JSON schema** (payload of `/thunder/detections`):
```json
{
  "timestamp": 1234567890.123,
  "frame_id": 42,
  "detections": [
    {"label": "person", "confidence": 0.92, "bbox": [x1,y1,x2,y2], "distance_m": 2.3}
  ]
}
```

---

## 6. Coordinate Frames (REP-105)

Standard ROS2 REP-105 frame tree used across the stack:

```
map
 └── odom
      └── base_link
           └── camera_color_optical_frame
```

| Frame | Origin | Used by |
|---|---|---|
| `map` | Fixed world origin, set at mapping start | LingTu SLAM, navigation goals |
| `odom` | Wheel odometry origin, drifts over time | Short-horizon motion estimation |
| `base_link` | Robot body center | brainstem, all sensor transforms |
| `camera_color_optical_frame` | RGB camera optical center (Z forward, X right, Y down) | frame_daemon detections, depth fusion, distance_m in Detection |

Detection `distance_m` values in `/thunder/detections` are computed in
`camera_color_optical_frame` and expressed as Euclidean distance from the
camera origin. Navigation goals are expressed in `map` frame.

---

## 7. Memory Layers

```
L0  Runtime Truth      — live read from dog-safety-service + brainstem gRPC each turn
L1  Working Memory     — rolling JSON conversation, 40 msgs + compression (ConversationManager)
L2  Session Memory     — session summary .md files in data/sessions/ (SessionMemory)
L3  Episodic Memory    — robot experience JSONL + reflection + knowledge (EpisodicMemory)
L4  Long-term (MemU)   — vector DB, semantic search (MemoryBridge)
```

L0 is prepended to the system prompt each turn as a `[运行时状态 HH:MM:SS]` block.
It is non-blocking: reads from a 30-second TTL cache; the cache is warmed by a
background `asyncio.to_thread()` call fired at the start of each turn.

---

## 8. Safety Invariants

These hold regardless of which component triggers them:

1. **E-STOP is tier-0**: `IntentRouter` detects stop keywords before any LLM call.
   ESTOP propagates to `dog-safety-service` via a fire-and-forget daemon thread
   (never blocks the local arm stop).

2. **Skill gate**: `BrainPipeline.execute_skill()` calls `query_estop_state()` via
   `asyncio.to_thread()` before every skill. If E-STOP is active the skill is
   blocked with `[安全锁定]` and the LLM is not called.

3. **askme is not a motion authority**: all motion commands go through
   `dog-safety-service` (gate) and `dog-control-service` (dispatch). askme never
   sends CAN frames or brainstem gRPC motion commands directly.

4. **Single control plane**: nav tasks go through `nav-gateway` → LingTu. askme
   does not call LingTu gRPC directly in production profiles.
