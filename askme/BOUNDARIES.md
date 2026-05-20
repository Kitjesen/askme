# Package Boundaries

This file is the quick map for deciding where robot-facing code belongs.

## One-Line Rule

```text
robot_interaction decides who is interacting and what the user wants.
ports define what upper layers may ask for.
interfaces define legacy backend registries and ABC contracts.
providers choose concrete implementations for this deployment.
robot implements concrete robot, hardware, and telemetry clients.
```

## Where To Put Code

| You are changing... | Start here | Do not start here |
| --- | --- | --- |
| User intent, address detection, interaction gate, wake/ignore/refuse decisions | `robot_interaction/` | `voice/`, `robot/`, `providers/` |
| A stable capability contract used by runtime/pipeline | `ports/` | `robot/` |
| Legacy backend registry or ABC contract | `interfaces/` | `runtime/`, `robot/`, `voice/`, `perception/` |
| Which implementation a deployment should use | `providers/` | `runtime/`, `pipeline/` |
| HTTP/SDK/serial/client code that talks to robot services or hardware | `robot/` | `robot_interaction/` |
| Runtime module wiring | `runtime/modules/` | `robot/` |
| Mechanical arm tool or MCP control | `ports/arm_control.py`, `providers/arm_control.py` | `robot/arm/` from upper layers |

## Dependency Direction

Allowed high-level direction:

```text
blueprints
  -> runtime / pipeline / voice / robot_interaction
  -> ports / interfaces
  -> providers
  -> robot / perception / external SDKs / hardware services
```

Important constraints:

- `robot_interaction` may depend on policies, text routing, and ports.
- `robot_interaction` must not import concrete robot clients.
- `providers` may import `ports` and concrete adapter packages such as `robot`.
- `providers` must not import `runtime`, `pipeline`, `blueprints`, `api`,
  `voice_gateway`, or `robot_interaction`.
- `ports` must not import providers, robot clients, SDKs, or runtime modules.
- `interfaces` must not import concrete `robot`, `voice`, or `perception`
  implementations or runtime modules; `interfaces/register_defaults.py` is
  only a compatibility startup facade and should delegate concrete registration
  to the owning layer.
- `runtime` and `pipeline` should depend on ports for provider-backed
  capabilities.
- Shared runtime telemetry such as OTA metrics belongs in `telemetry`, not in
  `robot`; robot OTA import paths are compatibility facades only.

## Current Provider-Backed Capabilities

| Capability | Port | Provider | Concrete implementation |
| --- | --- | --- | --- |
| Mechanical arm direct control and safety defaults | `ports/arm_control.py` | `providers/arm_control.py` | `robot/arm/arm_controller.py`, `robot/arm/safety.py` |
| Robot control | `ports/robot_control.py` | `providers/robot_control.py` | `robot/dog/control_client.py` |
| Safety / E-STOP | `ports/safety.py` | `providers/safety.py` | `robot/dog/safety_client.py` |
| Status LED | `ports/led.py` | `providers/led.py` | `robot/indicators/*` |
| Pub/sub data bus | `interfaces/bus.py` | `providers/telemetry/bus.py`, `providers/register_defaults.py` | `robot/telemetry/pulse.py`, `robot/telemetry/mock_pulse.py` |
| Perception | `ports/perception.py` | `providers/perception.py` | `perception/*` |
| Scene intelligence | `ports/perception.py` | `providers/perception.py` | `perception/scene_intelligence.py` |
| Audio frontend / ASR / TTS | `ports/voice.py` | `providers/voice.py` | `voice/orchestration`, `voice/input`, `voice/output` |
| Edge voice I/O | `ports/voice.py` | `providers/voice.py` | `voice/input`, `voice/output` |
| Voice runtime bridge | `ports/voice.py` | `providers/voice.py` / `providers/voice_runtime.py` | external askme-edge-service HTTP API |

## How To Read A Flow

For robot control:

```text
PipelineModule
  -> RobotControlPort
  -> providers.build_robot_control()
  -> DogControlClient
  -> dog-control service / hardware gateway
```

For user interaction:

```text
Voice/Text input
  -> AddressDetector / InteractionGate / RobotInteractionService / IntentRouter
  -> pipeline / skill routing
  -> runtime handoff or tool execution
  -> ports
  -> providers
  -> robot clients
```

If a file needs to cross one of these arrows in the opposite direction, stop and
add a port or a provider factory instead.

For voice input/output:

```text
VoiceModule / TextModule
  -> AudioFrontendPort / AudioRouterPort
  -> providers.build_audio_frontend()
  -> AudioAgent / AudioRouter / ASRManager / TTSEngine
  -> microphone, speaker, local/cloud speech providers
```

For runtime turn handoff:

```text
VoiceGatewayService
  -> VoiceTurnBridgePort
  -> providers.build_voice_runtime_bridge()
  -> providers.voice_runtime.VoiceRuntimeBridge
  -> external askme-edge-service
```
