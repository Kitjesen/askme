# askme.ports

## Purpose

`askme.ports` defines application-facing contracts that separate middle-layer
code from concrete hardware, SDK, and provider implementations. It should
contain small `Protocol` classes and narrow shared data contracts, not concrete
implementations.

Ports define what upper layers may ask for. Providers implement those
contracts.

## Does Not Own

Do not add business flow, provider selection, runtime lifecycle, hardware
clients, SDK clients, or transport adapters here. A port should say "what can
be requested", not "when should this happen" or "which implementation should
serve it".

## Public Entrypoints

Use the package facade for stable contract imports:

```python
from askme.ports import RobotControlPort, VoiceTurnBridgePort
```

Current contract modules:

- `runtime_executor/`: provider-neutral DTOs, typed failures, and the transport
  protocol for supervised external task execution.
- `speech_playback/`: targeted speech job, artifact, actor, priority, and
  provider-neutral playback lifecycle contracts.
- `arm_control.py`: `ArmControlPort` for mechanical-arm action, state, E-STOP,
  and lifecycle.
- `led.py`: `LedControllerPort` and `LedBridgePort` for status LED output and
  bridge tasks.
- `perception.py`: `VisionPort`, `InteractionPerceptionPort`,
  `ChangeMonitorPort`, and `SceneIntelligencePort`.
- `robot_control.py`: `RobotControlPort` for robot command/capability
  dispatch.
- `safety.py`: `SafetyPort` for E-STOP state and notification.
- `voice.py`: `AudioFrontendPort`, `AudioRouterPort`, `ASRProviderPort`,
  `TTSProviderPort`, `RealtimeVoiceFrontendPort`, `RealtimeApprovalPort`,
  `VoiceTurnBridgePort`, and `VoiceIOPort`. The realtime pair is the explicit
  optional S2S boundary; pipeline code must not discover those methods with
  ad-hoc `getattr` checks.

## Boundary Rules

- Ports may be imported by runtime, pipeline, API, MCP, tools, voice gateway,
  robot interaction, and providers.
- Ports must not import provider implementations, concrete robot clients,
  runtime modules, SDK packages, API/MCP transports, or product workflows.
- Keep protocols small enough that tests can provide fakes without constructing
  real hardware or cloud clients.
- Add a new port only when more than one upper-layer caller needs to depend on
  a capability that is currently hidden inside a concrete client.

## Common Changes

- Add a method to an existing port when it is part of the same stable
  capability contract.
- Add a new port module when the capability is a separate dependency boundary.
- Update `askme.ports.__init__` when a contract becomes public facade API.
- Coordinate with the provider owner before changing a port method signature;
  provider implementations and upper-layer fakes must move together.

## Verification

Run boundary and contract tests after changing this package:

```powershell
pytest tests\test_six_layer_package_boundaries.py tests\test_contract_voice_gate.py tests\test_product_contracts.py -q
```

For README-only boundary edits, run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

Ports are the migration target for code that previously imported concrete
provider, robot, voice, or perception implementations directly. Keep legacy
compatibility facades thin; do not move implementation code into `ports` to
preserve an old import path.
