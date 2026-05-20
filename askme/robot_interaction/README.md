# askme.robot_interaction

## Purpose

`askme.robot_interaction` owns human-robot interaction decisions: intent
routing, command/query split, dialog state, address detection, interaction
gating, trace payloads, scenario intent classification, and handoff metadata
for task execution.

This package receives text, optional voice/perception evidence, and routing
policy inputs. It returns interaction or intent decisions for channels,
pipeline, or runtime orchestration to act on.

## Does Not Own

This package must not open devices, call robot clients, execute tasks, select
providers, dispatch tools, run ASR/TTS, or own runtime execution. Use `ports`
and `providers` when an interaction decision needs a concrete robot
capability.

Concrete implementations and execution layers stay outside this package:

- HTTP clients for robot services;
- serial, motor, arm, LED, camera, or telemetry implementations;
- provider selection for a deployment;
- runtime task execution and tool dispatch.

## Public Entrypoints

Use the package facade for interaction decisions:

```python
from askme.robot_interaction import (
    AddressDetector,
    InteractionGate,
    IntentRouter,
    RobotInteractionService,
)
```

Current modules:

- `address_detector.py`: fast address/bystander filtering.
- `interaction_gate.py`: wake, ignore, refuse, clarify, answer, defer, and
  record decisions.
- `intent_router.py`: deterministic intent classification.
- `observability.py`: intent route trace payload helpers.
- `perception_context.py`: normalized perception evidence for the gate.
- `routing_policy.py`: configurable routing policy inputs.
- `scenario_intents.py`: scenario intent classification.
- `service.py`: stable `RobotInteractionService` facade.

## Boundary Rules

Files in this package must not import concrete implementation or execution
layers:

- `askme.robot` or `askme.robot.*`;
- `askme.runtime` or `askme.runtime.*`;
- `askme.providers` or `askme.providers.*`;
- `askme.pipeline` or `askme.pipeline.*`;
- `askme.tools` or `askme.tools.*`;
- `askme.mcp` or `askme.mcp.*`;
- `askme.api` or `askme.api.*`.

Allowed dependencies are local `askme.robot_interaction.*` modules, stable
contracts such as `askme.ports`, and narrow facade calls to
`askme.voice_gateway` when channel composition needs them.

`VoiceGatewayService` belongs to `askme.voice_gateway`. `VoiceTurnBridgePort`
belongs to `askme.ports`.

## Common Changes

- Update `AddressDetector` for direct-address, robot-name, question, command,
  or bystander speech heuristics.
- Update `InteractionGate` for wake/ignore/refuse/clarify/defer policy.
- Update `IntentRouter`, `RoutingPolicy`, or `scenario_intents.py` for
  deterministic route categories.
- Update `observability.py` when trace payload fields need to change.
- Update `RobotInteractionService` only when callers need a stable facade over
  routing behavior.

## Verification

Run the robot interaction lane tests after changing this package:

```powershell
pytest tests\test_interaction_gate.py tests\test_address_detector.py tests\test_contract_voice_gate.py tests\test_six_layer_package_boundaries.py -q
```

For README-only boundary edits, run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

Legacy `askme.interaction` imports remain facades. Historical voice-layer
modules such as `askme.voice.address_detector` and
`askme.voice.interaction_gate` should stay thin compatibility paths; new
interaction behavior belongs in `askme.robot_interaction`.

Example flow:

```text
"go to the entrance and look around"
  -> robot_interaction decides this is a task/command
  -> pipeline plans or routes it
  -> runtime/tools request a port
  -> providers choose concrete robot implementation
  -> robot client talks to hardware/service
```
