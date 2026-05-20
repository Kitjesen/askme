# Robot package

## Role

`askme.robot` owns concrete robot, hardware, telemetry, and robot-service
clients. It answers "how do we talk to this robot or hardware service?"

`askme.robot` is split by robot responsibility:

- `arm/`: standalone arm controller, direct joint commands, policy runner, serial bridge, and local arm safety checker.
- `dog/`: Thunder dog control/safety clients and runtime health aggregation.
- `indicators/`: LED controller and state-to-LED bridge.
- `telemetry/`: OTA latency metrics, pulse event bus, mock pulse, and pubsub base.

## What Does Not Belong Here

- user intent routing;
- wake/ignore/refuse decisions;
- product blueprint composition;
- runtime module orchestration;
- customer-facing skill or scenario policy.

Those belong in `robot_interaction`, `blueprints`, `runtime`, `pipeline`, or
`skills`.

## How Upper Layers Should Use This Package

Upper layers should normally not import concrete robot clients directly. Add or
use a port under `askme.ports`, then expose the concrete robot implementation
through `askme.providers`.

New code should import from the owner package, for example:

```python
from askme.robot.dog import DogControlClient
from askme.robot.telemetry import OTABridgeMetrics
```

Legacy imports remain available while callers are migrated.
