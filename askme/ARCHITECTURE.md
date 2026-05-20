# AskMe Package Architecture

AskMe is a product platform, not a script collection. Package layout must make
ownership and runtime boundaries obvious before a customer project is added.

For day-to-day code changes, start with `CODE_MAP.md`. It maps common change
types to owner packages, explains how the current package layout corresponds to
the six-layer voice-robot architecture, and lists the first files and tests to
use before editing.

## Top-Level Rules

- `api`: HTTP routes and request/response service adapters.
- `runtime`: module lifecycle, dependency wiring, task runtime, and handoff.
- `pipeline`: turn orchestration and field-operation workflows.
- `voice_gateway`: unified voice turn boundary and runtime bridge service.
- `robot_interaction`: intent routing, interaction traces, and interaction API.
- `ports`: interface contracts consumed above providers and hardware. Ports
  must not import provider implementations.
- `providers`: bottom-layer adapter factories for cloud, local, and hardware
  implementations. Providers may import ports and edge clients, but not
  product, runtime, pipeline, API, or interaction layers.
- `voice`: audio input/output, human interaction gate, and voice diagnostics.
- `memory`: knowledge, evidence, retrieval, indexing, and memory backends.
- `skills`: customer-visible capability packages and skill governance.
- `tools`: callable tool implementations exposed to agents and runtimes.
- `robot`: hardware/control adapters and robot-side safety clients.
- `perception`: camera/sensor perception bridges and environment snapshots.
- `space`: site map, semantic place, route, and managed-object knowledge.
- `llm`: provider adapters, model routing, and prompt rendering.
- `contracts`: stable schemas shared across modules.
- `compat`: shared compatibility helpers for staged package migrations.
- `audit`: append-only evidence, review, export, and query.

## Subpackage Rule

When a directory has more than about 8 implementation files or mixes runtime
responsibilities, split it into responsibility subpackages and leave only:

- `__init__.py`
- `README.md`
- responsibility subdirectories

Historical imports may be preserved with a compatibility alias in `__init__.py`,
but new code must use canonical imports from the responsibility subpackage.

## Current Canonical Examples

```python
from askme.voice.output.tts import TTSEngine
from askme.pipeline.field.field_operations import FieldOperationsService
from askme.pipeline.channels.voice_loop import VoiceLoop
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.tools.robot.move_tool import MoveRobotTool
from askme.memory.retrieval.bridge import MemoryBridge
from askme.ports import RobotControlPort
from askme.voice_gateway import VoiceGatewayService
from askme.robot_interaction import IntentRouter, RobotInteractionService
from askme.skills.core.skill_manager import SkillManager
from askme.skills.contracts.contracts import registered_skill_contracts
from askme.runtime.core.module import Runtime
from askme.runtime.task.handoff import RuntimeHandoffService
from askme.cognition.planning import CognitivePlanner
from askme.cognition.world import WorldStateService
```

## Perception And Cognition Boundary

`askme.cognition.world.WorldStateService` is the product world-state source used by
planning, safety preflight, and runtime handoff. `askme.perception.WorldState`
is only a perception-side tracking cache for observed objects.

Cross-boundary writes must go through `CognitionPerceptionSync`:

- Pulse `DetectionFrame` is the primary live sensor path.
- `ChangeEvent` JSONL remains a compatibility fallback.
- Perception world-state snapshots are internal sync input and must preserve
  `observed_at`/`last_seen` freshness before they enter cognition.
- `SceneIntelligence` is a memory/read API, not a live world-state owner.

Do not let LLM, tools, or perception modules write planner facts directly.

## Forbidden Growth Pattern

Do not add another flat file to a crowded package root because it is faster.
Choose the owner subpackage first. If no owner exists, create the owner package
and document it in that package's `README.md`.
